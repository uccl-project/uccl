// src/pybind_proxy.cc
#include "bench_utils.hpp"  // BenchEnv, init_env, destroy_env, print_block_latencies, Stats, compute_stats, print_summary
#include "proxy.hpp"
#include "py_cuda_shims.hpp"  // launch_gpu_issue_batched_commands_shim
#include "ring_buffer.cuh"
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <new>  // placement new
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

namespace py = pybind11;

// -------------------- Pinned ring buffer helpers (non-trivial type)
// --------------------
uintptr_t alloc_cmd_ring() {
  void* raw = nullptr;
  auto err = cudaMallocHost(&raw, sizeof(DeviceToHostCmdBuffer));
  if (err != cudaSuccess || raw == nullptr) {
    throw std::runtime_error("cudaMallocHost(DeviceToHostCmdBuffer) failed");
  }
  auto* rb = static_cast<DeviceToHostCmdBuffer*>(raw);
  new (rb) DeviceToHostCmdBuffer{};  // value-initialize non-trivial type
  return reinterpret_cast<uintptr_t>(rb);
}

void free_cmd_ring(uintptr_t addr) {
  if (!addr) return;
  auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(addr);
  rb->~DeviceToHostCmdBuffer();  // explicit dtor (non-trivial type)
  auto err = cudaFreeHost(static_cast<void*>(rb));
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaFreeHost(DeviceToHostCmdBuffer) failed");
  }
}

// -------------------- Thin Proxy wrapper (threaded) --------------------
class PyProxy {
 public:
  PyProxy(uintptr_t rb_addr, int block_idx, uintptr_t gpu_buffer_addr,
          size_t total_size, int rank, std::string const& peer_ip,
          bool pin_thread)
      : peer_ip_storage_{peer_ip},
        thread_{},
        mode_{Mode::None},
        running_{false} {
    Proxy::Config cfg;
    cfg.rb = reinterpret_cast<DeviceToHostCmdBuffer*>(rb_addr);
    cfg.block_idx = block_idx;
    cfg.gpu_buffer = reinterpret_cast<void*>(gpu_buffer_addr);
    cfg.total_size = total_size;
    cfg.rank = rank;
    cfg.peer_ip = peer_ip_storage_.empty() ? nullptr : peer_ip_storage_.c_str();
    cfg.pin_thread = pin_thread;
    proxy_ = std::make_unique<Proxy>(cfg);
  }

  ~PyProxy() {
    try {
      stop();
    } catch (...) {
    }
  }

  void start_sender() { start(Mode::Sender); }
  void start_remote() { start(Mode::Remote); }
  void start_local() { start(Mode::Local); }
  void start_dual() { start(Mode::Dual); }

  void stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    proxy_->set_progress_run(false);
    // Release the GIL while potentially blocking on join
    {
      py::gil_scoped_release release;
      if (thread_.joinable()) thread_.join();
    }
    running_.store(false, std::memory_order_release);
  }

  double avg_rdma_write_us() const { return proxy_->avg_rdma_write_us(); }
  double avg_wr_latency_us() const { return proxy_->avg_wr_latency_us(); }
  uint64_t completed_wr() const { return proxy_->completed_wr(); }

 private:
  enum class Mode { None, Sender, Remote, Local, Dual };

  void start(Mode m) {
    if (running_.load(std::memory_order_acquire)) {
      throw std::runtime_error("Proxy already running");
    }
    mode_ = m;
    proxy_->set_progress_run(true);
    running_.store(true, std::memory_order_release);

    // Do NOT use gil_scoped_release in this new native thread
    thread_ = std::thread([this]() {
      switch (mode_) {
        case Mode::Sender:
          proxy_->run_sender();
          break;
        case Mode::Remote:
          proxy_->run_remote();
          break;
        case Mode::Local:
          proxy_->run_local();
          break;
        case Mode::Dual:
          proxy_->run_dual();
          break;
        default:
          break;
      }
    });
  }

  std::string peer_ip_storage_;
  std::unique_ptr<Proxy> proxy_;
  std::thread thread_;
  Mode mode_;
  std::atomic<bool> running_;
};

// -------------------- Bench wrapper with granular API + stats/printing
// --------------------
class PyBench {
 public:
  PyBench() : running_{false}, have_t0_{false}, have_t1_{false} {
    init_env(env_);  // sets env_.blocks, env_.stream, env_.rbs, etc.
  }

  void timing_start() {
    t0_ = std::chrono::high_resolution_clock::now();
    have_t0_ = true;
  }
  void timing_stop() {
    t1_ = std::chrono::high_resolution_clock::now();
    have_t1_ = true;
  }

  uintptr_t ring_addr(int i) const {
    if (i < 0 || i >= env_.blocks) throw std::out_of_range("ring index");
    return reinterpret_cast<uintptr_t>(&env_.rbs[i]);
  }

  ~PyBench() {
    try {
      join_proxies();
    } catch (...) {
    }
    destroy_env(env_);
  }

  // --- Introspection / helpers ---
  py::dict env_info() const {
    py::dict d;
    d["blocks"] = env_.blocks;
    d["queue_size"] = kQueueSize;
    d["threads_per_block"] = kNumThPerBlock;
    d["iterations"] = kIterations;
    d["stream_addr"] = reinterpret_cast<uintptr_t>(env_.stream);
    d["rbs_addr"] = reinterpret_cast<uintptr_t>(env_.rbs);
    return d;
  }
  int blocks() const { return env_.blocks; }
  bool is_running() const { return running_.load(std::memory_order_acquire); }

  // --- Control plane: start proxies (LOCAL) ---
  void start_local_proxies(int rank = 0,
                           std::string const& peer_ip = std::string(),
                           bool pin_thread = true) {
    if (running_.load(std::memory_order_acquire)) {
      throw std::runtime_error("Proxies already running");
    }
    threads_.reserve(env_.blocks);
    for (int i = 0; i < env_.blocks; ++i) {
      threads_.emplace_back([this, i, rank, peer_ip, pin_thread]() {
        Proxy p{
            make_cfg(env_, i, /*rank*/ rank,
                     /*peer_ip*/ peer_ip.empty() ? nullptr : peer_ip.c_str())};
        p.run_local();  // consumes exactly kIterations commands for block i
      });
    }
    running_.store(true, std::memory_order_release);
  }

  // --- Data plane: launch the producer kernel ---
  void launch_gpu_issue_batched_commands() {
    // Mimic your benchmark: record t0_ before launch
    t0_ = std::chrono::high_resolution_clock::now();
    have_t0_ = true;

    const size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
    py::gil_scoped_release release;
    auto st = launch_gpu_issue_batched_commands_shim(
        env_.blocks, kNumThPerBlock, shmem_bytes, env_.stream, env_.rbs);
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("kernel launch failed: ") +
                               cudaGetErrorString(st));
    }
  }

  // --- Synchronize the stream (records t1 like your benchmark) ---
  void sync_stream() {
    py::gil_scoped_release release;
    auto st = cudaStreamSynchronize(env_.stream);
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                               cudaGetErrorString(st));
    }
    t1_ = std::chrono::high_resolution_clock::now();
    have_t1_ = true;
  }

  // --- Join all proxies (blocks until run_local finishes kIterations per
  // block) ---
  void join_proxies() {
    py::gil_scoped_release release;
    for (auto& t : threads_)
      if (t.joinable()) t.join();
    threads_.clear();
    running_.store(false, std::memory_order_release);
  }

  // --- Printing APIs you asked for ---
  void print_block_latencies() { ::print_block_latencies(env_); }

  // compute_stats using last t0_/t1_ (mirrors benchmark_main timing)
  Stats compute_stats() const {
    if (!have_t0_ || !have_t1_) {
      throw std::runtime_error(
          "compute_stats: missing t0/t1. Call launch_* then sync_stream() "
          "first.");
    }
    return ::compute_stats(env_, t0_, t1_);
  }

  // Print summary using a provided Stats (Python will pass the Stats you got
  // from compute_stats())
  void print_summary(Stats const& s) const { ::print_summary(env_, s); }

  // Convenience: compute + print in one call, using last t0/t1
  void print_summary_last() const { ::print_summary(env_, compute_stats()); }

  // Convenience: return only elapsed time (ms) for last t0/t1
  double last_elapsed_ms() const {
    if (!have_t0_ || !have_t1_) return 0.0;
    return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
  }

 private:
  BenchEnv env_;
  std::vector<std::thread> threads_;
  std::atomic<bool> running_;

  // timing like your benchmark main
  std::chrono::high_resolution_clock::time_point t0_{}, t1_{};
  bool have_t0_, have_t1_;
};

// -------------------- pybind11 module --------------------
PYBIND11_MODULE(pyproxy, m) {
  m.doc() = "Python bindings for RDMA proxy and granular benchmark control";

  // Pinned ring helpers (useful for standalone PyProxy use)
  m.def("alloc_cmd_ring", &alloc_cmd_ring,
        "Allocate pinned DeviceToHostCmdBuffer and return its address");
  m.def("free_cmd_ring", &free_cmd_ring,
        "Destroy and free a pinned DeviceToHostCmdBuffer by address");
  m.def("launch_gpu_issue_kernel", [](int blocks, int threads_per_block,
                                      uintptr_t stream_ptr, uintptr_t rb_ptr) {
    const size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
    auto* stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto* rbs = reinterpret_cast<DeviceToHostCmdBuffer*>(rb_ptr);
    auto st = launch_gpu_issue_batched_commands_shim(blocks, threads_per_block,
                                                     shmem_bytes, stream, rbs);
    if (st != cudaSuccess) {
      throw std::runtime_error("Kernel launch failed: " +
                               std::string(cudaGetErrorString(st)));
    }
  });
  m.def("sync_stream", []() {
    auto st = cudaDeviceSynchronize();  // Or use a shared stream if needed
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") +
                               cudaGetErrorString(st));
    }
  });

  // Opaque Stats type (so we can pass/return it without exposing fields)
  py::class_<Stats>(m, "Stats");

  // Fine-grained per-proxy wrapper (optional if you want manual control)
  py::class_<PyProxy>(m, "Proxy")
      .def(py::init<uintptr_t, int, uintptr_t, size_t, int, std::string const&,
                    bool>(),
           py::arg("rb_addr"), py::arg("block_idx"), py::arg("gpu_buffer_addr"),
           py::arg("total_size"), py::arg("rank"), py::arg("peer_ip"),
           py::arg("pin_thread") = true)
      .def("start_sender", &PyProxy::start_sender)
      .def("start_remote", &PyProxy::start_remote)
      .def("start_local", &PyProxy::start_local)
      .def("start_dual", &PyProxy::start_dual)
      .def("stop", &PyProxy::stop)
      .def("avg_rdma_write_us", &PyProxy::avg_rdma_write_us)
      .def("avg_wr_latency_us", &PyProxy::avg_wr_latency_us)
      .def("completed_wr", &PyProxy::completed_wr);

  // Granular bench control + printing/stats
  py::class_<PyBench>(m, "Bench")
      .def(py::init<>())
      .def("env_info", &PyBench::env_info)
      .def("blocks", &PyBench::blocks)
      .def("ring_addr", &PyBench::ring_addr)
      .def("timing_start", &PyBench::timing_start)
      .def("timing_stop", &PyBench::timing_stop)
      .def("is_running", &PyBench::is_running)
      .def("start_local_proxies", &PyBench::start_local_proxies,
           py::arg("rank") = 0, py::arg("peer_ip") = std::string(),
           py::arg("pin_thread") = true)
      .def("launch_gpu_issue_batched_commands",
           &PyBench::launch_gpu_issue_batched_commands)
      .def("sync_stream", &PyBench::sync_stream)
      .def("join_proxies", &PyBench::join_proxies)
      .def("print_block_latencies", &PyBench::print_block_latencies)
      .def("compute_stats", &PyBench::compute_stats)  // returns Stats
      .def("print_summary", &PyBench::print_summary)  // takes Stats
      .def("print_summary_last", &PyBench::print_summary_last)
      .def("last_elapsed_ms", &PyBench::last_elapsed_ms);
}