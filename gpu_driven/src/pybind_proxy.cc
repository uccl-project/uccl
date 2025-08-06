#include "bench_utils.hpp"
#include "proxy.hpp"
#include "py_cuda_shims.hpp"
#include "ring_buffer.cuh"
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#ifdef ENABLE_PROXY_CUDA_MEMCPY
#include "peer_copy_worker.hpp"
#endif

namespace py = pybind11;

uintptr_t alloc_cmd_ring() {
  void* raw = nullptr;
  auto err = cudaMallocHost(&raw, sizeof(DeviceToHostCmdBuffer));
  if (err != cudaSuccess || raw == nullptr) {
    throw std::runtime_error("cudaMallocHost(DeviceToHostCmdBuffer) failed");
  }
  auto* rb = static_cast<DeviceToHostCmdBuffer*>(raw);
  new (rb) DeviceToHostCmdBuffer{};
  return reinterpret_cast<uintptr_t>(rb);
}

void free_cmd_ring(uintptr_t addr) {
  if (!addr) return;
  auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(addr);
  rb->~DeviceToHostCmdBuffer();
  auto err = cudaFreeHost(static_cast<void*>(rb));
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaFreeHost(DeviceToHostCmdBuffer) failed");
  }
}

class UcclProxy {
  friend class PeerCopyManager;

 public:
  UcclProxy(uintptr_t rb_addr, int block_idx, uintptr_t gpu_buffer_addr,
            size_t total_size, int rank,
            std::string const& peer_ip = std::string())
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
    proxy_ = std::make_unique<Proxy>(cfg);
  }

  ~UcclProxy() {
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
    {
      py::gil_scoped_release release;
      if (thread_.joinable()) thread_.join();
    }
    running_.store(false, std::memory_order_release);
  }

 private:
  enum class Mode { None, Sender, Remote, Local, Dual };

  void start(Mode m) {
    if (running_.load(std::memory_order_acquire)) {
      throw std::runtime_error("Proxy already running");
    }
    mode_ = m;
    proxy_->set_progress_run(true);
    running_.store(true, std::memory_order_release);

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

class Bench {
 public:
  Bench()
      : running_{false}, have_t0_{false}, have_t1_{false}, done_evt_(nullptr) {
    init_env(env_);
    GPU_RT_CHECK(cudaEventCreateWithFlags(&done_evt_, cudaEventDisableTiming));
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

  ~Bench() {
    try {
      join_proxies();
    } catch (...) {
    }
    if (done_evt_) {
      cudaEventDestroy(done_evt_);
      done_evt_ = nullptr;
    }
    destroy_env(env_);
  }

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

  void start_local_proxies(int rank = 0,
                           std::string const& peer_ip = std::string()) {
    if (running_.load(std::memory_order_acquire)) {
      throw std::runtime_error("Proxies already running");
    }
    threads_.reserve(env_.blocks);
    for (int i = 0; i < env_.blocks; ++i) {
      threads_.emplace_back([this, i, rank, peer_ip]() {
        Proxy p{make_cfg(env_, i, rank,
                         peer_ip.empty() ? nullptr : peer_ip.c_str())};
        p.run_local();
      });
    }
    running_.store(true, std::memory_order_release);
  }

  void launch_gpu_issue_batched_commands() {
    timing_start();
    const size_t shmem_bytes = kQueueSize * 2 * sizeof(unsigned long long);
    py::gil_scoped_release release;
    auto st = launch_gpu_issue_batched_commands_shim(
        env_.blocks, kNumThPerBlock, shmem_bytes, env_.stream, env_.rbs);
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("kernel launch failed: ") +
                               cudaGetErrorString(st));
    }
    GPU_RT_CHECK(cudaEventRecord(done_evt_, env_.stream));
  }

  void sync_stream() {
    py::gil_scoped_release release;
    auto st = cudaStreamSynchronize(env_.stream);
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                               cudaGetErrorString(st));
    }
    timing_stop();
  }

  void sync_stream_interruptible(int poll_ms = 5, long long timeout_ms = -1) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
      {
        py::gil_scoped_release release;
        cudaError_t st = cudaEventQuery(done_evt_);
        if (st == cudaSuccess) break;
        if (st != cudaErrorNotReady) {
          (void)cudaGetLastError();
          throw std::runtime_error(std::string("cudaEventQuery failed: ") +
                                   cudaGetErrorString(st));
        }
      }
      {
        py::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() != 0) {
          throw py::error_already_set();
        }
      }
      if (timeout_ms >= 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
        if (elapsed.count() >= timeout_ms) {
          throw std::runtime_error("Stream sync timed out");
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }
    timing_stop();
  }
  void join_proxies() {
    py::gil_scoped_release release;
    for (auto& t : threads_)
      if (t.joinable()) t.join();
    threads_.clear();
    running_.store(false, std::memory_order_release);
  }

  void print_block_latencies() { ::print_block_latencies(env_); }

  Stats compute_stats() const {
    if (!have_t0_ || !have_t1_) {
      throw std::runtime_error(
          "compute_stats: missing t0/t1. Call launch_* then sync_stream() "
          "first.");
    }
    return ::compute_stats(env_, t0_, t1_);
  }

  void print_summary(Stats const& s) const { ::print_summary(env_, s); }
  void print_summary_last() const { ::print_summary(env_, compute_stats()); }
  double last_elapsed_ms() const {
    if (!have_t0_ || !have_t1_) return 0.0;
    return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
  }

 private:
  BenchEnv env_;
  std::vector<std::thread> threads_;
  std::atomic<bool> running_;
  std::chrono::high_resolution_clock::time_point t0_{}, t1_{};
  bool have_t0_, have_t1_;
  cudaEvent_t done_evt_;
};

#ifdef ENABLE_PROXY_CUDA_MEMCPY
class PeerCopyManager {
 public:
  explicit PeerCopyManager(int src_device = 0) {
    shared_.src_device = src_device;
    shared_.run.store(true, std::memory_order_release);
  }
  ~PeerCopyManager() { stop(); }

  // Start one peer_copy_worker per proxy
  void start_for_proxies(std::vector<UcclProxy*> const& proxies) {
    int const n = static_cast<int>(proxies.size());
    if (n <= 0) return;
    ctxs_.resize(n);
    threads_.reserve(n);
    for (int i = 0; i < n; ++i) {
      // Access the underlying Proxy's ring; UcclProxy declared us as a friend.
      threads_.emplace_back(peer_copy_worker, std::ref(shared_),
                            std::ref(ctxs_[i]),
                            std::ref(proxies[i]->proxy_->ring), i);
    }
  }

  void stop() {
    if (threads_.empty()) return;
    shared_.run.store(false, std::memory_order_release);
    for (auto& t : threads_)
      if (t.joinable()) t.join();
    threads_.clear();
    ctxs_.clear();
  }

 private:
  PeerCopyShared shared_;
  std::vector<PeerWorkerCtx> ctxs_;
  std::vector<std::thread> threads_;
};
#endif
PYBIND11_MODULE(gpu_driven, m) {
  m.doc() = "Python bindings for RDMA proxy and granular benchmark control";
  m.def("alloc_cmd_ring", &alloc_cmd_ring,
        "Allocate pinned DeviceToHostCmdBuffer and return its address");
  m.def("free_cmd_ring", &free_cmd_ring,
        "Destroy and free a pinned DeviceToHostCmdBuffer by address");
  m.def("launch_gpu_issue_kernel", [](int blocks, int threads_per_block,
                                      uintptr_t stream_ptr, uintptr_t rb_ptr) {
    const size_t shmem_bytes = kQueueSize * 2 * sizeof(unsigned long long);
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
    auto st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") +
                               cudaGetErrorString(st));
    }
  });
  m.def("set_device", [](int dev) {
    auto st = cudaSetDevice(dev);
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                               cudaGetErrorString(st));
    }
  });
  m.def("get_device", []() {
    int dev;
    auto st = cudaGetDevice(&dev);
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                               cudaGetErrorString(st));
    }
    return dev;
  });
  m.def("check_stream", [](uintptr_t stream_ptr) {
    auto* s = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaError_t st = cudaStreamQuery(s);
    return std::string(cudaGetErrorString(st));
  });

  m.def(
      "stream_query",
      [](uintptr_t stream_ptr) {
        auto* stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        auto st = cudaStreamQuery(stream);
        if (st == cudaSuccess) return std::string("done");
        if (st == cudaErrorNotReady) return std::string("not_ready");
        return std::string("error: ") + cudaGetErrorString(st);
      },
      py::arg("stream_ptr"));

  m.def("device_reset", []() {
    auto st = cudaDeviceReset();
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaDeviceReset failed: ") +
                               cudaGetErrorString(st));
    }
  });

  py::class_<Stats>(m, "Stats");
  py::class_<UcclProxy>(m, "Proxy")
      .def(py::init<uintptr_t, int, uintptr_t, size_t, int,
                    std::string const&>(),
           py::arg("rb_addr"), py::arg("block_idx"), py::arg("gpu_buffer_addr"),
           py::arg("total_size"), py::arg("rank") = 0,
           py::arg("peer_ip") = std::string())
      .def("start_sender", &UcclProxy::start_sender)
      .def("start_remote", &UcclProxy::start_remote)
      .def("start_local", &UcclProxy::start_local)
      .def("start_dual", &UcclProxy::start_dual)
      .def("stop", &UcclProxy::stop);

  py::class_<Bench>(m, "Bench")
      .def(py::init<>())
      .def("env_info", &Bench::env_info)
      .def("blocks", &Bench::blocks)
      .def("ring_addr", &Bench::ring_addr)
      .def("timing_start", &Bench::timing_start)
      .def("timing_stop", &Bench::timing_stop)
      .def("is_running", &Bench::is_running)
      .def("start_local_proxies", &Bench::start_local_proxies,
           py::arg("rank") = 0, py::arg("peer_ip") = std::string())
      .def("launch_gpu_issue_batched_commands",
           &Bench::launch_gpu_issue_batched_commands)
      .def("sync_stream", &Bench::sync_stream)
      .def("sync_stream_interruptible", &Bench::sync_stream_interruptible,
           py::arg("poll_ms") = 5, py::arg("timeout_ms") = -1,
           "Polls the stream and respects Ctrl-C / Python signals.")
      .def("join_proxies", &Bench::join_proxies)
      .def("print_block_latencies", &Bench::print_block_latencies)
      .def("compute_stats", &Bench::compute_stats)
      .def("print_summary", &Bench::print_summary)
      .def("print_summary_last", &Bench::print_summary_last)
      .def("last_elapsed_ms", &Bench::last_elapsed_ms);

#ifdef ENABLE_PROXY_CUDA_MEMCPY
  py::class_<PeerCopyManager>(m, "PeerCopyManager")
      .def(py::init<int>(), py::arg("src_device") = 0)
      .def("start_for_proxies",
           [](PeerCopyManager& mgr, py::iterable proxy_list) {
             std::vector<UcclProxy*> vec;
             for (py::handle h : proxy_list) {
               vec.push_back(h.cast<UcclProxy*>());
             }
             mgr.start_for_proxies(vec);
           })
      .def("stop", &PeerCopyManager::stop);
#endif
}