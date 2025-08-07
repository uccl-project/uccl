#include "uccl_bench.hpp"
#include "uccl_proxy.hpp"
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#ifdef ENABLE_PROXY_CUDA_MEMCPY
#include "peer_copy_manager.hpp"
#endif
#include "bench_utils.hpp"
#include "py_cuda_shims.hpp"
#include "ring_buffer.cuh"

namespace py = pybind11;

PYBIND11_MODULE(gpu_driven, m) {
  m.doc() = "Python bindings for RDMA proxy and granular benchmark control";
  m.def("alloc_cmd_ring", &alloc_cmd_ring);
  m.def("free_cmd_ring", &free_cmd_ring);
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
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") +
                               cudaGetErrorString(st));
  });
  m.def("set_device", [](int dev) {
    auto st = cudaSetDevice(dev);
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                               cudaGetErrorString(st));
  });
  m.def("get_device", []() {
    int dev;
    auto st = cudaGetDevice(&dev);
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                               cudaGetErrorString(st));
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
    if (st != cudaSuccess)
      throw std::runtime_error(std::string("cudaDeviceReset failed: ") +
                               cudaGetErrorString(st));
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
           py::arg("should_abort") = nullptr)
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
             for (py::handle h : proxy_list)
               vec.push_back(h.cast<UcclProxy*>());
             mgr.start_for_proxies(vec);
           })
      .def("stop", &PeerCopyManager::stop);
#endif
}