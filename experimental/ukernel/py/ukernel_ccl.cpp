#include "../src/ccl/coll_config.h"
#include "../src/ccl/coll_types.h"
#include "../src/ccl/executor.h"
#include <c10/cuda/CUDAStream.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

namespace UKernel {
namespace CCL {
namespace Python {

namespace {

torch::Tensor tensor_from_python(nb::handle obj, char const* arg_name) {
  PyObject* py_obj = obj.ptr();
  if (!THPVariable_Check(py_obj)) {
    throw std::invalid_argument(std::string(arg_name) +
                                " must be a torch.Tensor");
  }
  return THPVariable_Unpack(py_obj);
}

ScalarType to_scalar_type(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kUInt8:   return ScalarType::UInt8;
    case torch::kInt8:    return ScalarType::Int8;
    case torch::kInt16:   return ScalarType::Int16;
    case torch::kInt32:   return ScalarType::Int32;
    case torch::kInt64:   return ScalarType::Int64;
    case torch::kFloat16: return ScalarType::Float16;
    case torch::kFloat32: return ScalarType::Float32;
    case torch::kFloat64: return ScalarType::Float64;
    case torch::kBFloat16: return ScalarType::BFloat16;
    case torch::kBool:    return ScalarType::Bool;
    default: break;
  }
  throw std::invalid_argument("unsupported torch dtype for ukernel collective");
}

ReductionKind to_reduction(uint32_t v) {
  auto r = static_cast<ReductionKind>(v);
  switch (r) {
    case ReductionKind::Sum:
    case ReductionKind::Prod:
    case ReductionKind::Max:
    case ReductionKind::Min:
    case ReductionKind::BitwiseAnd:
      return r;
    default: break;
  }
  throw std::invalid_argument("unsupported reduction kind");
}

void validate_arithmetic_dtype(torch::ScalarType dtype, char const* what) {
  switch (dtype) {
    case torch::kInt8:  case torch::kInt32: case torch::kInt64:
    case torch::kFloat16: case torch::kFloat32: case torch::kFloat64:
    case torch::kBFloat16:
      return;
    default: break;
  }
  throw std::invalid_argument(
      std::string(what) +
      " supports int8/int32/int64/fp16/fp32/fp64/bf16");
}

// prepare() does peer setup, MR (re)registration and buffer resolution —
// expensive, and only needed once per (shape, pointer) combination.
std::string prepare_key(CollectiveConfig const& cfg, void const* input,
                        void const* output) {
  std::string k;
  k.reserve(96);
  auto add = [&k](uint64_t v) {
    k.append(reinterpret_cast<char const*>(&v), sizeof(v));
  };
  add(static_cast<uint64_t>(cfg.kind));
  add(cfg.input_bytes);
  add(cfg.output_bytes);
  add(cfg.tile_bytes);
  add(cfg.input_split_bytes.size());
  for (size_t v : cfg.input_split_bytes) add(v);
  add(cfg.output_split_bytes.size());
  for (size_t v : cfg.output_split_bytes) add(v);
  add(cfg.signal_group_tiles);
  add(reinterpret_cast<uintptr_t>(input));
  add(reinterpret_cast<uintptr_t>(output));
  return k;
}

}  // namespace

// The async methods mirror the SprayExecutor application API 1:1
// (submit -> handle, poll / wait / status / error_message, release);
// the sync collectives are thin submit+wait+release convenience
// wrappers. Contract for async use: the tensor must stay alive and
// unmodified until wait() observes completion, and every handle must be
// released exactly once (releasing a running handle raises, mirroring
// the C++ logic_error).
class ProcessGroup {
 public:
  ProcessGroup(int rank, int world_size, int gpu_id,
               std::string exchanger_ip = "127.0.0.1",
               int exchanger_port = 16998,
               int threads_per_block = 64,
               int blocks_per_worker = 1,
               size_t smem_size = 4096)
      : rank_(rank), world_size_(world_size), gpu_id_(gpu_id) {
    if (world_size_ < 2)
      throw std::invalid_argument("world_size must be >= 2");
    if (rank_ < 0 || rank_ >= world_size_)
      throw std::invalid_argument("rank out of range");

    SprayExecutorConfig cfg;
    cfg.gpu_id = gpu_id;
    cfg.rank = rank;
    cfg.world_size = world_size;
    cfg.exchanger_ip = exchanger_ip;
    cfg.exchanger_port = exchanger_port;
    // The exchanger elects a node leader by local_id; per-GPU index is the
    // per-node ordinal (distinct GPUs same-node, each node leading with 0
    // cross-node).
    cfg.local_id = gpu_id;
    cfg.threads_per_block = threads_per_block;
    cfg.blocks_per_worker = blocks_per_worker;
    cfg.smem_size = smem_size;
    cfg.max_device_fifos = 1;
    // Torch coexistence: let the persistent worker kernels exit after
    // 1ms of fifo idleness (relaunched on the next enqueue), so torch's
    // device-wide syncs (torch.cuda.synchronize(), .item(), D2H) can
    // pass between collectives. Bursts keep the kernel resident.
    cfg.device_idle_exit_us = 1000;
    {
      // Exchanger connect + peer meta exchange can block for seconds;
      // let Python handle signals meanwhile.
      nb::gil_scoped_release release;
      executor_ = SprayExecutor::create(cfg);
    }
  }

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int gpu_id() const { return gpu_id_; }

  uint64_t allreduce_submit(
      nb::handle tensor_handle,
      uint32_t reduction = static_cast<uint32_t>(ReductionKind::Sum),
      size_t tile_bytes = 64ull << 10, uint32_t signal_group_tiles = 1) {
    return allreduce_submit_tensor(
        tensor_from_python(tensor_handle, "tensor"), reduction, tile_bytes,
        signal_group_tiles);
  }

  uint64_t alltoall_submit(nb::handle tensor_handle,
                           size_t tile_bytes = 64ull << 10,
                           uint32_t signal_group_tiles = 1) {
    auto tensor = tensor_from_python(tensor_handle, "tensor");
    require_cuda_contiguous(tensor, "tensor");
    validate_arithmetic_dtype(tensor.scalar_type(), "alltoall");

    auto flat = tensor.view({-1});
    ScalarType dtype = to_scalar_type(tensor.scalar_type());
    size_t elem_bytes = static_cast<size_t>(flat.element_size());
    size_t bytes = static_cast<size_t>(flat.numel()) * elem_bytes;
    if (bytes % (static_cast<size_t>(world_size_) * elem_bytes) != 0)
      throw std::invalid_argument(
          "equal-split alltoall requires tensor bytes divisible by "
          "world_size * dtype_size");

    CollectiveConfig cfg;
    cfg.nranks = world_size_;
    cfg.rank = rank_;
    cfg.input_bytes = bytes;
    cfg.output_bytes = bytes;
    cfg.tile_bytes = tile_bytes;
    cfg.kind = CollKind::AllToAllPairwise;
    cfg.dtype = dtype;
    cfg.signal_group_tiles = signal_group_tiles;

    void* ptr = flat.data_ptr();
    return submit_collective(cfg, ptr, ptr);
  }

  bool poll(uint64_t h) { return executor_->poll(h); }

  bool wait(uint64_t h, uint64_t timeout_ms = 0) {
    return wait_collective(h, timeout_ms);
  }

  CollectiveOpStatus status(uint64_t h) { return executor_->status(h); }
  std::string error_message(uint64_t h) { return executor_->error_message(h); }
  void release(uint64_t h) { executor_->release(h); }

  void allreduce(nb::handle tensor_handle,
                 uint32_t reduction = static_cast<uint32_t>(ReductionKind::Sum),
                 size_t tile_bytes = 64ull << 10,
                 uint32_t signal_group_tiles = 1) {
    wait_and_release(
        allreduce_submit(tensor_handle, reduction, tile_bytes,
                         signal_group_tiles));
  }

  void alltoall(nb::handle tensor_handle,
                size_t tile_bytes = 64ull << 10,
                uint32_t signal_group_tiles = 1) {
    wait_and_release(alltoall_submit(tensor_handle, tile_bytes,
                                     signal_group_tiles));
  }

  void barrier() {
    if (!barrier_tensor_.defined()) {
      barrier_tensor_ = torch::ones(
          {static_cast<int64_t>(world_size_)},
          torch::TensorOptions()
              .dtype(torch::kFloat32)
              .device(c10::Device(c10::DeviceType::CUDA, gpu_id_)));
    }
    wait_and_release(allreduce_submit_tensor(
        barrier_tensor_, static_cast<uint32_t>(ReductionKind::Sum),
        static_cast<size_t>(barrier_tensor_.numel() *
                            barrier_tensor_.element_size()),
        1));
  }

 private:
  void require_cuda_contiguous(torch::Tensor const& t, char const* name) {
    if (!t.is_cuda() || t.device().index() != gpu_id_)
      throw std::invalid_argument(std::string(name) +
                                  " must be on this process group's GPU");
    if (!t.is_contiguous())
      throw std::invalid_argument(
          std::string(name) +
          " must be contiguous (call .contiguous() yourself first — "
          "collectives run in place and cannot silently copy back)");
  }

  void ensure_prepared(CollectiveConfig const& cfg, void* input,
                       void* output) {
    std::string key = prepare_key(cfg, input, output);
    if (prepared_keys_.find(key) != prepared_keys_.end()) return;
    if (prepared_keys_.size() >= kMaxPrepared) prepared_keys_.clear();
    {
      // prepare() does peer setup, MR (re)registration and buffer
      // resolution, which can block for tens of seconds on a cold
      // shape; let Python handle signals meanwhile (Ctrl+C is
      // delivered when the call returns, not mid-call).
      nb::gil_scoped_release release;
      executor_->prepare(cfg, input, output);
    }
    prepared_keys_.insert(std::move(key));
  }

  uint64_t allreduce_submit_tensor(torch::Tensor const& tensor,
                                   uint32_t reduction, size_t tile_bytes,
                                   uint32_t signal_group_tiles) {
    require_cuda_contiguous(tensor, "tensor");
    validate_arithmetic_dtype(tensor.scalar_type(), "allreduce");

    auto flat = tensor.view({-1});
    ScalarType dtype = to_scalar_type(tensor.scalar_type());
    size_t elem_bytes = static_cast<size_t>(flat.element_size());
    size_t bytes = static_cast<size_t>(flat.numel()) * elem_bytes;
    if (bytes % (static_cast<size_t>(world_size_) * elem_bytes) != 0)
      throw std::invalid_argument(
          "allreduce tensor bytes must be divisible by world_size * "
          "dtype_size");

    CollectiveConfig cfg;
    cfg.nranks = world_size_;
    cfg.rank = rank_;
    cfg.input_bytes = bytes;
    cfg.output_bytes = bytes;
    cfg.tile_bytes = tile_bytes;
    cfg.kind = CollKind::AllReduceRing;
    cfg.dtype = dtype;
    cfg.reduction = to_reduction(reduction);
    cfg.signal_group_tiles = signal_group_tiles;

    void* ptr = flat.data_ptr();
    return submit_collective(cfg, ptr, ptr);
  }

  // prepare (cached) + torch stream fence + executor submit. mu_ covers
  // only this section: SprayExecutor::submit reallocs its internal
  // scratch outside its own locks, while wait/poll/status are
  // documented concurrent-safe — waits are deliberately NOT locked so
  // in-flight collectives can overlap.
  uint64_t submit_collective(CollectiveConfig const& cfg, void* input,
                             void* output) {
    std::lock_guard<std::mutex> lock(mu_);
    ensure_prepared(cfg, input, output);
    // Inherit torch's stream ordering: any kernels the user launched on
    // the current stream (e.g. the in-place op producing this tensor)
    // must finish before our device workers read the buffers. A host
    // sync is the only correct fence with persistent worker kernels —
    // their memory accesses are not stream-ordered with new stream
    // waits.
    cudaError_t serr = cudaStreamSynchronize(
        c10::cuda::getCurrentCUDAStream().stream());
    if (serr != cudaSuccess)
      throw std::runtime_error(
          std::string("stream sync before collective failed: ") +
          cudaGetErrorString(serr));
    return executor_->submit(cfg, input, output);
  }

  // Wait in 20ms slices: release the GIL and check for pending Python
  // signals (Ctrl+C → KeyboardInterrupt) between slices, so a long
  // collective stays interruptible instead of pinning the process.
  // Mirrors SprayExecutor::wait semantics: returns false only on
  // Failed; with timeout_ms > 0 it may return true while still Running
  // (a timeout is not an error — use poll() to test completion). On a
  // pending signal the handle is NOT released: a Running run cannot be
  // released (the C++ release throws logic_error), and propagating the
  // interrupt takes priority.
  bool wait_collective(uint64_t h, uint64_t timeout_ms) {
    auto const deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    bool const use_deadline = timeout_ms > 0;
    for (;;) {
      {
        nb::gil_scoped_release release;
        executor_->wait(h, std::chrono::milliseconds(20));
      }
      if (PyErr_CheckSignals() != 0) throw nb::python_error();
      auto st = executor_->status(h);
      if (st != CollectiveOpStatus::Running)
        return st != CollectiveOpStatus::Failed;
      if (use_deadline && std::chrono::steady_clock::now() >= deadline)
        return true;
    }
  }

  void wait_and_release(uint64_t h) {
    wait_collective(h, 0);
    auto st = executor_->status(h);
    if (st != CollectiveOpStatus::Completed) {
      std::string msg = executor_->error_message(h);
      executor_->release(h);
      throw std::runtime_error(
          "ukernel collective failed" +
          (msg.empty() ? std::string{} : std::string(": ") + msg));
    }
    executor_->release(h);
  }

  static constexpr size_t kMaxPrepared = 64;

  int rank_;
  int world_size_;
  int gpu_id_;
  std::unique_ptr<SprayExecutor> executor_;
  torch::Tensor barrier_tensor_;
  std::unordered_set<std::string> prepared_keys_;
  std::mutex mu_;
};

}  // namespace Python
}  // namespace CCL
}  // namespace UKernel

NB_MODULE(TORCH_EXTENSION_NAME, m) {
  using UKernel::CCL::Python::ProcessGroup;

  nb::enum_<UKernel::CCL::CollectiveOpStatus>(m, "CollectiveOpStatus")
      .value("Queued", UKernel::CCL::CollectiveOpStatus::Queued)
      .value("Running", UKernel::CCL::CollectiveOpStatus::Running)
      .value("Completed", UKernel::CCL::CollectiveOpStatus::Completed)
      .value("Failed", UKernel::CCL::CollectiveOpStatus::Failed)
      .export_values();

  nb::class_<ProcessGroup>(m, "ProcessGroup")
      .def(nb::init<int, int, int, std::string, int, int, int, size_t>(),
           nb::arg("rank"), nb::arg("world_size"), nb::arg("gpu_id"),
           nb::arg("exchanger_ip") = "127.0.0.1",
           nb::arg("exchanger_port") = 16998,
           nb::arg("threads_per_block") = 64,
           nb::arg("blocks_per_worker") = 1,
           nb::arg("smem_size") = 4096)
      .def_prop_ro("rank", &ProcessGroup::rank)
      .def_prop_ro("world_size", &ProcessGroup::world_size)
      .def_prop_ro("gpu_id", &ProcessGroup::gpu_id)
      .def("allreduce_submit",
           [](ProcessGroup& self, nb::handle tensor, uint32_t reduction,
              size_t tile_bytes, uint32_t signal_group_tiles) {
             return self.allreduce_submit(tensor, reduction, tile_bytes,
                                          signal_group_tiles);
           },
           nb::arg("tensor"),
           nb::arg("reduction") = static_cast<uint32_t>(
               UKernel::CCL::ReductionKind::Sum),
           nb::arg("tile_bytes") = 64ull << 10,
           nb::arg("signal_group_tiles") = 1)
      .def("alltoall_submit",
           [](ProcessGroup& self, nb::handle tensor, size_t tile_bytes,
              uint32_t signal_group_tiles) {
             return self.alltoall_submit(tensor, tile_bytes,
                                         signal_group_tiles);
           },
           nb::arg("tensor"), nb::arg("tile_bytes") = 64ull << 10,
           nb::arg("signal_group_tiles") = 1)
      .def("poll", [](ProcessGroup& self, uint64_t h) { return self.poll(h); },
           nb::arg("handle"))
      .def("wait",
           [](ProcessGroup& self, uint64_t h, uint64_t timeout_ms) {
             return self.wait(h, timeout_ms);
           },
           nb::arg("handle"), nb::arg("timeout_ms") = 0)
      .def("status",
           [](ProcessGroup& self, uint64_t h) { return self.status(h); },
           nb::arg("handle"))
      .def("error_message",
           [](ProcessGroup& self, uint64_t h) {
             return self.error_message(h);
           },
           nb::arg("handle"))
      .def("release", [](ProcessGroup& self, uint64_t h) { self.release(h); },
           nb::arg("handle"))
      .def("allreduce",
           [](ProcessGroup& self, nb::handle tensor, uint32_t reduction,
              size_t tile_bytes, uint32_t signal_group_tiles) {
             self.allreduce(tensor, reduction, tile_bytes,
                            signal_group_tiles);
           },
           nb::arg("tensor"),
           nb::arg("reduction") = static_cast<uint32_t>(
               UKernel::CCL::ReductionKind::Sum),
           nb::arg("tile_bytes") = 64ull << 10,
           nb::arg("signal_group_tiles") = 1)
      .def("alltoall",
           [](ProcessGroup& self, nb::handle tensor, size_t tile_bytes,
              uint32_t signal_group_tiles) {
             self.alltoall(tensor, tile_bytes, signal_group_tiles);
           },
           nb::arg("tensor"), nb::arg("tile_bytes") = 64ull << 10,
           nb::arg("signal_group_tiles") = 1)
      .def("barrier", [](ProcessGroup& self) { self.barrier(); });
}
