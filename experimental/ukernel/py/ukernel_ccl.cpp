#include "../src/ccl/coll_config.h"
#include "../src/ccl/coll_types.h"
#include "../src/ccl/executor.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
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

std::vector<size_t> split_bytes_from_elements(
    std::vector<int64_t> const& splits, int world_size,
    size_t elem_size, size_t total_elems, char const* which) {
  if (splits.size() != static_cast<size_t>(world_size)) {
    throw std::invalid_argument(std::string(which) +
                                " count must equal world_size");
  }
  std::vector<size_t> out;
  out.reserve(splits.size());
  size_t sum = 0;
  for (auto s : splits) {
    if (s < 0)
      throw std::invalid_argument(std::string(which) + " must be non-negative");
    size_t bytes = static_cast<size_t>(s) * elem_size;
    sum += static_cast<size_t>(s);
    out.push_back(bytes);
  }
  if (sum != total_elems)
    throw std::invalid_argument(std::string("sum(") + which +
                                ") must equal tensor numel");
  return out;
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

  void allreduce(nb::handle tensor_handle,
                 uint32_t reduction = static_cast<uint32_t>(ReductionKind::Sum),
                 size_t tile_bytes = 64ull << 10,
                 uint32_t signal_group_tiles = 1) {
    auto tensor = tensor_from_python(tensor_handle, "tensor");
    allreduce_internal(tensor, reduction, tile_bytes, signal_group_tiles);
  }

  void alltoall(nb::handle tensor_handle,
                size_t tile_bytes = 64ull << 10,
                uint32_t signal_group_tiles = 1) {
    std::lock_guard<std::mutex> lock(mu_);
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
    submit_and_wait(cfg, ptr, ptr);
  }

  void alltoallv(nb::handle output_handle, nb::handle input_handle,
                 std::vector<int64_t> output_split_sizes,
                 std::vector<int64_t> input_split_sizes,
                 size_t tile_bytes = 64ull << 10,
                 uint32_t signal_group_tiles = 1) {
    std::lock_guard<std::mutex> lock(mu_);
    auto output = tensor_from_python(output_handle, "output");
    auto input = tensor_from_python(input_handle, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(input, "input");
    validate_arithmetic_dtype(output.scalar_type(), "alltoallv");
    validate_arithmetic_dtype(input.scalar_type(), "alltoallv");
    if (output.scalar_type() != input.scalar_type())
      throw std::invalid_argument("output and input must have the same dtype");

    auto out_flat = output.view({-1});
    auto in_flat = input.view({-1});
    ScalarType dtype = to_scalar_type(output.scalar_type());
    size_t elem_bytes = static_cast<size_t>(out_flat.element_size());
    size_t in_bytes = static_cast<size_t>(in_flat.numel()) * elem_bytes;
    size_t out_bytes = static_cast<size_t>(out_flat.numel()) * elem_bytes;

    auto in_split_bytes = split_bytes_from_elements(
        input_split_sizes, world_size_, elem_bytes,
        static_cast<size_t>(in_flat.numel()), "input_split_sizes");
    auto out_split_bytes = split_bytes_from_elements(
        output_split_sizes, world_size_, elem_bytes,
        static_cast<size_t>(out_flat.numel()), "output_split_sizes");

    size_t r = static_cast<size_t>(rank_);
    if (in_split_bytes[r] != out_split_bytes[r])
      throw std::invalid_argument(
          "input_split_sizes[rank] must equal output_split_sizes[rank]");

    CollectiveConfig cfg;
    cfg.nranks = world_size_;
    cfg.rank = rank_;
    cfg.input_bytes = in_bytes;
    cfg.output_bytes = out_bytes;
    cfg.input_split_bytes = std::move(in_split_bytes);
    cfg.output_split_bytes = std::move(out_split_bytes);
    cfg.tile_bytes = tile_bytes;
    cfg.kind = CollKind::AllToAllPairwise;
    cfg.dtype = dtype;
    cfg.signal_group_tiles = signal_group_tiles;

    submit_and_wait(cfg, in_flat.data_ptr(), out_flat.data_ptr());
  }

  void barrier() {
    if (!barrier_tensor_.defined()) {
      barrier_tensor_ = torch::ones(
          {static_cast<int64_t>(world_size_)},
          torch::TensorOptions()
              .dtype(torch::kFloat32)
              .device(c10::Device(c10::DeviceType::CUDA, gpu_id_)));
    }
    allreduce_internal(barrier_tensor_,
                       static_cast<uint32_t>(ReductionKind::Sum),
                       static_cast<size_t>(barrier_tensor_.numel() *
                                           barrier_tensor_.element_size()),
                       1);
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

  void submit_and_wait(CollectiveConfig const& cfg, void* input,
                       void* output) {
    ensure_prepared(cfg, input, output);
    auto h = executor_->submit(cfg, input, output);
    // Wait in slices: release the GIL and check for pending Python
    // signals (Ctrl+C → KeyboardInterrupt) between slices, so a long
    // collective stays interruptible instead of pinning the process.
    for (;;) {
      bool ok;
      {
        nb::gil_scoped_release release;
        ok = executor_->wait(h, std::chrono::milliseconds(20));
      }
      if (PyErr_CheckSignals() != 0) {
        executor_->release(h);
        throw nb::python_error();
      }
      auto st = executor_->status(h);
      if (st == CollectiveOpStatus::Running) continue;
      if (st != CollectiveOpStatus::Completed || !ok) {
        std::string msg = executor_->error_message(h);
        executor_->release(h);
        throw std::runtime_error(
            "ukernel collective failed" +
            (msg.empty() ? std::string{} : std::string(": ") + msg));
      }
      break;
    }
    executor_->release(h);
  }

  void allreduce_internal(torch::Tensor tensor, uint32_t reduction,
                          size_t tile_bytes, uint32_t signal_group_tiles) {
    std::lock_guard<std::mutex> lock(mu_);
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
    submit_and_wait(cfg, ptr, ptr);
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
      .def("alltoallv",
           [](ProcessGroup& self, nb::handle output, nb::handle input,
              std::vector<int64_t> output_split_sizes,
              std::vector<int64_t> input_split_sizes, size_t tile_bytes,
              uint32_t signal_group_tiles) {
             self.alltoallv(output, input,
                            std::move(output_split_sizes),
                            std::move(input_split_sizes), tile_bytes,
                            signal_group_tiles);
           },
           nb::arg("output"), nb::arg("input"),
           nb::arg("output_split_sizes"), nb::arg("input_split_sizes"),
           nb::arg("tile_bytes") = 64ull << 10,
           nb::arg("signal_group_tiles") = 1)
      .def("barrier", [](ProcessGroup& self) { self.barrier(); });
}
