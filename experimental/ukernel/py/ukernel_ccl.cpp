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
#include <thread>
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

void validate_allreduce_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kInt8:  case torch::kInt32: case torch::kInt64:
    case torch::kFloat16: case torch::kFloat32: case torch::kFloat64:
    case torch::kBFloat16:
      return;
    default: break;
  }
  throw std::invalid_argument(
      "allreduce supports int8/int32/int64/fp16/fp32/fp64/bf16");
}

void validate_alltoall_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kInt8:  case torch::kInt32: case torch::kInt64:
    case torch::kFloat16: case torch::kFloat32: case torch::kFloat64:
    case torch::kBFloat16:
      return;
    default: break;
  }
  throw std::invalid_argument(
      "alltoall supports int8/int32/int64/fp16/fp32/fp64/bf16");
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

void wait_collective(SprayExecutor& ex, CollectiveOpHandle h) {
  int spin = 0;
  while (ex.status(h) != CollectiveOpStatus::Completed) {
    if (++spin < 1000) continue;
    spin = 0;
    std::this_thread::yield();
  }
}

}  // namespace

class ProcessGroup {
 public:
  ProcessGroup(int rank, int world_size, int gpu_id,
               std::string exchanger_ip = "127.0.0.1",
               int exchanger_port = 6979,
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
    cfg.threads_per_block = threads_per_block;
    cfg.blocks_per_worker = blocks_per_worker;
    cfg.smem_size = smem_size;
    cfg.max_device_fifos = 1;
    executor_ = SprayExecutor::create(cfg);
  }

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int gpu_id() const { return gpu_id_; }

  void allreduce(nb::handle tensor_handle,
                 uint32_t reduction = static_cast<uint32_t>(UKernel::CCL::ReductionKind::Sum),
                 size_t tile_bytes = 64ull << 10) {
    auto tensor = tensor_from_python(tensor_handle, "tensor");
    allreduce_internal(tensor, reduction, tile_bytes);
  }

  void alltoall(nb::handle tensor_handle,
                size_t tile_bytes = 64ull << 10) {
    std::lock_guard<std::mutex> lock(mu_);
    auto tensor = tensor_from_python(tensor_handle, "tensor");
    if (!tensor.is_cuda() || tensor.device().index() != gpu_id_)
      throw std::invalid_argument("tensor must be on this process group's GPU");
    validate_alltoall_dtype(tensor.scalar_type());

    auto flat = tensor.contiguous().view({-1});
    ScalarType dtype = to_scalar_type(tensor.scalar_type());
    size_t bytes = static_cast<size_t>(flat.numel()) *
                   static_cast<size_t>(flat.element_size());
    size_t elem_bytes = static_cast<size_t>(flat.element_size());
    size_t denom = static_cast<size_t>(world_size_) * elem_bytes;
    if (bytes % denom != 0)
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

    void* ptr = flat.data_ptr();
    ensure_prepared(cfg, ptr, ptr);
    auto h = executor_->submit(cfg, ptr, ptr);
    wait_collective(*executor_, h);
    executor_->release(h);
  }

  void alltoallv(nb::handle output_handle, nb::handle input_handle,
                 std::vector<int64_t> output_split_sizes,
                 std::vector<int64_t> input_split_sizes,
                 size_t tile_bytes = 64ull << 10) {
    std::lock_guard<std::mutex> lock(mu_);
    auto output = tensor_from_python(output_handle, "output");
    auto input = tensor_from_python(input_handle, "input");
    if (!output.is_cuda() || output.device().index() != gpu_id_)
      throw std::invalid_argument("output must be on this process group's GPU");
    if (!input.is_cuda() || input.device().index() != gpu_id_)
      throw std::invalid_argument("input must be on this process group's GPU");
    validate_alltoall_dtype(output.scalar_type());
    validate_alltoall_dtype(input.scalar_type());
    if (output.scalar_type() != input.scalar_type())
      throw std::invalid_argument("output and input must have the same dtype");

    auto out_flat = output.contiguous().view({-1});
    auto in_flat = input.contiguous().view({-1});
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

    void* out_ptr = out_flat.data_ptr();
    void* in_ptr = in_flat.data_ptr();
    ensure_prepared(cfg, in_ptr, out_ptr);
    auto h = executor_->submit(cfg, in_ptr, out_ptr);
    wait_collective(*executor_, h);
    executor_->release(h);
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
                       static_cast<uint32_t>(UKernel::CCL::ReductionKind::Sum),
                       static_cast<size_t>(barrier_tensor_.numel() *
                                           barrier_tensor_.element_size()));
  }

 private:
  void ensure_prepared(CollectiveConfig const& cfg, void* input, void* output) {
    executor_->prepare(cfg, input, output);
    prepared_ = true;
  }

  void allreduce_internal(torch::Tensor tensor, uint32_t reduction,
                          size_t tile_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!tensor.is_cuda() || tensor.device().index() != gpu_id_)
      throw std::invalid_argument("tensor must be on this process group's GPU");
    validate_allreduce_dtype(tensor.scalar_type());

    auto flat = tensor.contiguous().view({-1});
    ScalarType dtype = to_scalar_type(tensor.scalar_type());
    size_t bytes = static_cast<size_t>(flat.numel()) *
                   static_cast<size_t>(flat.element_size());
    size_t elem_bytes = static_cast<size_t>(flat.element_size());
    if (bytes % (static_cast<size_t>(world_size_) * elem_bytes) != 0)
      throw std::invalid_argument(
          "allreduce tensor bytes must be divisible by world_size * dtype_size");

    CollectiveConfig cfg;
    cfg.nranks = world_size_;
    cfg.rank = rank_;
    cfg.input_bytes = bytes;
    cfg.output_bytes = bytes;
    cfg.tile_bytes = tile_bytes;
    cfg.kind = CollKind::AllReduceRing;
    cfg.dtype = dtype;
    cfg.reduction = to_reduction(reduction);

    void* ptr = flat.data_ptr();
    ensure_prepared(cfg, ptr, ptr);
    auto h = executor_->submit(cfg, ptr, ptr);
    wait_collective(*executor_, h);
    executor_->release(h);
  }

  int rank_;
  int world_size_;
  int gpu_id_;
  std::unique_ptr<SprayExecutor> executor_;
  torch::Tensor barrier_tensor_;
  bool prepared_ = false;
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
           nb::arg("exchanger_port") = 6979,
           nb::arg("threads_per_block") = 64,
           nb::arg("blocks_per_worker") = 1,
           nb::arg("smem_size") = 4096)
      .def_prop_ro("rank", &ProcessGroup::rank)
      .def_prop_ro("world_size", &ProcessGroup::world_size)
      .def_prop_ro("gpu_id", &ProcessGroup::gpu_id)
      .def("allreduce",
           [](ProcessGroup& self, nb::handle tensor, uint32_t reduction,
              size_t tile_bytes) {
             self.allreduce(tensor, reduction, tile_bytes);
           },
           nb::arg("tensor"),
           nb::arg("reduction") = static_cast<uint32_t>(UKernel::CCL::ReductionKind::Sum),
           nb::arg("tile_bytes") = 64ull << 10)
      .def("alltoall",
           [](ProcessGroup& self, nb::handle tensor, size_t tile_bytes) {
             self.alltoall(tensor, tile_bytes);
           },
           nb::arg("tensor"), nb::arg("tile_bytes") = 64ull << 10)
      .def("alltoallv",
           [](ProcessGroup& self, nb::handle output, nb::handle input,
              std::vector<int64_t> output_split_sizes,
              std::vector<int64_t> input_split_sizes, size_t tile_bytes) {
             self.alltoallv(output, input,
                            std::move(output_split_sizes),
                            std::move(input_split_sizes), tile_bytes);
           },
           nb::arg("output"), nb::arg("input"),
           nb::arg("output_split_sizes"), nb::arg("input_split_sizes"),
           nb::arg("tile_bytes") = 64ull << 10)
      .def("barrier", [](ProcessGroup& self) { self.barrier(); });
}
