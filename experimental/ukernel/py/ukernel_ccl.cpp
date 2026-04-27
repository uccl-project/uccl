#include "../include/gpu_rt.h"
#include "../include/transport.h"
#include "../src/ccl/backend/device_backend.h"
#include "../src/ccl/backend/transport_backend.h"
#include "../src/ccl/collective_types.h"
#include "../src/ccl/executor.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

Transport::PreferredTransport parse_transport(std::string const& value) {
  if (value == "auto") return Transport::PreferredTransport::Auto;
  if (value == "ipc") return Transport::PreferredTransport::Ipc;
  if (value == "uccl") return Transport::PreferredTransport::Uccl;
  if (value == "tcp") return Transport::PreferredTransport::Tcp;
  throw std::invalid_argument("unsupported transport: " + value);
}

ScalarType to_scalar_type(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kUInt8:
      return ScalarType::UInt8;
    case torch::kInt8:
      return ScalarType::Int8;
    case torch::kInt16:
      return ScalarType::Int16;
    case torch::kInt32:
      return ScalarType::Int32;
    case torch::kInt64:
      return ScalarType::Int64;
    case torch::kFloat16:
      return ScalarType::Float16;
    case torch::kFloat32:
      return ScalarType::Float32;
    case torch::kFloat64:
      return ScalarType::Float64;
    case torch::kBFloat16:
      return ScalarType::BFloat16;
    case torch::kBool:
      return ScalarType::Bool;
    default:
      break;
  }
  throw std::invalid_argument("unsupported torch dtype for ukernel collective");
}

void validate_collective_dtype(CollectiveKind collective,
                               torch::ScalarType dtype) {
  switch (collective) {
    case CollectiveKind::AllReduce:
      switch (dtype) {
        case torch::kInt8:
        case torch::kInt32:
        case torch::kInt64:
        case torch::kFloat16:
        case torch::kFloat32:
        case torch::kFloat64:
        case torch::kBFloat16:
          return;
        default:
          break;
      }
      throw std::invalid_argument(
          "allreduce currently supports int8/int32/int64/fp16/fp32/fp64/bf16 "
          "tensors");
    case CollectiveKind::AllToAll:
      switch (dtype) {
        case torch::kInt8:
        case torch::kInt32:
        case torch::kInt64:
        case torch::kFloat16:
        case torch::kFloat32:
        case torch::kFloat64:
        case torch::kBFloat16:
          return;
        default:
          break;
      }
      throw std::invalid_argument(
          "alltoall currently supports int8/int32/int64/fp16/fp32/fp64/bf16 "
          "tensors");
  }
}

ReductionKind parse_reduction_kind(uint32_t value) {
  switch (static_cast<ReductionKind>(value)) {
    case ReductionKind::Sum:
    case ReductionKind::Prod:
    case ReductionKind::Max:
    case ReductionKind::Min:
    case ReductionKind::BitwiseAnd:
      return static_cast<ReductionKind>(value);
    case ReductionKind::None:
      break;
  }
  throw std::invalid_argument("unsupported reduction kind");
}

std::vector<size_t> normalize_split_bytes(std::vector<int64_t> const& splits,
                                          int world_size, size_t elem_size,
                                          size_t total_elems,
                                          char const* which) {
  if (splits.size() != static_cast<size_t>(world_size)) {
    throw std::invalid_argument(std::string(which) +
                                " split count must equal world_size");
  }
  std::vector<size_t> out;
  out.reserve(splits.size());
  size_t sum_elems = 0;
  for (int64_t split : splits) {
    if (split < 0) {
      throw std::invalid_argument(std::string(which) +
                                  " split sizes must be non-negative");
    }
    size_t elems = static_cast<size_t>(split);
    sum_elems += elems;
    out.push_back(elems * elem_size);
  }
  if (sum_elems != total_elems) {
    throw std::invalid_argument(std::string("sum(") + which +
                                "_split_sizes) must equal tensor numel");
  }
  return out;
}

CollectiveBinding build_collective_memory(
    int rank, int world_size, void* input_ptr, size_t input_bytes,
    ScalarType input_dtype, void* output_ptr, size_t output_bytes,
    ScalarType output_dtype, void* staging_ptr, size_t staging_bytes,
    CollectiveBufferRoles const& roles) {
  CollectiveBinding binding;
  binding.registry = std::make_shared<BufferRegistry>();
  binding.registry->local_rank = rank;
  binding.roles = roles;
  binding.roles.validate();
  RegisteredBuffer& input =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Input));
  input.local_ptr = input_ptr;
  input.bytes = input_bytes;
  input.layout.sizes = {
      static_cast<int64_t>(input_bytes / scalar_type_size(input_dtype))};
  input.layout.strides = {1};
  input.layout.dtype = input_dtype;
  input.peer_views.resize(static_cast<size_t>(world_size));

  RegisteredBuffer& output =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Output));
  output.local_ptr = output_ptr;
  output.bytes = output_bytes;
  output.layout.sizes = {
      static_cast<int64_t>(output_bytes / scalar_type_size(output_dtype))};
  output.layout.strides = {1};
  output.layout.dtype = output_dtype;
  output.remotely_accessible =
      binding.buffer_id(CollectiveBufferRole::Output) ==
              binding.buffer_id(CollectiveBufferRole::Input)
          ? true
          : false;
  output.peer_views.resize(static_cast<size_t>(world_size));

  RegisteredBuffer& staging =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Scratch));
  staging.local_ptr = staging_ptr;
  staging.bytes = staging_bytes;
  staging.layout.sizes = {static_cast<int64_t>(staging_bytes)};
  staging.layout.strides = {1};
  staging.layout.dtype = ScalarType::UInt8;
  staging.peer_views.resize(static_cast<size_t>(world_size));

  for (int peer = 0; peer < world_size; ++peer) {
    input.peer_views[static_cast<size_t>(peer)].same_node = true;
    output.peer_views[static_cast<size_t>(peer)].same_node = true;
    staging.peer_views[static_cast<size_t>(peer)].same_node = true;
  }

  return binding;
}

struct WorkTensor {
  torch::Tensor work_flat;
};

struct BindingState {
  void* input_ptr = nullptr;
  size_t input_bytes = 0;
  ScalarType input_dtype = ScalarType::UInt8;
  void* output_ptr = nullptr;
  size_t output_bytes = 0;
  ScalarType output_dtype = ScalarType::UInt8;
  void* staging_ptr = nullptr;
  size_t staging_bytes = 0;
  BufferId input_buffer_id = kDefaultInputBufferId;
  BufferId output_buffer_id = kDefaultOutputBufferId;
  BufferId scratch_buffer_id = kDefaultScratchBufferId;

  bool matches(void* other_input_ptr, size_t other_input_bytes,
               ScalarType other_input_dtype, void* other_output_ptr,
               size_t other_output_bytes, ScalarType other_output_dtype,
               void* other_staging_ptr, size_t other_staging_bytes,
               CollectiveBufferRoles const& other_roles) const {
    return input_ptr == other_input_ptr && input_bytes == other_input_bytes &&
           input_dtype == other_input_dtype && output_ptr == other_output_ptr &&
           output_bytes == other_output_bytes &&
           output_dtype == other_output_dtype &&
           staging_ptr == other_staging_ptr &&
           staging_bytes == other_staging_bytes &&
           input_buffer_id == other_roles.input_buffer_id &&
           output_buffer_id == other_roles.output_buffer_id &&
           scratch_buffer_id == other_roles.scratch_buffer_id;
  }
};

struct InflightCollective {
  CollectiveOpHandle handle{};
  torch::Tensor input_tensor;
  torch::Tensor output_tensor;
  torch::Tensor input_work_tensor;
  torch::Tensor output_work_tensor;
  torch::Tensor staging_tensor;
};

}  // namespace

class ProcessGroup {
 public:
  ProcessGroup(int rank, int world_size, int gpu_id, std::string exchanger_ip,
               int exchanger_port, std::string transport = "auto",
               uint32_t device_task_capacity = 4096,
               uint32_t max_device_fifos = 8, uint32_t threads_per_block = 256,
               uint32_t fifo_capacity = 64, uint32_t smem_size = 0)
      : rank_(rank),
        world_size_(world_size),
        gpu_id_(gpu_id),
        transport_backend_(TransportBackendConfig{
            gpu_id,
            rank,
            world_size,
            std::make_shared<Transport::CommunicatorConfig>(
                Transport::CommunicatorConfig{
                    exchanger_ip,
                    exchanger_port,
                    rank,
                    "default",
                    parse_transport(transport),
                }),
        }),
        device_backend_(DeviceBackendConfig{
            device_task_capacity,
            max_device_fifos,
            threads_per_block,
            fifo_capacity,
            smem_size,
        }),
        executor_(
            ExecutorBackends{&transport_backend_, &device_backend_, nullptr},
            [this](int remote_rank, uint32_t remote_buffer_id, size_t offset,
                   size_t bytes,
                   void** out_ptr, int* out_device_idx) {
              if (out_ptr == nullptr) return false;
              UKernel::Transport::IPCItem ipc{};
              try {
                ipc = transport_backend_.communicator().get_ipc(remote_rank,
                                                                remote_buffer_id);
              } catch (...) {
                return false;
              }
              if (!ipc.valid || ipc.direct_ptr == nullptr) return false;
              if (offset > ipc.bytes || bytes > ipc.bytes - offset) {
                return false;
              }
              *out_ptr = reinterpret_cast<void*>(
                  reinterpret_cast<uintptr_t>(ipc.direct_ptr) +
                  ipc.base_offset + offset);
              if (out_device_idx != nullptr) {
                *out_device_idx = ipc.device_idx;
              }
              return true;
            }) {
    if (world_size_ < 2) {
      throw std::invalid_argument("world_size must be >= 2");
    }
    if (rank_ < 0 || rank_ >= world_size_) {
      throw std::invalid_argument("rank out of range");
    }
    GPU_RT_CHECK(gpuSetDevice(gpu_id_));
  }

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int gpu_id() const { return gpu_id_; }

  void allreduce(torch::Tensor tensor, size_t tile_bytes = 64ull << 10,
                 uint32_t num_flows = 2) {
    wait_handle(submit_allreduce(std::move(tensor),
                                 static_cast<uint32_t>(ReductionKind::Sum),
                                 tile_bytes, num_flows));
  }

  void alltoall(torch::Tensor tensor, size_t tile_bytes = 64ull << 10,
                uint32_t num_flows = 2) {
    wait_handle(submit_alltoall(std::move(tensor), tile_bytes, num_flows));
  }

  void alltoall_out(torch::Tensor output, torch::Tensor input,
                    size_t tile_bytes = 64ull << 10, uint32_t num_flows = 2) {
    wait_handle(submit_alltoall_out(std::move(output), std::move(input),
                                    tile_bytes, num_flows));
  }

  void alltoallv_out(torch::Tensor output, torch::Tensor input,
                     std::vector<int64_t> output_split_sizes,
                     std::vector<int64_t> input_split_sizes,
                     size_t tile_bytes = 64ull << 10, uint32_t num_flows = 2) {
    wait_handle(submit_alltoallv_out(
        std::move(output), std::move(input), std::move(output_split_sizes),
        std::move(input_split_sizes), tile_bytes, num_flows));
  }

  uint64_t submit_allreduce(torch::Tensor tensor, uint32_t reduction,
                            size_t tile_bytes = 64ull << 10,
                            uint32_t num_flows = 2) {
    return submit_collective(CollectiveKind::AllReduce, std::move(tensor),
                             parse_reduction_kind(reduction), tile_bytes,
                             num_flows);
  }

  uint64_t submit_alltoall(torch::Tensor tensor,
                           size_t tile_bytes = 64ull << 10,
                           uint32_t num_flows = 2) {
    return submit_alltoall_out(tensor, tensor, tile_bytes, num_flows);
  }

  uint64_t submit_alltoall_out(torch::Tensor output, torch::Tensor input,
                               size_t tile_bytes = 64ull << 10,
                               uint32_t num_flows = 2) {
    return submit_alltoallv_out(std::move(output), std::move(input), {}, {},
                                tile_bytes, num_flows);
  }

  uint64_t submit_alltoallv_out(torch::Tensor output, torch::Tensor input,
                                std::vector<int64_t> output_split_sizes,
                                std::vector<int64_t> input_split_sizes,
                                size_t tile_bytes = 64ull << 10,
                                uint32_t num_flows = 2) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!inflight_.empty()) {
      throw std::runtime_error(
          "only one inflight collective per ukernel ProcessGroup is currently "
          "supported");
    }
    GPU_RT_CHECK(gpuSetDevice(gpu_id_));

    bool require_even_split =
        input_split_sizes.empty() && output_split_sizes.empty();
    WorkTensor input_work = prepare_work_tensor(CollectiveKind::AllToAll, input,
                                                require_even_split);
    WorkTensor output_work = prepare_work_tensor(CollectiveKind::AllToAll,
                                                 output, require_even_split);
    ScalarType input_dtype = to_scalar_type(input.scalar_type());
    ScalarType output_dtype = to_scalar_type(output.scalar_type());
    if (input_dtype != output_dtype) {
      throw std::invalid_argument(
          "alltoall output and input must have the same dtype");
    }
    size_t input_bytes =
        static_cast<size_t>(input_work.work_flat.numel()) *
        static_cast<size_t>(input_work.work_flat.element_size());
    size_t output_bytes =
        static_cast<size_t>(output_work.work_flat.numel()) *
        static_cast<size_t>(output_work.work_flat.element_size());
    std::vector<size_t> input_split_bytes;
    std::vector<size_t> output_split_bytes;
    if (require_even_split) {
      if (input_bytes != output_bytes) {
        throw std::invalid_argument(
            "equal-split alltoall output and input must have the same byte "
            "size");
      }
    } else {
      size_t elem_size =
          static_cast<size_t>(input_work.work_flat.element_size());
      input_split_bytes = normalize_split_bytes(
          input_split_sizes, world_size_, elem_size,
          static_cast<size_t>(input_work.work_flat.numel()), "input");
      output_split_bytes = normalize_split_bytes(
          output_split_sizes, world_size_, elem_size,
          static_cast<size_t>(output_work.work_flat.numel()), "output");
      size_t self_index = static_cast<size_t>(rank_);
      if (input_split_bytes[self_index] != output_split_bytes[self_index]) {
        throw std::invalid_argument(
            "alltoallv requires input_split_sizes[rank] == "
            "output_split_sizes[rank]");
      }
    }

    CollectiveConfig config{};
    config.nranks = world_size_;
    config.rank = rank_;
    config.num_flows = num_flows;
    config.tensor_bytes = std::max(input_bytes, output_bytes);
    config.input_bytes = input_bytes;
    config.output_bytes = output_bytes;
    config.input_split_bytes = std::move(input_split_bytes);
    config.output_split_bytes = std::move(output_split_bytes);
    config.tile_bytes = tile_bytes;
    config.staging_bytes = tile_bytes * (world_size_ - 1);
    config.algorithm = AlgorithmKind::Pairwise;
    config.dtype = input_dtype;
    config.reduction = ReductionKind::Sum;

    CollectiveBufferRoles roles =
        input_work.work_flat.data_ptr() == output_work.work_flat.data_ptr()
            ? roles_
            : CollectiveBufferRoles{
                  kDefaultInputBufferId,
                  kDefaultScratchBufferId,
                  static_cast<BufferId>(kDefaultScratchBufferId + 1),
              };
    CollectivePlan plan =
        build_plan(make_plan_request(CollectiveKind::AllToAll, config, roles));
    torch::Tensor staging = ensure_staging(plan.staging_bytes_required);
    void* staging_ptr = staging.defined() ? staging.data_ptr() : nullptr;

    if (!binding_state_.matches(input_work.work_flat.data_ptr(), input_bytes,
                                input_dtype, output_work.work_flat.data_ptr(),
                                output_bytes, output_dtype, staging_ptr,
                                plan.staging_bytes_required, roles)) {
      binding_memory_ =
          std::make_shared<CollectiveBinding>(build_collective_memory(
              rank_, world_size_, input_work.work_flat.data_ptr(), input_bytes,
              input_dtype, output_work.work_flat.data_ptr(), output_bytes,
              output_dtype, staging_ptr, plan.staging_bytes_required, roles));
      binding_state_.input_ptr = input_work.work_flat.data_ptr();
      binding_state_.input_bytes = input_bytes;
      binding_state_.input_dtype = input_dtype;
      binding_state_.output_ptr = output_work.work_flat.data_ptr();
      binding_state_.output_bytes = output_bytes;
      binding_state_.output_dtype = output_dtype;
      binding_state_.staging_ptr = staging_ptr;
      binding_state_.staging_bytes = plan.staging_bytes_required;
      binding_state_.input_buffer_id = roles.input_buffer_id;
      binding_state_.output_buffer_id = roles.output_buffer_id;
      binding_state_.scratch_buffer_id = roles.scratch_buffer_id;
    }

    CollectiveOpHandle handle =
        executor_.submit(std::move(plan), binding_memory_);
    inflight_.emplace(handle.value, InflightCollective{
                                        handle,
                                        input,
                                        output,
                                        input_work.work_flat,
                                        output_work.work_flat,
                                        staging,
                                    });
    return handle.value;
  }

  uint64_t submit_barrier() {
    if (!barrier_tensor_.defined()) {
      barrier_tensor_ = torch::ones(
          {1}, torch::TensorOptions()
                   .dtype(torch::kInt32)
                   .device(c10::Device(c10::DeviceType::CUDA, gpu_id_)));
    } else {
      barrier_tensor_.fill_(1);
    }
    return submit_collective(CollectiveKind::AllReduce, barrier_tensor_,
                             ReductionKind::Sum, sizeof(int32_t), 1);
  }

  bool poll_handle(uint64_t handle_value) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = inflight_.find(handle_value);
    if (it == inflight_.end()) return true;
    if (!executor_.poll(it->second.handle)) return false;
    finalize_completed_locked(it);
    return true;
  }

  void wait_handle(uint64_t handle_value) {
    CollectiveOpHandle handle{};
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = inflight_.find(handle_value);
      if (it == inflight_.end()) return;
      handle = it->second.handle;
    }

    nb::gil_scoped_release release;
    executor_.wait(handle);

    std::lock_guard<std::mutex> lock(mu_);
    auto it = inflight_.find(handle_value);
    if (it == inflight_.end()) return;
    finalize_completed_locked(it);
  }

  std::string status_handle(uint64_t handle_value) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = inflight_.find(handle_value);
    if (it == inflight_.end()) return "released";
    switch (executor_.status(it->second.handle)) {
      case CollectiveOpStatus::Queued:
        return "queued";
      case CollectiveOpStatus::Running:
        return "running";
      case CollectiveOpStatus::Completed:
        return "completed";
      case CollectiveOpStatus::Failed:
        return "failed";
    }
    return "unknown";
  }

  bool same_host(int peer_rank) const {
    return transport_backend_.communicator().same_host(peer_rank);
  }

  std::string peer_transport(int peer_rank) const {
    switch (transport_backend_.communicator().peer_transport_kind(peer_rank)) {
      case Transport::PeerTransportKind::Unknown:
        return "unknown";
      case Transport::PeerTransportKind::Ipc:
        return "ipc";
      case Transport::PeerTransportKind::Uccl:
        return "uccl";
      case Transport::PeerTransportKind::Tcp:
        return "tcp";
    }
    return "unknown";
  }

 private:
  WorkTensor prepare_work_tensor(CollectiveKind collective,
                                 torch::Tensor const& tensor,
                                 bool require_even_alltoall_bytes = true) {
    if (!tensor.defined()) {
      throw std::invalid_argument("tensor must be defined");
    }
    if (!tensor.is_cuda()) {
      throw std::invalid_argument("tensor must be a CUDA tensor");
    }
    if (tensor.device().index() != gpu_id_) {
      throw std::invalid_argument(
          "tensor device does not match process group gpu_id");
    }
    validate_collective_dtype(collective, tensor.scalar_type());

    WorkTensor out;
    torch::Tensor flat = tensor.view({-1});
    size_t raw_bytes = static_cast<size_t>(flat.numel()) *
                       static_cast<size_t>(flat.element_size());
    size_t alignment = static_cast<size_t>(world_size_) *
                       static_cast<size_t>(flat.element_size());
    if (collective == CollectiveKind::AllToAll && require_even_alltoall_bytes &&
        alignment != 0 && (raw_bytes % alignment) != 0) {
      throw std::invalid_argument(
          "alltoall tensor byte size must be divisible by world_size * "
          "element_size");
    }
    out.work_flat = flat;
    return out;
  }

  torch::Tensor ensure_staging(size_t staging_bytes) {
    if (staging_bytes == 0) {
      staging_tensor_ = torch::Tensor();
      return staging_tensor_;
    }
    if (staging_tensor_.defined() &&
        staging_tensor_.device().index() == gpu_id_ &&
        static_cast<size_t>(staging_tensor_.numel()) >= staging_bytes) {
      return staging_tensor_;
    }
    staging_tensor_ =
        torch::empty({static_cast<int64_t>(staging_bytes)},
                     torch::TensorOptions()
                         .dtype(torch::kUInt8)
                         .device(c10::Device(c10::DeviceType::CUDA, gpu_id_)));
    return staging_tensor_;
  }

  uint64_t submit_collective(CollectiveKind collective, torch::Tensor tensor,
                             ReductionKind reduction, size_t tile_bytes,
                             uint32_t num_flows) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!inflight_.empty()) {
      throw std::runtime_error(
          "only one inflight collective per ukernel ProcessGroup is currently "
          "supported");
    }
    GPU_RT_CHECK(gpuSetDevice(gpu_id_));

    WorkTensor work = prepare_work_tensor(collective, tensor);
    ScalarType dtype = to_scalar_type(tensor.scalar_type());
    size_t tensor_bytes = static_cast<size_t>(work.work_flat.numel()) *
                          static_cast<size_t>(work.work_flat.element_size());

    CollectiveConfig config{};
    config.nranks = world_size_;
    config.rank = rank_;
    config.num_flows = num_flows;
    config.tensor_bytes = tensor_bytes;
    config.tile_bytes = tile_bytes;
    config.staging_bytes = tile_bytes * num_flows;
    config.algorithm = (collective == CollectiveKind::AllReduce)
                           ? AlgorithmKind::Ring
                           : AlgorithmKind::Pairwise;
    config.dtype = dtype;
    config.reduction = reduction;

    CollectivePlan plan =
        build_plan(make_plan_request(collective, config, roles_));
    torch::Tensor staging = ensure_staging(plan.staging_bytes_required);
    void* staging_ptr = staging.defined() ? staging.data_ptr() : nullptr;

    if (!binding_state_.matches(work.work_flat.data_ptr(), tensor_bytes, dtype,
                                work.work_flat.data_ptr(), tensor_bytes, dtype,
                                staging_ptr, plan.staging_bytes_required,
                                roles_)) {
      binding_memory_ =
          std::make_shared<CollectiveBinding>(build_collective_memory(
              rank_, world_size_, work.work_flat.data_ptr(), tensor_bytes,
              dtype, work.work_flat.data_ptr(), tensor_bytes, dtype,
              staging_ptr, plan.staging_bytes_required, roles_));
      binding_state_.input_ptr = work.work_flat.data_ptr();
      binding_state_.input_bytes = tensor_bytes;
      binding_state_.input_dtype = dtype;
      binding_state_.output_ptr = work.work_flat.data_ptr();
      binding_state_.output_bytes = tensor_bytes;
      binding_state_.output_dtype = dtype;
      binding_state_.staging_ptr = staging_ptr;
      binding_state_.staging_bytes = plan.staging_bytes_required;
      binding_state_.input_buffer_id = roles_.input_buffer_id;
      binding_state_.output_buffer_id = roles_.output_buffer_id;
      binding_state_.scratch_buffer_id = roles_.scratch_buffer_id;
    }

    CollectiveOpHandle handle =
        executor_.submit(std::move(plan), binding_memory_);
    inflight_.emplace(handle.value, InflightCollective{
                                        handle,
                                        tensor,
                                        tensor,
                                        work.work_flat,
                                        work.work_flat,
                                        staging,
                                    });
    return handle.value;
  }

  void finalize_completed_locked(
      std::unordered_map<uint64_t, InflightCollective>::iterator it) {
    CollectiveOpStatus status = executor_.status(it->second.handle);
    if (status == CollectiveOpStatus::Failed) {
      std::string error = executor_.error_message(it->second.handle);
      executor_.release(it->second.handle);
      inflight_.erase(it);
      throw std::runtime_error(error.empty() ? "collective failed" : error);
    }
    if (status != CollectiveOpStatus::Completed) return;

    executor_.release(it->second.handle);
    inflight_.erase(it);
  }

  int rank_;
  int world_size_;
  int gpu_id_;
  CollectiveBufferRoles roles_{};
  std::shared_ptr<CollectiveBinding> binding_memory_;
  CommunicatorTransportBackend transport_backend_;
  DeviceBackend device_backend_;
  Executor executor_;
  torch::Tensor staging_tensor_;
  torch::Tensor barrier_tensor_;
  BindingState binding_state_{};
  std::unordered_map<uint64_t, InflightCollective> inflight_;
  std::mutex mu_;
};

}  // namespace Python
}  // namespace CCL
}  // namespace UKernel

NB_MODULE(TORCH_EXTENSION_NAME, m) {
  using UKernel::CCL::Python::ProcessGroup;

  nb::class_<ProcessGroup>(m, "ProcessGroup")
      .def(nb::init<int, int, int, std::string, int, std::string, uint32_t,
                    uint32_t, uint32_t, uint32_t, uint32_t>(),
           nb::arg("rank"), nb::arg("world_size"), nb::arg("gpu_id"),
           nb::arg("exchanger_ip") = "127.0.0.1",
           nb::arg("exchanger_port") = 6979, nb::arg("transport") = "auto",
           nb::arg("device_task_capacity") = 4096,
           nb::arg("max_device_fifos") = 8, nb::arg("threads_per_block") = 256,
           nb::arg("fifo_capacity") = 64, nb::arg("smem_size") = 0)
      .def_prop_ro("rank", &ProcessGroup::rank)
      .def_prop_ro("world_size", &ProcessGroup::world_size)
      .def_prop_ro("gpu_id", &ProcessGroup::gpu_id)
      .def(
          "submit_allreduce",
          [](ProcessGroup& self, nb::handle tensor, uint32_t reduction,
             size_t tile_bytes, uint32_t num_flows) {
            return self.submit_allreduce(
                UKernel::CCL::Python::tensor_from_python(tensor, "tensor"),
                reduction, tile_bytes, num_flows);
          },
          nb::arg("tensor"),
          nb::arg("reduction") =
              static_cast<uint32_t>(UKernel::CCL::ReductionKind::Sum),
          nb::arg("tile_bytes") = 64ull << 10, nb::arg("num_flows") = 2)
      .def(
          "submit_alltoall",
          [](ProcessGroup& self, nb::handle tensor, size_t tile_bytes,
             uint32_t num_flows) {
            return self.submit_alltoall(
                UKernel::CCL::Python::tensor_from_python(tensor, "tensor"),
                tile_bytes, num_flows);
          },
          nb::arg("tensor"), nb::arg("tile_bytes") = 64ull << 10,
          nb::arg("num_flows") = 2)
      .def(
          "submit_alltoall_out",
          [](ProcessGroup& self, nb::handle output, nb::handle input,
             size_t tile_bytes, uint32_t num_flows) {
            return self.submit_alltoall_out(
                UKernel::CCL::Python::tensor_from_python(output, "output"),
                UKernel::CCL::Python::tensor_from_python(input, "input"),
                tile_bytes, num_flows);
          },
          nb::arg("output"), nb::arg("input"),
          nb::arg("tile_bytes") = 64ull << 10, nb::arg("num_flows") = 2)
      .def(
          "submit_alltoallv_out",
          [](ProcessGroup& self, nb::handle output, nb::handle input,
             std::vector<int64_t> output_split_sizes,
             std::vector<int64_t> input_split_sizes, size_t tile_bytes,
             uint32_t num_flows) {
            return self.submit_alltoallv_out(
                UKernel::CCL::Python::tensor_from_python(output, "output"),
                UKernel::CCL::Python::tensor_from_python(input, "input"),
                std::move(output_split_sizes), std::move(input_split_sizes),
                tile_bytes, num_flows);
          },
          nb::arg("output"), nb::arg("input"), nb::arg("output_split_sizes"),
          nb::arg("input_split_sizes"), nb::arg("tile_bytes") = 64ull << 10,
          nb::arg("num_flows") = 2)
      .def("submit_barrier", &ProcessGroup::submit_barrier)
      .def("poll_handle", &ProcessGroup::poll_handle, nb::arg("handle"))
      .def("wait_handle", &ProcessGroup::wait_handle, nb::arg("handle"))
      .def("status_handle", &ProcessGroup::status_handle, nb::arg("handle"))
      .def(
          "allreduce",
          [](ProcessGroup& self, nb::handle tensor, size_t tile_bytes,
             uint32_t num_flows) {
            self.allreduce(
                UKernel::CCL::Python::tensor_from_python(tensor, "tensor"),
                tile_bytes, num_flows);
          },
          nb::arg("tensor"), nb::arg("tile_bytes") = 64ull << 10,
          nb::arg("num_flows") = 2)
      .def(
          "alltoall",
          [](ProcessGroup& self, nb::handle tensor, size_t tile_bytes,
             uint32_t num_flows) {
            self.alltoall(
                UKernel::CCL::Python::tensor_from_python(tensor, "tensor"),
                tile_bytes, num_flows);
          },
          nb::arg("tensor"), nb::arg("tile_bytes") = 64ull << 10,
          nb::arg("num_flows") = 2)
      .def(
          "alltoall_out",
          [](ProcessGroup& self, nb::handle output, nb::handle input,
             size_t tile_bytes, uint32_t num_flows) {
            self.alltoall_out(
                UKernel::CCL::Python::tensor_from_python(output, "output"),
                UKernel::CCL::Python::tensor_from_python(input, "input"),
                tile_bytes, num_flows);
          },
          nb::arg("output"), nb::arg("input"),
          nb::arg("tile_bytes") = 64ull << 10, nb::arg("num_flows") = 2)
      .def(
          "alltoallv_out",
          [](ProcessGroup& self, nb::handle output, nb::handle input,
             std::vector<int64_t> output_split_sizes,
             std::vector<int64_t> input_split_sizes, size_t tile_bytes,
             uint32_t num_flows) {
            self.alltoallv_out(
                UKernel::CCL::Python::tensor_from_python(output, "output"),
                UKernel::CCL::Python::tensor_from_python(input, "input"),
                std::move(output_split_sizes), std::move(input_split_sizes),
                tile_bytes, num_flows);
          },
          nb::arg("output"), nb::arg("input"), nb::arg("output_split_sizes"),
          nb::arg("input_split_sizes"), nb::arg("tile_bytes") = 64ull << 10,
          nb::arg("num_flows") = 2)
      .def("same_host", &ProcessGroup::same_host, nb::arg("peer_rank"))
      .def("peer_transport", &ProcessGroup::peer_transport,
           nb::arg("peer_rank"));
}
