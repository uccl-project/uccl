#include "device_backend.h"
#include "../../device/task.h"
#include "../../device/worker.h"
#include <stdexcept>
#include <string>
#include <utility>

namespace UKernel {
namespace CCL {

namespace {

Device::DataType to_device_dtype(int dtype) {
  return static_cast<Device::DataType>(dtype);
}

Device::ReduceType to_device_reduce_type(int reduce_type) {
  return static_cast<Device::ReduceType>(reduce_type);
}

void validate_span(char const* what, size_t offset, size_t bytes, size_t capacity) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::invalid_argument(std::string(what) + " out of range");
  }
}

}  // namespace

DeviceBackend::DeviceBackend(UKernel::Device::WorkerPool* worker_pool,
                             CollectiveMemory memory,
                             int dtype,
                             int reduce_type)
    : worker_pool_(worker_pool),
      memory_(std::move(memory)),
      dtype_(dtype),
      reduce_type_(reduce_type) {
  if (worker_pool_ == nullptr) {
    throw std::invalid_argument("device backend requires a worker pool");
  }
  if (!Device::TaskManager::instance().inited()) {
    throw std::invalid_argument(
        "device backend requires TaskManager to be initialized externally");
  }
}

char const* DeviceBackend::name() const { return "device"; }

void DeviceBackend::validate(ExecutionPlan const& plan) const {
  if (plan.staging_bytes_required != 0 && memory_.staging.local_ptr == nullptr) {
    throw std::invalid_argument("device backend staging buffer is missing");
  }
  if (plan.staging_bytes_required > memory_.staging.bytes) {
    throw std::invalid_argument("device backend staging capacity is insufficient");
  }
}

bool DeviceBackend::supports(ExecOpKind kind) const {
  switch (kind) {
    case ExecOpKind::DeviceCopy:
    case ExecOpKind::DeviceReduce:
      return true;
    case ExecOpKind::TransportSend:
    case ExecOpKind::TransportRecv:
      return false;
  }
  return false;
}

BackendToken DeviceBackend::submit(ExecOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for device backend");
  }

  Device::TaskArgs args{};
  args.src = const_cast<void*>(resolve_const(op.src, op.tile.size_bytes));
  args.src2 = nullptr;
  args.dst = resolve_mutable(op.dst, op.tile.size_bytes);
  args.bytes = op.tile.size_bytes;
  args.src_rank = op.src.peer_rank;
  args.dst_rank = op.dst.peer_rank;
  args.src_device = -1;
  args.dst_device = -1;
  args.redType = to_device_reduce_type(reduce_type_);

  Device::TaskType task_type =
      (op.kind == ExecOpKind::DeviceReduce)
          ? Device::TaskType::CollReduce
          : Device::TaskType::CollCopy;

  uint32_t fifo_id = fifo_id_for(op);
  Device::Task task = Device::TaskManager::instance().create_task(
      args, task_type, to_device_dtype(dtype_), 0);

  uint64_t task_id = worker_pool_->enqueue(task, fifo_id);
  if (task_id == 0) {
    Device::TaskManager::instance().free_task_args(task.args_index());
    throw std::runtime_error("device backend failed to enqueue task");
  }

  BackendToken token{next_token_++};
  submitted_[token.value] = SubmittedTask{fifo_id, task_id, false};
  return token;
}

bool DeviceBackend::poll(BackendToken token) {
  auto it = submitted_.find(token.value);
  if (it == submitted_.end()) return true;
  bool done = worker_pool_->is_done(it->second.task_id, it->second.fifo_id);
  if (done && !it->second.completion_queued) {
    it->second.completion_queued = true;
    completed_tokens_.push_back(token.value);
  }
  return done;
}

bool DeviceBackend::try_pop_completed(BackendToken& token) {
  if (completed_tokens_.empty()) {
    for (auto& [token_value, submitted] : submitted_) {
      if (submitted.completion_queued) continue;
      if (!worker_pool_->is_done(submitted.task_id, submitted.fifo_id)) continue;
      submitted.completion_queued = true;
      completed_tokens_.push_back(token_value);
    }
  }

  if (completed_tokens_.empty()) return false;
  token.value = completed_tokens_.front();
  completed_tokens_.pop_front();
  return true;
}

void DeviceBackend::release(BackendToken token) { submitted_.erase(token.value); }

void* DeviceBackend::byte_offset(void* base, size_t offset) const {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* DeviceBackend::byte_offset(void const* base, size_t offset) const {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

void* DeviceBackend::resolve_mutable(BufferRef const& ref, size_t bytes) const {
  switch (ref.kind) {
    case BufferKind::Staging:
      if (memory_.staging.local_ptr == nullptr) {
        throw std::invalid_argument("device backend staging buffer is missing");
      }
      validate_span("device backend staging", ref.offset_bytes, bytes,
                    memory_.staging.bytes);
      return byte_offset(memory_.staging.local_ptr, ref.offset_bytes);
    case BufferKind::Tensor:
      if (memory_.tensor.local_ptr == nullptr) {
        throw std::invalid_argument("device backend local tensor is missing");
      }
      validate_span("device backend local tensor", ref.offset_bytes, bytes,
                    memory_.tensor.bytes);
      return byte_offset(memory_.tensor.local_ptr, ref.offset_bytes);
    case BufferKind::PeerTensor:
      break;
  }
  throw std::invalid_argument("device backend cannot write remote tensor");
}

void const* DeviceBackend::resolve_const(BufferRef const& ref, size_t bytes) const {
  switch (ref.kind) {
    case BufferKind::Staging:
      if (memory_.staging.local_ptr == nullptr) {
        throw std::invalid_argument("device backend staging buffer is missing");
      }
      validate_span("device backend staging", ref.offset_bytes, bytes,
                    memory_.staging.bytes);
      return byte_offset(memory_.staging.local_ptr, ref.offset_bytes);
    case BufferKind::Tensor:
      if (memory_.tensor.local_ptr == nullptr) {
        throw std::invalid_argument("device backend local tensor is missing");
      }
      validate_span("device backend local tensor", ref.offset_bytes, bytes,
                    memory_.tensor.bytes);
      return byte_offset(memory_.tensor.local_ptr, ref.offset_bytes);
    case BufferKind::PeerTensor:
      if (ref.peer_rank < 0 ||
          static_cast<size_t>(ref.peer_rank) >= memory_.tensor.peer_views.size()) {
        break;
      }
      if (!memory_.tensor.peer_views[static_cast<size_t>(ref.peer_rank)].peer_accessible ||
          memory_.tensor.peer_views[static_cast<size_t>(ref.peer_rank)].ptr == nullptr) {
        throw std::invalid_argument("device backend peer tensor is not directly accessible");
      }
      validate_span("device backend peer tensor", ref.offset_bytes, bytes,
                    memory_.tensor.bytes);
      return byte_offset(memory_.tensor.peer_views[static_cast<size_t>(ref.peer_rank)].ptr,
                         ref.offset_bytes);
  }
  throw std::invalid_argument("device backend invalid source reference");
}

uint32_t DeviceBackend::fifo_id_for(ExecOp const& op) {
  if (worker_pool_->num_fifos() == 0) {
    throw std::runtime_error("device backend worker pool has no FIFOs");
  }
  uint32_t lane_id = op.tile.flow_index;
  auto it = lane_fifo_assignments_.find(lane_id);
  if (it != lane_fifo_assignments_.end()) {
    return it->second;
  }
  uint32_t fifo_id = next_fifo_cursor_ % worker_pool_->num_fifos();
  next_fifo_cursor_ = (next_fifo_cursor_ + 1) % worker_pool_->num_fifos();
  lane_fifo_assignments_.emplace(lane_id, fifo_id);
  return fifo_id;
}

}  // namespace CCL
}  // namespace UKernel
