#include "device_backend.h"
#include "../../device/task.h"
#include "../../device/worker.h"
#include "../../include/gpu_rt.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace UKernel {
namespace CCL {

namespace {

void validate_span(char const* what, size_t offset, size_t bytes,
                   size_t capacity) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::invalid_argument(std::string(what) + " out of range");
  }
}

Device::DataType to_device_dtype(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Int8:
      return Device::DataType::Int8;
    case ScalarType::Int32:
      return Device::DataType::Int32;
    case ScalarType::Int64:
      return Device::DataType::Int64;
    case ScalarType::Float16:
      return Device::DataType::Fp16;
    case ScalarType::Float32:
      return Device::DataType::Fp32;
    case ScalarType::Float64:
      return Device::DataType::Fp64;
    case ScalarType::BFloat16:
      return Device::DataType::Bf16;
    case ScalarType::UInt8:
    case ScalarType::Int16:
    case ScalarType::Bool:
      break;
  }
  throw std::invalid_argument("device backend does not support this dtype");
}

Device::ReduceType to_device_reduce_type(ReductionKind reduction) {
  switch (reduction) {
    case ReductionKind::None:
      return Device::ReduceType::None;
    case ReductionKind::Sum:
      return Device::ReduceType::Sum;
    case ReductionKind::Prod:
      return Device::ReduceType::Prod;
    case ReductionKind::Max:
      return Device::ReduceType::Max;
    case ReductionKind::Min:
      return Device::ReduceType::Min;
    case ReductionKind::BitwiseAnd:
      return Device::ReduceType::BitwiseAnd;
  }
  return Device::ReduceType::None;
}

}  // namespace

DeviceBackend::DeviceBackend(std::shared_ptr<CollectiveMemory> memory,
                             DeviceBackendConfig const& config)
    : memory_(std::move(memory)), config_(config) {
  if (memory_ == nullptr) {
    throw std::invalid_argument("device backend requires collective memory");
  }
  if (config_.task_capacity == 0) {
    throw std::invalid_argument("device backend task_capacity must be positive");
  }
  if (config_.max_fifos == 0) {
    throw std::invalid_argument("device backend max_fifos must be positive");
  }
  if (config_.threads_per_block == 0) {
    throw std::invalid_argument(
        "device backend threads_per_block must be positive");
  }
  if (config_.fifo_capacity == 0) {
    throw std::invalid_argument(
        "device backend fifo_capacity must be positive");
  }

  int device = 0;
  GPU_RT_CHECK(gpuGetDevice(&device));
  local_device_idx_ = device;
  GPU_RT_CHECK(
      gpuDeviceGetAttribute(&sm_count_, gpuDevAttrMultiProcessorCount, device));
}

DeviceBackend::~DeviceBackend() {
  ensure_device_context();
  completed_tokens_.clear();
  submitted_.clear();
  active_flows_.clear();
  worker_pool_.reset();
  if (owns_task_manager_) {
    Device::TaskManager::instance().release();
    owns_task_manager_ = false;
  }
}

char const* DeviceBackend::name() const { return "device"; }

void DeviceBackend::validate(ExecutionPlan const& plan) const {
  if (plan.staging_bytes_required != 0 &&
      memory_->staging.local_ptr == nullptr) {
    throw std::invalid_argument("device backend staging buffer is missing");
  }
  if (plan.staging_bytes_required > memory_->staging.bytes) {
    throw std::invalid_argument(
        "device backend staging capacity is insufficient");
  }
  for (ExecOp const& op : plan.ops) {
    if (!supports(op.kind)) continue;
    (void)to_device_dtype(op.dtype);
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
  ensure_device_context();
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for device backend");
  }
  void const* src =
      op.resolved_src != nullptr ? op.resolved_src
                                 : resolve_const(op.src, op.tile.size_bytes);
  void* dst =
      op.resolved_dst != nullptr ? op.resolved_dst
                                 : resolve_mutable(op.dst, op.tile.size_bytes);
  ensure_runtime();

  Device::TaskArgs args{};
  args.src = const_cast<void*>(src);
  args.src2 = nullptr;
  args.dst = dst;
  args.bytes = op.tile.size_bytes;
  args.src_rank = (op.src.kind == BufferKind::PeerTensor ||
                   op.src.kind == BufferKind::PeerStaging)
                      ? op.src.rank
                      : memory_->tensor.local_rank;
  args.dst_rank = (op.dst.kind == BufferKind::PeerTensor ||
                   op.dst.kind == BufferKind::PeerStaging)
                      ? op.dst.rank
                      : memory_->tensor.local_rank;
  args.src_device =
      op.src_device >= 0 ? op.src_device : local_device_idx_;
  args.dst_device =
      op.dst_device >= 0 ? op.dst_device : local_device_idx_;
  args.set_red_type(::UKernel::CCL::to_device_reduce_type(op.reduction));

  Device::TaskType task_type = (op.kind == ExecOpKind::DeviceReduce)
                                   ? Device::TaskType::CollReduce
                                   : Device::TaskType::CollCopy;

  uint32_t flow_id = op.tile.flow_index;
  uint32_t fifo_id = acquire_fifo(flow_id, suggested_num_blocks(op));
  Device::DataType dtype = ::UKernel::CCL::to_device_dtype(op.dtype);
  Device::Task task =
      Device::TaskManager::instance().create_task(args, task_type, dtype, 0);

  uint64_t task_id = Device::WorkerPool::kInvalidTaskId;
  for (int retry = 0; retry < 1000 &&
                      task_id == Device::WorkerPool::kInvalidTaskId;
       ++retry) {
    task_id = worker_pool_->enqueue(task, fifo_id);
    if (task_id == Device::WorkerPool::kInvalidTaskId) {
      std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
  }
  if (task_id == Device::WorkerPool::kInvalidTaskId) {
    active_flows_[flow_id].inflight--;
    stop_flow(flow_id);
    Device::TaskManager::instance().free_task_args(task.args_index());
    throw std::runtime_error("device backend failed to enqueue task");
  }

  BackendToken token{next_token_++};
  submitted_[token.value] =
      SubmittedTask{fifo_id, task_id, flow_id, task.args_index(), false, false};
  return token;
}

bool DeviceBackend::poll(BackendToken token) {
  ensure_device_context();
  auto it = submitted_.find(token.value);
  if (it == submitted_.end()) return true;
  bool done = worker_pool_->is_done(it->second.task_id, it->second.fifo_id);
  if (done && !it->second.completion_queued) {
    release_task_args(it->second);
    it->second.completion_queued = true;
    completed_tokens_.push_back(token.value);
  }
  return done;
}

bool DeviceBackend::try_pop_completed(BackendToken& token) {
  ensure_device_context();
  if (completed_tokens_.empty()) {
    for (auto& [token_value, submitted] : submitted_) {
      if (submitted.completion_queued) continue;
      if (!worker_pool_->is_done(submitted.task_id, submitted.fifo_id))
        continue;
      release_task_args(submitted);
      submitted.completion_queued = true;
      completed_tokens_.push_back(token_value);
    }
  }

  if (completed_tokens_.empty()) return false;
  token.value = completed_tokens_.front();
  completed_tokens_.pop_front();
  return true;
}

void DeviceBackend::release(BackendToken token) {
  ensure_device_context();
  auto it = submitted_.find(token.value);
  if (it == submitted_.end()) return;
  uint32_t flow_id = it->second.flow_id;
  bool const completed = it->second.args_released;
  submitted_.erase(it);
  auto flow_it = active_flows_.find(flow_id);
  if (flow_it == active_flows_.end()) return;
  if (flow_it->second.inflight > 0) {
    --flow_it->second.inflight;
  }
  if (!completed && flow_it->second.inflight == 0) {
    stop_flow(flow_id);
  }
}

void DeviceBackend::stop(uint32_t flow_id) {
  ensure_device_context();
  stop_flow(flow_id);
}

void* DeviceBackend::byte_offset(void* base, size_t offset) const {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* DeviceBackend::byte_offset(void const* base, size_t offset) const {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

void* DeviceBackend::resolve_mutable(BufferRef const& ref, size_t bytes) const {
  switch (ref.kind) {
    case BufferKind::Staging:
      if (memory_->staging.local_ptr == nullptr) {
        throw std::invalid_argument("device backend staging buffer is missing");
      }
      validate_span("device backend staging", ref.offset_bytes, bytes,
                    memory_->staging.bytes);
      return byte_offset(memory_->staging.local_ptr, ref.offset_bytes);
    case BufferKind::Tensor:
      if (memory_->tensor.local_ptr == nullptr) {
        throw std::invalid_argument("device backend local tensor is missing");
      }
      validate_span("device backend local tensor", ref.offset_bytes, bytes,
                    memory_->tensor.bytes);
      return byte_offset(memory_->tensor.local_ptr, ref.offset_bytes);
    case BufferKind::PeerTensor:
    case BufferKind::PeerStaging:
      throw std::invalid_argument(
          "device backend requires resolved runtime pointer for remote dst");
  }
  throw std::invalid_argument("device backend invalid destination reference");
}

void const* DeviceBackend::resolve_const(BufferRef const& ref,
                                         size_t bytes) const {
  switch (ref.kind) {
    case BufferKind::Staging:
      if (memory_->staging.local_ptr == nullptr) {
        throw std::invalid_argument("device backend staging buffer is missing");
      }
      validate_span("device backend staging", ref.offset_bytes, bytes,
                    memory_->staging.bytes);
      return byte_offset(memory_->staging.local_ptr, ref.offset_bytes);
    case BufferKind::Tensor:
      if (memory_->tensor.local_ptr == nullptr) {
        throw std::invalid_argument("device backend local tensor is missing");
      }
      validate_span("device backend local tensor", ref.offset_bytes, bytes,
                    memory_->tensor.bytes);
      return byte_offset(memory_->tensor.local_ptr, ref.offset_bytes);
    case BufferKind::PeerTensor:
    case BufferKind::PeerStaging:
      throw std::invalid_argument(
          "device backend requires resolved runtime pointer for remote src");
  }
  throw std::invalid_argument("device backend invalid source reference");
}

void DeviceBackend::ensure_runtime() {
  ensure_device_context();
  if (!Device::TaskManager::instance().inited()) {
    Device::TaskManager::instance().init(config_.task_capacity);
    owns_task_manager_ = true;
  }
  if (worker_pool_ != nullptr) {
    return;
  }
  Device::WorkerPool::Config cfg;
  cfg.numMaxWorkers = config_.max_fifos;
  cfg.threadsPerBlock = config_.threads_per_block;
  cfg.fifoCapacity = config_.fifo_capacity;
  cfg.smemSize = config_.smem_size;
  worker_pool_ = std::make_unique<Device::WorkerPool>(cfg);
  free_fifos_.clear();
  for (uint32_t fifo_id = 0; fifo_id < cfg.numMaxWorkers; ++fifo_id) {
    free_fifos_.push_back(fifo_id);
  }
}

void DeviceBackend::ensure_device_context() const {
  int current_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&current_device));
  if (current_device != local_device_idx_) {
    GPU_RT_CHECK(gpuSetDevice(local_device_idx_));
  }
}

void DeviceBackend::release_task_args(SubmittedTask& task) {
  if (task.args_released) {
    return;
  }
  if (worker_pool_ != nullptr) {
    worker_pool_->retireTask(task.fifo_id, task.task_id);
  }
  Device::TaskManager::instance().free_task_args(task.args_id);
  task.args_released = true;
}

uint32_t DeviceBackend::acquire_fifo(uint32_t flow_id, uint32_t num_blocks) {
  auto it = active_flows_.find(flow_id);
  if (it != active_flows_.end()) {
    ++it->second.inflight;
    return it->second.fifo_id;
  }
  if (worker_pool_ == nullptr) {
    throw std::runtime_error("device backend worker runtime is not initialized");
  }
  if (free_fifos_.empty()) {
    throw std::runtime_error("device backend has no available FIFO slots");
  }

  uint32_t fifo_id = free_fifos_.front();
  free_fifos_.pop_front();
  if (!worker_pool_->createWorker(fifo_id, num_blocks)) {
    free_fifos_.push_front(fifo_id);
    throw std::runtime_error("device backend failed to create worker");
  }
  worker_pool_->waitWorker(fifo_id);
  active_flows_.emplace(flow_id, ActiveFlow{fifo_id, 1});
  return fifo_id;
}

void DeviceBackend::stop_flow(uint32_t flow_id) {
  if (worker_pool_ == nullptr) return;
  auto it = active_flows_.find(flow_id);
  if (it == active_flows_.end() || it->second.inflight != 0) {
    return;
  }
  uint32_t fifo_id = it->second.fifo_id;
  worker_pool_->destroyWorker(fifo_id);
  for (auto& [_, submitted] : submitted_) {
    if (submitted.flow_id != flow_id || submitted.args_released) {
      continue;
    }
    release_task_args(submitted);
  }
  free_fifos_.push_back(fifo_id);
  active_flows_.erase(it);
}

uint32_t DeviceBackend::suggested_num_blocks(ExecOp const& op) const {
  size_t bytes_per_block = (op.kind == ExecOpKind::DeviceReduce) ? (1u << 20)
                                                                 : (4u << 20);
  uint32_t blocks = static_cast<uint32_t>(
      std::max<size_t>(1, (op.tile.size_bytes + bytes_per_block - 1) /
                              bytes_per_block));
  return std::min<uint32_t>(std::max<uint32_t>(1, blocks),
                            static_cast<uint32_t>(std::max(1, sm_count_)));
}

}  // namespace CCL
}  // namespace UKernel
