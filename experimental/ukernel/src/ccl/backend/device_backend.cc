#include "device_backend.h"
#include "../utils.h"
#include "../../device/task.h"
#include "../../device/worker.h"
#include "../../include/gpu_rt.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace UKernel {
namespace CCL {

namespace {

Device::DataType to_device_dtype(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Int8:    return Device::DataType::Int8;
    case ScalarType::Int32:   return Device::DataType::Int32;
    case ScalarType::Int64:   return Device::DataType::Int64;
    case ScalarType::Float16: return Device::DataType::Fp16;
    case ScalarType::Float32: return Device::DataType::Fp32;
    case ScalarType::Float64: return Device::DataType::Fp64;
    case ScalarType::BFloat16: return Device::DataType::Bf16;
    default: break;
  }
  throw std::invalid_argument("device backend does not support this dtype");
}

Device::ReduceType to_reduce_type(ReductionKind reduction) {
  switch (reduction) {
    case ReductionKind::None:       return Device::ReduceType::None;
    case ReductionKind::Sum:        return Device::ReduceType::Sum;
    case ReductionKind::Prod:       return Device::ReduceType::Prod;
    case ReductionKind::Max:        return Device::ReduceType::Max;
    case ReductionKind::Min:        return Device::ReduceType::Min;
    case ReductionKind::BitwiseAnd: return Device::ReduceType::BitwiseAnd;
  }
  return Device::ReduceType::None;
}

}  // namespace

DeviceBackend::DeviceBackend(DeviceBackendConfig const& config)
    : config_(config) {
  if (config_.task_capacity == 0)
    throw std::invalid_argument("device backend task_capacity must be positive");
  if (config_.max_fifos == 0)
    throw std::invalid_argument("device backend max_fifos must be positive");
  if (config_.threads_per_block == 0)
    throw std::invalid_argument("device backend threads_per_block must be positive");
  if (config_.fifo_capacity == 0)
    throw std::invalid_argument("device backend fifo_capacity must be positive");

  int device = 0;
  GPU_RT_CHECK(gpuGetDevice(&device));
  local_device_idx_ = device;
  GPU_RT_CHECK(
      gpuDeviceGetAttribute(&sm_count_, gpuDevAttrMultiProcessorCount, device));
  submitted_per_fifo_.resize(config_.max_fifos);
}

DeviceBackend::~DeviceBackend() {
  for (auto& m : submitted_per_fifo_) m.clear();
  active_streams_.clear();
  worker_pool_.reset();
  if (owns_task_manager_) {
    Device::TaskManager::instance().release();
    owns_task_manager_ = false;
  }
}

char const* DeviceBackend::name() const { return "device"; }

// ── validate — one-time init ──────────────────────────────────────────

void DeviceBackend::validate(CollectivePlan const& plan,
                              CollectiveBinding& binding) {
  int current_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&current_device));
  if (current_device != local_device_idx_)
    GPU_RT_CHECK(gpuSetDevice(local_device_idx_));

  if (plan.staging_bytes_required != 0 &&
      binding.role_buffer(CollectiveBufferRole::Scratch).local_ptr == nullptr)
    throw std::invalid_argument("device backend staging buffer is missing");
  if (plan.staging_bytes_required >
      binding.role_buffer(CollectiveBufferRole::Scratch).bytes)
    throw std::invalid_argument("device backend staging capacity insufficient");

  // Remember reduction dtype from binding (for DeviceRecvReduce kernel dispatch).
  reduction_dtype_ = binding.role_buffer(CollectiveBufferRole::Input).layout.dtype;
  reduction_kind_ = plan.reduction;

  // Pre-warm runtime & workers.
  ensure_runtime();
  for (Op const& op : plan.ops) {
    uint32_t fid = op.stream_index;
    if (active_streams_.count(fid) == 0)
      acquire_fifo(fid, suggested_num_blocks(op));
  }
}

// ── submit ─────────────────────────────────────────────────────────────

void DeviceBackend::set_signal_buffers(std::vector<GpuSignalPeer> const& bufs) {
  gpu_signal_bufs_ = bufs;
}

bool DeviceBackend::supports(OpKind kind) const {
  return kind == OpKind::DeviceCopy || kind == OpKind::DeviceReduce ||
         kind == OpKind::DeviceSend || kind == OpKind::DeviceRecvReduce ||
         kind == OpKind::DeviceRecv;
}

BackendToken DeviceBackend::submit(Op const& op, OpBindings const& bind,
                                    CollectiveBinding& binding) {
  if (!supports(op.kind))
    throw std::invalid_argument("unsupported op kind for device backend");
  if (worker_pool_ == nullptr)
    throw std::runtime_error("device backend not initialized — call validate() first");

  void const* src = bind.resolved_src
      ? bind.resolved_src
      : byte_offset(binding.role_buffer(
          buf_role(op.kind, true, op.copy_from_staging)).local_ptr,
          op.src_off);
  void* dst = bind.resolved_dst
      ? bind.resolved_dst
      : byte_offset(binding.role_buffer(
          buf_role(op.kind, false, op.copy_from_staging)).local_ptr,
          op.dst_off);

  Device::TaskArgs args{};
  args.src = const_cast<void*>(src);
  args.src2 = nullptr;
  args.dst = dst;
  args.bytes = op.bytes;
  args.src_rank = op.src_peer != ~0u ? static_cast<int>(op.src_peer) : binding.local_rank();
  args.dst_rank = op.dst_peer != ~0u ? static_cast<int>(op.dst_peer) : binding.local_rank();
  args.src_device = bind.src_device >= 0 ? bind.src_device : local_device_idx_;
  args.dst_device = bind.dst_device >= 0 ? bind.dst_device : local_device_idx_;
  args.set_red_type(to_reduce_type(
      (op.kind == OpKind::DeviceReduce ||
       op.kind == OpKind::DeviceRecvReduce)
          ? reduction_kind_ : ReductionKind::None));

  // SM IPC: resolve signal buffer from gpu_signal_bufs_.
  if (op.kind == OpKind::DeviceSend && op.dst_peer != ~0u &&
      static_cast<size_t>(op.dst_peer) < gpu_signal_bufs_.size()) {
    args.src2 = gpu_signal_bufs_[op.dst_peer].remote;
  } else if ((op.kind == OpKind::DeviceRecvReduce ||
              op.kind == OpKind::DeviceRecv) &&
             op.src_peer != ~0u &&
             static_cast<size_t>(op.src_peer) < gpu_signal_bufs_.size()) {
    args.src2 = gpu_signal_bufs_[op.src_peer].local;
  }
  if (op.kind == OpKind::DeviceSend || op.kind == OpKind::DeviceRecvReduce ||
      op.kind == OpKind::DeviceRecv) {
    uint8_t red = static_cast<uint8_t>(to_reduce_type(
        op.kind == OpKind::DeviceRecvReduce
            ? reduction_kind_ : ReductionKind::None));
    args.redTypeRaw = (bind.signal_seq << 8) | red;
  }

  Device::TaskType task_type;
  switch (op.kind) {
    case OpKind::DeviceReduce:         task_type = Device::TaskType::CollReduce; break;
    case OpKind::DeviceRecvReduce:   task_type = Device::TaskType::CollRecvReduce; break;
    case OpKind::DeviceSend:     task_type = Device::TaskType::CollSend; break;
    case OpKind::DeviceRecv:     task_type = Device::TaskType::CollRecv; break;
    default:                           task_type = Device::TaskType::CollCopy; break;
  }

  ScalarType op_dtype = (op.kind == OpKind::DeviceReduce ||
                         op.kind == OpKind::DeviceRecvReduce)
      ? reduction_dtype_ : ScalarType::Int8;

  uint32_t stream_id = op.stream_index;

  auto fit = active_streams_.find(stream_id);
  uint32_t fifo_id = (fit != active_streams_.end())
      ? fit->second.fifo_id
      : acquire_fifo(stream_id, suggested_num_blocks(op));

  Device::Task task =
      Device::TaskManager::instance().create_task(
          args, task_type,
          to_device_dtype(op_dtype), 0);

  uint64_t task_id = worker_pool_->enqueue(task, fifo_id);
  if (task_id == Device::WorkerPool::kInvalidTaskId) {
    Device::TaskManager::instance().free_task_args(task.args_index());
    return BackendToken{0};
  }

  BackendToken token{next_token_++};
  submitted_per_fifo_[fifo_id][token.value] = TaskRec{task_id, stream_id, task.args_index()};
  active_streams_[stream_id].pending++;
  return token;
}

// ── drain ──────────────────────────────────────────────────────────────

size_t DeviceBackend::drain(BackendToken* out, size_t max_count) {
  int current_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&current_device));
  if (current_device != local_device_idx_)
    GPU_RT_CHECK(gpuSetDevice(local_device_idx_));

  size_t count = 0;
  streams_to_stop_.clear();
  for (uint32_t fid = 0; fid < submitted_per_fifo_.size() && count < max_count; ++fid) {
    auto& fifo_map = submitted_per_fifo_[fid];
    if (fifo_map.empty()) continue;
    auto it = fifo_map.begin();
    while (it != fifo_map.end() && count < max_count) {
      TaskRec& rec = it->second;
      if (!worker_pool_->is_done(rec.task_id, fid)) { ++it; continue; }
      release_task_args(rec.args_id);
      out[count].value = it->first;
      out[count].failed = false;
      ++count;

      uint32_t stream_id = rec.stream_id;
      it = fifo_map.erase(it);

      auto fi = active_streams_.find(stream_id);
      if (fi != active_streams_.end() && fi->second.pending > 0) {
        --fi->second.pending;
        if (fi->second.pending == 0)
          streams_to_stop_.push_back(stream_id);
      }
    }
  }
  for (uint32_t stream_id : streams_to_stop_)
    stop_stream(stream_id);
  return count;
}

// ── stop ───────────────────────────────────────────────────────────────

void DeviceBackend::stop(uint32_t stream_id) {
  int current_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&current_device));
  if (current_device != local_device_idx_)
    GPU_RT_CHECK(gpuSetDevice(local_device_idx_));
  stop_stream(stream_id);
}

void DeviceBackend::stop_stream(uint32_t stream_id) {
  if (worker_pool_ == nullptr) return;
  auto it = active_streams_.find(stream_id);
  if (it == active_streams_.end() || it->second.pending != 0)
    return;
  uint32_t fifo_id = it->second.fifo_id;
  worker_pool_->destroyWorker(fifo_id);
  // Release any remaining task args for this stream.
  for (auto& [token_val, rec] : submitted_per_fifo_[fifo_id])
    release_task_args(rec.args_id);
  submitted_per_fifo_[fifo_id].clear();
  free_fifos_.push_back(fifo_id);
  active_streams_.erase(it);
}

// ── internal ───────────────────────────────────────────────────────────

void DeviceBackend::release_task_args(uint32_t args_id) {
  Device::TaskManager::instance().free_task_args(args_id);
}

void DeviceBackend::ensure_runtime() {
  if (!Device::TaskManager::instance().inited()) {
    Device::TaskManager::instance().init(config_.task_capacity);
    owns_task_manager_ = true;
  }
  if (worker_pool_ != nullptr) return;
  Device::WorkerPool::Config cfg;
  cfg.numMaxWorkers = config_.max_fifos;
  cfg.threadsPerBlock = config_.threads_per_block;
  cfg.fifoCapacity = config_.fifo_capacity;
  cfg.smemSize = config_.smem_size;
  worker_pool_ = std::make_unique<Device::WorkerPool>(cfg);
  free_fifos_.clear();
  for (uint32_t fid = 0; fid < cfg.numMaxWorkers; ++fid)
    free_fifos_.push_back(fid);
}

uint32_t DeviceBackend::acquire_fifo(uint32_t stream_id, uint32_t num_blocks) {
  auto it = active_streams_.find(stream_id);
  if (it != active_streams_.end())
    return it->second.fifo_id;
  if (worker_pool_ == nullptr)
    throw std::runtime_error("device runtime not initialized");
  if (free_fifos_.empty())
    throw std::runtime_error("no available FIFO slots");

  uint32_t fifo_id = free_fifos_.back();
  free_fifos_.pop_back();
  if (!worker_pool_->createWorker(fifo_id, num_blocks)) {
    free_fifos_.push_back(fifo_id);
    throw std::runtime_error("failed to create worker");
  }
  worker_pool_->waitWorker(fifo_id);
  active_streams_.emplace(stream_id, StreamRec{fifo_id, 0});
  return fifo_id;
}

uint32_t DeviceBackend::suggested_num_blocks(Op const& op) const {
  size_t elem_bytes = (op.kind == OpKind::DeviceReduce ||
                       op.kind == OpKind::DeviceRecvReduce)
      ? scalar_type_size(reduction_dtype_) : 1;
  size_t bytes_per_block = (op.kind == OpKind::DeviceReduce) ? (128u << 10) : (1u << 20);
  if (elem_bytes > 1) bytes_per_block *= elem_bytes;
  uint32_t blocks = static_cast<uint32_t>(std::max<size_t>(
      1, (op.bytes + bytes_per_block - 1) / bytes_per_block));
  return std::min<uint32_t>(std::max<uint32_t>(1, blocks),
                            static_cast<uint32_t>(std::max(1, sm_count_)));
}

}  // namespace CCL
}  // namespace UKernel
