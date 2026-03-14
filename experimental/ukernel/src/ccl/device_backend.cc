#include "device_backend.h"
#include <stdexcept>

namespace UKernel {
namespace CCL {

PersistentKernelBackend::PersistentKernelBackend(
    UKernel::Device::PersistentKernel<UKernel::Device::Task>& kernel,
    CollectiveBuffers buffers, UKernel::Device::DataType dtype,
    UKernel::Device::ReduceType reduce_type,
    UKernel::Device::TransferPath transfer_path, uint32_t num_blocks)
    : kernel_(kernel),
      buffers_(buffers),
      dtype_(dtype),
      reduce_type_(reduce_type),
      transfer_path_(transfer_path),
      num_blocks_(num_blocks == 0 ? 1 : num_blocks) {}

char const* PersistentKernelBackend::name() const {
  return "persistent-kernel";
}

bool PersistentKernelBackend::supports(ExecutionOpKind kind) const {
  switch (kind) {
    case ExecutionOpKind::PkCopy:
    case ExecutionOpKind::PkReduce:
      return true;
    case ExecutionOpKind::RdmaSend:
    case ExecutionOpKind::RdmaRecv:
    case ExecutionOpKind::CeCopy:
    case ExecutionOpKind::EventWait:
    case ExecutionOpKind::Barrier:
      return false;
  }
  return false;
}

BackendToken PersistentKernelBackend::submit(ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for persistent-kernel backend");
  }

  UKernel::Device::CollArgs args{};
  args.src = const_cast<void*>(resolve_src(op.src_role, op.chunk.offset_bytes));
  args.src2 = nullptr;
  args.dst = resolve_dst(op.dst_role, op.chunk.offset_bytes);
  args.bytes = op.chunk.size_bytes;
  args.op_id = op.op_id;
  args.step_id = static_cast<uint32_t>(op.op_id);
  args.chunk_id = op.chunk.chunk_index;
  args.completion_cookie = static_cast<uint32_t>(op.op_id);
  args.src_rank = op.src_rank;
  args.dst_rank = op.dst_rank;
  args.src_device = 0;
  args.dst_device = 0;
  args.flags = 0;
  args.redType = op.kind == ExecutionOpKind::PkReduce
                     ? reduce_type_
                     : UKernel::Device::ReduceType::None;
  args.requested_path = transfer_path_;
  args.resolved_path = UKernel::Device::TransferPath::Auto;

  uint32_t block_id = op.chunk.channel_id % num_blocks_;
  UKernel::Device::TaskType task_type =
      op.kind == ExecutionOpKind::PkReduce
          ? UKernel::Device::TaskType::CollReduce
          : UKernel::Device::TaskType::CollCopy;
  UKernel::Device::Task task = UKernel::Device::TaskManager::instance()
                                    .create_coll_task(args, task_type, dtype_,
                                                      block_id);
  uint64_t task_id = kernel_.submit(task);

  BackendToken token{next_token_++};
  submitted_[token.value] = SubmittedTask{block_id, task_id};
  return token;
}

bool PersistentKernelBackend::poll(BackendToken token) {
  auto it = submitted_.find(token.value);
  if (it == submitted_.end()) return true;
  return kernel_.is_done(it->second.block_id, it->second.task_id);
}

void PersistentKernelBackend::release(BackendToken token) {
  submitted_.erase(token.value);
}

void* PersistentKernelBackend::byte_offset(void* base, size_t offset) const {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* PersistentKernelBackend::byte_offset(void const* base,
                                                 size_t offset) const {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

void* PersistentKernelBackend::resolve_dst(BufferRole role, size_t offset) const {
  switch (role) {
    case BufferRole::FinalOutput:
      return byte_offset(buffers_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(buffers_.recv_staging, offset);
    case BufferRole::None:
    case BufferRole::LocalInput:
    case BufferRole::RemoteInput:
    case BufferRole::RemoteReduced:
      throw std::invalid_argument("invalid dst buffer role for persistent-kernel backend");
  }
  throw std::invalid_argument("unknown dst buffer role");
}

void const* PersistentKernelBackend::resolve_src(BufferRole role,
                                                 size_t offset) const {
  switch (role) {
    case BufferRole::LocalInput:
      return byte_offset(buffers_.local_input, offset);
    case BufferRole::RemoteInput:
      return byte_offset(buffers_.remote_input, offset);
    case BufferRole::RemoteReduced:
      return byte_offset(buffers_.remote_reduced, offset);
    case BufferRole::RecvStaging:
      return byte_offset(buffers_.recv_staging, offset);
    case BufferRole::FinalOutput:
      return byte_offset(buffers_.final_output, offset);
    case BufferRole::None:
      throw std::invalid_argument("invalid src buffer role for persistent-kernel backend");
  }
  throw std::invalid_argument("unknown src buffer role");
}

CopyEngineBackend::CopyEngineBackend(
    CollectiveBuffers buffers, int dst_device, int src_device,
    gpuStream_t stream)
    : buffers_(buffers) {
  int current_device = 0;
  GPU_RT_CHECK(gpuGetDevice(&current_device));
  dst_device_ = dst_device >= 0 ? dst_device : current_device;
  src_device_ = src_device >= 0 ? src_device : dst_device_;

  if (stream != nullptr) {
    stream_ = stream;
  } else {
    set_device(dst_device_);
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&stream_, gpuStreamNonBlocking));
    owns_stream_ = true;
    set_device(current_device);
  }
}

CopyEngineBackend::~CopyEngineBackend() {
  for (auto& kv : submitted_) {
    set_device(dst_device_);
    gpuEventDestroy(kv.second.event);
  }
  if (owns_stream_ && stream_ != nullptr) {
    set_device(dst_device_);
    gpuStreamDestroy(stream_);
  }
}

char const* CopyEngineBackend::name() const { return "copy-engine"; }

bool CopyEngineBackend::supports(ExecutionOpKind kind) const {
  return kind == ExecutionOpKind::CeCopy;
}

BackendToken CopyEngineBackend::submit(ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for copy-engine backend");
  }

  void* dst = resolve_dst(op.dst_role, op.chunk.offset_bytes);
  void const* src = resolve_src(op.src_role, op.chunk.offset_bytes);

  int current_device = 0;
  GPU_RT_CHECK(gpuGetDevice(&current_device));
  set_device(dst_device_);
  if (src_device_ == dst_device_) {
    GPU_RT_CHECK(
        gpuMemcpyAsync(dst, src, op.chunk.size_bytes, gpuMemcpyDeviceToDevice,
                       stream_));
  } else {
    GPU_RT_CHECK(gpuMemcpyPeerAsync(dst, dst_device_, const_cast<void*>(src),
                                    src_device_, op.chunk.size_bytes, stream_));
  }

  gpuEvent_t event = nullptr;
  GPU_RT_CHECK(gpuEventCreateWithFlags(&event, gpuEventDisableTiming));
  GPU_RT_CHECK(gpuEventRecord(event, stream_));
  set_device(current_device);

  BackendToken token{next_token_++};
  submitted_[token.value] = SubmittedCopy{event};
  ++submissions_;
  return token;
}

bool CopyEngineBackend::poll(BackendToken token) {
  auto it = submitted_.find(token.value);
  if (it == submitted_.end()) return true;

  int current_device = 0;
  GPU_RT_CHECK(gpuGetDevice(&current_device));
  set_device(dst_device_);
  gpuError_t status = gpuEventQuery(it->second.event);
  set_device(current_device);
  if (status == gpuSuccess) return true;
  if (status == gpuErrorNotReady) return false;
  GPU_RT_CHECK(status);
  return false;
}

void CopyEngineBackend::release(BackendToken token) {
  auto it = submitted_.find(token.value);
  if (it == submitted_.end()) return;

  int current_device = 0;
  GPU_RT_CHECK(gpuGetDevice(&current_device));
  set_device(dst_device_);
  GPU_RT_CHECK(gpuEventDestroy(it->second.event));
  set_device(current_device);
  submitted_.erase(it);
}

void* CopyEngineBackend::byte_offset(void* base, size_t offset) const {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* CopyEngineBackend::byte_offset(void const* base,
                                           size_t offset) const {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

void* CopyEngineBackend::resolve_dst(BufferRole role, size_t offset) const {
  switch (role) {
    case BufferRole::FinalOutput:
      return byte_offset(buffers_.final_output, offset);
    case BufferRole::RecvStaging:
      return byte_offset(buffers_.recv_staging, offset);
    case BufferRole::None:
    case BufferRole::LocalInput:
    case BufferRole::RemoteInput:
    case BufferRole::RemoteReduced:
      throw std::invalid_argument("invalid dst buffer role for copy-engine backend");
  }
  throw std::invalid_argument("unknown dst buffer role");
}

void const* CopyEngineBackend::resolve_src(BufferRole role,
                                           size_t offset) const {
  switch (role) {
    case BufferRole::LocalInput:
      return byte_offset(buffers_.local_input, offset);
    case BufferRole::RemoteInput:
      return byte_offset(buffers_.remote_input, offset);
    case BufferRole::RemoteReduced:
      return byte_offset(buffers_.remote_reduced, offset);
    case BufferRole::RecvStaging:
      return byte_offset(buffers_.recv_staging, offset);
    case BufferRole::FinalOutput:
      return byte_offset(buffers_.final_output, offset);
    case BufferRole::None:
      throw std::invalid_argument("invalid src buffer role for copy-engine backend");
  }
  throw std::invalid_argument("unknown src buffer role");
}

void CopyEngineBackend::set_device(int device) const {
  GPU_RT_CHECK(gpuSetDevice(device));
}

}  // namespace CCL
}  // namespace UKernel
