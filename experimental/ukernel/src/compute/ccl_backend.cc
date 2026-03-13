#include "ccl_backend.h"
#include <stdexcept>

namespace UKernel {
namespace Compute {

ComputePersistentKernelBackend::ComputePersistentKernelBackend(
    PersistentKernel<Task>& kernel, void* dst_base, void const* src_base,
    DataType dtype, ReduceType reduce_type, TransferPath transfer_path,
    uint32_t num_blocks, void* staging_base)
    : kernel_(kernel),
      dst_base_(dst_base),
      src_base_(src_base),
      staging_base_(staging_base),
      dtype_(dtype),
      reduce_type_(reduce_type),
      transfer_path_(transfer_path),
      num_blocks_(num_blocks == 0 ? 1 : num_blocks) {}

char const* ComputePersistentKernelBackend::name() const {
  return "compute-persistent";
}

bool ComputePersistentKernelBackend::supports(
    UKernel::CCL::ExecutionOpKind kind) const {
  switch (kind) {
    case UKernel::CCL::ExecutionOpKind::PkCopy:
    case UKernel::CCL::ExecutionOpKind::PkReduce:
      return true;
    case UKernel::CCL::ExecutionOpKind::RdmaSend:
    case UKernel::CCL::ExecutionOpKind::RdmaRecv:
    case UKernel::CCL::ExecutionOpKind::CeCopy:
    case UKernel::CCL::ExecutionOpKind::EventWait:
    case UKernel::CCL::ExecutionOpKind::Barrier:
      return false;
  }
  return false;
}

UKernel::CCL::BackendToken ComputePersistentKernelBackend::submit(
    UKernel::CCL::ExecutionOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for compute backend");
  }

  CollArgs args{};
  bool stage_for_reduce =
      (op.flags & static_cast<uint32_t>(UKernel::CCL::ExecutionOpFlags::StageForReduce)) !=
      0;
  void* copy_dst =
      stage_for_reduce && staging_base_ != nullptr
          ? byte_offset(staging_base_, op.chunk.offset_bytes)
          : byte_offset(dst_base_, op.chunk.offset_bytes);
  void const* reduce_src =
      staging_base_ != nullptr
          ? byte_offset(staging_base_, op.chunk.offset_bytes)
          : byte_offset(src_base_, op.chunk.offset_bytes);

  args.src = const_cast<void*>(byte_offset(src_base_, op.chunk.offset_bytes));
  args.src2 = nullptr;
  args.dst = op.kind == UKernel::CCL::ExecutionOpKind::PkCopy
                 ? copy_dst
                 : byte_offset(dst_base_, op.chunk.offset_bytes);
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
  if (op.kind == UKernel::CCL::ExecutionOpKind::PkReduce) {
    args.src = const_cast<void*>(reduce_src);
    args.redType = reduce_type_;
  } else {
    args.redType = ReduceType::None;
  }
  args.requested_path = transfer_path_;
  args.resolved_path = TransferPath::Auto;

  uint32_t block_id = op.chunk.channel_id % num_blocks_;
  TaskType task_type = op.kind == UKernel::CCL::ExecutionOpKind::PkReduce
                           ? TaskType::CollReduce
                           : TaskType::CollCopy;
  Task task =
      TaskManager::instance().create_coll_task(args, task_type, dtype_, block_id);
  uint64_t task_id = kernel_.submit(task);

  UKernel::CCL::BackendToken token{next_token_++};
  submitted_[token.value] = SubmittedTask{block_id, task_id};
  return token;
}

bool ComputePersistentKernelBackend::poll(UKernel::CCL::BackendToken token) {
  auto it = submitted_.find(token.value);
  if (it == submitted_.end()) return true;
  return kernel_.is_done(it->second.block_id, it->second.task_id);
}

void ComputePersistentKernelBackend::release(UKernel::CCL::BackendToken token) {
  submitted_.erase(token.value);
}

void* ComputePersistentKernelBackend::byte_offset(void* base, size_t offset) const {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* ComputePersistentKernelBackend::byte_offset(void const* base,
                                                        size_t offset) const {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

}  // namespace Compute
}  // namespace UKernel
