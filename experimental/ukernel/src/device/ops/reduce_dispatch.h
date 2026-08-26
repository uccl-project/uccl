#pragma once

#include "ops.h"
#include "task.h"

namespace UKernel {
namespace Device {

// run_reduce<T> — the ILP/TMA reduce template. It lives in this header so
// the persistent-kernel TU only ever CALLS the per-dtype dispatch
// functions below (cross-TU, relocatable device code): inlining the full
// dtype matrix into the worker kernel blew the 255-register budget (the
// standalone fp32 reduce needs ~254 registers at ILP=16, so the union of
// several instantiations spilled ~6.6KB and cut the pure-reduce rate to
// ~1/4 of the spill-free direct launch: 245 vs 935 GB/s at BLK=32 on
// B300). Each dispatch instantiation lives in reduce_dispatch.cu with its
// own register allocation.
template <typename T>
__device__ __forceinline__ void run_reduce(TaskArgs const& a, uint32_t block_id,
                                           uint32_t num_blocks,
                                           void* smem_buf) {
  T* dst = reinterpret_cast<T*>(a.dst);
  T const* src = reinterpret_cast<T const*>(a.src);
  const uint64_t total_count = static_cast<uint64_t>(a.bytes) / sizeof(T);

  const uint64_t max_threads_per_block = 1024;
  if (blockDim.x > max_threads_per_block) return;

  // Round each block's chunk to the 16B vector width so every block
  // starts at a 16B-aligned offset. Shard counts like 16MB/12 ranks =
  // 349,524 floats / 8 blocks = 43,690 (not a multiple of 4) put odd
  // blocks at 16B-misaligned addresses, and the vectorized reduce/copy
  // (TypedVec/Vec, 16B-aligned types) faults the SM with a misaligned
  // address (NVRM Xid 13). Mirrors run_typed_copy's partition.
  constexpr uint64_t kVecElems = 16 / sizeof(T);  // 4 for float
  const uint64_t count_per_block =
      (total_count / num_blocks / kVecElems) * kVecElems;
  const uint64_t block_offset = block_id * count_per_block;
  const uint64_t my_count = (block_id + 1 == num_blocks)
                                ? (total_count - block_offset)
                                : count_per_block;

  // Same-host fused reduce+copy: the accumulator (dst) was written by
  // the peer GPU over IPC, so reads must bypass the destination L2
  // (ld.global.cv) to see the peer's contribution. Cross-node proxy
  // tasks and plain reduces read locally-written data and keep the
  // cached path.
  bool const peer_dst = a.reduce_copy() && !a.rdma_fused_proxy();
  if constexpr (is_fast_reduce_dtype<T>()) {
    read_reduce_store<T>(dst + block_offset, src + block_offset,
                         static_cast<size_t>(my_count), a.red_type(),
                         smem_buf, peer_dst);
    if (a.reduce_copy() && !a.rdma_fused_proxy()) {
      // Fused reduce+copy: forward the just-reduced shard to the next
      // rank's accumulation buffer (device LD/ST write to peer, the
      // alltoall-proven direction). Same block partition as the reduce.
      // RDMA proxy mode skips this: the host will RDMA-put `dst` after
      // the device notifies it through the D2H ring.
      T* dst2 = reinterpret_cast<T*>(a.dst2);
      copy<T>(dst2 + block_offset, dst + block_offset,
              static_cast<size_t>(my_count), smem_buf);
    }
  } else {
    // dtype outside the fast set: generic scalar reduce (correct for all
    // ops, no ILP/TMA instantiation).
    read_reduce_store_generic<T>(dst + block_offset, src + block_offset,
                                 static_cast<size_t>(my_count),
                                 a.red_type(), peer_dst);
    if (a.reduce_copy() && !a.rdma_fused_proxy()) {
      T* dst2 = reinterpret_cast<T*>(a.dst2);
      copy<T>(dst2 + block_offset, dst + block_offset,
              static_cast<size_t>(my_count), smem_buf);
    }
  }
}

// Per-dtype reduce entry points, defined in reduce_dispatch.cu. The
// persistent kernels call these (cross-TU, never inlined).
#define UK_DECLARE_REDUCE_DISPATCH(suffix)                                   \
  __device__ void dispatch_reduce_##suffix(TaskArgs const& args,             \
                                           uint32_t block_id,                \
                                           uint32_t num_blocks,              \
                                           void* smem_buf)
UK_DECLARE_REDUCE_DISPATCH(fp32);
UK_DECLARE_REDUCE_DISPATCH(fp16);
UK_DECLARE_REDUCE_DISPATCH(bf16);
UK_DECLARE_REDUCE_DISPATCH(int32);
UK_DECLARE_REDUCE_DISPATCH(int8);
UK_DECLARE_REDUCE_DISPATCH(int64);
UK_DECLARE_REDUCE_DISPATCH(fp64);
#undef UK_DECLARE_REDUCE_DISPATCH

}  // namespace Device
}  // namespace UKernel
