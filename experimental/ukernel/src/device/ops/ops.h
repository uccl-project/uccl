#pragma once

#include "high_perf.h"
#include "reduce_ops.h"
#include "reg_ops.h"
#include "tma_ops.h"
#include <type_traits>

// Reduce ILP (16B loads in flight per thread) is a BUILD-TIME knob so the
// fully-unrolled kernel stays cheap to compile: REDUCE_ILP=4|8|16 (default
// 4). A runtime dispatch over 4/8/16 tripled cicc+ptxas time to ~20 min
// per file on B300; with a single compile-time value it is ~1-2 min.
#ifndef UK_REDUCE_ILP
#define UK_REDUCE_ILP 4
#endif
#if UK_REDUCE_ILP != 4 && UK_REDUCE_ILP != 8 && UK_REDUCE_ILP != 16
#error "UK_REDUCE_ILP must be 4, 8 or 16"
#endif

// Fast reduce set: (dtype, op) combos that get the ILP-vectorized kernel.
// Everything else falls back to a correct scalar loop, so these knobs
// only trade compile time / peak per-SM throughput, never correctness.
// The default covers the common ML set (fp32/fp16/bf16 x Sum); builds
// that need every combo pass UK_REDUCE_FAST_DTYPES=127 UK_REDUCE_FAST_OPS=31.
// This is the compile-time lever for the sm_103 ILP=16 blowup: the full
// 7-dtype x 5-op matrix instantiates ~35 unrolled ILP loops per file
// (50+ min on B300); the default fast set instantiates 3.
#ifndef UK_REDUCE_FAST_DTYPES  // 1=fp32 2=fp16 4=bf16 8=int32 16=int8 32=int64 64=fp64
#define UK_REDUCE_FAST_DTYPES 7
#endif
#ifndef UK_REDUCE_FAST_OPS  // 1=Sum 2=Prod 4=Max 8=Min 16=BitwiseAnd
#define UK_REDUCE_FAST_OPS 1
#endif

template <typename T>
__device__ constexpr bool is_fast_reduce_dtype() {
  if constexpr (std::is_same_v<T, float>)
    return (UK_REDUCE_FAST_DTYPES & 1) != 0;
  else if constexpr (std::is_same_v<T, __half>)
    return (UK_REDUCE_FAST_DTYPES & 2) != 0;
  else if constexpr (std::is_same_v<T, nv_bfloat16>)
    return (UK_REDUCE_FAST_DTYPES & 4) != 0;
  else if constexpr (std::is_same_v<T, int32_t>)
    return (UK_REDUCE_FAST_DTYPES & 8) != 0;
  else if constexpr (std::is_same_v<T, int8_t>)
    return (UK_REDUCE_FAST_DTYPES & 16) != 0;
  else if constexpr (std::is_same_v<T, int64_t>)
    return (UK_REDUCE_FAST_DTYPES & 32) != 0;
  else if constexpr (std::is_same_v<T, double>)
    return (UK_REDUCE_FAST_DTYPES & 64) != 0;
  else
    return false;
}

// Large-task TMA bulk reduce (sm_90+): REDUCE_SMEM_KB is the dynamic
// shared-memory budget per block (src+dst chunk buffers + mbarriers);
// TMA_REDUCE=1 enables the cp.async.bulk chunked path instead of the
// register/ILP-limited vector loop. Both are build-time so the kernel
// and the launch config stay consistent (no runtime size mismatch).
#ifndef UK_TMA_REDUCE
#define UK_TMA_REDUCE 0
#endif
#ifndef UK_REDUCE_SMEM_KB
#define UK_REDUCE_SMEM_KB 4
#endif
#define UK_REDUCE_SMEM_BYTES (UK_REDUCE_SMEM_KB * 1024)

namespace UKernel {
namespace Device {
namespace {

// Vector type: 32B on SM80+, 16B otherwise
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
using Vec = ulonglong4;
static constexpr int kVEC_BYTES = 32;
#else
using Vec = uint4;
static constexpr int kVEC_BYTES = 16;
#endif

}  // anonymous namespace

template <typename T>
__device__ __forceinline__ void copy(void* dst, void const* src, size_t count,
                                     void* smem_buf) {
  int tid = threadIdx.x;
  int nthread = blockDim.x;
  (void)smem_buf;  // TMA bulk copy removed: it hangs on peer-mapped
                   // destinations and gains nothing over the vectorized
                   // loop for local ones.

  // The vectorized path needs both pointers 16B-aligned (Vec is a
  // 16B-aligned type; on SM80+ it is 32 bytes wide but still only
  // requires 16B alignment). Misaligned inputs — e.g. a fused
  // reduce+copy block offset when shard/num_blocks is not a vector
  // multiple — fault the SM with a misaligned address (NVRM Xid 13),
  // so fall back to a coalesced scalar loop.
  if (((reinterpret_cast<uintptr_t>(dst) |
        reinterpret_cast<uintptr_t>(src)) &
       0xF) == 0) {
    // Vectorized copy (kVEC_BYTES-byte loads through read-only cache)
    constexpr int NELTS_PER_VEC = kVEC_BYTES / (int)sizeof(T);
    size_t nvec = count / NELTS_PER_VEC;

    Vec const* src_v = reinterpret_cast<Vec const*>(src);
    Vec* dst_v = reinterpret_cast<Vec*>(dst);

    for (size_t vi = tid; vi < nvec; vi += nthread) dst_v[vi] = src_v[vi];

    // Scalar tail
    if constexpr (NELTS_PER_VEC > 1) {
      size_t base = nvec * NELTS_PER_VEC;
      T* dst_t = static_cast<T*>(dst);
      T const* src_t = static_cast<T const*>(src);
      for (size_t i = base + tid; i < count; i += nthread)
        dst_t[i] = src_t[i];
    }
  } else {
    T* dst_t = static_cast<T*>(dst);
    T const* src_t = static_cast<T const*>(src);
    for (size_t i = static_cast<size_t>(tid); i < count; i += nthread)
      dst_t[i] = src_t[i];
  }
}

// 16B-aligned typed vector for wide reduce loads/stores. All supported
// reduce dtypes have sizeof(T) in {1,2,4,8}, so N = 16/sizeof(T) is
// 16/8/4/2 and the struct is exactly 16B.
template <typename T, int N>
struct alignas(16) TypedVec {
  T e[N];
};

// 16B volatile load (ld.global.cv, bypasses L2). A fused reduce+copy
// accumulator is written by the peer GPU over IPC; on PCIe systems the
// destination GPU's L2 can hold stale lines of it, so normal cached
// reads miss the peer's contribution (observed on L40S: ~2% sparse
// wrong sums, run-varying). NVLink/Blackwell systems are coherent, so
// this only bites PCIe setups.
template <typename T, int N>
__device__ __forceinline__ TypedVec<T, N> typedvec_ld_cv(
    TypedVec<T, N> const* p) {
  static_assert(sizeof(TypedVec<T, N>) == 16);
  TypedVec<T, N> v;
  uint4 tmp = __ldcv(reinterpret_cast<uint4 const*>(p));
  *reinterpret_cast<uint4*>(&v) = tmp;
  return v;
}

// Bulk vectorized dst[i] = dst[i] op src[i], compile-time op so the
// element loop fully unrolls and memory traffic stays 16B wide per
// thread (NCCL-style wide reduce; the previous scalar loop was neither
// vectorized nor coalesced).
template <typename T, int N, ReduceType op>
__device__ __forceinline__ void vec_read_reduce_store(T* dst, T const* src,
                                                      size_t count, int tid,
                                                      int nthread,
                                                      bool peer_dst = false) {
  using V = TypedVec<T, N>;
  V const* svp = reinterpret_cast<V const*>(src);
  V* dvp = reinterpret_cast<V*>(dst);
  size_t nvec = count / static_cast<size_t>(N);
  // ILP: each thread keeps kVecInFlight independent 16B loads in flight
  // before storing. The element-wise reduce is latency-bound at low
  // unroll (measured: U=1 needs ~16 blocks for 150 GB/s, U=4 reaches
  // ~185 GB/s with 8 blocks on A40 — the DRAM ceiling), so fewer blocks
  // saturate the memory system. U = UK_REDUCE_ILP: B300's HBM3e
  // latency-bandwidth product is ~5x A40's, so it needs more bytes in
  // flight per block (rebuild with REDUCE_ILP=8/16 to sweep).
  constexpr int kVecInFlight = UK_REDUCE_ILP;
  size_t stride = static_cast<size_t>(nthread) * kVecInFlight;
  for (size_t base = static_cast<size_t>(tid); base < nvec;
       base += stride) {
    V sv[kVecInFlight];
    V dv[kVecInFlight];
    size_t idx[kVecInFlight];
#pragma unroll
    for (int u = 0; u < kVecInFlight; ++u)
      idx[u] = base + static_cast<size_t>(u) * nthread;
#pragma unroll
    for (int u = 0; u < kVecInFlight; ++u)
      if (idx[u] < nvec) sv[u] = svp[idx[u]];
#pragma unroll
    for (int u = 0; u < kVecInFlight; ++u)
      if (idx[u] < nvec) {
        dv[u] = peer_dst ? typedvec_ld_cv(&dvp[idx[u]]) : dvp[idx[u]];
#pragma unroll
        for (int e = 0; e < N; ++e)
          dv[u].e[e] = apply_reduce(dv[u].e[e], sv[u].e[e], op);
      }
#pragma unroll
    for (int u = 0; u < kVecInFlight; ++u)
      if (idx[u] < nvec) dvp[idx[u]] = dv[u];
  }
  size_t base = nvec * static_cast<size_t>(N);
  for (size_t i = base + tid; i < count; i += nthread)
    dst[i] = apply_reduce(peer_dst ? __ldcv(&dst[i]) : dst[i], src[i], op);
}

#if __CUDA_ARCH__ >= 900 && UK_TMA_REDUCE
// Chunked cp.async.bulk reduce for large tasks: per chunk, bulk-load src
// and dst into shared memory (mbarrier-tracked), reduce in smem, bulk-
// store dst back. In-flight bytes per block are bounded by smem, not by
// per-thread registers — this is the lever to reach native-class
// per-block throughput at low block counts. TmaSemaphores are carved out
// of the same smem buffer (mbarriers must live in shared memory).
template <typename T, ReduceType op>
__device__ __forceinline__ void tma_bulk_reduce_chunk(
    T* dst_t, T const* src_t, char* smem, size_t off_bytes, size_t len_bytes,
    int tid, int nthread) {
  // cp.async.bulk (TMA) on this B300/CUDA combo silently truncates
  // transfers above ~47KB (measured: 48640 OK, 49152 delivers only the
  // first ~256B, mbarrier completes on the partial data — wrong results
  // in the allreduce). Cap the chunk well below that; larger in-flight
  // bytes then need a multi-slot pipeline, not a bigger single chunk.
  constexpr size_t kChunkUncapped =
      ((UK_REDUCE_SMEM_BYTES - 2 * sizeof(TmaSemaphore)) / 2) &
      ~static_cast<size_t>(31);
  constexpr size_t kMaxChunkBytes = 32 * 1024;
  constexpr size_t kChunkBytes =
      kChunkUncapped < kMaxChunkBytes ? kChunkUncapped : kMaxChunkBytes;
  T* smem_src = reinterpret_cast<T*>(smem);
  T* smem_dst = reinterpret_cast<T*>(smem + kChunkBytes);
  TmaSemaphore* sem_src = reinterpret_cast<TmaSemaphore*>(
      smem + 2 * kChunkBytes);
  TmaSemaphore* sem_dst = reinterpret_cast<TmaSemaphore*>(
      smem + 2 * kChunkBytes + sizeof(TmaSemaphore));
  size_t const e0 = off_bytes / sizeof(T);

  // mbarrier.init per chunk (count=1, phase 0). Phase-toggling across
  // chunks hangs at high chunk counts (observed at 512 chunks/tile), and
  // the barrier must be re-armed for every arrive.expect_tx — a fresh
  // init per chunk is what the earlier runs validated.
  if (tid == 0) {
    tma_init_semaphore(*sem_src, 1);
    tma_init_semaphore(*sem_dst, 1);
  }
  __syncthreads();
  if (tid == 0) {
    tma_load<T>(smem_src, src_t + e0, len_bytes, *sem_src);
    tma_load<T>(smem_dst, dst_t + e0, len_bytes, *sem_dst);
  }
  __syncthreads();
  if (tid == 0) {
    tma_wait(*sem_src, 0);
    tma_wait(*sem_dst, 0);
  }
  __syncthreads();
  size_t const n = len_bytes / sizeof(T);
  // Vectorize the smem reduce: one 16B TypedVec per thread per iteration
  // instead of scalar element ops (4x fewer smem accesses / loop iters on
  // fp32). kChunkBytes is 32B-aligned so both chunk buffers are 16B
  // aligned for TypedVec; the scalar tail covers any remainder.
  constexpr int kVec = 16 / static_cast<int>(sizeof(T));
  if (kVec > 1) {
    using V = TypedVec<T, kVec>;
    V* sv = reinterpret_cast<V*>(smem_src);
    V* dv = reinterpret_cast<V*>(smem_dst);
    size_t const nvec = n / static_cast<size_t>(kVec);
    for (size_t i = static_cast<size_t>(tid); i < nvec;
         i += static_cast<size_t>(nthread)) {
      V const s = sv[i];
      V d = dv[i];
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        d.e[e] = apply_reduce(d.e[e], s.e[e], op);
      dv[i] = d;
    }
    for (size_t i = nvec * static_cast<size_t>(kVec) +
                    static_cast<size_t>(tid);
         i < n; i += static_cast<size_t>(nthread))
      smem_dst[i] = apply_reduce(smem_dst[i], smem_src[i], op);
  } else {
    for (size_t i = static_cast<size_t>(tid); i < n;
         i += static_cast<size_t>(nthread))
      smem_dst[i] = apply_reduce(smem_dst[i], smem_src[i], op);
  }
  __syncthreads();
  if (tid == 0) {
    tma_store<T>(dst_t + e0, smem_dst, len_bytes);
    // Wait for THIS thread's bulk-store group before the next chunk's
    // TMA loads reuse the smem buffer (non-issuing threads have no group
    // and return immediately; the barrier below fences the reuse).
    tma_wait_group<0>();
    // Make the async-proxy writes visible to generic-proxy agents (host
    // memsets, other blocks' loads) — without this, a store can land after
    // a later memset/load and corrupt data (observed: warmup's store raced
    // the pre-timing memset, leaving dst = warmup + rounds instead of
    // rounds).
    tma_fence_async_global();
  }
  __syncthreads();
}

template <typename T, ReduceType op>
__device__ __forceinline__ void tma_bulk_reduce(void* dst, void const* src,
                                                size_t count,
                                                void* smem_buf) {
  constexpr size_t kChunkBytes =
      ((UK_REDUCE_SMEM_BYTES - 2 * sizeof(TmaSemaphore)) / 2) &
      ~static_cast<size_t>(31);
  if (kChunkBytes < 32) return;  // smem too small; caller falls back

  char* smem = static_cast<char*>(smem_buf);
  int tid = threadIdx.x;
  int nthread = blockDim.x;

  T* dst_t = static_cast<T*>(dst);
  T const* src_t = static_cast<T const*>(src);
  size_t bytes = count * sizeof(T);

  size_t off = 0;
  while (off + kChunkBytes <= bytes) {
    tma_bulk_reduce_chunk<T, op>(dst_t, src_t, smem, off, kChunkBytes, tid,
                                 nthread);
    off += kChunkBytes;
  }
  if (off < bytes) {
    // Tail (< kChunkBytes): TMA bulk on odd-sized final chunks produced
    // garbage (observed: 130KB wrong clusters per 512KB block slice,
    // starting at the tail chunk). Fall back to the ILP vector path —
    // the tail is at most one chunk, so the throughput impact is nil.
    constexpr int kVec = 16 / static_cast<int>(sizeof(T));
    size_t tail_count = (bytes - off) / sizeof(T);
    if (kVec > 1 && (reinterpret_cast<uintptr_t>(dst_t + off / sizeof(T)) &
                     0xF) == 0 &&
        (reinterpret_cast<uintptr_t>(src_t + off / sizeof(T)) & 0xF) == 0) {
      vec_read_reduce_store<T, kVec, op>(dst_t + off / sizeof(T),
                                         src_t + off / sizeof(T), tail_count,
                                         tid, nthread);
    } else {
      for (size_t i = off / sizeof(T); i < count; ++i)
        dst_t[i] = apply_reduce(dst_t[i], src_t[i], op);
    }
  }
}
#endif

template <typename T, ReduceType op>
__device__ __forceinline__ void read_reduce_store_op(void* dst, void const* src,
                                                     size_t count,
                                                     void* smem_buf,
                                                     bool peer_dst = false) {
  int tid = threadIdx.x;
  int nthread = blockDim.x;
  size_t bytes = count * sizeof(T);

  // cp.async.bulk needs multiple-of-16 size and 16B-aligned addresses;
  // a 4-byte allreduce used to crash here. The mbarrier is carved after
  // the payload, so the payload must leave room for it inside the actual
  // dynamic-smem budget (UK_REDUCE_SMEM_BYTES matches the launch config).
  if (!peer_dst && is_tma_supported() && smem_buf != nullptr &&
      bytes + sizeof(TmaSemaphore) <= UK_REDUCE_SMEM_BYTES &&
      bytes % 16 == 0 &&
      (reinterpret_cast<uintptr_t>(dst) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(src) & 0xF) == 0) {
    T* dst_ptr = static_cast<T*>(dst);
    T const* src_ptr = static_cast<T const*>(src);
    T* temp_result = static_cast<T*>(smem_buf);

    if (tid == 0) {
      // mbarrier carved out of smem after the payload (a stack mbarrier
      // is invalid — previously hung/crashed on sub-4KB allreduces).
      TmaSemaphore* sem = reinterpret_cast<TmaSemaphore*>(
          static_cast<char*>(smem_buf) + bytes);
      tma_init_semaphore(*sem, 1);
      tma_load<T>(smem_buf, dst_ptr, bytes, *sem);
      tma_wait(*sem, 0);
    }
    __syncthreads();

    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = (tid + 1 == nthread) ? count : start + chunk;
    for (size_t i = start; i < end; ++i)
      temp_result[i] = apply_reduce(temp_result[i], src_ptr[i], op);

    __syncthreads();

    if (tid == 0) {
      tma_store<T>(dst, smem_buf, bytes);
      tma_wait_group<0>();
    }
    return;
  }

  T* dst_ptr = static_cast<T*>(dst);
  T const* src_ptr = static_cast<T const*>(src);
#if __CUDA_ARCH__ >= 900 && UK_TMA_REDUCE
  // Large-task TMA bulk path: chunks of smem-sized blocks via cp.async.bulk.
  if (smem_buf != nullptr &&
      (reinterpret_cast<uintptr_t>(dst_ptr) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(src_ptr) & 0xF) == 0) {
    tma_bulk_reduce<T, op>(dst_ptr, src_ptr, count, smem_buf);
    return;
  }
#endif
  constexpr int kVec = 16 / static_cast<int>(sizeof(T));
  if (kVec > 1 &&
      (reinterpret_cast<uintptr_t>(dst_ptr) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(src_ptr) & 0xF) == 0) {
    vec_read_reduce_store<T, kVec, op>(dst_ptr, src_ptr, count, tid, nthread,
                                       peer_dst);
  } else {
    // Unaligned pointers (odd tile/block offsets): coalesced scalar path.
    for (size_t i = tid; i < count; i += nthread)
      dst_ptr[i] = apply_reduce(peer_dst ? __ldcv(&dst_ptr[i]) : dst_ptr[i],
                                src_ptr[i], op);
  }
}

// Generic scalar fallbacks for (dtype, op) combos outside the fast set.
// Correct for every dtype/op; one cheap instantiation per dtype instead
// of the unrolled ILP/TMA bodies.
template <typename T>
__device__ __forceinline__ void read_reduce_store_generic(void* dst,
                                                          void const* src,
                                                          size_t count,
                                                          ReduceType op,
                                                          bool peer_dst = false) {
  int tid = threadIdx.x;
  int nthread = blockDim.x;
  T* dst_ptr = static_cast<T*>(dst);
  T const* src_ptr = static_cast<T const*>(src);
  for (size_t i = static_cast<size_t>(tid); i < count;
       i += static_cast<size_t>(nthread))
    dst_ptr[i] = apply_reduce(peer_dst ? __ldcv(&dst_ptr[i]) : dst_ptr[i],
                              src_ptr[i], op);
}

// dst[i] = dst[i] op src[i] over [0, count). Runtime op dispatched to a
// compile-time specialization so the vector loop fully unrolls. Only the
// ops in UK_REDUCE_FAST_OPS instantiate the heavy path; the rest fall
// back to the generic scalar loop.
template <typename T>
__device__ __forceinline__ void read_reduce_store(void* dst, void const* src,
                                                  size_t count, ReduceType op,
                                                  void* smem_buf,
                                                  bool peer_dst = false) {
  switch (op) {
#if UK_REDUCE_FAST_OPS & 1
    case ReduceType::Sum:
      read_reduce_store_op<T, ReduceType::Sum>(dst, src, count, smem_buf,
                                               peer_dst);
      break;
#endif
#if UK_REDUCE_FAST_OPS & 2
    case ReduceType::Prod:
      read_reduce_store_op<T, ReduceType::Prod>(dst, src, count, smem_buf,
                                                peer_dst);
      break;
#endif
#if UK_REDUCE_FAST_OPS & 4
    case ReduceType::Max:
      read_reduce_store_op<T, ReduceType::Max>(dst, src, count, smem_buf,
                                               peer_dst);
      break;
#endif
#if UK_REDUCE_FAST_OPS & 8
    case ReduceType::Min:
      read_reduce_store_op<T, ReduceType::Min>(dst, src, count, smem_buf,
                                               peer_dst);
      break;
#endif
#if UK_REDUCE_FAST_OPS & 16
    case ReduceType::BitwiseAnd:
      read_reduce_store_op<T, ReduceType::BitwiseAnd>(dst, src, count,
                                                      smem_buf);
      break;
#endif
    default:
      read_reduce_store_generic<T>(dst, src, count, op);
      break;
  }
}

}  // namespace Device
}  // namespace UKernel
