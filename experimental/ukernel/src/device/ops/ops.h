#pragma once

#include "high_perf.h"
#include "reduce_ops.h"
#include "reg_ops.h"
#include "tma_ops.h"

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
  size_t bytes = count * sizeof(T);

  // TMA path for small messages (hardware async copy, up to 4KB)
  if (is_tma_supported() && smem_buf != nullptr && bytes <= 4096) {
    if (tid == 0) {
      TmaSemaphore sem;
      tma_init_semaphore(sem, 0);
      tma_load<T>(smem_buf, src, bytes, sem);
      tma_wait_group<0>();
      tma_store<T>(dst, smem_buf, bytes);
    }
    __syncthreads();
    return;
  }

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
    for (size_t i = base + tid; i < count; i += nthread) dst_t[i] = src_t[i];
  }
}

// 16B-aligned typed vector for wide reduce loads/stores. All supported
// reduce dtypes have sizeof(T) in {1,2,4,8}, so N = 16/sizeof(T) is
// 16/8/4/2 and the struct is exactly 16B.
template <typename T, int N>
struct alignas(16) TypedVec {
  T e[N];
};

// Bulk vectorized dst[i] = dst[i] op src[i], compile-time op so the
// element loop fully unrolls and memory traffic stays 16B wide per
// thread (NCCL-style wide reduce; the previous scalar loop was neither
// vectorized nor coalesced).
template <typename T, int N, ReduceType op>
__device__ __forceinline__ void vec_read_reduce_store(T* dst, T const* src,
                                                      size_t count, int tid,
                                                      int nthread) {
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
        dv[u] = dvp[idx[u]];
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
    dst[i] = apply_reduce(dst[i], src[i], op);
}

template <typename T, ReduceType op>
__device__ __forceinline__ void read_reduce_store_op(void* dst, void const* src,
                                                     size_t count,
                                                     void* smem_buf) {
  int tid = threadIdx.x;
  int nthread = blockDim.x;
  size_t bytes = count * sizeof(T);

  if (is_tma_supported() && smem_buf != nullptr && bytes <= 4096) {
    T* dst_ptr = static_cast<T*>(dst);
    T const* src_ptr = static_cast<T const*>(src);
    T* temp_result = static_cast<T*>(smem_buf);

    if (tid == 0) {
      TmaSemaphore sem_dst;
      tma_init_semaphore(sem_dst, 0);
      tma_load<T>(smem_buf, dst_ptr, bytes, sem_dst);
      tma_wait_group<0>();
    }
    __syncthreads();

    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = (tid + 1 == nthread) ? count : start + chunk;
    for (size_t i = start; i < end; ++i)
      temp_result[i] = apply_reduce(temp_result[i], src_ptr[i], op);

    __syncthreads();

    if (tid == 0) tma_store<T>(dst, smem_buf, bytes);
    return;
  }

  T* dst_ptr = static_cast<T*>(dst);
  T const* src_ptr = static_cast<T const*>(src);
  constexpr int kVec = 16 / static_cast<int>(sizeof(T));
  if (kVec > 1 &&
      (reinterpret_cast<uintptr_t>(dst_ptr) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(src_ptr) & 0xF) == 0) {
    vec_read_reduce_store<T, kVec, op>(dst_ptr, src_ptr, count, tid, nthread);
  } else {
    // Unaligned pointers (odd tile/block offsets): coalesced scalar path.
    for (size_t i = tid; i < count; i += nthread)
      dst_ptr[i] = apply_reduce(dst_ptr[i], src_ptr[i], op);
  }
}

// dst[i] = dst[i] op src[i] over [0, count). Runtime op dispatched to a
// compile-time specialization so the vector loop fully unrolls.
template <typename T>
__device__ __forceinline__ void read_reduce_store(void* dst, void const* src,
                                                  size_t count, ReduceType op,
                                                  void* smem_buf) {
  switch (op) {
    case ReduceType::Sum:
      read_reduce_store_op<T, ReduceType::Sum>(dst, src, count, smem_buf);
      break;
    case ReduceType::Prod:
      read_reduce_store_op<T, ReduceType::Prod>(dst, src, count, smem_buf);
      break;
    case ReduceType::Max:
      read_reduce_store_op<T, ReduceType::Max>(dst, src, count, smem_buf);
      break;
    case ReduceType::Min:
      read_reduce_store_op<T, ReduceType::Min>(dst, src, count, smem_buf);
      break;
    case ReduceType::BitwiseAnd:
      read_reduce_store_op<T, ReduceType::BitwiseAnd>(dst, src, count,
                                                      smem_buf);
      break;
    default:
      read_reduce_store_op<T, ReduceType::Sum>(dst, src, count, smem_buf);
      break;
  }
}

}  // namespace Device
}  // namespace UKernel
