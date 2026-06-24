#pragma once

#include "high_perf.h"
#include "reduce_ops.h"
#include "reg_ops.h"
#include "tma_ops.h"

namespace UKernel {
namespace Device {
namespace {

// ── Vector type: 32B on SM80+, 16B otherwise ──────────────────────────
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

  // ── TMA path for small messages (hardware async copy, up to 4KB) ──
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

  // ── Vectorized copy (kVEC_BYTES-byte loads through read-only cache) ──
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

template <typename T>
__device__ __forceinline__ void read_reduce_store(void* dst, void const* src,
                                                  size_t count, ReduceType op,
                                                  void* smem_buf) {
  int tid = threadIdx.x;
  int nthread = blockDim.x;
  size_t bytes = count * sizeof(T);

  if (is_tma_supported() && smem_buf != nullptr && bytes <= 4096) {
    T* dst_ptr = static_cast<T*>(dst);
    const T* src_ptr = static_cast<const T*>(src);
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
    for (size_t i = start; i < end; ++i) {
      temp_result[i] = apply_reduce(temp_result[i], src_ptr[i], op);
    }

    __syncthreads();

    if (tid == 0) {
      tma_store<T>(dst, smem_buf, bytes);
    }
  } else {
    T* dst_ptr = static_cast<T*>(dst);
    const T* src_ptr = static_cast<const T*>(src);
    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    if (start >= count) {
      return;
    }
    size_t end = (start + chunk < count) ? (start + chunk) : count;
    for (size_t i = start; i < end; ++i) {
      dst_ptr[i] = apply_reduce(dst_ptr[i], src_ptr[i], op);
    }
  }
}

}  // namespace Device
}  // namespace UKernel
