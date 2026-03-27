#pragma once

#include "high_perf.h"
#include "reduce_ops.h"
#include "reg_ops.h"
#include "tma_ops.h"

namespace UKernel {
namespace Device {

template <typename T>
__device__ __forceinline__ void copy(void* dst, void const* src, size_t count,
                                     void* smem_buf) {
  int tid = threadIdx.x;
  int nthread = blockDim.x;
  size_t bytes = count * sizeof(T);

  if (is_tma_supported() && smem_buf != nullptr && bytes <= 4096) {
    if (tid == 0) {
      TmaSemaphore sem;
      tma_init_semaphore(sem, 0);
      tma_load<T>(smem_buf, src, bytes, sem);
      tma_wait_group<0>();
      tma_store<T>(dst, smem_buf, bytes);
    }
    __syncthreads();
  } else {
    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    if (start >= count) {
      return;
    }
    size_t end = (start + chunk < count) ? (start + chunk) : count;
    for (size_t i = start; i < end; ++i) {
      static_cast<T*>(dst)[i] = static_cast<const T*>(src)[i];
    }
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
