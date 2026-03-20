#pragma once

#include "reg_ops.h"
#include "reduce_ops.h"
#include "tma_ops.h"
#include "high_perf.h"

namespace UKernel {
namespace Device {

template <typename T>
__device__ __forceinline__ void copy(
    void* dst, const void* src, size_t count,
    int tid, int nthread, void* smem_buf) {
    if (is_tma_supported() && smem_buf != nullptr) {
        TmaSemaphore sem;
        tma_init_semaphore(sem, 0);
        tma_load<T>(smem_buf, src, count * sizeof(T), sem);
        tma_wait_group<0>();
        tma_store<T>(dst, smem_buf, count * sizeof(T));
    } else {
        reg_copy<T>(static_cast<T*>(dst), static_cast<const T*>(src), count, tid, nthread);
    }
}

template <typename T, ReduceType op>
__device__ __forceinline__ void read_reduce_store(
    void* dst, const void* src, size_t count,
    void* smem_buf, int tid, int nthread) {
    if (is_tma_supported() && smem_buf != nullptr) {
        T* dst_ptr = static_cast<T*>(dst);
        const T* src_ptr = static_cast<const T*>(src);
        
        // Load destination values first if we need to combine with them
        TmaSemaphore sem_dst;
        tma_init_semaphore(sem_dst, 0);
        tma_load<T>(smem_buf, dst_ptr, count * sizeof(T), sem_dst);
        tma_wait_group<0>();
        
        T* temp_result = static_cast<T*>(smem_buf);
        
        // Perform the reduction operation with source data
        for (size_t i = tid; i < count; i += nthread) {
            T src_val = src_ptr[i];  // Directly read from source
            temp_result[i] = apply_reduce(temp_result[i], src_val, op);  // Combine dst[i] with src[i]
        }

        __syncthreads();

        // Store the result back to destination
        tma_store<T>(dst, smem_buf, count * sizeof(T));
    } else {
        reg_reduce<T>(static_cast<T*>(dst), static_cast<const T*>(src), count, tid, nthread, op);
    }
}

}  // namespace Device
}  // namespace UKernel