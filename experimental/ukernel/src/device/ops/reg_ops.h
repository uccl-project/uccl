#pragma once

#include "reduce_ops.h"
#include <type_traits>

namespace UKernel {
namespace Device {

template <typename T>
__device__ __forceinline__ void reg_copy(T* dst, const T* src, size_t count, int tid, int nthread) {
    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = (tid + 1 == nthread) ? count : start + chunk;
    for (size_t i = start; i < end; ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
__device__ __forceinline__ void reg_reduce(T* dst, const T* src, size_t count, int tid, int nthread, ReduceType op) {
    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = start + chunk < count ? start + chunk : count;
    for (size_t i = start; i < end; ++i) dst[i] = apply_reduce(dst[i], src[i], op);
}

template <typename T>
__device__ __forceinline__ void reg_multi_reduce(T* dst, const T* const* srcs, uint32_t n_src,
                                                size_t count, int tid, int nthread, ReduceType op) {
    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = start + chunk < count ? start + chunk : count;
    for (size_t i = start; i < end; ++i) {
        T val = srcs[0][i];
        for (uint32_t s = 1; s < n_src; ++s) val = apply_reduce(val, srcs[s][i], op);
        dst[i] = val;
    }
}

template <typename T>
__device__ __forceinline__ void reg_fill(T* dst, T val, size_t count, int tid, int nthread) {
    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = start + chunk < count ? start + chunk : count;
    for (size_t i = start; i < end; ++i) dst[i] = val;
}

template <typename T, int N>
__device__ __forceinline__ void reg_copy_vectorized(T* dst, const T* src, size_t count, int tid, int nthread) {
    constexpr size_t kVecSize = N * sizeof(T);
    size_t n_vec = count / N;
    if (n_vec == 0) {
        reg_copy(dst, src, count, tid, nthread);
        return;
    }
    size_t chunk = (n_vec + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = start + chunk < n_vec ? start + chunk : n_vec;

    using VecT = typename std::aligned_storage<N * sizeof(T), kVecSize>::type;
    auto* dst_vec = reinterpret_cast<VecT*>(dst);
    auto* src_vec = reinterpret_cast<const VecT*>(src);

    for (size_t i = start; i < end; ++i) dst_vec[i] = src_vec[i];
    size_t remaining = count - n_vec * N;
    if (remaining > 0 && tid == 0) {
        for (size_t i = 0; i < remaining; ++i) dst[n_vec * N + i] = src[n_vec * N + i];
    }
}

template <int N>
__device__ __forceinline__ void reg_copy_float4(float4* dst, const float4* src, size_t count, int tid, int nthread) {
    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = start + chunk < count ? start + chunk : count;
    for (size_t i = start; i < end; ++i) dst[i] = src[i];
}

template <typename T>
__device__ __forceinline__ void reg_to_shmem_async(void* smem, const T* reg_src, size_t count) {
    const char* src = reinterpret_cast<const char*>(reg_src);
    char* dst = static_cast<char*>(smem);
    constexpr size_t kAlign = 16;
    size_t aligned_count = (count * sizeof(T)) & ~(kAlign - 1);
    for (size_t i = 0; i < aligned_count; i += kAlign) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "l"(dst + i), "l"(src + i), "n"(kAlign));
    }
    if (aligned_count < count * sizeof(T)) {
        size_t remaining = count * sizeof(T) - aligned_count;
        for (size_t i = 0; i < remaining; ++i) dst[aligned_count + i] = src[aligned_count + i];
    }
    asm volatile("cp.async.commit_group;");
}

template <typename T>
__device__ __forceinline__ void reg_from_shmem_async(T* reg_dst, const void* smem, size_t count) {
    const char* src = static_cast<const char*>(smem);
    char* dst = reinterpret_cast<char*>(reg_dst);
    constexpr size_t kAlign = 16;
    size_t aligned_count = (count * sizeof(T)) & ~(kAlign - 1);
    for (size_t i = 0; i < aligned_count; i += kAlign) {
        asm volatile("cp.async.ca.global.shared [%0], [%1], %2;" :: "l"(dst + i), "l"(src + i), "n"(kAlign));
    }
    if (aligned_count < count * sizeof(T)) {
        size_t remaining = count * sizeof(T) - aligned_count;
        for (size_t i = 0; i < remaining; ++i) dst[aligned_count + i] = src[aligned_count + i];
    }
    asm volatile("cp.async.commit_group;");
}

template <typename T>
__device__ __forceinline__ void reg_shmem_copy_async(void* dst_smem, const void* src_smem, size_t bytes) {
    const char* src = static_cast<const char*>(src_smem);
    char* dst = static_cast<char*>(dst_smem);
    constexpr size_t kAlign = 16;
    size_t aligned_bytes = bytes & ~(kAlign - 1);
    for (size_t i = 0; i < aligned_bytes; i += kAlign) {
        asm volatile("cp.async.ca.shared.shared [%0], [%1], %2;" :: "l"(dst + i), "l"(src + i), "n"(kAlign));
    }
    if (aligned_bytes < bytes) {
        for (size_t i = aligned_bytes; i < bytes; ++i) dst[i] = src[i];
    }
    asm volatile("cp.async.commit_group;");
}

template <typename T>
__device__ __forceinline__ void reg_unroll_copy(T* dst, const T* src, size_t count, int tid, int nthread) {
    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = start + chunk < count ? start + chunk : count;
    constexpr size_t kUnroll = 4;
    size_t i = start;
    size_t unroll_end = start + ((end - start) / kUnroll) * kUnroll;
    for (; i < unroll_end; i += kUnroll) {
        dst[i] = src[i];
        dst[i + 1] = src[i + 1];
        dst[i + 2] = src[i + 2];
        dst[i + 3] = src[i + 3];
    }
    for (; i < end; ++i) dst[i] = src[i];
}

}  // namespace Device
}  // namespace UKernel