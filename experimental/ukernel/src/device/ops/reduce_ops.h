#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace UKernel {
namespace Device {

template <typename T>
__device__ __forceinline__ T apply_reduce(T a, T b, ReduceType op);

template <>
__device__ __forceinline__ float apply_reduce(float a, float b, ReduceType op) {
    if (op == ReduceType::Sum) return a + b;
    if (op == ReduceType::Prod) return a * b;
    if (op == ReduceType::Max) return fmaxf(a, b);
    if (op == ReduceType::Min) return fminf(a, b);
    return b;
}

template <>
__device__ __forceinline__ __half apply_reduce(__half a, __half b, ReduceType op) {
#if __CUDA_ARCH__ >= 700
    if (op == ReduceType::Sum) return __hadd(a,b);
    if (op == ReduceType::Prod) return __hmul(a,b);
    if (op == ReduceType::Max) return __hmax(a,b);
    if (op == ReduceType::Min) return __hmin(a,b);
#endif
    return b;
}

template <>
__device__ __forceinline__ int32_t apply_reduce(int32_t a, int32_t b, ReduceType op) {
    if (op == ReduceType::Sum) return a + b;
    if (op == ReduceType::Prod) return a * b;
    if (op == ReduceType::Max) return (a > b) ? a : b;
    if (op == ReduceType::Min) return (a < b) ? a : b;
    if (op == ReduceType::BitwiseAnd) return a & b;
    return b;
}

template <>
__device__ __forceinline__ int64_t apply_reduce(int64_t a, int64_t b, ReduceType op) {
    if (op == ReduceType::Sum) return a + b;
    if (op == ReduceType::Prod) return a * b;
    if (op == ReduceType::Max) return (a > b) ? a : b;
    if (op == ReduceType::Min) return (a < b) ? a : b;
    if (op == ReduceType::BitwiseAnd) return a & b;
    return b;
}

template <>
__device__ __forceinline__ double apply_reduce(double a, double b, ReduceType op) {
    if (op == ReduceType::Sum) return a + b;
    if (op == ReduceType::Prod) return a * b;
    if (op == ReduceType::Max) return fmax(a,b);
    if (op == ReduceType::Min) return fmin(a,b);
    return b;
}

template <>
__device__ __forceinline__ nv_bfloat16 apply_reduce(nv_bfloat16 a, nv_bfloat16 b, ReduceType op) {
#if __CUDA_ARCH__ >= 800
    if (op == ReduceType::Sum) return __hadd(a, b);
    if (op == ReduceType::Prod) return __hmul(a, b);
    if (op == ReduceType::Max) return __hmax(a, b);
    if (op == ReduceType::Min) return __hmin(a, b);
#endif
    return b;
}

template <>
__device__ __forceinline__ int8_t apply_reduce(int8_t a, int8_t b, ReduceType op) {
    if (op == ReduceType::Sum) return a + b;
    if (op == ReduceType::Prod) return a * b;
    if (op == ReduceType::Max) return (a > b) ? a : b;
    if (op == ReduceType::Min) return (a < b) ? a : b;
    if (op == ReduceType::BitwiseAnd) return a & b;
    return b;
}

}  // namespace Device
}  // namespace UKernel