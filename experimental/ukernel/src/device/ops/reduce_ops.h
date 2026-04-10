#pragma once

#include "../task.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_bf16.h>
#endif

#if defined(__HIP_PLATFORM_AMD__)
using nv_bfloat16 = hip_bfloat16;
#endif

namespace UKernel {
namespace Device {

template <typename T>
__device__ __forceinline__ T apply_reduce(T a, T b, ReduceType op);

#if defined(__HIP_PLATFORM_AMD__)
// ROCm compatibility fallback only: keep this implementation simple and
// obviously correct so fp16/bf16 reduce paths work on AMD before we add a
// tuned native implementation.
__device__ __forceinline__ __half apply_reduce_half_amd_naive(__half a,
                                                              __half b,
                                                              ReduceType op) {
  float af = __half2float(a);
  float bf = __half2float(b);
  if (op == ReduceType::Sum) return __float2half(af + bf);
  if (op == ReduceType::Prod) return __float2half(af * bf);
  if (op == ReduceType::Max) return __float2half(fmaxf(af, bf));
  if (op == ReduceType::Min) return __float2half(fminf(af, bf));
  return b;
}

// ROCm compatibility fallback only: keep this implementation simple and
// obviously correct so fp16/bf16 reduce paths work on AMD before we add a
// tuned native implementation.
__device__ __forceinline__ nv_bfloat16 apply_reduce_bf16_amd_naive(
    nv_bfloat16 a, nv_bfloat16 b, ReduceType op) {
  float af = static_cast<float>(a);
  float bf = static_cast<float>(b);
  if (op == ReduceType::Sum) return __float2bfloat16(af + bf);
  if (op == ReduceType::Prod) return __float2bfloat16(af * bf);
  if (op == ReduceType::Max) return __float2bfloat16(fmaxf(af, bf));
  if (op == ReduceType::Min) return __float2bfloat16(fminf(af, bf));
  return b;
}
#endif

template <>
__device__ __forceinline__ float apply_reduce(float a, float b, ReduceType op) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Prod) return a * b;
  if (op == ReduceType::Max) return fmaxf(a, b);
  if (op == ReduceType::Min) return fminf(a, b);
  return b;
}

template <>
__device__ __forceinline__ __half apply_reduce(__half a, __half b,
                                               ReduceType op) {
#if defined(__HIP_PLATFORM_AMD__)
  return apply_reduce_half_amd_naive(a, b, op);
#elif __CUDA_ARCH__ >= 700
  if (op == ReduceType::Sum) return __hadd(a, b);
  if (op == ReduceType::Prod) return __hmul(a, b);
  if (op == ReduceType::Max) return __hmax(a, b);
  if (op == ReduceType::Min) return __hmin(a, b);
#endif
  return b;
}

template <>
__device__ __forceinline__ int32_t apply_reduce(int32_t a, int32_t b,
                                                ReduceType op) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Prod) return a * b;
  if (op == ReduceType::Max) return (a > b) ? a : b;
  if (op == ReduceType::Min) return (a < b) ? a : b;
  if (op == ReduceType::BitwiseAnd) return a & b;
  return b;
}

template <>
__device__ __forceinline__ int64_t apply_reduce(int64_t a, int64_t b,
                                                ReduceType op) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Prod) return a * b;
  if (op == ReduceType::Max) return (a > b) ? a : b;
  if (op == ReduceType::Min) return (a < b) ? a : b;
  if (op == ReduceType::BitwiseAnd) return a & b;
  return b;
}

template <>
__device__ __forceinline__ double apply_reduce(double a, double b,
                                               ReduceType op) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Prod) return a * b;
  if (op == ReduceType::Max) return fmax(a, b);
  if (op == ReduceType::Min) return fmin(a, b);
  return b;
}

template <>
__device__ __forceinline__ nv_bfloat16 apply_reduce(nv_bfloat16 a,
                                                    nv_bfloat16 b,
                                                    ReduceType op) {
#if defined(__HIP_PLATFORM_AMD__)
  return apply_reduce_bf16_amd_naive(a, b, op);
#elif __CUDA_ARCH__ >= 800
  if (op == ReduceType::Sum) return __hadd(a, b);
  if (op == ReduceType::Prod) return __hmul(a, b);
  if (op == ReduceType::Max) return __hmax(a, b);
  if (op == ReduceType::Min) return __hmin(a, b);
#endif
  return b;
}

template <>
__device__ __forceinline__ int8_t apply_reduce(int8_t a, int8_t b,
                                               ReduceType op) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Prod) return a * b;
  if (op == ReduceType::Max) return (a > b) ? a : b;
  if (op == ReduceType::Min) return (a < b) ? a : b;
  if (op == ReduceType::BitwiseAnd) return a & b;
  return b;
}

}  // namespace Device
}  // namespace UKernel
