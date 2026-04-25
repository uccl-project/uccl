#pragma once

#include "reduce_ops.h"
#include "task.h"

namespace UKernel {
namespace Device {

constexpr int kWarpSize = 32;

#if __CUDA_ARCH__ >= 900

template <typename T, ReduceType op>
__device__ __forceinline__ T warp_reduce(T val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val = apply_reduce(val, __shfl_xor_sync(0xffffffff, val, offset), op);
  }
  return val;
}

template <ReduceType op>
__device__ __forceinline__ float warp_reduce(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = apply_reduce(val, __shfl_xor_sync(0xffffffff, val, offset), op);
  }
  return val;
}

template <typename T, ReduceType op>
__device__ __forceinline__ T block_reduce(T val, int tid) {
  static __shared__ T smem[32];
  int lane = tid % kWarpSize;
  int warp = tid / kWarpSize;

  val = warp_reduce<T, op>(val);

  if (lane == 0) smem[warp] = val;
  __syncthreads();

  if (warp == 0) {
    val = (tid < blockDim.x / kWarpSize) ? smem[lane] : T{};
    val = warp_reduce<T, op>(val);
  }
  return val;
}

template <typename T, ReduceType op>
__device__ __forceinline__ T block_reduce_sum(T* smem, T val, int tid) {
  int lane = tid % kWarpSize;
  int warp = tid / kWarpSize;

  val = warp_reduce<T, op>(val);

  if (lane == 0) smem[warp] = val;
  __syncthreads();

  if (warp == 0) {
    val = (lane < blockDim.x / kWarpSize) ? smem[lane] : T{};
    val = warp_reduce<T, op>(val);
  }
  return val;
}

__device__ __forceinline__ unsigned lanemask_lt() {
  unsigned mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
  return mask;
}

template <typename T, ReduceType op>
__device__ __forceinline__ T warp_scan(T val) {
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    T other = __shfl_up_sync(0xffffffff, val, offset);
    val = apply_reduce(val, other, op);
  }
  return val;
}

template <typename T, ReduceType op>
__device__ __forceinline__ T warp_scan_add(T val) {
  T result = warp_scan<T, op>(val);
  return __shfl_up_sync(0xffffffff, result, 1);
}

#else

template <typename T, ReduceType op>
__device__ __forceinline__ T warp_reduce(T val) {
  return val;
}

template <ReduceType op>
__device__ __forceinline__ float warp_reduce(float val) {
  return val;
}

template <typename T, ReduceType op>
__device__ __forceinline__ T block_reduce(T val, int tid) {
  return val;
}

template <typename T, ReduceType op>
__device__ __forceinline__ T block_reduce_sum(T* smem, T val, int tid) {
  return val;
}

__device__ __forceinline__ unsigned lanemask_lt() { return 0; }

template <typename T, ReduceType op>
__device__ __forceinline__ T warp_scan(T val) {
  return val;
}

template <typename T, ReduceType op>
__device__ __forceinline__ T warp_scan_add(T val) {
  return val;
}

#endif

}  // namespace Device
}  // namespace UKernel