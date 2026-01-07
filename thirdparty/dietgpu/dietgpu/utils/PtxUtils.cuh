/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

namespace dietgpu {

__device__ __forceinline__ unsigned int
getBitfield(uint8_t val, int pos, int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;"
      : "=r"(ret)
      : "r"((uint32_t)val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ unsigned int
getBitfield(uint16_t val, int pos, int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;"
      : "=r"(ret)
      : "r"((uint32_t)val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ unsigned int
getBitfield(unsigned int val, int pos, int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ uint64_t
getBitfield(uint64_t val, int pos, int len) {
  uint64_t ret;
  asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ unsigned int
setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
  unsigned int ret;
  asm("bfi.b32 %0, %1, %2, %3, %4;"
      : "=r"(ret)
      : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__ uint32_t rotateLeft(uint32_t v, uint32_t shift) {
  uint32_t out;
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(out)
      : "r"(v), "r"(v), "r"(shift));
  return out;
}

__device__ __forceinline__ uint32_t rotateRight(uint32_t v, uint32_t shift) {
  uint32_t out;
  asm("shf.r.clamp.b32 %0, %1, %2, %3;"
      : "=r"(out)
      : "r"(v), "r"(v), "r"(shift));
  return out;
}

__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

template <typename T>
__device__ inline T warpReduceAllMin(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(0xffffffff, val);
#else
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val = min(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  }

  return val;
#endif
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllMax(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xffffffff, val);
#else
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  }

  return val;
#endif
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(0xffffffff, val);
#else
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  }

  return val;
#endif
}

} // namespace dietgpu
