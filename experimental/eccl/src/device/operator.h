#pragma once

#include "c2d_fifo_device.h"
#include "task.h"
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

namespace eccl {

inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

__device__ void run_copy(OpTask const& t);

template <typename T>
__device__ __forceinline__ T apply_red(OpRedType op, T a, T b);

template <typename T>
__device__ void run_reduce_inplace(OpTask const& t);

template <typename T>
__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<T> fifo,
                                     bool* should_stop = nullptr);

}  // namespace eccl
