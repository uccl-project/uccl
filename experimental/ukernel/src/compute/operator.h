#pragma once

#include "c2d_fifo_device.h"
#include "task.h"
#include <cassert>
#include <cstdint>

namespace UKernel {
namespace Compute {

inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

__device__ void run_copy(CollArgs const& a);

template <typename T>
__device__ __forceinline__ T apply_red(ReduceType op, T a, T b);

template <typename T>
__device__ void run_reduce_inplace(CollArgs const& a);

template <typename T>
__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<T>* fifos,
                                     CollArgs* d_coll, MoeArgs* d_moe,
                                     bool* should_stop = nullptr);

}  // namespace Compute
}  // namespace UKernel
