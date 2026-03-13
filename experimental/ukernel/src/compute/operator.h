#pragma once

#include "c2d_fifo_device.h"
#include "d2c_fifo_device.hpp"
#include "sm_fifo_device.h"
#include "task.h"
#include <cassert>
#include <cstdint>

namespace UKernel {
namespace Compute {

inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

__device__ void run_copy(CollArgs const& a);
__device__ void run_copy_register(CollArgs const& a);
__device__ void run_copy_tma(CollArgs const& a);

template <typename T>
__device__ __forceinline__ T apply_red(ReduceType op, T a, T b);

template <typename T>
__device__ void run_reduce_inplace(CollArgs const& a);

__global__ void basePersistentKernel(
    mscclpp::C2DDeviceHandle<Task>* fifos,
    mscclpp::SmDeviceHandle<Task>* sm_fifos,
    mscclpp::FifoDeviceHandle* d2c_fifo, CollArgs* d_coll, MoeArgs* d_moe,
    GemmArgs* d_gemm, bool* should_stop = nullptr);

}  // namespace Compute
}  // namespace UKernel
