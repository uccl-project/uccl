#pragma once

#include "c2d_fifo_device.h"
#include "d2c_fifo_device.hpp"
#include "task.h"
#include <cassert>
#include <cstdint>

namespace UKernel {
namespace Device {

inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

// Device-side helpers and the base kernel used by the persistent collective
// runtime. This stays in compute/ because it is an execution backend detail.
__device__ void run_copy(CollArgs const& a);
__device__ void run_copy_register(CollArgs const& a);
__device__ void run_copy_tma(CollArgs const& a);

template <typename T>
__device__ __forceinline__ T apply_red(ReduceType op, T a, T b);

template <typename T>
__device__ void run_reduce_inplace(CollArgs const& a);

__global__ void basePersistentKernel(
    mscclpp::C2DDeviceHandle<Task>* fifos, mscclpp::FifoDeviceHandle* d2c_fifo,
    CollArgs* d_coll, GemmArgs* d_gemm, bool* should_stop = nullptr);

}  // namespace Device
}  // namespace UKernel
