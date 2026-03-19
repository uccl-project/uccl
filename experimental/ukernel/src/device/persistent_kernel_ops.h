#pragma once

#include "c2d_fifo_device.h"
#include "task.h"
#include <cassert>
#include <cstdint>

namespace UKernel {
namespace Device {

__device__ void run_copy_register(TaskArgs const& a, uint32_t block_id,
                                  uint32_t num_blocks);

template <typename T>
__device__ __forceinline__ T apply_red(ReduceType op, T a, T b);

template <typename T>
__device__ void run_reduce_inplace(TaskArgs const& a, uint32_t block_id,
                                  uint32_t num_blocks);

__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                     TaskArgs* d_task_args,
                                     bool* should_stop);

}  // namespace Device
}  // namespace UKernel
