#pragma once

#include "c2d_fifo_device.h"
#include "task.h"
#include <cassert>
#include <cstdint>

namespace UKernel {
namespace Device {

__device__ void run_copy(TaskArgs const& a, uint32_t block_id,
                         uint32_t num_blocks, void* smem_buf = nullptr);

template <typename T, ReduceType op>
__device__ void run_reduce(TaskArgs const& a, uint32_t block_id,
                           uint32_t num_blocks, void* smem_buf = nullptr);

__global__ void singlePersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                       TaskArgs* d_task_args,
                                       bool* should_stop);

__global__ void multiPersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                     TaskArgs* d_task_args,
                                     bool* should_stop,
                                     uint32_t* d_readyFlag);

}  // namespace Device
}  // namespace UKernel