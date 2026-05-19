#pragma once

#include "c2d_fifo_device.h"
#include "task.h"
#include <cassert>
#include <cstdint>

// SM occupancy measurement — 5 timestamps per dispatched task
struct SmTimestamp { uint64_t t[5]; };

namespace UKernel {
namespace Device {

struct alignas(16) MultiBlockSync {
  uint32_t publishedPhase;
  uint32_t completedBlocks;
  uint32_t command;
  uint32_t hasCurrentArgs;
  Task currentTask;
  TaskArgs currentArgs;
};

__device__ __forceinline__ void run_copy(TaskArgs const& a, uint32_t block_id,
                                         uint32_t num_blocks,
                                         void* smem_buf = nullptr);

template <typename T>
__device__ __forceinline__ void run_reduce(TaskArgs const& a, uint32_t block_id,
                                           uint32_t num_blocks,
                                           void* smem_buf = nullptr);

// SM measurement buffers appended as kernel args. nullptr = skip recording.
__global__ void singlePersistentKernel(
    mscclpp::C2DDeviceHandle<Task>* c2d_fifos, TaskArgs* d_task_args,
    bool* should_stop,
    SmTimestamp* d_sm_ts, uint32_t* d_sm_count);

__global__ void multiPersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                      TaskArgs* d_task_args, bool* should_stop,
                                      MultiBlockSync* d_sync,
                                      SmTimestamp* d_sm_ts, uint32_t* d_sm_count);

__global__ void benchDispatchNopKernel();
__global__ void benchDispatchCopyFp32Kernel(TaskArgs args);
__global__ void benchDispatchReduceFp32Kernel(TaskArgs args);

}  // namespace Device
}  // namespace UKernel
