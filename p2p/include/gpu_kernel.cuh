#ifndef GPU_KERNEL_CUH
#define GPU_KERNEL_CUH

#include "common.hpp"
#include "ring_buffer.cuh"

__global__ void gpu_issue_batched_commands(DeviceToHostCmdBuffer* rbs);

#endif  // GPU_KERNEL_CUH