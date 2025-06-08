#ifndef GPU_KERNEL_CUH
#define GPU_KERNEL_CUH

#include "common.hpp"
#include "ring_buffer.cuh"

__global__ void gpu_issue_batched_commands(RingBuffer* rbs, void** d_ptrs);

#ifdef MEASURE_PER_OP_LATENCY
// __device__ unsigned long long cycle_accum[kNumThBlocks];
// __device__ unsigned int op_count[kNumThBlocks];
#endif

#endif  // GPU_KERNEL_CUH