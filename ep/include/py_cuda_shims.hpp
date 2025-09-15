#pragma once
#include "ring_buffer.cuh"
#include <cuda_runtime.h>

// Launch the GPU command-producer kernel via a CU shim.
// Returns cudaSuccess or a CUDA error.
cudaError_t launch_gpu_issue_batched_commands_shim(int blocks,
                                                   int threads_per_block,
                                                   size_t shmem_bytes,
                                                   cudaStream_t stream,
                                                   DeviceToHostCmdBuffer* rbs);