#include "gpu_kernel.cuh"
#include "py_cuda_shims.hpp"
#include "ring_buffer.cuh"
#include <cuda_runtime.h>

cudaError_t launch_gpu_issue_batched_commands_shim(int blocks,
                                                   int threads_per_block,
                                                   size_t shmem_bytes,
                                                   cudaStream_t stream,
                                                   DeviceToHostCmdBuffer* rbs) {
  gpu_issue_batched_commands<<<blocks, threads_per_block, shmem_bytes,
                               stream>>>(rbs);
  return cudaGetLastError();
}