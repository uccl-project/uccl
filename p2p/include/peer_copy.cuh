// peer_copy.cuh
#pragma once

#include "copy_ring.hpp"
#include <cuda_runtime.h>

cudaError_t launch_peer_bulk_copy(void* dst_ptr, int dst_dev, void* src_ptr,
                                  int src_dev, size_t bytes,
                                  cudaStream_t stream = 0);

cudaError_t launch_peer_bulk_copy2(CopyTask const* host_tasks, int num_tasks,
                                   cudaStream_t stream, int src_device,
                                   CopyTask*& d_tasks);

__global__ void peer_copy_kernel_vec_batched(CopyTask const* __restrict__ tasks,
                                             int num_tasks,
                                             int tasks_per_block);