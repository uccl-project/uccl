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