// peer_copy.cuh
#pragma once

#include <cuda_runtime.h>

cudaError_t launch_peer_bulk_copy(void* dst_ptr, int dst_dev, void* src_ptr,
                                  int src_dev, size_t bytes,
                                  cudaStream_t stream = 0);
