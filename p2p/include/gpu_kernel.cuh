#ifndef GPU_KERNEL_CUH
#define GPU_KERNEL_CUH

#include "ring_buffer.hpp"

// Kernel: GPU pushes commands into the ring buffer asynchronously
__global__ void push_kernel(RingBuffer* rb, void* src, size_t bytes, 
                            uint32_t dst_rank, uint32_t dst_gpu, int iters);

#endif // GPU_KERNEL_CUH