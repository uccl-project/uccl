#include "ring_buffer.hpp"
#include "gpu_kernel.cuh"

__global__ void push_kernel(RingBuffer* rb, void* src, size_t bytes, uint32_t dst_rank, uint32_t dst_gpu, int iters) {
    // Single thread/block pushes "iters" commands to the ring buffer
    if (threadIdx.x + blockIdx.x != 0) return;

    for (int i = 0; i < iters; ++i) {
        // acquire slot
        uint64_t head = atomicAdd(reinterpret_cast<unsigned long long*>(&rb->head), 1ULL);
        uint32_t idx  = head & QUEUE_MASK;
        rb->buf[idx].dst_rank = dst_rank;
        rb->buf[idx].dst_gpu  = dst_gpu;
        rb->buf[idx].src_ptr  = src;
        rb->buf[idx].bytes    = bytes;
        __threadfence_system(); // ensure writes visible to CPU
    }
}