#include "ring_buffer.cuh"
#include <new>

RingBuffer* create_ring_buffer(CUstream_st* stream) {
    void* host_mem = nullptr;
    CHECK_CUDA(cudaHostAlloc(&host_mem, sizeof(RingBuffer), cudaHostAllocPortable | cudaHostAllocMapped));
    auto* rb = new (host_mem) RingBuffer;
    rb->head.store(0, std::memory_order_relaxed);
    rb->tail.store(0, std::memory_order_relaxed);
    // Obtain device‑visible pointer
    void* dev_ptr = nullptr;
    CHECK_CUDA(cudaHostGetDevicePointer(&dev_ptr, host_mem, 0));
    // Optionally warm‑up mapping
    if (stream) CHECK_CUDA(cudaStreamSynchronize(stream));
    return reinterpret_cast<RingBuffer*>(dev_ptr); // return device pointer for GPU
}