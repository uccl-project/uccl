#ifndef RING_BUFFER_HPP
#define RING_BUFFER_HPP
#include "common.hpp"

// Host‑pinned lock‑free ring buffer (single producer GPU, single consumer CPU)
struct alignas(128) RingBuffer {
    std::atomic<uint64_t> head;  // producer writes
    std::atomic<uint64_t> tail;  // consumer writes
    TransferCmd           buf[QUEUE_SIZE];
};

RingBuffer* create_ring_buffer(CUstream_st* stream = nullptr);

#endif // RING_BUFFER_HPP