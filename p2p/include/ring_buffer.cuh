#ifndef RING_BUFFER_CUH
#define RING_BUFFER_CUH

#include "common.hpp"  // For TransferCmd, QUEUE_SIZE
#include <atomic>
#include <cuda.h>

// Host-pinned lock-free ring buffer (single-producer GPU, single-consumer CPU)
struct alignas(128) RingBuffer {
  uint64_t head;  // Next slot to produce
  uint64_t tail;  // Next slot to consume
  volatile TransferCmd buf[QUEUE_SIZE];
  uint64_t cycle_accum;
  uint64_t op_count;
};

// Allocates a host-pinned ring buffer and returns device-visible pointer
RingBuffer* create_ring_buffer(CUstream_st* stream = nullptr);

#endif  // RING_BUFFER_CUH