#ifndef RING_BUFFER_CUH
#define RING_BUFFER_CUH

#include "common.hpp"  // For TransferCmd, QUEUE_SIZE
#include <atomic>
#include <cuda.h>

// Host-pinned lock-free ring buffer (single-producer GPU, single-consumer CPU)
struct alignas(128) RingBuffer {
  uint64_t head;  // Next slot to produce
  uint64_t tail;  // Next slot to consume
  TransferCmd buf[kQueueSize];
  uint64_t cycle_accum;
  uint64_t op_count;
  uint64_t cycle_start;
  uint64_t cycle_end;
};

#endif  // RING_BUFFER_CUH