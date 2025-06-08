#include "proxy.hpp"
#include "ring_buffer.cuh"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#define kIterations 10000000
#define kQueueSize 128
#define kQueueMask (kQueueSize - 1)

static inline bool pin_thread_to_cpu(int cpu) {
  int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpu < 0 || cpu >= num_cpus) return false;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);

  pthread_t current_thread = pthread_self();
  return !pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}


void cpu_consume(RingBuffer* rb, int block_idx) {
  // printf("CPU thread for block %d started\n", block_idx);
  pin_thread_to_cpu(block_idx);

  uint64_t my_tail = 0;
  for (int seen = 0; seen < kIterations; ++seen) {
    // TODO: here, if CPU caches fifo->head, it may not see the updates from
    // GPU.
    while (rb->head == my_tail) {
#ifdef DEBUG_PRINT
      if (block_idx == 0) {
        printf(
            "CPU thread for block %d, waiting for head to advance: my_tail: "
            "%lu, head: %llu\n",
            block_idx, my_tail, fifo->head);
      }
#endif
      /* spin */
    }
    uint64_t idx = my_tail & kQueueMask;
    uint64_t cmd;
    do {
      cmd = rb->buf[idx].cmd;
      _mm_pause();  // Avoid hammering the cacheline.
    } while (cmd == 0);

#ifdef DEBUG_PRINT
    printf(
        "CPU thread for block %d, seen: %d, my_head: %llu, my_tail: %lu, "
        "consuming cmd %llu\n",
        block_idx, seen, fifo->head, my_tail,
        static_cast<unsigned long long>(cmd));
#endif
    uint64_t expected_cmd =
        (static_cast<uint64_t>(block_idx) << 32) | (seen + 1);
    if (cmd != expected_cmd) {
      fprintf(stderr, "Error: block %d, expected cmd %llu, got %llu\n",
              block_idx, static_cast<unsigned long long>(expected_cmd),
              static_cast<unsigned long long>(cmd));
      exit(1);
    }
    rb->buf[idx].cmd = 0;
    // std::atomic_thread_fence(std::memory_order_release);
    my_tail++;
    rb->tail = my_tail;
    // _mm_clflush(&(fifo->tail));
  }
}