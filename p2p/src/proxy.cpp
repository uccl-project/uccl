#include "proxy.hpp"
#include "common.hpp"
#include "rdma.hpp"
#include "ring_buffer.cuh"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

static inline bool pin_thread_to_cpu(int cpu) {
  int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpu < 0 || cpu >= num_cpus) return false;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);

  pthread_t current_thread = pthread_self();
  return !pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

inline uint64_t load_volatile_u64(uint64_t volatile* addr) {
  uint64_t val;
  asm volatile("movq %1, %0" : "=r"(val) : "m"(*addr) : "memory");
  return val;
}

void cpu_consume(RingBuffer* rb, int block_idx, void* gpu_buffer,
                 size_t total_size, int rank, char const* peer_ip) {
  printf("CPU thread for block %d started\n", block_idx);
  pin_thread_to_cpu(block_idx);

  RDMAConnectionInfo local_info, remote_info;
  ensure_thread_qp(gpu_buffer, total_size, &local_info, rank);

  modify_qp_to_init();

  printf("Local RDMA info: addr=0x%lx, rkey=0x%x\n", local_info.addr,
         local_info.rkey);
  exchange_connection_info(rank, peer_ip, block_idx, &local_info, &remote_info);
  printf("Exchanged remote_addr: 0x%lx, remote_rkey: 0x%x\n", remote_info.addr,
         remote_info.rkey);

  modify_qp_to_rtr(&remote_info);
  modify_qp_to_rts(&local_info);

  remote_addr = remote_info.addr;
  remote_rkey = remote_info.rkey;

  uint64_t my_tail = 0;
  auto total_rdma_write_durations =
      std::chrono::duration<double, std::micro>::zero();
  for (int seen = 0; seen < kIterations;) {
    // Force loading rb->head from DRAM.
    while (load_volatile_u64(&rb->head) == my_tail) {
#ifdef DEBUG_PRINT
      if (block_idx == 0) {
        printf(
            "CPU thread for block %d, waiting for head to advance: my_tail: "
            "%lu, head: %lu\n",
            block_idx, my_tail, rb->head);
      }
#endif
      /* spin */
      _mm_pause();
    }
    uint64_t idx = my_tail & kQueueMask;
    uint64_t cmd;
    do {
      cmd = rb->buf[idx].cmd;
      _mm_pause();  // Avoid hammering the cacheline.
    } while (cmd == 0);
#ifdef DEBUG_PRINT
    printf(
        "CPU thread for block %d, seen: %d, my_head: %lu, my_tail: %lu, "
        "consuming cmd %llu\n",
        block_idx, seen, rb->head, my_tail,
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

    if (false && rank == 0) {
      rdma_write_stub(gpu_buffer, total_size);
      poll_completion();
      printf("Polling completions done. %d out of %d\n", seen, kIterations);
    } else if (rb->head - my_tail == 1) {
      // Record time
      auto start = std::chrono::high_resolution_clock::now();
      post_rdma_async(gpu_buffer, total_size, seen);
      auto end = std::chrono::high_resolution_clock::now();
      total_rdma_write_durations +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      rb->buf[idx].cmd = 0;
      // std::atomic_thread_fence(std::memory_order_release);
      my_tail++;
      rb->tail = my_tail;
      // Initiate flush to DRAM
      _mm_clwb(&(rb->tail));
      // Ensure that flush to DRAM completes.
      _mm_sfence();
      ++seen;
    } else {
      int batch_size = rb->head - my_tail;
      // if (batch_size > kBatchSize) {
      //   batch_size = kBatchSize;
      // }
      // printf("Posting %d commands, original: %ld\n", batch_size,
      //  rb->head - my_tail);
      auto start = std::chrono::high_resolution_clock::now();
      post_rdma_async_chained(gpu_buffer, total_size, batch_size);
      auto end = std::chrono::high_resolution_clock::now();
      total_rdma_write_durations +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      for (int i = 0; i < batch_size; ++i) {
        uint64_t idx = (my_tail + i) & kQueueMask;
        rb->buf[idx].cmd = 0;
      }
      // std::atomic_thread_fence(std::memory_order_release);
      my_tail += batch_size;
      rb->tail = my_tail;
      // Initiate flush to DRAM
      _mm_clwb(&(rb->tail));
      // Ensure that flush to DRAM completes.
      _mm_sfence();
      seen += batch_size;
    }
  }

  printf("CPU thread for block %d finished consuming %d commands\n", block_idx,
         kIterations);

  printf("Average rdma write duration: %.2f us\n",
         total_rdma_write_durations.count() / kIterations);
}

void cpu_consume_local(RingBuffer* rb, int block_idx) {
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
            "%lu, head: %lu\n",
            block_idx, my_tail, rb->head);
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
        "CPU thread for block %d, seen: %d, my_head: %lu, my_tail: %lu, "
        "consuming cmd %llu\n",
        block_idx, seen, rb->head, my_tail,
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

    for (int i = 0; i < 100; ++i) {
      _mm_pause();
    }

    rb->buf[idx].cmd = 0;
    // std::atomic_thread_fence(std::memory_order_release);
    my_tail++;
    rb->tail = my_tail;
    // _mm_clflush(&(fifo->tail));
  }
}