#include "proxy.hpp"
#include "common.hpp"
#include "rdma.hpp"
#include "ring_buffer.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

inline uint64_t load_volatile_u64(uint64_t volatile* addr) {
  uint64_t val;
  asm volatile("movq %1, %0" : "=r"(val) : "m"(*addr) : "memory");
  return val;
}

void cpu_proxy(RingBuffer* rb, int block_idx, void* gpu_buffer,
               size_t total_size, int rank, char const* peer_ip) {
  printf("CPU thread for block %d started\n", block_idx);
  pin_thread_to_cpu(block_idx);

  ibv_cq* cq = create_per_thread_cq();
  RDMAConnectionInfo local_info, remote_info;
  create_per_thread_qp(gpu_buffer, total_size, &local_info, rank, cq);

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

  std::unordered_set<uint64_t> finished_wrs;
  std::mutex finished_wrs_mutex;

#ifdef SEPARATE_POLLING
  std::vector<std::thread> cq_threads;
  int cpu_id = kPollingThreadStartPort + block_idx;
  cq_threads.emplace_back(per_thread_polling, cpu_id, cq, &finished_wrs,
                          &finished_wrs_mutex);
#endif

  uint64_t my_tail = 0;
  auto total_rdma_write_durations =
      std::chrono::duration<double, std::micro>::zero();

  for (size_t seen = 0; my_tail < kIterations;) {
    poll_completions(cq, finished_wrs, finished_wrs_mutex);
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

#ifdef DEBUG_PRINT
    printf(
        "CPU thread for block %d, seen: %d, my_head: %lu, my_tail: %lu, "
        "consuming cmd %llu\n",
        block_idx, seen, rb->head, my_tail,
        static_cast<unsigned long long>(cmd));
#endif

    uint64_t cur_head = load_volatile_u64(&rb->head);
    size_t batch_size = cur_head - seen;

    if (batch_size > kQueueSize) {
      fprintf(stderr, "Error: batch_size %zu exceeds kQueueSize %d\n",
              batch_size, kQueueSize);
      exit(1);
    }

    if (batch_size < 0) {
      fprintf(stderr, "Error: batch_size %zu is negative\n", batch_size);
      exit(1);
    }

    std::vector<uint64_t> wrs_to_post;
    for (size_t i = seen; i < cur_head; ++i) {
      uint64_t cmd;
      do {
        cmd = rb->buf[i & kQueueMask].cmd;
        _mm_pause();  // Avoid hammering the cacheline.
      } while (cmd == 0);
      uint64_t expected_cmd =
          (static_cast<uint64_t>(block_idx) << 32) | (i + 1);
      if (cmd != expected_cmd) {
        fprintf(stderr, "Error: block %d, expected cmd %llu, got %llu\n",
                block_idx, static_cast<unsigned long long>(expected_cmd),
                static_cast<unsigned long long>(cmd));
        exit(1);
      }
      wrs_to_post.push_back(i);
    }
    seen = cur_head;

    if (wrs_to_post.size() != batch_size) {
      fprintf(stderr, "Error: wrs_to_post size %zu exceeds batch size %zu\n",
              wrs_to_post.size(), batch_size);
      exit(1);
    }

    if (!wrs_to_post.empty()) {
      auto start = std::chrono::high_resolution_clock::now();
      post_rdma_async_chained(gpu_buffer, total_size, batch_size, wrs_to_post,
                              cq, finished_wrs, finished_wrs_mutex);
      auto end = std::chrono::high_resolution_clock::now();
      total_rdma_write_durations +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }

    // This assumes we don't have EFA NICs.
#ifdef ASSUME_WR_IN_ORDER
    {
      std::lock_guard<std::mutex> lock(finished_wrs_mutex);
      for (auto i : finished_wrs) {
        rb->buf[i & kQueueMask].cmd = 0;
      }
      my_tail += finished_wrs.size();
      finished_wrs.clear();
    }
#else
    for (size_t i = my_tail; i < cur_head; ++i) {
      // uint64_t cmd = rb->buf[i & kQueueMask].cmd;
      {
        std::lock_guard<std::mutex> lock(finished_wrs_mutex);
        if (finished_wrs.count(i)) {
          rb->buf[i & kQueueMask].cmd = 0;
          my_tail++;
          finished_wrs.erase(i);
        } else {
          break;
        }
      }
    }
#endif
    rb->tail = my_tail;
    _mm_clwb(&(rb->tail));
    _mm_sfence();
  }

  printf(
      "CPU thread for block %d finished consuming %d commands, my_tail: %ld\n",
      block_idx, kIterations, my_tail);

  printf("Average rdma write duration: %.2f us\n",
         total_rdma_write_durations.count() / kIterations);

  if (check_cq_completion()) g_progress_run.store(false);
#ifdef SEPARATE_POLLING
  for (auto& t : cq_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
#endif
  printf("CPU thread for block %d finished, joined all CQ threads\n",
         block_idx);
}

void cpu_proxy_local(RingBuffer* rb, int block_idx) {
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
    my_tail++;
    rb->tail = my_tail;
  }
}