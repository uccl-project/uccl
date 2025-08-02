#pragma once
#include "common.hpp"
#include "ring_buffer.cuh"
#include <atomic>
#include <mutex>
#include <thread>

// Shared across all peer-copy workers on a process
struct PeerCopyShared {
  // Controls the worker loop
  std::atomic<bool> run{true};

  // P2P enable flags (once per GPU pair)
  std::once_flag peer_ok_flag[NUM_GPUS][NUM_GPUS];

  // Source GPU for receiving host-side staging to device
  int src_device = 0;
};

struct PeerWorkerCtx {
  // Counters / timings
  uint64_t async_memcpy_count = 0;
  uint64_t prev_completed_async_memcpy_count = 0;
  uint64_t async_memcpy_total_time = 0;
  uint64_t highest_issued_wr_id = 0;

  // Batch buffers
  CopyTask tasks[RECEIVER_BATCH_SIZE];
  uint64_t task_wrs[RECEIVER_BATCH_SIZE];

  // CUDA resources
  gpuStream_t stream = nullptr;
  CopyTask* d_tasks = nullptr;  // device buffer for tasks
};

void peer_copy_worker(PeerCopyShared& shared, PeerWorkerCtx& ctx,
                      CopyRingBuffer& ring, int idx);