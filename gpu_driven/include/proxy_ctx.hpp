#pragma once
#include "util/gpu_rt.h"
#include <infiniband/verbs.h>
#include <atomic>
#include <unordered_map>
#include <vector>

struct ProxyCtx {
  // RDMA objects
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_mr* mr = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_qp* ack_qp = nullptr;
  ibv_qp* recv_ack_qp = nullptr;

  // Remote memory
  uintptr_t remote_addr = 0;
  uint32_t remote_rkey = 0;
  uint32_t rkey = 0;

  // Progress/accounting
  std::atomic<uint64_t> posted{0};
  std::atomic<uint64_t> completed{0};
  std::atomic<bool> progress_run{true};

  // ACK receive ring
  std::vector<uint64_t> ack_recv_buf;
  ibv_mr* ack_recv_mr = nullptr;
  uint64_t largest_completed_wr = 0;
  bool has_received_ack = false;

  // For batched WR bookkeeping (largest_wr -> component wr_ids)
  std::unordered_map<uint64_t, std::vector<uint64_t>> wr_id_to_wr_ids;

  // ACK send counters (optional atomics)
  std::atomic<uint64_t> send_ack_posted{0};
  std::atomic<uint64_t> send_ack_completed{0};

  // GPU copy helpers (moved from function-static thread_local)
  gpuStream_t copy_stream = nullptr;
  bool peer_enabled[NUM_GPUS][NUM_GPUS] = {};
  size_t pool_index = 0;

  // Optional: per-GPU destination buffers if you previously used a global
  void* per_gpu_device_buf[NUM_GPUS] = {nullptr};
};