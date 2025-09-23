#pragma once
#include "util/gpu_rt.h"
#include <infiniband/verbs.h>
#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

template <typename Key>
class TokenCounter {
 public:
  using MapType = std::map<Key, size_t>;
  void Add(Key const& key, size_t k) { counter_[key] += k; }
  size_t Get(Key const& key) const {
    auto it = counter_.find(key);
    return (it == counter_.end()) ? 0 : it->second;
  }
  void Reset(Key const& key) { counter_[key] = 0; }
  void Clear() { counter_.clear(); }

 private:
  MapType counter_;
};

using DispatchTokenKey = std::tuple<int, int, int>;
using CombineTokenKey = std::pair<int, int>;

struct WriteStruct {
  int expert_idx;
  int dst_rank;
  bool is_combine;
  int low_latency_buffer_idx;
};

struct ProxyCtx {
  // RDMA objects
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_mr* mr = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_qp* ack_qp = nullptr;
  ibv_qp* recv_ack_qp = nullptr;

  uint32_t dst_qpn;
  uint32_t dst_ack_qpn;
  struct ibv_ah* dst_ah = nullptr;

  // Remote memory
  uintptr_t remote_addr = 0;  // Base address of remote rdma_buffer
  uint64_t remote_len = 0;
  uint32_t remote_rkey = 0;
  uint32_t rkey = 0;

  // Buffer offset within rdma_buffer for address translation
  uintptr_t dispatch_recv_data_offset =
      0;  // offset of dispatch_rdma_recv_data_buffer from rdma_buffer base

  // Atomic operations buffer (GPU memory for receiving old values)
  uint32_t* atomic_old_values_buf =
      nullptr;  // GPU buffer for atomic old values
  static constexpr size_t kMaxAtomicOps =
      1024;  // Maximum concurrent atomic operations

  // Progress/accounting
  std::atomic<uint64_t> completed{0};
  std::atomic<bool> progress_run{true};

  // ACK receive ring
  std::vector<uint64_t> ack_recv_buf;
  ibv_mr* ack_recv_mr = nullptr;

  // GPU copy helpers (moved from function-static thread_local)
  gpuStream_t copy_stream = nullptr;
  bool peer_enabled[MAX_NUM_GPUS][MAX_NUM_GPUS] = {};
  size_t pool_index = 0;

  // Optional: per-GPU destination buffers if you previously used a global
  void* per_gpu_device_buf[MAX_NUM_GPUS] = {nullptr};

  uint32_t tag = 0;

  TokenCounter<DispatchTokenKey> dispatch_token_counter;
  TokenCounter<CombineTokenKey> combine_token_counter;

  /* low_latency_buffer_idx, expert_idx, dst_rank */
  std::unordered_map<uint64_t, WriteStruct> wr_id_to_write_struct;
  TokenCounter<DispatchTokenKey> dispatch_sent_counter;
  TokenCounter<DispatchTokenKey> combine_sent_counter;
};
