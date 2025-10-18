#pragma once
#include "barrier_local.hpp"
#include "util/gpu_rt.h"
#include <infiniband/verbs.h>
#include <atomic>
#include <map>
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
using NormalTokenKey = std::pair<int, int>;

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
  std::vector<ibv_qp*> data_qps_by_ring;
  std::vector<uint32_t> dst_data_qpn_by_ring;
  // std::vector<ibv_cq*> extra_cqs;
  ibv_qp* ack_qp = nullptr;
  ibv_qp* recv_ack_qp = nullptr;
  int numa_node = -1;

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

  // For batched WR bookkeeping (largest_wr -> component wr_ids)
  std::unordered_map<uint64_t, std::vector<uint64_t>> wr_id_to_wr_ids{};

  // GPU copy helpers (moved from function-static thread_local)
  gpuStream_t copy_stream = nullptr;
  bool peer_enabled[MAX_NUM_GPUS][MAX_NUM_GPUS] = {};
  size_t pool_index = 0;

  // Optional: per-GPU destination buffers if you previously used a global
  void* per_gpu_device_buf[MAX_NUM_GPUS] = {nullptr};

  uint32_t tag = 0;

  TokenCounter<DispatchTokenKey> dispatch_token_counter;
  TokenCounter<CombineTokenKey> combine_token_counter;
  TokenCounter<NormalTokenKey> normal_token_counter;

  /* low_latency_buffer_idx, expert_idx, dst_rank */
  std::unordered_map<uint64_t, WriteStruct> wr_id_to_write_struct;
  TokenCounter<DispatchTokenKey> dispatch_sent_counter;
  TokenCounter<DispatchTokenKey> combine_sent_counter;
  TokenCounter<NormalTokenKey> normal_sent_counter;

  // Async-barrier state (single inflight assumed)
  bool barrier_inflight = false;
  uint64_t barrier_seq = 0;
  uint64_t barrier_wr = 0;

  // Rank-0 bookkeeping
  std::vector<uint8_t> barrier_arrived;  // size = num_ranks; 1 if arrival seen
  int barrier_arrival_count = 0;         // arrivals seen (include self)

  // Followers: release flag from rank-0
  bool barrier_released = false;
  uint64_t barrier_release_seq = 0;

  // Intra-node (shared-memory) barrier state
  LocalBarrier* lb = nullptr;  // mapped shared barrier block (per node+thread)
  bool lb_owner = false;  // we created the shm (so we should unlink on destroy)
  int num_local_ranks = 0;    // #local ranks on this node
  int node_leader_rank = -1;  // lowest global rank on this node
  int local_rank = -1;        // convenience mirror of cfg_.local_rank
  int thread_idx = -1;        // thread index used in shm name

  static constexpr int kNotifyGpuCounter = 100;
  int notify_gpu_counter = kNotifyGpuCounter;
};
