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

#ifdef USE_DMABUF
// Describes one chunk of a GPU-buffer MR registration.
struct MRChunk {
  uintptr_t base;  // virtual (iova) start address of this chunk
  size_t len;
  ibv_mr* mr;
};

// Describes one chunk on the remote side (exchanged during connection setup).
struct RemoteMRChunk {
  uintptr_t base;
  size_t len;
  uint32_t rkey;
};

// A single segment produced by ProxyCtx::split_for_chunks().  Each segment is
// wholly contained within one local MR chunk AND one remote MR chunk.
struct WRSegment {
  uintptr_t laddr;
  uint32_t lkey;
  uintptr_t raddr;
  uint32_t rkey;
  uint32_t len;
};
#endif  // USE_DMABUF

struct ProxyCtx {
  // RDMA objects
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_mr* mr = nullptr;
  ibv_cq* cq = nullptr;
  ibv_cq_ex* cq_ex = nullptr;
  ibv_qp* qp = nullptr;
  std::vector<ibv_qp*> data_qps_by_channel;
  std::vector<uint32_t> dst_data_qpn_by_ring;
  // std::vector<ibv_cq*> extra_cqs;
  ibv_qp* ack_qp = nullptr;
  ibv_qp* recv_ack_qp = nullptr;
  int numa_node = -1;
  int gid_index = -1;

  uint32_t dst_qpn;
  uint32_t dst_ack_qpn;
  struct ibv_ah* dst_ah = nullptr;

  // Remote memory
  uintptr_t remote_addr = 0;  // Base address of remote rdma_buffer
  uint64_t remote_len = 0;
  uint32_t remote_rkey = 0;
  uint32_t rkey = 0;

#ifdef USE_DMABUF
  // Chunked MR support — populated when the GPU buffer exceeds the per-MR
  // size limit (e.g. 2 GiB limit with full IOMMU translation using DMA-BUF).
  // When empty, the single ctx.mr / ctx.rkey / ctx.remote_rkey are
  // used directly.
  std::vector<MRChunk> gpu_mr_chunks;
  std::vector<RemoteMRChunk> remote_mr_chunks;

  // Look up the local lkey for the MR chunk containing |addr|.
  uint32_t lkey_for(uintptr_t addr) const {
    for (auto& c : gpu_mr_chunks) {
      if (addr >= c.base && addr < c.base + c.len) return c.mr->lkey;
    }
    return mr->lkey;  // single-MR path or zero-length SGE
  }

  // Look up the remote rkey for the chunk containing |addr|.
  uint32_t rkey_for(uintptr_t addr) const {
    for (auto& c : remote_mr_chunks) {
      if (addr >= c.base && addr < c.base + c.len) return c.rkey;
    }
    return remote_rkey;  // single-MR path or zero-length SGE
  }

  // Split an (laddr, raddr, total_len) range into segments where each
  // segment is wholly contained within one local MR chunk AND one remote
  // MR chunk.  When chunking is not active (≤1 chunk on either side),
  // returns a single segment covering the full range.
  std::vector<WRSegment> split_for_chunks(uintptr_t laddr, uintptr_t raddr,
                                          uint32_t total_len) const {
    if (gpu_mr_chunks.size() <= 1 && remote_mr_chunks.size() <= 1) {
      return {{laddr, lkey_for(laddr), raddr, rkey_for(raddr), total_len}};
    }
    std::vector<WRSegment> segs;
    uint32_t off = 0;
    while (off < total_len) {
      uint32_t remain = total_len - off;
      uintptr_t la = laddr + off;
      uintptr_t ra = raddr + off;
      uint32_t seg_len = remain;
      for (auto const& c : gpu_mr_chunks) {
        if (la >= c.base && la < c.base + c.len) {
          uint32_t avail = static_cast<uint32_t>(c.base + c.len - la);
          if (avail < seg_len) seg_len = avail;
          break;
        }
      }
      for (auto const& c : remote_mr_chunks) {
        if (ra >= c.base && ra < c.base + c.len) {
          uint32_t avail = static_cast<uint32_t>(c.base + c.len - ra);
          if (avail < seg_len) seg_len = avail;
          break;
        }
      }
      segs.push_back({la, lkey_for(la), ra, rkey_for(ra), seg_len});
      off += seg_len;
    }
    return segs;
  }
#endif  // USE_DMABUF

  // Atomic buffer (separate from main RDMA buffer)
  ibv_mr* atomic_buffer_mr = nullptr;       // MR for local atomic_buffer_ptr
  uintptr_t remote_atomic_buffer_addr = 0;  // Remote atomic_buffer_ptr address
  uint64_t remote_atomic_buffer_len = 0;    // Remote atomic_buffer_ptr length
  uint32_t remote_atomic_buffer_rkey = 0;   // Remote atomic_buffer_ptr rkey

  // Buffer offset within rdma_buffer for address translation
  uintptr_t dispatch_recv_data_offset =
      0;  // offset of dispatch_rdma_recv_data_buffer from rdma_buffer base

  // Local scratch buffer for native RDMA atomics (e.g.,
  // IBV_WR_ATOMIC_FETCH_AND_ADD). The NIC writes the "old value" into this
  // local buffer; it must be registered and 8-byte aligned because verbs
  // atomics are 64-bit.
  uint64_t* atomic_old_values_buf = nullptr;
  ibv_mr* atomic_old_values_mr = nullptr;
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
  TokenCounter<NormalTokenKey> normal_token_counter;

  /* low_latency_buffer_idx, expert_idx, dst_rank */
  std::unordered_map<uint64_t, WriteStruct> wr_id_to_write_struct;
  TokenCounter<DispatchTokenKey> dispatch_sent_counter;
  TokenCounter<DispatchTokenKey> combine_sent_counter;
  TokenCounter<NormalTokenKey> normal_sent_counter;

  // Async-barrier state (single inflight assumed)
  bool barrier_inflight = false;
  uint64_t barrier_seq = 0;
  int barrier_wr = -1;

  bool quiet_inflight = false;
  int quiet_wr = -1;

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

  std::unordered_map<uint64_t, uint8_t> next_seq_per_index;
  inline uint64_t seq_key(int dst_rank, size_t index) {
    // assumes dst_rank fits 32 bits; if index > 32 bits, prefer Pair Hash below
    return (static_cast<uint64_t>(static_cast<uint32_t>(dst_rank)) << 32) ^
           static_cast<uint64_t>(static_cast<uint32_t>(index));
  }
};

// Return the CQ as ibv_cq* for polling/destroy. When EFA we store cq_ex and
// destroy with ibv_destroy_cq(ibv_cq_ex_to_cq(cq_ex)).
inline ibv_cq* get_cq(ProxyCtx& S) {
  return S.cq_ex ? ibv_cq_ex_to_cq(S.cq_ex) : S.cq;
}
