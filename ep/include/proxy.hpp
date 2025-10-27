#ifndef PROXY_HPP
#define PROXY_HPP

#include "common.hpp"
#include "proxy_ctx.hpp"
#include "rdma.hpp"
#include "ring_buffer.cuh"
#include "util/gpu_rt.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "d2h_queue_host.hpp"
#include <deque>
#include <set>
#include <tuple>

struct PeerMeta {
  int rank;
  uintptr_t ptr;
  size_t nbytes;
  std::string ip;
};

class Proxy {
 public:
  enum class Mode { Sender, Remote, Local, Dual };

  struct Config {
    std::vector<d2hq::HostD2HHandle> d2h_queues;
    int thread_idx = 0;
    void* gpu_buffer = nullptr;
    size_t total_size = 0;
    int rank = 0;
    int node_idx = -1;
    int local_rank = -1;
    char const* peer_ip = nullptr;
    bool pin_thread = true;
    int num_experts = 0;
    int num_ranks = 0;
    int num_nodes = 0;
  };

  explicit Proxy(Config const& cfg) : cfg_(cfg) {
    // Initialize state tracking for each ring buffer
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring_tails_.resize(cfg_.d2h_queues.size(), 0);
    ring_seen_.resize(cfg_.d2h_queues.size(), 0);
#endif
  }

  void set_progress_run(bool run) {
    ctx_.progress_run.store(run, std::memory_order_release);
  }

  // Set the offset of dispatch_rdma_recv_data_buffer within rdma_buffer
  void set_dispatch_recv_data_offset(uintptr_t offset) {
    ctx_.dispatch_recv_data_offset = offset;
  }

  void set_atomic_buffer_ptr(void* ptr) { atomic_buffer_ptr_ = ptr; }

  void run_sender();
  void run_remote();
  void run_local();
  void run_dual();
  void pin_thread_to_cpu_wrapper();
  void pin_thread_to_numa_wrapper();
  void destroy(bool free_gpu_buffer);

  double avg_rdma_write_us() const;
  double avg_wr_latency_us() const;
  uint64_t completed_wr() const;

  void set_peers_meta(std::vector<PeerMeta> const& peers);
  void set_bench_d2h_channel_addrs(std::vector<uintptr_t> const& addrs);

  CopyRingBuffer ring;

 private:
  friend class FifoProxy;  // Allow FifoProxy to access private methods
  ProxyCtx ctx_;
  void init_common();
  void init_sender();
  void init_remote();

  void notify_gpu_completion(uint64_t& my_tail);
  void post_gpu_command(uint64_t& my_tail, size_t& seen);
  void post_gpu_commands_mixed(std::vector<uint64_t> const& wrs_to_post,
                               std::vector<TransferCmd> const& cmds_to_post);
  void post_barrier_msg(int dst_rank, bool ack, uint64_t seq);
  void send_barrier(uint64_t wr);
  void barrier_check();
  void quiet(std::vector<uint64_t> wrs, std::vector<TransferCmd> cmds);
  void quiet_cq();

  // Subset support functions
  void init_subsets();
  void quiet_subset(int subset_id);
  int get_subset_id(int rank) const;
  void send_barrier_subset(uint64_t wr, int subset_id);
  void barrier_check_subset(int subset_id);
  void post_barrier_msg_subset(int dst_rank, bool ack, uint64_t seq, int subset_id);
  Config cfg_;
  RDMAConnectionInfo local_info_{}, remote_info_{};

  // Completion tracking
  std::unordered_set<uint64_t> acked_wrs_;
  std::unordered_map<uint64_t, std::chrono::high_resolution_clock::time_point>
      wr_id_to_start_time_;
  uint64_t completion_count_ = 0;
  uint64_t wr_time_total_us_ = 0;

  // Sender loop aggregates
  std::chrono::duration<double, std::micro> total_rdma_write_durations_ =
      std::chrono::duration<double, std::micro>::zero();

  std::vector<PeerMeta> peers_;
  std::vector<std::unique_ptr<ProxyCtx>> ctxs_for_all_ranks_;
  std::vector<RDMAConnectionInfo> local_infos_, remote_infos_;
  std::vector<ProxyCtx*> ctx_by_tag_;
  void* atomic_buffer_ptr_;
  std::vector<TransferCmd> postponed_atomics_;
  std::vector<uint64_t> postponed_wr_ids_;

  // Subset synchronization data structures (Phase 2 & 3)
  std::vector<std::vector<int>> all_subsets_;  // all_subsets_[subset_id] = {ranks...}
  std::vector<int> rank_to_subset_;             // rank_to_subset_[rank] = subset_id
  std::vector<std::unordered_set<uint64_t>> pending_wrs_by_subset_;  // [subset_id] -> {wr_ids}

  // Note: SubsetBarrierState is now in ProxyCtx (proxy_ctx.hpp) for access from rdma.cpp

#ifdef USE_MSCCLPP_FIFO_BACKEND
  std::vector<uint64_t> fifo_seq_;
  std::vector<std::deque<uint64_t>> fifo_pending_;
#else
  std::vector<uint64_t> ring_tails_;
  std::vector<size_t> ring_seen_;
#endif
};

#endif  // PROXY_HPP
