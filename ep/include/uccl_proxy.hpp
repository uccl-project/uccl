#pragma once
#include "bench_utils.hpp"
#include "fifo.hpp"
#include "proxy.hpp"
#include "ring_buffer.cuh"
#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class PeerCopyManager;

class UcclProxy {
  friend class PeerCopyManager;

 public:
  UcclProxy(int thread_idx, uintptr_t gpu_buffer_addr, size_t total_size,
            int rank, int node_idx, int local_rank, int num_experts = 0,
            int num_ranks = 0, int num_nodes = 0, bool use_normal_mode = false,
            bool is_intranode = false,
            bool gpu_buffer_is_host_allocated = false);
  ~UcclProxy();

  void start_sender();
  void start_remote();
  void start_local();
  void start_dual();
  void stop();
  int get_listen_port() const { return proxy_->get_listen_port(); }

  // Set the offset of dispatch_rdma_recv_data_buffer within rdma_buffer
  void set_dispatch_recv_data_offset(uintptr_t offset) {
    proxy_->set_dispatch_recv_data_offset(offset);
  }

  void* get_atomic_buffer_ptr() {
    if (!atomic_buffer_ptr_) {
      fprintf(stderr, "Error: atomic_buffer_ptr_ is not set yet\n");
      std::abort();
    }
    return atomic_buffer_ptr_;
  }

  void set_atomic_buffer_ptr(void* ptr) {
    // printf("Set atomic_buffer_ptr_ to %p\n", ptr);
    atomic_buffer_ptr_ = ptr;
    proxy_->set_atomic_buffer_ptr(atomic_buffer_ptr_);
  }

  // Calculate and set dispatch_recv_data_offset automatically based on layout
  // parameters
  void calculate_and_set_dispatch_recv_data_offset(int num_tokens, int hidden,
                                                   int num_experts) {
    // Calculate layout parameters (same logic as ep_config.hpp and test)
    int num_scales = hidden / 128;
    size_t num_bytes_per_dispatch_msg =
        4 + std::max(hidden * 2, hidden + num_scales * 4);
    size_t dispatch_send_buffer_bytes = num_tokens * num_bytes_per_dispatch_msg;
    size_t combine_send_buffer_bytes =
        num_experts * num_tokens * hidden * 2;  // sizeof(bfloat16)
    size_t send_buffer_bytes =
        std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
    size_t dispatch_recv_count_buffer_bytes = num_experts * 4;
    size_t signaling_buffer_bytes_aligned =
        ((dispatch_recv_count_buffer_bytes + 127) / 128) * 128;
    uintptr_t dispatch_recv_data_offset =
        signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2;
    proxy_->set_dispatch_recv_data_offset(dispatch_recv_data_offset);
    proxy_->cfg_.num_experts = num_experts;
  }

  std::vector<uint64_t> get_d2h_channel_addrs() const;
  int thread_idx() const noexcept { return thread_idx_; }
  void* gpu_buffer_addr() const noexcept { return gpu_buffer_addr_; }
  double avg_rdma_write_us() const { return proxy_->avg_rdma_write_us(); }
  double avg_wr_latency_us() const { return proxy_->avg_wr_latency_us(); }
  void set_peers_meta(std::vector<PeerMeta> const& peers);
  void set_bench_d2h_channel_addrs(std::vector<uintptr_t> const& addrs) {
    proxy_->set_bench_d2h_channel_addrs(addrs);
  }

 private:
  enum class Mode { None, Sender, Remote, Local, Dual };
  void start(Mode m);

  std::unique_ptr<Proxy> proxy_;
  std::thread thread_;
  Mode mode_;
  std::atomic<bool> running_;
  std::vector<uintptr_t> d2h_channel_addrs_;
  int thread_idx_;
  void* gpu_buffer_addr_;
  std::vector<PeerMeta> peers_;
  int local_rank_;
  void* atomic_buffer_ptr_;
  int node_idx_;
  bool is_intranode_;
  std::vector<d2hq::HostD2HHandle> d2h_queues;
  std::vector<std::unique_ptr<mscclpp::Fifo>> fifos;
};

// ============================================================================
// FIFO-based Proxy Wrapper
// ============================================================================

// Python-facing FIFO proxy wrapper that wraps the real Proxy class
class FifoProxy {
 public:
  FifoProxy(int thread_idx, uintptr_t gpu_buffer_addr, size_t total_size,
            int rank, int node_idx, int local_rank, bool is_intranode = false);
  ~FifoProxy();

  void set_fifo(mscclpp::Fifo* fifo);
  void set_peers_meta(std::vector<PeerMeta> const& meta);

  void start_sender();
  void start_remote();
  void stop();
  int get_listen_port() const { return proxy_->get_listen_port(); }

  double avg_wr_latency_us() const;
  uint64_t processed_count() const;

  int thread_idx;

 private:
  void run_sender();
  void run_remote();

  mscclpp::Fifo* fifo_;
  std::unique_ptr<Proxy> proxy_;  // Underlying Proxy for RDMA operations
  std::unique_ptr<std::thread> thread_;
  std::atomic<bool> stop_flag_;

  uintptr_t gpu_buffer_addr_;
  size_t total_size_;
  int rank_;
  int node_idx_;
  int local_rank_;
  bool is_intranode_;
};
