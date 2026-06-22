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

class UcclProxy {
 public:
  UcclProxy(int thread_idx, uintptr_t gpu_buffer_addr, size_t total_size,
            int rank, int node_idx, int local_rank, int num_experts = 0,
            int num_ranks = 0, int num_nodes = 0, bool use_normal_mode = false,
            bool is_intranode = false,
            bool gpu_buffer_is_host_allocated = false,
            bool owns_gpu_buffer = true);
  ~UcclProxy();

  void start_sender();
  void start_remote();
  void start_local();
  void start_dual();
  void stop();
  int get_listen_port() const { return proxy_->get_listen_port(); }

  void* get_atomic_buffer_ptr() {
    if (!atomic_buffer_ptr_) {
      fprintf(stderr, "Error: atomic_buffer_ptr_ is not set yet\n");
      std::abort();
    }
    return atomic_buffer_ptr_;
  }

  uintptr_t get_atomic_buffer_addr() {
    return reinterpret_cast<uintptr_t>(get_atomic_buffer_ptr());
  }

  size_t get_atomic_buffer_bytes() const { return kAtomicBufferSize; }

  void set_atomic_buffer_ptr(void* ptr) {
    // printf("Set atomic_buffer_ptr_ to %p\n", ptr);
    atomic_buffer_ptr_ = ptr;
    proxy_->set_atomic_buffer_ptr(atomic_buffer_ptr_);
  }

  void set_atomic_buffer_addr(uintptr_t addr) {
    set_atomic_buffer_ptr(reinterpret_cast<void*>(addr));
  }

  std::vector<uint64_t> get_d2h_channel_addrs() const;
  std::vector<uint64_t> get_d2h_channel_device_addrs() const;
  std::vector<uint64_t> get_d2h_channel_handle_addrs() const;
  int thread_idx() const noexcept { return thread_idx_; }
  void* gpu_buffer_addr() const noexcept { return gpu_buffer_addr_; }
  bool use_normal_mode() const noexcept { return proxy_->cfg_.use_normal_mode; }
  double avg_rdma_write_us() const { return proxy_->avg_rdma_write_us(); }
  double avg_wr_latency_us() const { return proxy_->avg_wr_latency_us(); }
  void set_peers_meta(std::vector<PeerMeta> const& peers);
  void set_bench_d2h_channel_addrs(std::vector<uintptr_t> const& addrs) {
    proxy_->set_bench_d2h_channel_addrs(addrs);
  }
  void notify_proxy_thread_adaptive_sleeper() {
    proxy_->notify_proxy_thread_adaptive_sleeper();
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
  bool atomic_buffer_is_host_allocated_ =
      false;  // true => cudaFreeHost, false => cudaFree
  int node_idx_;
  bool is_intranode_;
  bool owns_gpu_buffer_;
  std::vector<d2hq::HostD2HHandle> d2h_queues;
  std::vector<std::unique_ptr<mscclpp::Fifo>> fifos;
  void* d2h_device_handle_objs_{nullptr};
  std::vector<uint64_t> d2h_device_handle_addrs_;
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
