#pragma once
#include "bench_utils.hpp"
#include "proxy.hpp"
#include "ring_buffer.cuh"
#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>

class PeerCopyManager;

class UcclProxy {
  friend class PeerCopyManager;

 public:
  UcclProxy(uintptr_t rb_addr, int block_idx, uintptr_t gpu_buffer_addr,
            size_t total_size, int rank, std::string const& peer_ip = {});
  ~UcclProxy();

  void start_sender();
  void start_remote();
  void start_local();
  void start_dual();
  void stop();

  // Set the offset of dispatch_rdma_recv_data_buffer within rdma_buffer
  void set_dispatch_recv_data_offset(uintptr_t offset) {
    proxy_->set_dispatch_recv_data_offset(offset);
  }

  void* get_atomic_buffer_ptr() {
    if (!atomic_buffer_ptr_) {
      // For intranode mode, return a dummy pointer (atomic operations not needed)
      static int dummy_atomic_buffer = 0;
      return &dummy_atomic_buffer;
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
  }

  uintptr_t rb_addr() const noexcept { return rb_; }
  int block_idx() const noexcept { return block_idx_; }
  void* gpu_buffer_addr() const noexcept { return gpu_buffer_addr_; }
  void set_peers_meta(std::vector<PeerMeta> const& peers);

 private:
  enum class Mode { None, Sender, Remote, Local, Dual };
  void start(Mode m);

  std::string peer_ip_storage_;
  std::unique_ptr<Proxy> proxy_;
  std::thread thread_;
  Mode mode_;
  std::atomic<bool> running_;
  uintptr_t rb_;
  int block_idx_;
  void* gpu_buffer_addr_;
  std::vector<PeerMeta> peers_;
  int local_rank_;
  void* atomic_buffer_ptr_;
};
