#pragma once
#include "fifo.hpp"
#include "proxy.hpp"
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Python-facing FIFO proxy wrapper that wraps the real Proxy class
class FifoProxy {
 public:
  FifoProxy(int thread_idx, uintptr_t gpu_buffer_addr, size_t total_size,
            int rank, int node_idx, int local_rank, std::string const& peer_ip);
  ~FifoProxy();

  void set_fifo(mscclpp::Fifo* fifo);
  void set_peers_meta(
      std::vector<std::tuple<int, uintptr_t, size_t, std::string>> const& meta);

  void start_sender();
  void start_remote();
  void stop();

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
  std::string peer_ip_;

  std::vector<std::tuple<int, uintptr_t, size_t, std::string>> peers_meta_;
};
