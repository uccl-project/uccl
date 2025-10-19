#pragma once
#include "bench_utils_fifo.hpp"  // This includes bench_utils.hpp which includes necessary headers
#include "common.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

// Forward declaration from uccl_bench.hpp
struct EnvInfo;

class BenchFifo {
 public:
  BenchFifo();
  ~BenchFifo();

  EnvInfo env_info() const;
  int blocks() const;
  int num_proxies() const;
  bool is_running() const;

  // Get pointer to Fifo object for proxy setup
  mscclpp::Fifo* get_fifo(int i) const;

  void timing_start();
  void timing_stop();

  void launch_gpu_issue_batched_commands();
  void sync_stream();

  void sync_stream_interruptible(
      int poll_ms = 5, long long timeout_ms = -1,
      std::function<bool()> const& should_abort = nullptr);

  void join_proxies();
  void print_block_latencies();

  Stats compute_stats() const;
  void print_summary(Stats const& s) const;
  void print_summary_last() const;
  double last_elapsed_ms() const;

 private:
  BenchEnvFifo env_;
  std::vector<std::thread> threads_;
  std::atomic<bool> running_;

  std::chrono::high_resolution_clock::time_point t0_, t1_;
  bool have_t0_, have_t1_;
  cudaEvent_t done_evt_;
};
