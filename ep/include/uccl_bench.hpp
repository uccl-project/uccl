#pragma once

#include "bench_utils.hpp"  // BenchEnv, Stats, helpers
#include "ring_buffer.cuh"  // DeviceToHostCmdBuffer etc.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

struct EnvInfo {
  int blocks;
  int queue_size;
  int threads_per_block;
  int iterations;
  uintptr_t stream_addr;
  uintptr_t rbs_addr;
};

class Bench {
 public:
  Bench();
  ~Bench();

  EnvInfo env_info() const;
  int blocks() const;
  int num_proxies() const;
  bool is_running() const;
  uintptr_t ring_addr(int i) const;

  void timing_start();
  void timing_stop();

  // pure C++ (no pybind)
  void start_local_proxies(int rank = 0, std::string const& peer_ip = {});
  void launch_gpu_issue_batched_commands();
  void sync_stream();

  // Supports optional cancellation callback (called in the poll loop).
  // Return true from should_abort() to abort; or throw from the callback.
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
  BenchEnv env_;
  std::vector<std::thread> threads_;
  std::atomic<bool> running_;
  std::chrono::high_resolution_clock::time_point t0_{}, t1_{};
  bool have_t0_{false}, have_t1_{false};
  cudaEvent_t done_evt_{nullptr};
};

// ============================================================================
// FIFO-based Benchmark Class
// ============================================================================

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