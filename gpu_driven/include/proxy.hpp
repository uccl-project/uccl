#ifndef PROXY_HPP
#define PROXY_HPP

#include "common.hpp"
#include "rdma.hpp"
#include "ring_buffer.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

struct ProxyCtx {
  DeviceToHostCmdBuffer*
      rb_host;  // host pointer (CPU visible address of RingBuffer)
  int my_rank;  // rank id for this proxy (if simulating multiple)
};

class Proxy {
 public:
  enum class Mode { Sender, Remote, Local };

  struct Config {
    DeviceToHostCmdBuffer* rb = nullptr;
    int block_idx = 0;
    void* gpu_buffer = nullptr;
    size_t total_size = 0;
    int rank = 0;
    char const* peer_ip = nullptr;
    CopyRingBuffer* ring = nullptr;
    bool pin_thread = true;
  };

  explicit Proxy(Config const& cfg) : cfg_(cfg) {
    bool const gdr_ok = GdrSupportInitOnce();
    if (!gdr_ok) {
      std::fprintf(stderr,
                   "Error: GPUDirect RDMA module is not loaded. Please load "
                   "nvidia_peermem or nv_peer_mem!\n");
      std::abort();
    }
  }

  void run_sender();
  void run_remote();
  void run_local();

  double avg_rdma_write_us() const;
  double avg_wr_latency_us() const;
  uint64_t completed_wr() const;

 private:
  void init_common();
  void init_sender();
  void init_remote();

  void sender_loop();
  void notify_gpu_completion(uint64_t& my_tail);
  void post_gpu_command(uint64_t& my_tail, size_t& seen);

  Config cfg_;
  ibv_cq* cq_ = nullptr;
  RDMAConnectionInfo local_info_{}, remote_info_{};

  // Completion tracking
  std::unordered_set<uint64_t> finished_wrs_;
  std::mutex finished_wrs_mutex_;

  std::unordered_map<uint64_t, std::chrono::high_resolution_clock::time_point>
      wr_id_to_start_time_;
  uint64_t completion_count_ = 0;
  uint64_t wr_time_total_us_ = 0;

  // Sender loop aggregates
  std::chrono::duration<double, std::micro> total_rdma_write_durations_ =
      std::chrono::duration<double, std::micro>::zero();
};

#endif  // PROXY_HPP