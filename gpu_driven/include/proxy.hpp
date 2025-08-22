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

class Proxy {
 public:
  enum class Mode { Sender, Remote, Local, Dual };

  struct Config {
    DeviceToHostCmdBuffer* rb = nullptr;
    int block_idx = 0;
    void* gpu_buffer = nullptr;
    size_t total_size = 0;
    int rank = 0;
    char const* peer_ip = nullptr;
    bool pin_thread = true;
  };

  explicit Proxy(Config const& cfg) : cfg_(cfg) {
    const size_t total_size = kRemoteBufferSize;
    for (int d = 0; d < NUM_GPUS; ++d) {
      GPU_RT_CHECK(gpuSetDevice(d));
      void* buf = nullptr;
      GPU_RT_CHECK(gpuMalloc(&buf, total_size));
      ctx_.per_gpu_device_buf[d] = buf;
    }
    GPU_RT_CHECK(gpuSetDevice(0));
  }

  void set_progress_run(bool run) {
    ctx_.progress_run.store(run, std::memory_order_release);
  }
  
  // Set the offset of dispatch_rdma_recv_data_buffer within rdma_buffer
  void set_dispatch_recv_data_offset(uintptr_t offset) {
    ctx_.dispatch_recv_data_offset = offset;
  }

  void run_sender();
  void run_remote();
  void run_local();
  void run_dual();
  void pin_thread();

  double avg_rdma_write_us() const;
  double avg_wr_latency_us() const;
  uint64_t completed_wr() const;
  CopyRingBuffer ring;

 private:
  ProxyCtx ctx_;
  void init_common();
  void init_sender();
  void init_remote();

  void notify_gpu_completion(uint64_t& my_tail);
  void post_gpu_command(uint64_t& my_tail, size_t& seen);
  void post_gpu_commands_mixed(const std::vector<uint64_t>& wrs_to_post,
                              const std::vector<TransferCmd>& cmds_to_post);
  void post_atomic_operations(const std::vector<uint64_t>& wrs_to_post,
                             const std::vector<TransferCmd>& cmds_to_post);

  Config cfg_;
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