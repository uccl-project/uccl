#pragma once
#include "common.hpp"
#include "proxy.hpp"
#include "ring_buffer.cuh"
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

struct BenchEnv {
  DeviceToHostCmdBuffer* rbs = nullptr;
  int blocks = kNumThBlocks;
  cudaStream_t stream = nullptr;
  cudaDeviceProp prop{};
};

inline void init_env(BenchEnv& env, int blocks = kNumThBlocks, int device = 0,
                     bool quiet = false) {
  env.blocks = blocks;
  cudaSetDevice(device);
  cudaCheckErrors("cudaSetDevice failed");
  cudaGetDeviceProperties(&env.prop, device);
  cudaCheckErrors("cudaGetDeviceProperties failed");
  if (!quiet) {
    std::printf("clock rate: %d kHz\n", env.prop.clockRate);
  }

  cudaStreamCreate(&env.stream);
  cudaCheckErrors("cudaStreamCreate failed");

  cudaHostAlloc(&env.rbs,
                sizeof(DeviceToHostCmdBuffer) * static_cast<size_t>(blocks),
                cudaHostAllocMapped);
  cudaCheckErrors("cudaHostAlloc failed");

  for (int i = 0; i < blocks; ++i) {
    env.rbs[i].head = 0;
    env.rbs[i].tail = 0;
#ifdef MEASURE_PER_OP_LATENCY
    env.rbs[i].cycle_accum = 0ULL;
    env.rbs[i].op_count = 0ULL;
    env.rbs[i].cycle_start = 0ULL;
    env.rbs[i].cycle_end = 0ULL;
#endif
    for (uint32_t j = 0; j < kQueueSize; ++j) {
      env.rbs[i].buf[j].cmd = 0ULL;
    }
  }
}

inline void destroy_env(BenchEnv& env) {
  if (env.rbs) {
    cudaFreeHost(env.rbs);
    env.rbs = nullptr;
  }
  if (env.stream) {
    cudaStreamDestroy(env.stream);
    env.stream = nullptr;
  }
}

inline Proxy::Config make_cfg(BenchEnv const& env, int block_idx, int rank,
                              char const* peer_ip, void* gpu_buffer = nullptr,
                              size_t total_size = 0,
                              CopyRingBuffer* ring = nullptr,
                              bool pin_thread = true) {
  Proxy::Config cfg{};
  cfg.rb = &env.rbs[block_idx];
  cfg.block_idx = block_idx;
  cfg.rank = rank;
  cfg.peer_ip = peer_ip;
  cfg.gpu_buffer = gpu_buffer;
  cfg.total_size = total_size;
  cfg.ring = ring;
  cfg.pin_thread = pin_thread;
  return cfg;
}

inline size_t shmem_bytes_local() {
  return kQueueSize * sizeof(unsigned long long);
}
inline size_t shmem_bytes_remote() {
  return kQueueSize * 2 * sizeof(unsigned long long);
}

inline double mops_to_gbps(double mops) {
  return mops * 1e6 * kObjectSize * 8 / 1e9;
}

inline void* alloc_gpu_buffer(size_t total_size) {
  void* p = nullptr;
#ifdef USE_GRACE_HOPPER
  cudaMallocHost(&p, total_size);
#else
  cudaMalloc(&p, total_size);
#endif
  cudaCheckErrors("alloc_gpu_buffer failed");
  return p;
}
inline void free_gpu_buffer(void* p) {
  if (!p) return;
#ifdef USE_GRACE_HOPPER
  cudaFreeHost(p);
#else
  cudaFree(p);
#endif
  cudaCheckErrors("free gpu_buffer failed");
}

struct Stats {
  unsigned int tot_ops = 0;
  unsigned long long tot_cycles = 0ULL;
  double wall_ms = 0.0;
  double wall_ms_gpu = 0.0;  // valid when MEASURE_PER_OP_LATENCY
  double throughput_mops = 0.0;
};

inline Stats compute_stats(BenchEnv const& env,
                           std::chrono::high_resolution_clock::time_point t0,
                           std::chrono::high_resolution_clock::time_point t1) {
  Stats s{};
#ifdef MEASURE_PER_OP_LATENCY
  for (int b = 0; b < env.blocks; ++b) {
    s.tot_cycles += env.rbs[b].cycle_accum;
    s.tot_ops += env.rbs[b].op_count;
  }
#else
  s.tot_ops = static_cast<unsigned int>(env.blocks) *
              static_cast<unsigned int>(kIterations);
#endif

  s.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

#ifdef MEASURE_PER_OP_LATENCY
  s.wall_ms_gpu = (env.rbs[0].cycle_end - env.rbs[0].cycle_start) * 1000.0 /
                  static_cast<double>(env.prop.clockRate) / 1000.0;

  if (s.tot_ops > 0 && s.wall_ms_gpu > 0.0) {
    s.throughput_mops =
        static_cast<double>(s.tot_ops) / (s.wall_ms_gpu * 1000.0);
  } else {
    s.throughput_mops = 0.0;
  }
#else
  if (s.wall_ms > 0.0) {
    s.throughput_mops = static_cast<double>(env.blocks) *
                        static_cast<double>(kIterations) / (s.wall_ms * 1000.0);
  } else {
    s.throughput_mops = 0.0;
  }
#endif
  return s;
}

inline void print_block_latencies(BenchEnv const& env) {
#ifdef MEASURE_PER_OP_LATENCY
  std::printf("\nPer-block avg latency:\n");
  for (int b = 0; b < env.blocks; ++b) {
    if (env.rbs[b].op_count == 0) {
      std::printf("  Block %d : N/A (0 ops)\n", b);
      continue;
    }
    double const us = static_cast<double>(env.rbs[b].cycle_accum) * 1000.0 /
                      static_cast<double>(env.prop.clockRate) /
                      static_cast<double>(env.rbs[b].op_count);
    std::printf("  Block %d : %.3f µs over %lu ops\n", b, us,
                env.rbs[b].op_count);
  }
#endif
}

inline void print_summary(BenchEnv const& env, Stats const& s) {
#ifdef MEASURE_PER_OP_LATENCY
  if (s.tot_ops > 0) {
    double const avg_us = static_cast<double>(s.tot_cycles) * 1000.0 /
                          static_cast<double>(env.prop.clockRate) /
                          static_cast<double>(s.tot_ops);
    std::printf("\nOverall avg GPU-measured latency  : %.3f µs\n", avg_us);
  } else {
    std::printf("\nOverall avg GPU-measured latency  : N/A (0 ops)\n");
  }
  std::printf("Total cycles                      : %llu\n", s.tot_cycles);
#endif

  std::printf("Total ops                         : %u\n", s.tot_ops);
#ifdef MEASURE_PER_OP_LATENCY
  std::printf("End-to-end wall-clock time        : %.3f ms\n", s.wall_ms_gpu);
#else
  std::printf("End-to-end wall-clock time        : %.3f ms\n", s.wall_ms);
#endif
  std::printf("Ops Throughput                    : %.2f Mops\n",
              s.throughput_mops);
  std::printf("Total Throughput                  : %.2f Gbps\n",
              mops_to_gbps(s.throughput_mops));
}