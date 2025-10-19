#pragma once
#include "bench_utils.hpp"  // Include common utilities to avoid duplication
#include "common.hpp"
#include "fifo.hpp"
#include "proxy.hpp"
#include "util/gpu_rt.h"
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

struct BenchEnvFifo {
  std::vector<std::unique_ptr<mscclpp::Fifo>> fifos;
  mscclpp::FifoDeviceHandle* d_fifo_handles = nullptr;
  int blocks = kNumThBlocks;
  gpuStream_t stream = nullptr;
  gpuDeviceProp prop{};

  // Metrics per FIFO
  uint64_t* cycle_start = nullptr;
  uint64_t* cycle_end = nullptr;
  uint64_t* cycle_accum = nullptr;
  uint32_t* op_count = nullptr;
};

inline void init_env_fifo(BenchEnvFifo& env, int blocks = kNumThBlocks,
                          int device = -1, uint32_t fifo_size = 2048) {
  env.blocks = blocks;
  if (device == -1) gpuGetDevice(&device);
  GPU_RT_CHECK(gpuGetDeviceProperties(&env.prop, device));
  GPU_RT_CHECK(gpuStreamCreate(&env.stream));

  // Create FIFOs (one per SM/block)
  env.fifos.reserve(blocks);
  std::vector<mscclpp::FifoDeviceHandle> host_handles;
  for (int i = 0; i < blocks; ++i) {
    env.fifos.push_back(std::make_unique<mscclpp::Fifo>(fifo_size));
    host_handles.push_back(env.fifos[i]->deviceHandle());
  }

  // Copy device handles to GPU
  GPU_RT_CHECK(cudaMalloc(&env.d_fifo_handles,
                          sizeof(mscclpp::FifoDeviceHandle) * blocks));
  GPU_RT_CHECK(cudaMemcpy(env.d_fifo_handles, host_handles.data(),
                          sizeof(mscclpp::FifoDeviceHandle) * blocks,
                          cudaMemcpyHostToDevice));

#ifdef MEASURE_PER_OP_LATENCY
  // Allocate metrics on device
  GPU_RT_CHECK(cudaMallocManaged(&env.cycle_start, sizeof(uint64_t) * blocks));
  GPU_RT_CHECK(cudaMallocManaged(&env.cycle_end, sizeof(uint64_t) * blocks));
  GPU_RT_CHECK(cudaMallocManaged(&env.cycle_accum, sizeof(uint64_t) * blocks));
  GPU_RT_CHECK(cudaMallocManaged(&env.op_count, sizeof(uint32_t) * blocks));

  for (int i = 0; i < blocks; ++i) {
    env.cycle_start[i] = 0;
    env.cycle_end[i] = 0;
    env.cycle_accum[i] = 0;
    env.op_count[i] = 0;
  }
#endif
}

inline void destroy_env_fifo(BenchEnvFifo& env) {
  if (env.d_fifo_handles) {
    GPU_RT_CHECK(cudaFree(env.d_fifo_handles));
    env.d_fifo_handles = nullptr;
  }

#ifdef MEASURE_PER_OP_LATENCY
  if (env.cycle_start) {
    GPU_RT_CHECK(cudaFree(env.cycle_start));
    env.cycle_start = nullptr;
  }
  if (env.cycle_end) {
    GPU_RT_CHECK(cudaFree(env.cycle_end));
    env.cycle_end = nullptr;
  }
  if (env.cycle_accum) {
    GPU_RT_CHECK(cudaFree(env.cycle_accum));
    env.cycle_accum = nullptr;
  }
  if (env.op_count) {
    GPU_RT_CHECK(cudaFree(env.op_count));
    env.op_count = nullptr;
  }
#endif

  env.fifos.clear();

  if (env.stream) {
    GPU_RT_CHECK(gpuStreamDestroy(env.stream));
    env.stream = nullptr;
  }
}

inline mscclpp::Fifo* get_fifo(BenchEnvFifo const& env, int thread_idx) {
  if (thread_idx < 0 || thread_idx >= (int)env.fifos.size()) {
    return nullptr;
  }
  return env.fifos[thread_idx].get();
}

inline size_t shmem_bytes_fifo() {
  return kQueueSize * sizeof(unsigned long long);
}

// Note: mops_to_gbps, alloc_gpu_buffer, free_gpu_buffer, and Stats
// are defined in bench_utils.hpp (included above)

inline Stats compute_stats_fifo(
    BenchEnvFifo const& env, std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1) {
  Stats s{};
#ifdef MEASURE_PER_OP_LATENCY
  for (int b = 0; b < env.blocks; ++b) {
    s.tot_cycles += env.cycle_accum[b];
    s.tot_ops += env.op_count[b];
  }
#else
  s.tot_ops = static_cast<unsigned int>(env.blocks) *
              static_cast<unsigned int>(kIterations);
#endif

  s.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

#ifdef MEASURE_PER_OP_LATENCY
  s.wall_ms_gpu = (env.cycle_end[0] - env.cycle_start[0]) * 1000.0 /
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

inline void print_block_latencies_fifo(BenchEnvFifo const& env) {
#ifdef MEASURE_PER_OP_LATENCY
  std::printf("\nPer-block avg latency:\n");
  for (int b = 0; b < env.blocks; ++b) {
    if (env.op_count[b] == 0) {
      std::printf("  Block %d : N/A (0 ops)\n", b);
      continue;
    }
    double const us = static_cast<double>(env.cycle_accum[b]) * 1000.0 /
                      static_cast<double>(env.prop.clockRate) /
                      static_cast<double>(env.op_count[b]);
    std::printf("  Block %d : %.3f µs over %u ops\n", b, us, env.op_count[b]);
  }
#endif
}

inline void print_summary_fifo(BenchEnvFifo const& env, Stats const& s) {
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

// Note: ceil_div and align are defined in bench_utils.hpp (included above)
