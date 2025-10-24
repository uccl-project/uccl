#pragma once
#include "common.hpp"
#include "d2h_queue_host.hpp"
#include "fifo.hpp"
#include "proxy.hpp"
#include "ring_buffer.cuh"
#include "util/gpu_rt.h"
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

struct BenchEnv {
  int blocks = kNumThBlocks;
  gpuStream_t stream = nullptr;
  gpuDeviceProp prop{};

  // Unified D2H handles visible to the application
  std::vector<d2hq::HostD2HHandle> d2h_queues;  // pointers to storage

  DeviceToHostCmdBuffer* rbs = nullptr;
  std::vector<std::unique_ptr<mscclpp::Fifo>> fifos;
  mscclpp::FifoDeviceHandle* d_fifo_handles = nullptr;
};

inline void init_env(BenchEnv& env, int blocks = kNumThBlocks,
                     int device = -1) {
  env.blocks = blocks;
  if (device == -1) gpuGetDevice(&device);
  GPU_RT_CHECK(gpuGetDeviceProperties(&env.prop, device));
  GPU_RT_CHECK(gpuStreamCreate(&env.stream));

#ifndef USE_MSCCLPP_FIFO_BACKEND
#ifdef USE_GRACE_HOPPER
  GPU_RT_CHECK(cudaMallocManaged(
      &env.rbs, sizeof(DeviceToHostCmdBuffer) * static_cast<size_t>(blocks)));
#else
  GPU_RT_CHECK(gpuHostAlloc(
      &env.rbs, sizeof(DeviceToHostCmdBuffer) * static_cast<size_t>(blocks),
      gpuHostAllocMapped));
#endif

  for (int i = 0; i < blocks; ++i) {
    new (&env.rbs[i]) DeviceToHostCmdBuffer();
    env.rbs[i].head = 0;
    env.rbs[i].tail = 0;
#ifdef MEASURE_PER_OP_LATENCY
    env.rbs[i].cycle_accum = 0ULL;
    env.rbs[i].op_count = 0ULL;
    env.rbs[i].cycle_start = 0ULL;
    env.rbs[i].cycle_end = 0ULL;
#endif
    for (uint32_t j = 0; j < kQueueSize; ++j) {
      env.rbs[i].volatile_clear_cmd_type(j);
    }
  }

  // Build backend-agnostic handles for the ring buffers
  d2hq::init_d2h_from_ring(env.rbs, static_cast<size_t>(blocks),
                           env.d2h_queues);

#else
  abort();
  env.fifos.clear();
  env.fifos.reserve(blocks);

  std::vector<mscclpp::FifoDeviceHandle> host_handles;
  host_handles.reserve(blocks);

  for (int i = 0; i < blocks; ++i) {
    // Use default FIFO size unless you want to pass one in
    auto fifo =
        std::make_unique<mscclpp::Fifo>((int)mscclpp::DEFAULT_FIFO_SIZE);
    host_handles.push_back(fifo->deviceHandle());
    env.fifos.push_back(std::move(fifo));
  }

  // Optional: copy device handles to GPU if your kernels need them
  if (!host_handles.empty()) {
    GPU_RT_CHECK(cudaMalloc(&env.d_fifo_handles,
                            sizeof(mscclpp::FifoDeviceHandle) * blocks));
    GPU_RT_CHECK(cudaMemcpy(env.d_fifo_handles, host_handles.data(),
                            sizeof(mscclpp::FifoDeviceHandle) * blocks,
                            cudaMemcpyHostToDevice));
  }

  // Build backend-agnostic handles for the FIFOs (3-arg overload)
  d2hq::init_d2h_from_fifo(env.fifos, env.d2h_queues);
#endif
}

inline void destroy_env(BenchEnv& env) {
#ifndef USE_MSCCLPP_FIFO_BACKEND
  // Ring backend
  if (env.rbs) {
#ifdef USE_GRACE_HOPPER
    GPU_RT_CHECK(cudaFree(env.rbs));
#else
    GPU_RT_CHECK(gpuFreeHost(env.rbs));
#endif
    env.rbs = nullptr;
  }
#else
  // FIFO backend
  if (env.d_fifo_handles) {
    GPU_RT_CHECK(cudaFree(env.d_fifo_handles));
    env.d_fifo_handles = nullptr;
  }
  env.fifos.clear();
#endif

  env.d2h_queues.clear();

  if (env.stream) {
    GPU_RT_CHECK(gpuStreamDestroy(env.stream));
    env.stream = nullptr;
  }
}

inline Proxy::Config make_cfg(BenchEnv const& env, int thread_idx, int rank,
                              char const* peer_ip, void* gpu_buffer = nullptr,
                              size_t total_size = 0, bool pin_thread = true) {
  Proxy::Config cfg{};
  // expose unified handle only
  cfg.d2h_queues.push_back(env.d2h_queues.at(thread_idx));
  cfg.thread_idx = thread_idx;
  cfg.rank = rank;
  cfg.peer_ip = peer_ip;
  cfg.gpu_buffer = gpu_buffer;
  cfg.total_size = total_size;
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
  GPU_RT_CHECK(gpuHostAlloc(&p, total_size, 0));
#else
  GPU_RT_CHECK(gpuMalloc(&p, total_size));
#endif
  return p;
}

inline void free_gpu_buffer(void* p) {
  if (!p) return;
#ifdef USE_GRACE_HOPPER
  GPU_RT_CHECK(gpuFreeHost(p));
#else
  GPU_RT_CHECK(gpuFree(p));
#endif
}

struct Stats {
  unsigned int tot_ops = 0;
  unsigned long long tot_cycles = 0ULL;
  double wall_ms = 0.0;
  double wall_ms_gpu = 0.0;  // valid when MEASURE_PER_OP_LATENCY
  double throughput_mops = 0.0;
  double avg_wr_latency_us = 0.0;
  double avg_rdma_write_us = 0.0;
};

inline Stats compute_stats(BenchEnv const& env,
                           std::chrono::high_resolution_clock::time_point t0,
                           std::chrono::high_resolution_clock::time_point t1) {
  Stats s{};
#ifdef MEASURE_PER_OP_LATENCY
#ifndef USE_MSCCLPP_FIFO_BACKEND
  // Ring path: read per-block GPU counters from ring buffer
  for (int b = 0; b < env.blocks; ++b) {
    s.tot_cycles += env.rbs[b].cycle_accum;
    s.tot_ops += env.rbs[b].op_count;
  }
#else
  // FIFO path: no per-slot counters available here by default
  s.tot_cycles = 0ULL;
  s.tot_ops = 0U;
#endif
#else
  s.tot_ops = static_cast<unsigned int>(env.blocks) *
              static_cast<unsigned int>(kIterations);
#endif

  s.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

#ifdef MEASURE_PER_OP_LATENCY
#ifndef USE_MSCCLPP_FIFO_BACKEND
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
    s.wall_ms_gpu = s.wall_ms;  // report same for visibility
  } else {
    s.throughput_mops = 0.0;
    s.wall_ms_gpu = 0.0;
  }
#endif
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
#ifndef USE_MSCCLPP_FIFO_BACKEND
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
#else
  std::printf(
      "\nPer-block avg latency: N/A (FIFO backend has no ring counters)\n");
#endif
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

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align(dtype_t a, dtype_t b) {
  return ceil_div<dtype_t>(a, b) * b;
}

// ============================================================================
// FIFO-based Benchmark Environment
// ============================================================================

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
