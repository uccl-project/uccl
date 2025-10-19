// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef FIFO_KERNEL_CUH_
#define FIFO_KERNEL_CUH_

#include "../../include/fifo_device.hpp"
#include <cstdint>

// Metrics structure (matching benchmark_fifo.cpp)
struct ThreadMetrics {
  uint64_t push_count;
  uint64_t total_cycles;
  uint64_t max_latency_cycles;
  uint64_t min_latency_cycles;
};

// FIFO throughput test kernel
__global__ void fifoThroughputKernel(
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t test_duration_ms,
    uint32_t warmup_iterations,
    bool measure_latency,
    bool volatile* stop_flag) {
  
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= num_threads) return;
  
  // Initialize metrics
  metrics[tid].push_count = 0;
  metrics[tid].total_cycles = 0;
  metrics[tid].max_latency_cycles = 0;
  metrics[tid].min_latency_cycles = UINT64_MAX;
  
  // Warmup phase
  for (uint32_t i = 0; i < warmup_iterations && !(*stop_flag); i++) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;  // Use tid+1 to ensure non-zero
    trigger.snd = i;
    
    fifo.push(trigger);
  }
  
  __syncthreads();
  
  // Test phase
  uint64_t test_start = clock64();
  uint64_t test_duration_cycles = (uint64_t)test_duration_ms * 1500000ULL; // ~1.5GHz
  
  while (!(*stop_flag)) {
    uint64_t current_time = clock64();
    if (current_time - test_start > test_duration_cycles) {
      break;
    }
    
    // Create trigger
    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;  // Use tid+1 to ensure non-zero
    trigger.snd = metrics[tid].push_count;
    
    // Measure latency if requested
    uint64_t push_start = 0;
    if (measure_latency) {
      push_start = clock64();
    }
    
    // Push to FIFO
    fifo.push(trigger);
    
    if (measure_latency) {
      uint64_t push_end = clock64();
      uint64_t latency = push_end - push_start;
      
      metrics[tid].total_cycles += latency;
      metrics[tid].max_latency_cycles = 
          max(metrics[tid].max_latency_cycles, latency);
      metrics[tid].min_latency_cycles = 
          min(metrics[tid].min_latency_cycles, latency);
    }
    
    // Increment counter
    metrics[tid].push_count++;
  }
}

// FIFO latency stress test kernel - multiple warps competing
__global__ void fifoLatencyStressKernel(
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t num_iterations,
    bool volatile* stop_flag) {
  
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= num_threads) return;
  
  // Initialize metrics
  metrics[tid].push_count = 0;
  metrics[tid].total_cycles = 0;
  metrics[tid].max_latency_cycles = 0;
  metrics[tid].min_latency_cycles = UINT64_MAX;
  
  __syncthreads();
  
  for (uint32_t i = 0; i < num_iterations && !(*stop_flag); i++) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;
    trigger.snd = i;
    
    uint64_t push_start = clock64();
    
    // Push to FIFO (this may spin if FIFO is full)
    fifo.push(trigger);
    
    uint64_t push_end = clock64();
    uint64_t latency = push_end - push_start;
    
    metrics[tid].total_cycles += latency;
    metrics[tid].max_latency_cycles = 
        max(metrics[tid].max_latency_cycles, latency);
    metrics[tid].min_latency_cycles = 
        min(metrics[tid].min_latency_cycles, latency);
    metrics[tid].push_count++;
  }
}

// Burst test - threads push as fast as possible
__global__ void fifoBurstKernel(
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t burst_size,
    bool volatile* stop_flag) {
  
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= num_threads) return;
  
  metrics[tid].push_count = 0;
  metrics[tid].total_cycles = 0;
  metrics[tid].max_latency_cycles = 0;
  metrics[tid].min_latency_cycles = UINT64_MAX;
  
  __syncthreads();
  
  uint64_t burst_start = clock64();
  
  for (uint32_t i = 0; i < burst_size && !(*stop_flag); i++) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;
    trigger.snd = i;
    
    uint64_t push_start = clock64();
    fifo.push(trigger);
    uint64_t push_end = clock64();
    
    uint64_t latency = push_end - push_start;
    metrics[tid].total_cycles += latency;
    metrics[tid].max_latency_cycles = 
        max(metrics[tid].max_latency_cycles, latency);
    metrics[tid].min_latency_cycles = 
        min(metrics[tid].min_latency_cycles, latency);
    metrics[tid].push_count++;
  }
  
  uint64_t burst_end = clock64();
  
  // Store total burst duration in first thread
  if (tid == 0) {
    metrics[tid].total_cycles = burst_end - burst_start;
  }
}

#endif  // FIFO_KERNEL_CUH_

