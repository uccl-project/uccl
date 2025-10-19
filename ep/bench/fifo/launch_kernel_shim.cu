// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "launch_kernel_shim.hpp"
#include "fifo_kernel.cuh"

// Kernel launch implementations
void launchFifoKernel(
    dim3 grid,
    dim3 block,
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t test_duration_ms,
    uint32_t warmup_iterations,
    bool measure_latency,
    bool volatile* stop_flag) {
  
  fifoThroughputKernel<<<grid, block>>>(
      fifo, metrics, num_threads, test_duration_ms,
      warmup_iterations, measure_latency, stop_flag);
}

void launchFifoLatencyStressKernel(
    dim3 grid,
    dim3 block,
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t num_iterations,
    bool volatile* stop_flag) {
  
  fifoLatencyStressKernel<<<grid, block>>>(
      fifo, metrics, num_threads, num_iterations, stop_flag);
}

void launchFifoBurstKernel(
    dim3 grid,
    dim3 block,
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t burst_size,
    bool volatile* stop_flag) {
  
  fifoBurstKernel<<<grid, block>>>(
      fifo, metrics, num_threads, burst_size, stop_flag);
}

