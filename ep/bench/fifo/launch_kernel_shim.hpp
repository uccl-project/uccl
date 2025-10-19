// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef LAUNCH_KERNEL_SHIM_HPP_
#define LAUNCH_KERNEL_SHIM_HPP_

#include "../../include/fifo_device.hpp"
#include <cstdint>
#include <cuda_runtime.h>

// Forward declarations of metrics structure
struct ThreadMetrics;

// Kernel launch wrappers (implemented in .cu file)
void launchFifoKernel(
    dim3 grid,
    dim3 block,
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t test_duration_ms,
    uint32_t warmup_iterations,
    bool measure_latency,
    bool volatile* stop_flag);

void launchFifoLatencyStressKernel(
    dim3 grid,
    dim3 block,
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t num_iterations,
    bool volatile* stop_flag);

void launchFifoBurstKernel(
    dim3 grid,
    dim3 block,
    mscclpp::FifoDeviceHandle fifo,
    ThreadMetrics* metrics,
    uint32_t num_threads,
    uint32_t burst_size,
    bool volatile* stop_flag);

#endif  // LAUNCH_KERNEL_SHIM_HPP_

