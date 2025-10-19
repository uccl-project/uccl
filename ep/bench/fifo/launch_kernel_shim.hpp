#ifndef LAUNCH_KERNEL_SHIM_HPP_
#define LAUNCH_KERNEL_SHIM_HPP_

#include "../../include/fifo_device.hpp"
#include <cstdint>
#include <cuda_runtime.h>

// Forward declarations of metrics structure
struct ThreadMetrics;

// Kernel launch wrappers (implemented in .cu file)
void launchFifoKernel(dim3 grid, dim3 block, mscclpp::FifoDeviceHandle* fifos,
                      ThreadMetrics* metrics, uint32_t num_threads,
                      uint32_t test_duration_ms, uint32_t warmup_iterations,
                      bool volatile* stop_flag, float gpu_clock_ghz,
                      uint32_t batch_size, int num_fifos,
                      uint64_t* latency_samples, int max_samples);

void launchFifoLatencyKernel(
    dim3 grid, dim3 block, mscclpp::FifoDeviceHandle* fifos,
    ThreadMetrics* metrics, uint32_t num_threads, uint32_t test_duration_ms,
    uint32_t warmup_iterations, bool volatile* stop_flag, float gpu_clock_ghz,
    int num_fifos, uint64_t* latency_samples, int max_samples);

void launchFifoBurstKernel(dim3 grid, dim3 block,
                           mscclpp::FifoDeviceHandle* fifos,
                           ThreadMetrics* metrics, uint32_t num_threads,
                           uint32_t test_duration_ms,
                           uint32_t warmup_iterations, bool volatile* stop_flag,
                           float gpu_clock_ghz, int num_fifos,
                           uint64_t* latency_samples, int max_samples);

#endif  // LAUNCH_KERNEL_SHIM_HPP_
