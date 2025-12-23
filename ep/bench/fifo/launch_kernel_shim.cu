#include "fifo_kernel.cuh"
#include "launch_kernel_shim.hpp"

// Kernel launch implementations
void launchFifoKernel(dim3 grid, dim3 block, mscclpp::FifoDeviceHandle* fifos,
                      ThreadMetrics* metrics, uint32_t num_threads,
                      uint32_t test_duration_ms, uint32_t warmup_iterations,
                      bool volatile* stop_flag, float gpu_clock_ghz,
                      uint32_t batch_size, int num_fifos,
                      uint64_t* latency_samples, int max_samples) {
  fifoThroughputKernel<<<grid, block>>>(
      fifos, metrics, num_threads, test_duration_ms, warmup_iterations,
      stop_flag, gpu_clock_ghz, batch_size, num_fifos, latency_samples,
      max_samples);
}

void launchFifoLatencyKernel(
    dim3 grid, dim3 block, mscclpp::FifoDeviceHandle* fifos,
    ThreadMetrics* metrics, uint32_t num_threads, uint32_t test_duration_ms,
    uint32_t warmup_iterations, bool volatile* stop_flag, float gpu_clock_ghz,
    int num_fifos, uint64_t* latency_samples, int max_samples) {
  fifoLatencyKernel<<<grid, block>>>(
      fifos, metrics, num_threads, test_duration_ms, warmup_iterations,
      stop_flag, gpu_clock_ghz, num_fifos, latency_samples, max_samples);
}

void launchFifoBurstKernel(dim3 grid, dim3 block,
                           mscclpp::FifoDeviceHandle* fifos,
                           ThreadMetrics* metrics, uint32_t num_threads,
                           uint32_t test_duration_ms,
                           uint32_t warmup_iterations, bool volatile* stop_flag,
                           float gpu_clock_ghz, int num_fifos,
                           uint64_t* latency_samples, int max_samples) {
  fifoBurstKernel<<<grid, block>>>(
      fifos, metrics, num_threads, test_duration_ms, warmup_iterations,
      stop_flag, gpu_clock_ghz, num_fifos, latency_samples, max_samples);
}

void launchFifoRandomKernel(
    dim3 grid, dim3 block, mscclpp::FifoDeviceHandle* fifos,
    ThreadMetrics* metrics, uint32_t num_threads, uint32_t test_duration_ms,
    uint32_t warmup_iterations, bool volatile* stop_flag, float gpu_clock_ghz,
    int num_fifos, uint64_t* latency_samples, int max_samples) {
  fifoRandomKernel<<<grid, block>>>(
      fifos, metrics, num_threads, test_duration_ms, warmup_iterations,
      stop_flag, gpu_clock_ghz, num_fifos, latency_samples, max_samples);
}

void launchFifoControlledMopsKernel(
    dim3 grid, dim3 block, mscclpp::FifoDeviceHandle* fifos,
    ThreadMetrics* metrics, uint32_t num_threads, uint32_t test_duration_ms,
    uint32_t warmup_iterations, bool volatile* stop_flag, float gpu_clock_ghz,
    int num_fifos, uint64_t* latency_samples, int max_samples,
    uint64_t sleep_cycles) {
  fifoControlledMopsKernel<<<grid, block>>>(
      fifos, metrics, num_threads, test_duration_ms, warmup_iterations,
      stop_flag, gpu_clock_ghz, num_fifos, latency_samples, max_samples,
      sleep_cycles);
}
