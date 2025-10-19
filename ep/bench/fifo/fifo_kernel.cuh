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

// FIFO throughput test kernel with batching support
__global__ void fifoThroughputKernel(
    mscclpp::FifoDeviceHandle* fifos, ThreadMetrics* metrics,
    uint32_t num_threads, uint32_t test_duration_ms, uint32_t warmup_iterations,
    bool volatile* stop_flag, float gpu_clock_ghz, uint32_t batch_size,
    int num_fifos, uint64_t* latency_samples, int max_samples) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_threads) return;

  // Each block uses its own FIFO
  mscclpp::FifoDeviceHandle& fifo = fifos[blockIdx.x % num_fifos];

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

  // Test phase with batching - maintains high in-flight count
  uint64_t test_start = clock64();
  // Convert test duration from ms to cycles using actual GPU clock rate
  uint64_t test_duration_cycles =
      (uint64_t)(test_duration_ms * gpu_clock_ghz * 1000000.0f);

  // Use a circular buffer to track head positions of in-flight requests
  // This allows us to maintain a consistently high number of in-flight requests
  // Max batch size is 256 to fit in local memory
  constexpr uint32_t MAX_BATCH_SIZE = 256;
  uint64_t head_buffer[MAX_BATCH_SIZE];
  uint32_t effective_batch_size =
      batch_size < MAX_BATCH_SIZE ? batch_size : MAX_BATCH_SIZE;
  uint32_t head_write_idx = 0;
  uint32_t head_read_idx = 0;
  uint32_t inflight_count = 0;
  uint32_t sample_idx = 0;

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
    push_start = clock64();

    // Push to FIFO and get the head position
    uint64_t head = fifo.push(trigger);

    // Store head in circular buffer
    head_buffer[head_write_idx] = head;
    head_write_idx = (head_write_idx + 1) % effective_batch_size;
    inflight_count++;

    uint64_t push_end = clock64();
    uint64_t latency = push_end - push_start;

    metrics[tid].total_cycles += latency;
    metrics[tid].max_latency_cycles =
        max(metrics[tid].max_latency_cycles, latency);
    metrics[tid].min_latency_cycles =
        min(metrics[tid].min_latency_cycles, latency);

    // Store latency sample for percentile calculation
    if (latency_samples && sample_idx < max_samples) {
      latency_samples[tid * max_samples + sample_idx] = latency;
      sample_idx++;
    }

    // Increment counter
    metrics[tid].push_count++;

    // Once we reach batch_size, start polling the oldest request
    // This maintains batch_size in-flight requests at all times
    if (inflight_count > effective_batch_size) {
      uint64_t oldest_head = head_buffer[head_read_idx];
      head_read_idx = (head_read_idx + 1) % effective_batch_size;

      // Wait for the oldest request to be consumed by the host
      fifo.sync(oldest_head);
      inflight_count--;
    }
  }
}

// FIFO RTT latency test kernel - measures round-trip time by polling after each
// push
__global__ void fifoLatencyKernel(mscclpp::FifoDeviceHandle* fifos,
                                  ThreadMetrics* metrics, uint32_t num_threads,
                                  uint32_t test_duration_ms,
                                  uint32_t warmup_iterations,
                                  bool volatile* stop_flag, float gpu_clock_ghz,
                                  int num_fifos, uint64_t* latency_samples,
                                  int max_samples) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_threads) return;

  // Each block uses its own FIFO
  mscclpp::FifoDeviceHandle& fifo = fifos[blockIdx.x % num_fifos];

  // Initialize metrics
  metrics[tid].push_count = 0;
  metrics[tid].total_cycles = 0;
  metrics[tid].max_latency_cycles = 0;
  metrics[tid].min_latency_cycles = UINT64_MAX;

  // Warmup phase - don't measure latency during warmup
  for (uint32_t i = 0; i < warmup_iterations && !(*stop_flag); i++) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;
    trigger.snd = i;

    uint64_t head = fifo.push(trigger);
    fifo.sync(head);  // Wait for host to consume
  }

  __syncthreads();

  // Test phase - measure RTT latency
  uint64_t test_start = clock64();
  uint64_t test_duration_cycles =
      (uint64_t)(test_duration_ms * gpu_clock_ghz * 1000000.0f);
  uint64_t iteration = 0;
  uint32_t sample_idx = 0;

  while (!(*stop_flag)) {
    uint64_t current_time = clock64();
    if (current_time - test_start > test_duration_cycles) {
      break;
    }

    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;
    trigger.snd = warmup_iterations + iteration;

    uint64_t rtt_start = clock64();

    // Push to FIFO and get the head position
    uint64_t head = fifo.push(trigger);

    // Poll the just-pushed request to measure RTT (round-trip time)
    // This waits until the host proxy has consumed the trigger
    fifo.sync(head);

    uint64_t rtt_end = clock64();
    uint64_t latency = rtt_end - rtt_start;

    metrics[tid].total_cycles += latency;
    metrics[tid].max_latency_cycles =
        max(metrics[tid].max_latency_cycles, latency);
    metrics[tid].min_latency_cycles =
        min(metrics[tid].min_latency_cycles, latency);
    metrics[tid].push_count++;

    // Store latency sample for percentile calculation
    if (latency_samples && sample_idx < max_samples) {
      latency_samples[tid * max_samples + sample_idx] = latency;
      sample_idx++;
    }

    iteration++;
  }
}

// Burst test - threads push as fast as possible without polling
__global__ void fifoBurstKernel(mscclpp::FifoDeviceHandle* fifos,
                                ThreadMetrics* metrics, uint32_t num_threads,
                                uint32_t test_duration_ms,
                                uint32_t warmup_iterations,
                                bool volatile* stop_flag, float gpu_clock_ghz,
                                int num_fifos, uint64_t* latency_samples,
                                int max_samples) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_threads) return;

  // Each block uses its own FIFO
  mscclpp::FifoDeviceHandle& fifo = fifos[blockIdx.x % num_fifos];

  metrics[tid].push_count = 0;
  metrics[tid].total_cycles = 0;
  metrics[tid].max_latency_cycles = 0;
  metrics[tid].min_latency_cycles = UINT64_MAX;

  // Warmup phase - don't measure during warmup
  for (uint32_t i = 0; i < warmup_iterations && !(*stop_flag); i++) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;
    trigger.snd = i;
    fifo.push(trigger);
  }

  __syncthreads();

  // Test phase - measure burst performance
  uint64_t test_start = clock64();
  uint64_t test_duration_cycles =
      (uint64_t)(test_duration_ms * gpu_clock_ghz * 1000000.0f);
  uint64_t iteration = 0;
  uint32_t sample_idx = 0;

  while (!(*stop_flag)) {
    uint64_t current_time = clock64();
    if (current_time - test_start > test_duration_cycles) {
      break;
    }

    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;
    trigger.snd = warmup_iterations + iteration;

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

    // Store latency sample for percentile calculation
    if (latency_samples && sample_idx < max_samples) {
      latency_samples[tid * max_samples + sample_idx] = latency;
      sample_idx++;
    }

    iteration++;
  }
}

// Random FIFO selection kernel - each thread randomly picks a FIFO for each
// push
__global__ void fifoRandomKernel(mscclpp::FifoDeviceHandle* fifos,
                                 ThreadMetrics* metrics, uint32_t num_threads,
                                 uint32_t test_duration_ms,
                                 uint32_t warmup_iterations,
                                 bool volatile* stop_flag, float gpu_clock_ghz,
                                 int num_fifos, uint64_t* latency_samples,
                                 int max_samples) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_threads) return;

  // Initialize metrics
  metrics[tid].push_count = 0;
  metrics[tid].total_cycles = 0;
  metrics[tid].max_latency_cycles = 0;
  metrics[tid].min_latency_cycles = UINT64_MAX;

  // Simple LCG random number generator (per-thread state)
  // Use tid and clock for seed to ensure different sequences per thread
  uint64_t rng_state = tid * 0x5DEECE66Dull + clock64();

  // Warmup phase - don't measure during warmup
  for (uint32_t i = 0; i < warmup_iterations && !(*stop_flag); i++) {
    // Update RNG state (Linear Congruential Generator)
    rng_state = rng_state * 0x5DEECE66Dull + 0xBull;
    uint32_t fifo_idx = (uint32_t)(rng_state >> 16) % num_fifos;

    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;
    trigger.snd = i;

    fifos[fifo_idx].push(trigger);
  }

  __syncthreads();

  // Test phase - measure burst performance with random FIFO selection
  uint64_t test_start = clock64();
  uint64_t test_duration_cycles =
      (uint64_t)(test_duration_ms * gpu_clock_ghz * 1000000.0f);
  uint64_t iteration = 0;
  uint32_t sample_idx = 0;

  while (!(*stop_flag)) {
    uint64_t current_time = clock64();
    if (current_time - test_start > test_duration_cycles) {
      break;
    }

    // Randomly select a FIFO for this push
    rng_state = rng_state * 0x5DEECE66Dull + 0xBull;
    uint32_t fifo_idx = (uint32_t)(rng_state >> 16) % num_fifos;

    mscclpp::ProxyTrigger trigger;
    trigger.fst = tid + 1;
    trigger.snd = warmup_iterations + iteration;

    uint64_t push_start = clock64();
    fifos[fifo_idx].push(trigger);
    uint64_t push_end = clock64();

    uint64_t latency = push_end - push_start;
    metrics[tid].total_cycles += latency;
    metrics[tid].max_latency_cycles =
        max(metrics[tid].max_latency_cycles, latency);
    metrics[tid].min_latency_cycles =
        min(metrics[tid].min_latency_cycles, latency);
    metrics[tid].push_count++;

    // Store latency sample for percentile calculation
    if (latency_samples && sample_idx < max_samples) {
      latency_samples[tid * max_samples + sample_idx] = latency;
      sample_idx++;
    }

    iteration++;
  }
}

#endif  // FIFO_KERNEL_CUH_
