/*
 * FIFO Performance Benchmark
 * Measures FIFO dispatch throughput and GPU-side latency
 * To run the benchmark:
 *    make
 *    torchrun --nproc_per_node=8 --standalone launch_fifo.py [-l] [-b]
 */

#include "../../include/fifo.hpp"
#include "../../include/gpu_utils.hpp"
#include "launch_kernel_shim.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

using namespace mscclpp;

// Configuration
struct BenchmarkConfig {
  uint32_t num_threads;        // Number of GPU threads pushing to FIFO
  uint32_t fifo_size;          // FIFO size
  uint32_t test_duration_ms;   // Test duration in milliseconds
  uint32_t warmup_iterations;  // Number of warmup iterations
  uint32_t batch_size;   // Batch size for in-flight requests (default: 32)
  float gpu_clock_ghz;   // GPU clock rate in GHz
  bool measure_latency;  // Whether to measure GPU-side latency
  int mode;              // 0: throughput, 1: latency, 2: burst
};

// Metrics collected from GPU
struct ThreadMetrics {
  uint64_t push_count;          // Number of successful pushes
  uint64_t total_cycles;        // Total cycles spent (for latency calculation)
  uint64_t max_latency_cycles;  // Maximum latency observed
  uint64_t min_latency_cycles;  // Minimum latency observed
};

// Maximum number of latency samples per thread for percentile calculation
constexpr int MAX_LATENCY_SAMPLES = 10000;

// Host-side proxy that polls and pops from multiple FIFOs
class MultiFifoProxy {
 public:
  MultiFifoProxy(std::vector<Fifo*> fifos)
      : fifos_(fifos), stop_(false), processed_count_(0) {}

  void start() { thread_ = std::thread(&MultiFifoProxy::run, this); }

  void stop() {
    stop_ = true;
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  uint64_t getProcessedCount() const { return processed_count_; }

 private:
  void run() {
    while (!stop_) {
      // Round-robin poll all FIFOs
      for (auto* fifo : fifos_) {
        ProxyTrigger trigger = fifo->poll();

        // Check if trigger is valid (fst != 0)
        if (trigger.fst != 0) {
          // Flip back the MSB that was set by the device
          trigger.snd ^= ((uint64_t)1 << (uint64_t)63);

          // Process the trigger (in real use, this would dispatch work)
          processed_count_++;

          // Pop the trigger
          fifo->pop();
        }
      }
    }
  }

  std::vector<Fifo*> fifos_;
  std::thread thread_;
  std::atomic<bool> stop_;
  std::atomic<uint64_t> processed_count_;
};

// Print throughput results
void printThroughputResults(
    std::vector<ThreadMetrics> const& metrics, uint64_t processed_count,
    double duration_sec, BenchmarkConfig const& config,
    std::vector<uint64_t> const* latency_samples = nullptr) {
  uint64_t total_pushes = 0;
  for (auto const& m : metrics) {
    total_pushes += m.push_count;
  }

  double throughput_mops = total_pushes / duration_sec / 1e6;
  double proxy_throughput_mops = processed_count / duration_sec / 1e6;

  printf("Threads: %4u | FIFO Size: %4u | ", config.num_threads,
         config.fifo_size);
  printf("GPU Pushes: %6.2f Mops/s | Proxy Processed: %6.2f Mops/s",
         throughput_mops, proxy_throughput_mops);

  if (config.measure_latency && total_pushes > 0) {
    uint64_t total_cycles = 0;
    uint64_t max_latency = 0;
    uint64_t min_latency = UINT64_MAX;

    for (auto const& m : metrics) {
      total_cycles += m.total_cycles;
      max_latency = std::max(max_latency, m.max_latency_cycles);
      if (m.min_latency_cycles > 0) {
        min_latency = std::min(min_latency, m.min_latency_cycles);
      }
    }

    double avg_cycles = (double)total_cycles / total_pushes;
    // Convert cycles to nanoseconds using actual GPU clock rate
    double avg_latency_ns = avg_cycles / config.gpu_clock_ghz;
    double max_latency_ns = max_latency / config.gpu_clock_ghz;
    double min_latency_ns =
        (min_latency == UINT64_MAX) ? 0 : min_latency / config.gpu_clock_ghz;

    // Calculate 99th percentile if we have samples
    double p99_latency_ns = 0;
    if (latency_samples && !latency_samples->empty()) {
      std::vector<uint64_t> sorted_samples = *latency_samples;
      std::sort(sorted_samples.begin(), sorted_samples.end());
      size_t p99_idx = (sorted_samples.size() * 99) / 100;
      if (p99_idx >= sorted_samples.size()) p99_idx = sorted_samples.size() - 1;
      p99_latency_ns = sorted_samples[p99_idx] / config.gpu_clock_ghz;
    }

    printf(" | RTT Latency (ns) - Avg: %.0f, Min: %.0f, P99: %.0f, Max: %.0f",
           avg_latency_ns, min_latency_ns, p99_latency_ns, max_latency_ns);
  }

  printf("\n");
}

// Run single benchmark test
void runBenchmark(BenchmarkConfig const& config) {
  constexpr int NUM_FIFOS = 32;
  constexpr int NUM_PROXIES = 4;
  constexpr int FIFOS_PER_PROXY = NUM_FIFOS / NUM_PROXIES;

  // Create 32 FIFOs
  std::vector<std::unique_ptr<Fifo>> fifos;
  std::vector<FifoDeviceHandle> deviceHandles;
  for (int i = 0; i < NUM_FIFOS; i++) {
    fifos.push_back(std::make_unique<Fifo>(config.fifo_size));
    deviceHandles.push_back(fifos[i]->deviceHandle());
  }

  // Copy device handles to GPU
  FifoDeviceHandle* d_fifo_handles;
  cudaMalloc(&d_fifo_handles, sizeof(FifoDeviceHandle) * NUM_FIFOS);
  cudaMemcpy(d_fifo_handles, deviceHandles.data(),
             sizeof(FifoDeviceHandle) * NUM_FIFOS, cudaMemcpyHostToDevice);

  // Allocate device metrics
  ThreadMetrics* d_metrics;
  cudaMalloc(&d_metrics, sizeof(ThreadMetrics) * config.num_threads);
  cudaMemset(d_metrics, 0, sizeof(ThreadMetrics) * config.num_threads);

  // Allocate latency samples buffer (for percentile calculation)
  uint64_t* d_latency_samples = nullptr;
  if (config.measure_latency) {
    cudaMalloc(&d_latency_samples,
               sizeof(uint64_t) * config.num_threads * MAX_LATENCY_SAMPLES);
    cudaMemset(d_latency_samples, 0,
               sizeof(uint64_t) * config.num_threads * MAX_LATENCY_SAMPLES);
  }

  // Stop flag
  bool* d_stop_flag;
  cudaMallocManaged(&d_stop_flag, sizeof(bool));
  *d_stop_flag = false;

  // Start 4 proxy threads, each managing 8 FIFOs
  std::vector<std::unique_ptr<MultiFifoProxy>> proxies;
  for (int i = 0; i < NUM_PROXIES; i++) {
    std::vector<Fifo*> proxy_fifos;
    for (int j = 0; j < FIFOS_PER_PROXY; j++) {
      proxy_fifos.push_back(fifos[i * FIFOS_PER_PROXY + j].get());
    }
    proxies.push_back(std::make_unique<MultiFifoProxy>(proxy_fifos));
    proxies.back()->start();
  }

  // Launch GPU kernel
  dim3 grid(NUM_FIFOS);
  dim3 block((config.num_threads + grid.x - 1) / grid.x);

  auto start_time = std::chrono::high_resolution_clock::now();

  if (config.mode == 0) {
    // Use throughput kernel with batching
    launchFifoKernel(grid, block, d_fifo_handles, d_metrics, config.num_threads,
                     config.test_duration_ms, config.warmup_iterations,
                     d_stop_flag, config.gpu_clock_ghz, config.batch_size,
                     NUM_FIFOS, d_latency_samples, MAX_LATENCY_SAMPLES);
  } else if (config.mode == 1) {
    // Use RTT latency kernel - polls after each push to measure round-trip time
    launchFifoLatencyKernel(grid, block, d_fifo_handles, d_metrics,
                            config.num_threads, config.test_duration_ms,
                            config.warmup_iterations, d_stop_flag,
                            config.gpu_clock_ghz, NUM_FIFOS, d_latency_samples,
                            MAX_LATENCY_SAMPLES);
  } else if (config.mode == 2) {
    // Use burst kernel - threads push as fast as possible without polling
    launchFifoBurstKernel(grid, block, d_fifo_handles, d_metrics,
                          config.num_threads, config.test_duration_ms,
                          config.warmup_iterations, d_stop_flag,
                          config.gpu_clock_ghz, NUM_FIFOS, d_latency_samples,
                          MAX_LATENCY_SAMPLES);
  }

  // Wait for test duration (or kernel completion for latency mode)
  std::this_thread::sleep_for(
      std::chrono::milliseconds(config.test_duration_ms + 500));

  // Signal stop
  *d_stop_flag = true;
  cudaDeviceSynchronize();

  auto end_time = std::chrono::high_resolution_clock::now();
  double duration_sec =
      std::chrono::duration<double>(end_time - start_time).count();

  // Stop all proxies and collect total processed count
  uint64_t total_processed = 0;
  for (auto& proxy : proxies) {
    proxy->stop();
    total_processed += proxy->getProcessedCount();
  }

  // Copy metrics back
  std::vector<ThreadMetrics> h_metrics(config.num_threads);
  cudaMemcpy(h_metrics.data(), d_metrics,
             sizeof(ThreadMetrics) * config.num_threads,
             cudaMemcpyDeviceToHost);

  // Collect latency samples if measuring latency
  std::vector<uint64_t> latency_samples;
  if (config.measure_latency && d_latency_samples) {
    std::vector<uint64_t> all_samples(config.num_threads * MAX_LATENCY_SAMPLES);
    cudaMemcpy(all_samples.data(), d_latency_samples,
               sizeof(uint64_t) * config.num_threads * MAX_LATENCY_SAMPLES,
               cudaMemcpyDeviceToHost);

    // Filter out zero samples (unused slots)
    for (auto const& sample : all_samples) {
      if (sample > 0) {
        latency_samples.push_back(sample);
      }
    }
  }

  // Print results
  printThroughputResults(h_metrics, total_processed, duration_sec, config,
                         latency_samples.empty() ? nullptr : &latency_samples);

  // Cleanup
  cudaFree(d_metrics);
  cudaFree(d_stop_flag);
  cudaFree(d_fifo_handles);
  if (d_latency_samples) {
    cudaFree(d_latency_samples);
  }
}

int main(int argc, char** argv) {
  // Get LOCAL_RANK from environment (set by torchrun)
  int local_rank = 0;
  char const* local_rank_env = std::getenv("LOCAL_RANK");
  if (local_rank_env != nullptr) {
    local_rank = std::atoi(local_rank_env);
  }

  // Get WORLD_SIZE from environment (total number of processes)
  int world_size = 1;
  char const* world_size_env = std::getenv("WORLD_SIZE");
  if (world_size_env != nullptr) {
    world_size = std::atoi(world_size_env);
  }

  // Initialize CUDA with the appropriate device
  cudaSetDevice(local_rank);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, local_rank);

  // Get GPU clock rate (clockRate is in kHz)
  float gpu_clock_ghz = prop.clockRate / 1000000.0f;

  printf("========================================\n");
  printf("FIFO Performance Benchmark\n");
  printf("========================================\n");
  printf("Rank: %d/%d\n", local_rank, world_size);
  printf("GPU %d: %s\n", local_rank, prop.name);
  printf("SM count: %d\n", prop.multiProcessorCount);
  printf("GPU Clock: %.2f GHz\n", gpu_clock_ghz);
  printf("Configuration: 32 FIFOs, 4 Proxy Threads (8 FIFOs/proxy)\n\n");

  BenchmarkConfig config = {.num_threads = 32,
                            .fifo_size = 2048,
                            .test_duration_ms = 3000,
                            .warmup_iterations = 100,
                            .batch_size = 32,
                            .gpu_clock_ghz = gpu_clock_ghz,
                            .measure_latency = true,
                            .mode = 0};

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-l") {
      config.mode = 1;
    } else if (std::string(argv[i]) == "-b") {
      config.mode = 2;
    }
  }

  // Test configurations
  std::vector<uint32_t> thread_counts = {1, 32, 64, 128, 256, 512};
  std::vector<uint32_t> fifo_sizes = {2048, 4096};

  if (config.mode == 0) {
    // Throughput tests
    printf("--- FIFO Dispatch Throughput Tests ---\n");
    printf(
        "(Testing different thread counts and FIFO sizes with batch size "
        "%u)\n\n",
        config.batch_size);
  } else if (config.mode == 1) {
    // RTT Latency tests
    printf("--- FIFO RTT Latency Tests ---\n");
    printf("(Measuring round-trip time: push + host processing + poll)\n\n");
  } else if (config.mode == 2) {
    // Burst tests
    printf("--- FIFO Burst Tests ---\n");
    printf("(Testing different thread counts and FIFO sizes)\n\n");
  }

  for (auto fifo_size : fifo_sizes) {
    printf("FIFO Size: %u\n", fifo_size);
    printf("-----------------------------------\n");
    for (auto num_threads : thread_counts) {
      if (num_threads > fifo_size) continue;
      config.num_threads = num_threads;
      config.fifo_size = fifo_size;
      runBenchmark(config);
    }
    printf("\n");
  }

  printf("\n========================================\n");
  printf("Benchmark Complete\n");
  printf("========================================\n");

  return 0;
}
