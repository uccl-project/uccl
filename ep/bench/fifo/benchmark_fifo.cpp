/*
 * FIFO Performance Benchmark
 * Measures FIFO dispatch throughput and GPU-side latency
 * To run the benchmark:
 *    make
 *    ./benchmark_fifo
 *    ./benchmark_fifo -l
 */

#include "../../include/fifo.hpp"
#include "../../include/gpu_utils.hpp"
#include "launch_kernel_shim.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
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
  bool measure_latency;        // Whether to measure GPU-side latency
  bool verbose;
};

// Metrics collected from GPU
struct ThreadMetrics {
  uint64_t push_count;          // Number of successful pushes
  uint64_t total_cycles;        // Total cycles spent (for latency calculation)
  uint64_t max_latency_cycles;  // Maximum latency observed
  uint64_t min_latency_cycles;  // Minimum latency observed
};

// Host-side proxy that polls and pops from FIFO
class FifoProxy {
 public:
  FifoProxy(Fifo* fifo) : fifo_(fifo), stop_(false), processed_count_(0) {}

  void start() { thread_ = std::thread(&FifoProxy::run, this); }

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
      ProxyTrigger trigger = fifo_->poll();

      // Check if trigger is valid (fst != 0)
      if (trigger.fst != 0) {
        // Flip back the MSB that was set by the device
        trigger.snd ^= ((uint64_t)1 << (uint64_t)63);

        // Process the trigger (in real use, this would dispatch work)
        processed_count_++;

        // Pop the trigger
        fifo_->pop();
      } else {
        // Brief sleep to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      }
    }
  }

  Fifo* fifo_;
  std::thread thread_;
  std::atomic<bool> stop_;
  std::atomic<uint64_t> processed_count_;
};

// Update tail cache periodically
void tailFlusher(Fifo* /*fifo*/, std::atomic<bool>* stop,
                 int flush_interval_us) {
  while (!(*stop)) {
    // In the old API, we would call fifo->flushTail(false)
    // But it's deprecated and no-op now. The tail is automatically managed.
    std::this_thread::sleep_for(std::chrono::microseconds(flush_interval_us));
  }
}

// Print throughput results
void printThroughputResults(std::vector<ThreadMetrics> const& metrics,
                            uint64_t processed_count, double duration_sec,
                            BenchmarkConfig const& config) {
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
    // Assuming ~1.5 GHz GPU clock (adjust based on your GPU)
    double avg_latency_ns = avg_cycles / 1.5;
    double max_latency_ns = max_latency / 1.5;
    double min_latency_ns = (min_latency == UINT64_MAX) ? 0 : min_latency / 1.5;

    printf(" | Latency (ns) - Avg: %.0f, Min: %.0f, Max: %.0f", avg_latency_ns,
           min_latency_ns, max_latency_ns);
  }

  printf("\n");
}

// Run single benchmark test
void runBenchmark(BenchmarkConfig const& config) {
  // Create FIFO
  Fifo fifo(config.fifo_size);

  // Get device handle
  FifoDeviceHandle deviceHandle = fifo.deviceHandle();

  // Allocate device metrics
  ThreadMetrics* d_metrics;
  cudaMalloc(&d_metrics, sizeof(ThreadMetrics) * config.num_threads);
  cudaMemset(d_metrics, 0, sizeof(ThreadMetrics) * config.num_threads);

  // Stop flag
  bool* d_stop_flag;
  cudaMallocManaged(&d_stop_flag, sizeof(bool));
  *d_stop_flag = false;

  // Start host proxy thread
  FifoProxy proxy(&fifo);
  proxy.start();

  // Start tail flusher thread (even though flushTail is deprecated,
  // we simulate periodic background work)
  std::atomic<bool> stop_flusher(false);
  std::thread flusher_thread(tailFlusher, &fifo, &stop_flusher, 100);

  // Launch GPU kernel
  dim3 block(256);
  dim3 grid((config.num_threads + block.x - 1) / block.x);

  auto start_time = std::chrono::high_resolution_clock::now();

  launchFifoKernel(grid, block, deviceHandle, d_metrics, config.num_threads,
                   config.test_duration_ms, config.warmup_iterations,
                   config.measure_latency, d_stop_flag);

  // Wait for test duration
  std::this_thread::sleep_for(
      std::chrono::milliseconds(config.test_duration_ms + 500));

  // Signal stop
  *d_stop_flag = true;
  cudaDeviceSynchronize();

  auto end_time = std::chrono::high_resolution_clock::now();
  double duration_sec =
      std::chrono::duration<double>(end_time - start_time).count();

  // Stop proxy and flusher
  proxy.stop();
  stop_flusher = true;
  flusher_thread.join();

  // Copy metrics back
  std::vector<ThreadMetrics> h_metrics(config.num_threads);
  cudaMemcpy(h_metrics.data(), d_metrics,
             sizeof(ThreadMetrics) * config.num_threads,
             cudaMemcpyDeviceToHost);

  // Print results
  printThroughputResults(h_metrics, proxy.getProcessedCount(), duration_sec,
                         config);

  // Cleanup
  cudaFree(d_metrics);
  cudaFree(d_stop_flag);
}

int main(int argc, char** argv) {
  // Initialize CUDA
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  printf("========================================\n");
  printf("FIFO Performance Benchmark\n");
  printf("========================================\n");
  printf("GPU: %s\n", prop.name);
  printf("SM count: %d\n\n", prop.multiProcessorCount);

  // Parse command line arguments
  bool verbose = false;
  bool latency_mode = false;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-v") {
      verbose = true;
    } else if (std::string(argv[i]) == "-l") {
      latency_mode = true;
    }
  }

  // Test configurations
  std::vector<uint32_t> thread_counts = {32, 64, 128, 256, 512};
  std::vector<uint32_t> fifo_sizes = {128, 256, 512, 1024};

  // Throughput tests
  printf("--- FIFO Dispatch Throughput Tests ---\n");
  printf("(Testing different thread counts and FIFO sizes)\n\n");

  for (auto fifo_size : fifo_sizes) {
    printf("FIFO Size: %u\n", fifo_size);
    printf("-----------------------------------\n");

    for (auto num_threads : thread_counts) {
      if (num_threads > fifo_size) {
        // Skip configurations where threads exceed FIFO size
        continue;
      }

      BenchmarkConfig config = {.num_threads = num_threads,
                                .fifo_size = fifo_size,
                                .test_duration_ms = 5000,
                                .warmup_iterations = 100,
                                .measure_latency = false,
                                .verbose = verbose};

      runBenchmark(config);
    }
    printf("\n");
  }

  // Latency tests
  if (latency_mode) {
    printf("\n--- GPU-Side Latency Tests ---\n");
    printf("(Measuring push latency with different FIFO sizes)\n\n");

    uint32_t test_threads = 128;  // Fixed thread count for latency tests

    for (auto fifo_size : fifo_sizes) {
      BenchmarkConfig config = {.num_threads = test_threads,
                                .fifo_size = fifo_size,
                                .test_duration_ms = 3000,
                                .warmup_iterations = 100,
                                .measure_latency = true,
                                .verbose = verbose};

      runBenchmark(config);
    }
  }

  printf("\n========================================\n");
  printf("Benchmark Complete\n");
  printf("========================================\n");

  return 0;
}
