/*
 * Ring Buffer Lock Throughput Test
 * Goal: Test GPU-CPU communication throughput with spinlock synchronization
 * Architecture: Each CPU proxy manages 8 warps and 1 ring buffer
 * Synchronization: Warps use spinlock to access shared ring buffer
 */

#include "../include/common.hpp"
#include "../include/ring_buffer.cuh"
#include <cuda/atomic>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

// Configuration
constexpr uint32_t WARPS_PER_PROXY = 8;
constexpr uint32_t MAX_WARPS = 1024;
constexpr uint32_t MAX_PROXIES = MAX_WARPS / WARPS_PER_PROXY;

// Test configuration
struct TestConfig {
  uint32_t num_warps;
  uint32_t num_proxies;
  uint32_t test_duration_ms;
  uint32_t ops_per_warp;
  uint32_t payload_size;
  bool verbose;
};

// Metrics for each warp
struct WarpMetrics {
  uint64_t successful_ops;
  uint64_t failed_ops;
  uint64_t total_cycles;
};

// Spinlock implementation for GPU
static __device__ void device_mutex_lock_system(uint32_t* mutex) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> lock(*mutex);
  // Spin until the mutex is acquired
  while (lock.exchange(1, cuda::memory_order_acquire) != 0)
    ;
}

static __device__ void device_mutex_unlock_system(uint32_t* mutex) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> lock(*mutex);
  lock.store(0, cuda::memory_order_release);
}

// GPU kernel for throughput test
__global__ void lock_throughput_kernel(DeviceToHostCmdBuffer** ring_buffers,
                                       uint32_t* spinlocks, TestConfig config,
                                       WarpMetrics* metrics,
                                       bool volatile* stop_flag) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / 32;
  const uint32_t lane_id = tid % 32;

  // Only thread 0 of each warp writes
  if (lane_id != 0 || warp_id >= config.num_warps) return;

  const uint32_t warps_per_proxy_actual =
      (config.num_warps + config.num_proxies - 1) / config.num_proxies;
  const uint32_t proxy_id = warp_id / warps_per_proxy_actual;
  if (proxy_id >= config.num_proxies) return;

  DeviceToHostCmdBuffer* ring_buffer = ring_buffers[proxy_id];
  uint32_t* spinlock = &spinlocks[proxy_id];

  // Initialize metrics
  metrics[warp_id].successful_ops = 0;
  metrics[warp_id].failed_ops = 0;
  metrics[warp_id].total_cycles = 0;

  // Create dummy transfer command
  TransferCmd dummy_cmd;
  dummy_cmd.cmd = 1;
  dummy_cmd.dst_rank = proxy_id;
  dummy_cmd.dst_gpu = 0;
  dummy_cmd.src_ptr = nullptr;
  dummy_cmd.bytes = config.payload_size;
  dummy_cmd.req_rptr = warp_id;
  dummy_cmd.req_lptr = 0;
  dummy_cmd.sm_id = blockIdx.x;
  dummy_cmd.lane_id = lane_id;
  dummy_cmd.message_idx = 0;
  dummy_cmd.is_atomic = false;
  dummy_cmd.value = warp_id;
  dummy_cmd.is_combine = false;

  // Test loop
  uint64_t test_start = clock64();
  uint64_t test_duration_cycles =
      (uint64_t)config.test_duration_ms * 1980000;  // ~1.98GHz

  while (!(*stop_flag)) {
    uint64_t current_time = clock64();
    if (current_time - test_start > test_duration_cycles) break;

    uint64_t op_start = clock64();

    // Acquire spinlock
    device_mutex_lock_system(spinlock);

    // Try to push to ring buffer
    dummy_cmd.message_idx = metrics[warp_id].successful_ops;
    bool success = ring_buffer->push(dummy_cmd);

    // Release spinlock
    device_mutex_unlock_system(spinlock);

    uint64_t op_end = clock64();

    // Update metrics
    if (success) {
      metrics[warp_id].successful_ops++;
    } else {
      metrics[warp_id].failed_ops++;
    }
    metrics[warp_id].total_cycles += (op_end - op_start);

    // Small backoff if failed
    if (!success) {
      __nanosleep(100);
    }
  }
}

// CPU proxy thread function
void cpu_proxy_thread(DeviceToHostCmdBuffer* ring_buffer, int proxy_id,
                      bool volatile* stop_flag,
                      std::atomic<uint64_t>* processed_count, bool verbose) {
  uint64_t processed = 0;
  TransferCmd cmd;

  if (verbose) {
    printf("CPU proxy %d started\n", proxy_id);
  }

  while (!*stop_flag) {
    if (ring_buffer->pop(cmd)) {
      processed++;

      // Verify dummy content if needed
      if (verbose && processed % 10000 == 0) {
        printf("Proxy %d: processed %lu ops (last from warp %d)\n", proxy_id,
               processed, cmd.value);
      }
    } else {
      // Small sleep to avoid busy waiting
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  }

  processed_count->store(processed);
  if (verbose) {
    printf("CPU proxy %d stopped, processed %lu ops\n", proxy_id, processed);
  }
}

// Print test results
void print_results(std::vector<WarpMetrics> const& metrics,
                   std::vector<std::atomic<uint64_t>> const& proxy_counts,
                   double duration_sec, TestConfig config) {
  uint64_t total_successful = 0;
  uint64_t total_failed = 0;
  uint64_t total_cycles = 0;

  for (auto const& m : metrics) {
    total_successful += m.successful_ops;
    total_failed += m.failed_ops;
    total_cycles += m.total_cycles;
  }

  uint64_t total_processed = 0;
  for (auto const& count : proxy_counts) {
    total_processed += count.load();
  }

  double gpu_throughput = total_successful / duration_sec;
  double cpu_throughput = total_processed / duration_sec;
  double avg_cycles_per_op = total_cycles / (double)total_successful;

  printf("\n===== Test Results =====\n");
  printf("Configuration:\n");
  printf("  Warps: %u\n", config.num_warps);
  printf("  Proxies: %u\n", config.num_proxies);
  printf("  Warps per proxy: %u (actual)\n",
         (config.num_warps + config.num_proxies - 1) / config.num_proxies);
  printf("  Payload size: %u bytes\n", config.payload_size);
  printf("  Test duration: %.2f seconds\n", duration_sec);

  printf("\nGPU Side:\n");
  printf("  Successful ops: %lu\n", total_successful);
  printf("  Failed ops: %lu (%.2f%%)\n", total_failed,
         100.0 * total_failed / (total_successful + total_failed));
  printf("  Throughput: %.2f Mops/sec\n", gpu_throughput / 1e6);
  printf("  Avg cycles per op: %.0f\n", avg_cycles_per_op);

  printf("\nCPU Side:\n");
  printf("  Total processed: %lu\n", total_processed);
  printf("  Throughput: %.2f Mops/sec\n", cpu_throughput / 1e6);

  if (total_processed != total_successful) {
    printf("\nWarning: GPU/CPU mismatch (GPU: %lu, CPU: %lu)\n",
           total_successful, total_processed);
  }
}

// Run single test
void run_test(uint32_t num_warps, uint32_t payload_size = 64,
              bool verbose = false, uint32_t force_proxies = 0) {
  // Calculate number of proxies needed
  uint32_t num_proxies =
      force_proxies > 0 ? force_proxies
                        : (num_warps + WARPS_PER_PROXY - 1) / WARPS_PER_PROXY;

  TestConfig config = {.num_warps = num_warps,
                       .num_proxies = num_proxies,
                       .test_duration_ms = 5000,  // 5 seconds
                       .ops_per_warp = 0,         // Unlimited
                       .payload_size = payload_size,
                       .verbose = verbose};

  printf("\nStarting test with %u warps, %u proxies\n", num_warps, num_proxies);

  // Allocate ring buffers (pinned memory for GPU access)
  std::vector<DeviceToHostCmdBuffer*> h_ring_buffers(num_proxies);
  DeviceToHostCmdBuffer** d_ring_buffers;
  cudaMallocManaged(&d_ring_buffers,
                    sizeof(DeviceToHostCmdBuffer*) * num_proxies);

  for (uint32_t i = 0; i < num_proxies; i++) {
    void* rb_ptr;
    cudaMallocHost(&rb_ptr, sizeof(DeviceToHostCmdBuffer));
    h_ring_buffers[i] = new (rb_ptr) DeviceToHostCmdBuffer();
    d_ring_buffers[i] = h_ring_buffers[i];
  }

  // Allocate spinlocks (one per proxy/ring buffer)
  uint32_t* d_spinlocks;
  cudaMallocManaged(&d_spinlocks, sizeof(uint32_t) * num_proxies);
  cudaMemset(d_spinlocks, 0, sizeof(uint32_t) * num_proxies);

  // Allocate metrics
  WarpMetrics* d_metrics;
  cudaMallocManaged(&d_metrics, sizeof(WarpMetrics) * num_warps);
  cudaMemset(d_metrics, 0, sizeof(WarpMetrics) * num_warps);

  // Stop flag
  bool* d_stop;
  cudaMallocManaged(&d_stop, sizeof(bool));
  *d_stop = false;

  // Start CPU proxy threads
  bool volatile h_stop_flag = false;
  std::vector<std::atomic<uint64_t>> proxy_counts(num_proxies);
  for (auto& count : proxy_counts) {
    count.store(0);
  }

  std::vector<std::thread> proxy_threads;
  for (uint32_t i = 0; i < num_proxies; i++) {
    proxy_threads.emplace_back(cpu_proxy_thread, h_ring_buffers[i], i,
                               &h_stop_flag, &proxy_counts[i], verbose);
  }

  // Launch GPU kernel
  dim3 grid((num_warps * 32 + 255) / 256);
  dim3 block(256);

  auto start_time = std::chrono::high_resolution_clock::now();

  lock_throughput_kernel<<<grid, block>>>(d_ring_buffers, d_spinlocks, config,
                                          d_metrics, d_stop);

  // Wait for test duration
  std::this_thread::sleep_for(
      std::chrono::milliseconds(config.test_duration_ms));

  // Signal stop
  *d_stop = true;
  cudaDeviceSynchronize();
  h_stop_flag = true;

  // Wait for CPU threads
  for (auto& t : proxy_threads) {
    t.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double duration_sec =
      std::chrono::duration<double>(end_time - start_time).count();

  // Copy metrics back
  std::vector<WarpMetrics> h_metrics(num_warps);
  cudaMemcpy(h_metrics.data(), d_metrics, sizeof(WarpMetrics) * num_warps,
             cudaMemcpyDeviceToHost);

  // Print results
  print_results(h_metrics, proxy_counts, duration_sec, config);

  // Cleanup
  for (auto* rb : h_ring_buffers) {
    rb->~DeviceToHostCmdBuffer();
    cudaFreeHost(rb);
  }
  cudaFree(d_ring_buffers);
  cudaFree(d_spinlocks);
  cudaFree(d_metrics);
  cudaFree(d_stop);
}

int main(int argc, char** argv) {
  // Initialize CUDA
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Using GPU: %s\n", prop.name);
  printf("SM count: %d\n", prop.multiProcessorCount);

  // Parse command line arguments
  bool verbose = false;
  if (argc > 1 && std::string(argv[1]) == "-v") {
    verbose = true;
  }

  // Test with 4 proxies, varying warp counts
  std::vector<uint32_t> warp_counts = {32, 64, 128, 256, 512, 1024};

  printf(
      "\n========== Ring Buffer Lock Throughput Test (4 Proxies) ==========\n");
  printf("Architecture: %u warps per proxy, spinlock synchronization\n",
         WARPS_PER_PROXY);
  printf("Fixed: 4 CPU proxies\n\n");

  for (auto num_warps : warp_counts) {
    // Force 4 proxies
    run_test(num_warps, 64, verbose, 4);
  }

  return 0;
}
