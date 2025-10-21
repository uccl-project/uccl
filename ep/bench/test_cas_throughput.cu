/*
 * CAS-based Ring Buffer Throughput Test
 * Goal: Test GPU-CPU communication throughput with CAS-based ring buffer
 * sharing Architecture: Multiple warps share ring buffers via CAS
 * (atomic_set_and_commit) Similar to production uccl_ibgda.cuh design
 */

#include "../include/common.hpp"
#include "../include/ring_buffer.cuh"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

// Configuration
constexpr uint32_t MAX_WARPS = 1024;
constexpr uint32_t MAX_RING_BUFFERS = 32;

// Test configuration
struct TestConfig {
  uint32_t num_warps;
  uint32_t num_ring_buffers;
  uint32_t num_proxies;
  uint32_t test_duration_ms;
  uint32_t payload_size;
  bool verbose;
};

// Metrics for each warp
struct WarpMetrics {
  uint64_t successful_ops;
  uint64_t failed_attempts;  // CAS failures
  uint64_t total_cycles;
};

// GPU kernel - warps compete for ring buffers using CAS
__global__ void cas_throughput_kernel(DeviceToHostCmdBuffer** ring_buffers,
                                      uint32_t num_ring_buffers,
                                      TestConfig config, WarpMetrics* metrics,
                                      bool volatile* stop_flag) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / 32;
  const uint32_t lane_id = tid % 32;
  const uint32_t sm_id = blockIdx.x;

  // Only thread 0 of each warp participates (like production code)
  if (lane_id != 0 || warp_id >= config.num_warps) return;

  // Select ring buffer based on SM ID (like production: sm_id % num_rings)
  uint32_t ring_idx = sm_id % num_ring_buffers;
  DeviceToHostCmdBuffer* ring_buffer = ring_buffers[ring_idx];

  // Initialize metrics
  metrics[warp_id].successful_ops = 0;
  metrics[warp_id].failed_attempts = 0;
  metrics[warp_id].total_cycles = 0;

  // Create dummy transfer command
  TransferCmd dummy_cmd;
  dummy_cmd.dst_rank = ring_idx;
  dummy_cmd.bytes = config.payload_size;
  dummy_cmd.req_rptr = warp_id;
  dummy_cmd.req_lptr = 0;
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
    uint32_t attempts = 0;

    // Use atomic_set_and_commit (CAS-based) from ring_buffer.cuh

    // This function internally does CAS competition
    bool success = ring_buffer->atomic_set_and_commit(dummy_cmd);

    uint64_t op_end = clock64();

    // Update metrics
    if (success) {
      metrics[warp_id].successful_ops++;
    }
    // Note: atomic_set_and_commit always succeeds eventually (it loops
    // internally) So we count cycles instead of failures
    metrics[warp_id].total_cycles += (op_end - op_start);
  }
}

// CPU proxy thread - uses head/tail batch processing like proxy.cpp
void cpu_proxy_thread(DeviceToHostCmdBuffer* ring_buffer, int proxy_id,
                      bool volatile* stop_flag,
                      std::atomic<uint64_t>* processed_count, bool verbose) {
  uint64_t processed = 0;
  uint64_t my_tail = 0;
  size_t seen = 0;

  if (verbose) {
    printf("CPU proxy %d started (head/tail batch mode)\n", proxy_id);
  }

  while (!*stop_flag) {
    // Force load head from DRAM (like proxy.cpp)
    uint64_t cur_head = ring_buffer->volatile_head();
    if (cur_head == my_tail) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      continue;
    }

    // Batch processing
    size_t batch_size = cur_head - seen;
    std::vector<TransferCmd> cmds_to_process;
    cmds_to_process.reserve(batch_size);

    // Collect batch of commands (like proxy.cpp)
    for (size_t i = seen; i < cur_head; ++i) {
      CmdType cmd = ring_buffer->volatile_load_cmd_type(i);
      if (cmd == CmdType::EMPTY) break;

      TransferCmd& cmd_entry = ring_buffer->load_cmd_entry(i);
      cmds_to_process.push_back(cmd_entry);
      seen = i + 1;
    }

    // Process the batch
    if (!cmds_to_process.empty()) {
      processed += cmds_to_process.size();

      // Update tail to mark commands as processed
      my_tail += cmds_to_process.size();
      ring_buffer->cpu_volatile_store_tail(my_tail);

      if (verbose && processed % 10000 == 0) {
        printf("Proxy %d: processed %lu ops (batch size: %zu)\n", proxy_id,
               processed, cmds_to_process.size());
      }
    }
  }

  processed_count->store(processed);
  if (verbose) {
    printf("CPU proxy %d stopped, processed %lu ops\n", proxy_id, processed);
  }
}

// Print results
void print_results(std::vector<WarpMetrics> const& metrics,
                   std::vector<std::atomic<uint64_t>> const& proxy_counts,
                   double duration_sec, TestConfig config) {
  uint64_t total_processed = 0;
  for (auto const& count : proxy_counts) {
    total_processed += count.load();
  }

  uint64_t total_successful = 0;
  uint64_t total_cycles = 0;
  for (auto const& m : metrics) {
    total_successful += m.successful_ops;
    total_cycles += m.total_cycles;
  }

  double msg_throughput = total_processed / duration_sec;
  double avg_cycles =
      total_successful > 0 ? total_cycles / (double)total_successful : 0;

  printf(
      "Warps: %3u, Rings: %2u, Proxies: %2u, Throughput: %.2f Mops/sec, Avg "
      "cycles/op: %.0f\n",
      config.num_warps, config.num_ring_buffers, config.num_proxies,
      msg_throughput / 1e6, avg_cycles);
}

// Run test
void run_test(uint32_t num_warps, uint32_t num_ring_buffers,
              uint32_t num_proxies, uint32_t payload_size = 64,
              bool verbose = false) {
  TestConfig config = {.num_warps = num_warps,
                       .num_ring_buffers = num_ring_buffers,
                       .num_proxies = num_proxies,
                       .test_duration_ms = 5000,  // 5 seconds
                       .payload_size = payload_size,
                       .verbose = verbose};

  // Allocate ring buffers
  std::vector<DeviceToHostCmdBuffer*> h_ring_buffers(num_ring_buffers);
  DeviceToHostCmdBuffer** d_ring_buffers;
  cudaMallocManaged(&d_ring_buffers,
                    sizeof(DeviceToHostCmdBuffer*) * num_ring_buffers);

  for (uint32_t i = 0; i < num_ring_buffers; i++) {
    void* rb_ptr;
    cudaMallocHost(&rb_ptr, sizeof(DeviceToHostCmdBuffer));
    h_ring_buffers[i] = new (rb_ptr) DeviceToHostCmdBuffer();
    d_ring_buffers[i] = h_ring_buffers[i];
  }

  // Allocate metrics
  WarpMetrics* d_metrics;
  cudaMallocManaged(&d_metrics, sizeof(WarpMetrics) * num_warps);
  cudaMemset(d_metrics, 0, sizeof(WarpMetrics) * num_warps);

  // Stop flag
  bool* d_stop;
  cudaMallocManaged(&d_stop, sizeof(bool));
  *d_stop = false;

  // Start CPU proxy threads (one per ring buffer)
  bool volatile h_stop_flag = false;
  std::vector<std::atomic<uint64_t>> proxy_counts(num_proxies);
  for (auto& count : proxy_counts) {
    count.store(0);
  }

  std::vector<std::thread> proxy_threads;
  for (uint32_t i = 0; i < num_proxies; i++) {
    // Each proxy handles ring_buffers/proxies buffers
    uint32_t ring_idx = i % num_ring_buffers;
    proxy_threads.emplace_back(cpu_proxy_thread, h_ring_buffers[ring_idx], i,
                               &h_stop_flag, &proxy_counts[i], verbose);
  }

  // Launch GPU kernel
  dim3 grid((num_warps * 32 + 255) / 256);
  dim3 block(256);

  auto start_time = std::chrono::high_resolution_clock::now();

  cas_throughput_kernel<<<grid, block>>>(d_ring_buffers, num_ring_buffers,
                                         config, d_metrics, d_stop);

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

  printf(
      "\n========== CAS-based Ring Buffer Throughput Test (Head/Tail Batch "
      "Mode) ==========\n");
  printf("Architecture: Multiple warps share ring buffers via CAS\n");
  printf("CPU Polling: Head/tail batch processing (like proxy.cpp)\n");
  printf("              (Production-like design)\n\n");

  // Test different configurations
  // Format: num_warps, num_ring_buffers, num_proxies

  printf("--- Testing with 4 ring buffers, 4 proxies ---\n");
  run_test(32, 4, 4, 64, verbose);
  run_test(64, 4, 4, 64, verbose);
  run_test(128, 4, 4, 64, verbose);
  run_test(256, 4, 4, 64, verbose);

  printf("\n--- Testing with 8 ring buffers, 8 proxies ---\n");
  run_test(32, 8, 8, 64, verbose);
  run_test(64, 8, 8, 64, verbose);
  run_test(128, 8, 8, 64, verbose);
  run_test(256, 8, 8, 64, verbose);

  printf("\n--- Testing with 16 ring buffers, 16 proxies ---\n");
  run_test(64, 16, 16, 64, verbose);
  run_test(128, 16, 16, 64, verbose);
  run_test(256, 16, 16, 64, verbose);

  return 0;
}
