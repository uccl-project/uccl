/*
 * Multi Ring Buffer Throughput Test
 * Goal: Test GPU-CPU communication throughput with multiple ring buffers
 * Architecture: Each warp has its own ring buffer (no lock contention)
 *               Each CPU proxy polls 8 ring buffers
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
constexpr uint32_t RINGS_PER_PROXY = 8;
constexpr uint32_t MAX_WARPS = 1024;

// Test configuration
struct TestConfig {
  uint32_t num_warps;
  uint32_t num_proxies;
  uint32_t test_duration_ms;
  uint32_t payload_size;
  bool verbose;
};

// Metrics for each warp
struct WarpMetrics {
  uint64_t successful_ops;
  uint64_t failed_ops;
  uint64_t total_cycles;
};

// GPU kernel for throughput test - each warp writes to its own ring buffer
__global__ void multi_ring_throughput_kernel(
    DeviceToHostCmdBuffer** ring_buffers, TestConfig config,
    WarpMetrics* metrics, bool volatile* stop_flag) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / 32;
  const uint32_t lane_id = tid % 32;

  // Only thread 0 of each warp writes
  if (lane_id != 0 || warp_id >= config.num_warps) return;

  // Each warp has its own ring buffer
  DeviceToHostCmdBuffer* ring_buffer = ring_buffers[warp_id];

  // Initialize metrics
  metrics[warp_id].successful_ops = 0;
  metrics[warp_id].failed_ops = 0;
  metrics[warp_id].total_cycles = 0;

  // Create dummy transfer command
  TransferCmd dummy_cmd;
  dummy_cmd.dst_rank = warp_id / RINGS_PER_PROXY;
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

    bool success = ring_buffer->push(dummy_cmd);

    uint64_t op_end = clock64();

    // Update metrics
    if (success) {
      metrics[warp_id].successful_ops++;
    } else {
      metrics[warp_id].failed_ops++;
      // Small backoff if ring buffer is full
      __nanosleep(100);
    }
    metrics[warp_id].total_cycles += (op_end - op_start);
  }
}

// CPU proxy thread function - uses head/tail batch processing like proxy.cpp
void cpu_proxy_thread_multi(std::vector<DeviceToHostCmdBuffer*> ring_buffers,
                            int proxy_id, bool volatile* stop_flag,
                            std::atomic<uint64_t>* processed_count,
                            bool verbose) {
  uint64_t processed = 0;

  // Per-ring buffer state tracking (like proxy.cpp)
  std::vector<uint64_t> ring_tails(ring_buffers.size(), 0);
  std::vector<size_t> ring_seen(ring_buffers.size(), 0);

  if (verbose) {
    printf("CPU proxy %d started, polling %zu ring buffers (head/tail mode)\n",
           proxy_id, ring_buffers.size());
  }

  while (!*stop_flag) {
    bool found_work = false;

    // Process each ring buffer using head/tail logic
    for (size_t rb_idx = 0; rb_idx < ring_buffers.size(); rb_idx++) {
      auto* ring_buffer = ring_buffers[rb_idx];
      uint64_t& my_tail = ring_tails[rb_idx];
      size_t& seen = ring_seen[rb_idx];

      // Force load head from DRAM (like proxy.cpp)
      uint64_t cur_head = ring_buffer->volatile_head();
      if (cur_head == my_tail) {
        continue;  // No new work in this ring
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
        found_work = true;
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

    // Small sleep if no work found in any buffer
    if (!found_work) {
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
  uint64_t total_processed = 0;
  for (auto const& count : proxy_counts) {
    total_processed += count.load();
  }

  // Calculate pure data transfer throughput (messages/sec)
  double msg_throughput = total_processed / duration_sec;

  // Calculate actual rings per proxy
  uint32_t rings_per_proxy =
      (config.num_warps + config.num_proxies - 1) / config.num_proxies;

  printf(
      "Warps: %3u, Proxies: %2u, Rings/proxy: %2u, Throughput: %.2f Mops/sec\n",
      config.num_warps, config.num_proxies, rings_per_proxy,
      msg_throughput / 1e6);
}

// Run single test
void run_test(uint32_t num_warps, uint32_t payload_size = 64,
              bool verbose = false) {
  // Each warp gets its own ring buffer
  // Each proxy polls RINGS_PER_PROXY ring buffers
  uint32_t num_proxies = (num_warps + RINGS_PER_PROXY - 1) / RINGS_PER_PROXY;

  TestConfig config = {.num_warps = num_warps,
                       .num_proxies = num_proxies,
                       .test_duration_ms = 5000,  // 5 seconds
                       .payload_size = payload_size,
                       .verbose = verbose};

  // Allocate ring buffers (one per warp, pinned memory for GPU access)
  std::vector<DeviceToHostCmdBuffer*> h_ring_buffers(num_warps);
  DeviceToHostCmdBuffer** d_ring_buffers;
  cudaMallocManaged(&d_ring_buffers,
                    sizeof(DeviceToHostCmdBuffer*) * num_warps);

  for (uint32_t i = 0; i < num_warps; i++) {
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

  // Start CPU proxy threads
  bool volatile h_stop_flag = false;
  std::vector<std::atomic<uint64_t>> proxy_counts(num_proxies);
  for (auto& count : proxy_counts) {
    count.store(0);
  }

  std::vector<std::thread> proxy_threads;
  for (uint32_t p = 0; p < num_proxies; p++) {
    // Assign ring buffers to this proxy
    std::vector<DeviceToHostCmdBuffer*> proxy_rings;
    uint32_t start_idx = p * RINGS_PER_PROXY;
    uint32_t end_idx = std::min(start_idx + RINGS_PER_PROXY, num_warps);

    for (uint32_t i = start_idx; i < end_idx; i++) {
      proxy_rings.push_back(h_ring_buffers[i]);
    }

    proxy_threads.emplace_back(cpu_proxy_thread_multi, proxy_rings, p,
                               &h_stop_flag, &proxy_counts[p], verbose);
  }

  // Launch GPU kernel
  dim3 grid((num_warps * 32 + 255) / 256);
  dim3 block(256);

  auto start_time = std::chrono::high_resolution_clock::now();

  multi_ring_throughput_kernel<<<grid, block>>>(d_ring_buffers, config,
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
  cudaFree(d_metrics);
  cudaFree(d_stop);
}

// Run test with fixed number of proxies
void run_test_fixed_proxies(uint32_t num_warps, uint32_t payload_size,
                            bool verbose, uint32_t fixed_proxies) {
  // Each warp gets its own ring buffer
  // Proxies divide the work evenly
  uint32_t num_proxies = fixed_proxies;

  TestConfig config = {.num_warps = num_warps,
                       .num_proxies = num_proxies,
                       .test_duration_ms = 5000,  // 5 seconds
                       .payload_size = payload_size,
                       .verbose = verbose};

  // Allocate ring buffers (one per warp, pinned memory for GPU access)
  std::vector<DeviceToHostCmdBuffer*> h_ring_buffers(num_warps);
  DeviceToHostCmdBuffer** d_ring_buffers;
  cudaMallocManaged(&d_ring_buffers,
                    sizeof(DeviceToHostCmdBuffer*) * num_warps);

  for (uint32_t i = 0; i < num_warps; i++) {
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

  // Start CPU proxy threads
  bool volatile h_stop_flag = false;
  std::vector<std::atomic<uint64_t>> proxy_counts(num_proxies);
  for (auto& count : proxy_counts) {
    count.store(0);
  }

  std::vector<std::thread> proxy_threads;
  uint32_t rings_per_proxy = (num_warps + num_proxies - 1) / num_proxies;

  for (uint32_t p = 0; p < num_proxies; p++) {
    // Assign ring buffers to this proxy (evenly distributed)
    std::vector<DeviceToHostCmdBuffer*> proxy_rings;
    uint32_t start_idx = p * rings_per_proxy;
    uint32_t end_idx = std::min(start_idx + rings_per_proxy, num_warps);

    for (uint32_t i = start_idx; i < end_idx; i++) {
      proxy_rings.push_back(h_ring_buffers[i]);
    }

    proxy_threads.emplace_back(cpu_proxy_thread_multi, proxy_rings, p,
                               &h_stop_flag, &proxy_counts[p], verbose);
  }

  // Launch GPU kernel
  dim3 grid((num_warps * 32 + 255) / 256);
  dim3 block(256);

  auto start_time = std::chrono::high_resolution_clock::now();

  multi_ring_throughput_kernel<<<grid, block>>>(d_ring_buffers, config,
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

  // Fixed 4 proxies, each handling 8 ring buffers
  uint32_t num_proxies = 4;
  uint32_t num_warps = num_proxies * RINGS_PER_PROXY;  // 32

  printf(
      "\n========== Multi Ring Buffer Throughput Test (Head/Tail Batch Mode) "
      "==========\n");
  printf("Architecture: 1 ring buffer per warp (no locks)\n");
  printf("CPU Polling: Head/tail batch processing (like proxy.cpp)\n\n");
  std::vector<uint32_t> warp_counts = {32, 64, 128, 256};
  for (auto warps : warp_counts) {
    run_test_fixed_proxies(warps, 64, verbose, num_proxies);
  }

  return 0;
}
