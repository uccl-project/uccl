/*
 * Flat Combining throughput Test
 * Goal: Find actual throughput limits
 * Target: Explore if we can reach 7Mops/s per GPU
 * TODO: @Yihan Perf-tuning on tput first, then test bandwidth
 */

#include "../include/common.hpp"
#include "../include/ring_buffer_fc.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

using namespace uccl::flat_combining;

// Test configuration
struct ThroughputConfig {
  uint32_t num_warps;
  uint32_t num_proxies;
  uint32_t payload_size;
  uint32_t test_duration_ms;  // Fixed test time
  uint32_t warmup_iterations;
  bool verbose;
};

// Simple throughput tracking per warp
struct WarpMetrics {
  uint64_t request_count;
};

// Simplified payload creation - just fill with pattern
__device__ void create_simple_payload(uint32_t warp_id, uint32_t request_id,
                                      uint32_t payload_size, uint8_t* buffer) {
  // Simple pattern: warp_id in first byte, then sequential
  buffer[0] = (uint8_t)warp_id;
  for (uint32_t i = 1; i < payload_size; i++) {
    buffer[i] = (uint8_t)(i & 0xFF);
  }
}

// Throughput test kernel - simplified for pure throughput measurement
__global__ void fc_throughput_kernel(
    FCRingBufferManager* mgr_ptr, DeviceToHostCmdBuffer** ring_buffers,
    PublicationList* pub_list, uint32_t* warp_to_proxy_map,
    uint32_t* proxy_to_combiner_map, uint8_t** payload_buffers,
    uint32_t* payload_write_ptrs, ThroughputConfig config, WarpMetrics* metrics,
    bool volatile* stop_flag) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / 32;
  const uint32_t lane_id = tid % 32;

  if (lane_id != 0 || warp_id >= config.num_warps) return;

  // Initialize manager
  if (warp_id == 0) {
    const uint32_t payload_buffer_size = 128 * 1024 * 1024;
    mgr_ptr->init(ring_buffers, config.num_proxies, config.num_warps, pub_list,
                  warp_to_proxy_map, proxy_to_combiner_map, payload_buffers,
                  payload_buffer_size, payload_write_ptrs);
  }
  __syncthreads();

  uint32_t proxy_id = warp_to_proxy_map[warp_id];
  bool is_combiner = (warp_id == proxy_to_combiner_map[proxy_id]);

  // Initialize metrics
  if (!is_combiner) {
    metrics[warp_id].request_count = 0;
  }

  if (is_combiner) {
    // Combiner logic - unchanged from original
    mgr_ptr->run_combiner(warp_id, proxy_id, stop_flag);
  } else {
    // Producer logic - NO PAYLOAD COPYING for pure FC testing

    // Quick warmup
    for (uint32_t i = 0; i < config.warmup_iterations && !(*stop_flag); i++) {
      TransferCmd cmd = {};
      cmd.cmd = 1;
      cmd.bytes = config.payload_size;  // Just record size, no actual data
      cmd.message_idx = i;
      mgr_ptr->submit_request(warp_id, cmd);  // Use no-payload version
    }

    // Latency measurement phase
    uint64_t test_start = clock64();
    uint64_t test_duration_cycles =
        (uint64_t)config.test_duration_ms * 1980000;  // Approx 1.98GHz

    while (!(*stop_flag)) {
      uint64_t current_time = clock64();
      if (current_time - test_start > test_duration_cycles) break;

      // Create and submit request - metadata only
      TransferCmd cmd = {};
      cmd.cmd = 1;
      cmd.bytes = config.payload_size;  // Just record size
      cmd.message_idx = metrics[warp_id].request_count;

      mgr_ptr->submit_request(warp_id, cmd);  // No payload copying

      // Record metrics - only count requests
      metrics[warp_id].request_count++;
    }
  }
}

// Simple CPU proxy
void simple_cpu_proxy(DeviceToHostCmdBuffer* ring_buffer, int proxy_id,
                      bool volatile* stop_flag, uint64_t* processed_count) {
  uint64_t processed = 0;
  TransferCmd cmd;

  while (!*stop_flag) {
    if (ring_buffer->pop(cmd)) {
      processed++;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  }

  *processed_count = processed;
}

// Print throughput results
void print_throughput_results(std::vector<WarpMetrics> const& metrics,
                              double duration_sec) {
  uint64_t total_requests = 0;

  // Aggregate producer metrics (skip combiners)
  for (auto const& m : metrics) {
    if (m.request_count > 0) {
      total_requests += m.request_count;
    }
  }

  if (total_requests == 0) {
    printf("No requests completed!\n");
    return;
  }

  // Calculate throughput
  double throughput_ops = total_requests / duration_sec;

  printf("%5zu | %15.2f\n", metrics.size(), throughput_ops / 1e6);
}

// Run single test configuration
void run_throughput_test(uint32_t num_warps, uint32_t num_proxies,
                         uint32_t payload_size) {
  ThroughputConfig config = {.num_warps = num_warps,
                             .num_proxies = num_proxies,
                             .payload_size = payload_size,
                             .test_duration_ms = 5000,  // 5 second tests
                             .warmup_iterations = 50,
                             .verbose = false};

  // Allocate GPU memory (same as original)
  PublicationRecord* d_records;
  cudaMalloc(&d_records, sizeof(PublicationRecord) * MAX_WARPS);
  cudaMemset(d_records, 0, sizeof(PublicationRecord) * MAX_WARPS);

  PublicationList* d_pub_list;
  cudaMalloc(&d_pub_list, sizeof(PublicationList));
  PublicationList h_pub_list = {d_records, config.num_warps};
  cudaMemcpy(d_pub_list, &h_pub_list, sizeof(PublicationList),
             cudaMemcpyHostToDevice);

  // Ring buffers
  std::vector<DeviceToHostCmdBuffer*> h_ring_buffers(config.num_proxies);
  DeviceToHostCmdBuffer** d_ring_buffers;
  cudaMallocManaged(&d_ring_buffers,
                    sizeof(DeviceToHostCmdBuffer*) * config.num_proxies);

  for (uint32_t i = 0; i < config.num_proxies; i++) {
    void* rb_ptr;
    cudaMallocHost(&rb_ptr, sizeof(DeviceToHostCmdBuffer));
    h_ring_buffers[i] = new (rb_ptr) DeviceToHostCmdBuffer();
    d_ring_buffers[i] = h_ring_buffers[i];
  }

  // Mapping
  uint32_t* d_warp_to_proxy;
  uint32_t* d_proxy_to_combiner;
  cudaMalloc(&d_warp_to_proxy, sizeof(uint32_t) * config.num_warps);
  cudaMalloc(&d_proxy_to_combiner, sizeof(uint32_t) * config.num_proxies);

  std::vector<uint32_t> h_warp_to_proxy(config.num_warps);
  std::vector<uint32_t> h_proxy_to_combiner(config.num_proxies);

  // Fixed mapping to avoid dead warps - ensure at least one producer per active
  // proxy
  uint32_t active_proxies = std::min(config.num_proxies, config.num_warps);

  // Map warps to active proxies only
  for (uint32_t w = 0; w < config.num_warps; w++) {
    h_warp_to_proxy[w] = w % active_proxies;
  }

  // Set combiners - first warp mapped to each proxy becomes combiner
  for (uint32_t p = 0; p < config.num_proxies; p++) {
    if (p < config.num_warps) {
      h_proxy_to_combiner[p] = p;  // Warp p is combiner for proxy p
    } else {
      h_proxy_to_combiner[p] = INVALID_COMBINER;  // No combiner for this proxy
    }
  }

  cudaMemcpy(d_warp_to_proxy, h_warp_to_proxy.data(),
             sizeof(uint32_t) * config.num_warps, cudaMemcpyHostToDevice);
  cudaMemcpy(d_proxy_to_combiner, h_proxy_to_combiner.data(),
             sizeof(uint32_t) * config.num_proxies, cudaMemcpyHostToDevice);

  // Payload buffers
  const uint32_t payload_buffer_size = 128 * 1024 * 1024;
  std::vector<uint8_t*> h_payload_buffers(config.num_proxies);
  uint8_t** d_payload_buffers;
  cudaMallocManaged(&d_payload_buffers, sizeof(uint8_t*) * config.num_proxies);

  for (uint32_t i = 0; i < config.num_proxies; i++) {
    cudaMallocHost(&h_payload_buffers[i], payload_buffer_size);
    d_payload_buffers[i] = h_payload_buffers[i];
  }

  uint32_t* d_payload_write_ptrs;
  cudaMalloc(&d_payload_write_ptrs, sizeof(uint32_t) * config.num_proxies);
  cudaMemset(d_payload_write_ptrs, 0, sizeof(uint32_t) * config.num_proxies);

  // Manager and metrics
  FCRingBufferManager* d_mgr;
  cudaMalloc(&d_mgr, sizeof(FCRingBufferManager));

  WarpMetrics* d_metrics;
  cudaMallocManaged(&d_metrics, sizeof(WarpMetrics) * config.num_warps);
  cudaMemset(d_metrics, 0, sizeof(WarpMetrics) * config.num_warps);

  bool* d_stop;
  cudaMallocManaged(&d_stop, sizeof(bool));
  *d_stop = false;

  // Start CPU proxies
  bool volatile h_stop_flag = false;
  std::vector<uint64_t> proxy_processed(config.num_proxies, 0);
  std::vector<std::thread> proxy_threads;

  for (uint32_t i = 0; i < config.num_proxies; i++) {
    proxy_threads.emplace_back(simple_cpu_proxy, h_ring_buffers[i], i,
                               &h_stop_flag, &proxy_processed[i]);
  }

  // Launch kernel
  dim3 grid((config.num_warps * 32 + 255) / 256);
  dim3 block(256);

  auto start_time = std::chrono::high_resolution_clock::now();

  fc_throughput_kernel<<<grid, block>>>(
      d_mgr, d_ring_buffers, d_pub_list, d_warp_to_proxy, d_proxy_to_combiner,
      d_payload_buffers, d_payload_write_ptrs, config, d_metrics, d_stop);

  // Wait for test duration
  std::this_thread::sleep_for(
      std::chrono::milliseconds(config.test_duration_ms + 500));

  *d_stop = true;
  cudaDeviceSynchronize();
  h_stop_flag = true;

  for (auto& t : proxy_threads) {
    t.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double duration_sec =
      std::chrono::duration<double>(end_time - start_time).count();

  // Print results
  std::vector<WarpMetrics> h_metrics(config.num_warps);
  cudaMemcpy(h_metrics.data(), d_metrics,
             sizeof(WarpMetrics) * config.num_warps, cudaMemcpyDeviceToHost);

  print_throughput_results(h_metrics, duration_sec);

  // Cleanup
  cudaFree(d_records);
  cudaFree(d_pub_list);
  cudaFree(d_ring_buffers);
  cudaFree(d_warp_to_proxy);
  cudaFree(d_proxy_to_combiner);
  cudaFree(d_payload_buffers);
  cudaFree(d_payload_write_ptrs);
  cudaFree(d_mgr);
  cudaFree(d_metrics);
  cudaFree(d_stop);

  for (auto* rb : h_ring_buffers) {
    rb->~DeviceToHostCmdBuffer();
    cudaFreeHost(rb);
  }
  for (auto* pb : h_payload_buffers) {
    cudaFreeHost(pb);
  }
}

int main(int argc, char** argv) {
  // Initialize CUDA
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  // Test parameters
  // Throughput test - find saturation point
  std::vector<uint32_t> warp_counts = {64, 128, 256, 512, 1024};
  uint32_t payload_size = 32768;  // Fixed payload size
  uint32_t num_proxies = 4;

  printf("Warps | Throughput (Mops/s)\n");
  printf("------|------------------\n");

  for (auto num_warps : warp_counts) {
    run_throughput_test(num_warps, num_proxies, payload_size);
  }
  return 0;
}