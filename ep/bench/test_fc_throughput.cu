/*
 * Flat Combining throughput Test
 * Goal: Find actual throughput limits 
 * Target: Explore if we can reach 7Mops/s per GPU
 * TODO: @Yihan Perf-tuning on tput first, then test bandwidth
 */

#include "../include/ring_buffer_fc.cuh"
#include "../include/common.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <thread>
#include <string>

using namespace uccl::flat_combining;

// Test configuration
struct ThroughputConfig {
    uint32_t num_warps;
    uint32_t num_proxies;
    uint32_t payload_size;
    uint32_t test_duration_ms;      // Fixed test time
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
    FCRingBufferManager* mgr_ptr,
    DeviceToHostCmdBuffer** ring_buffers,
    PublicationList* pub_list,
    uint32_t* warp_to_proxy_map,
    uint32_t* proxy_to_combiner_map,
    uint8_t** payload_buffers,
    uint32_t* payload_write_ptrs,
    ProxyWarpList* proxy_warp_lists,
    ThroughputConfig config,
    WarpMetrics* metrics,
    volatile bool* stop_flag
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;

    if (lane_id != 0 || warp_id >= config.num_warps) return;

    // Initialize manager
    if (warp_id == 0) {
        const uint32_t payload_buffer_size = 128 * 1024 * 1024;
        mgr_ptr->init(ring_buffers, config.num_proxies, config.num_warps,
                      pub_list, warp_to_proxy_map, proxy_to_combiner_map,
                      payload_buffers, payload_buffer_size, payload_write_ptrs,
                      proxy_warp_lists);
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
        uint64_t test_duration_cycles = (uint64_t)config.test_duration_ms * 1980000; // Approx 1.98GHz

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

// CPU proxy with head/tail batch processing like proxy.cpp
void simple_cpu_proxy(DeviceToHostCmdBuffer* ring_buffer, int proxy_id,
                     volatile bool* stop_flag, uint64_t* processed_count) {
    uint64_t processed = 0;
    uint64_t my_tail = 0;
    size_t seen = 0;

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
            uint64_t cmd = ring_buffer->volatile_load_cmd(i);
            // Non-blocking: break if cmd not ready
            if (cmd == 0) break;

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
        }
    }

    *processed_count = processed;
}

// Print throughput results
void print_throughput_results(const std::vector<WarpMetrics>& metrics,
                             double duration_sec) {
    uint64_t total_requests = 0;

    // Aggregate producer metrics (skip combiners)
    for (const auto& m : metrics) {
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

    printf("%5zu | %15.2f\n",
           metrics.size(),
           throughput_ops / 1e6);
}

// Run single test configuration
void run_throughput_test(uint32_t num_warps, uint32_t num_proxies, uint32_t payload_size, bool use_contiguous_mapping = false) {
    ThroughputConfig config = {
        .num_warps = num_warps,
        .num_proxies = num_proxies,
        .payload_size = payload_size,
        .test_duration_ms = 5000,  // 5 second tests
        .warmup_iterations = 50,
        .verbose = false
    };

    // Allocate GPU memory 
    PublicationRecord* d_records;
    cudaMalloc(&d_records, sizeof(PublicationRecord) * MAX_WARPS);
    cudaMemset(d_records, 0, sizeof(PublicationRecord) * MAX_WARPS);

    PublicationList* d_pub_list;
    cudaMalloc(&d_pub_list, sizeof(PublicationList));
    PublicationList h_pub_list = {d_records, config.num_warps};
    cudaMemcpy(d_pub_list, &h_pub_list, sizeof(PublicationList), cudaMemcpyHostToDevice);

    // Ring buffers
    std::vector<DeviceToHostCmdBuffer*> h_ring_buffers(config.num_proxies);
    DeviceToHostCmdBuffer** d_ring_buffers;
    cudaMallocManaged(&d_ring_buffers, sizeof(DeviceToHostCmdBuffer*) * config.num_proxies);

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

    // Create proxy warp lists for optimized combiner scanning
    std::vector<ProxyWarpList> h_proxy_warp_lists(config.num_proxies);
    for (auto& list : h_proxy_warp_lists) {
        list.init();
    }

    // Initialize warp mapping and build proxy warp lists
    init_warp_mapping(config.num_warps, config.num_proxies,
                      h_warp_to_proxy.data(), h_proxy_to_combiner.data(),
                      h_proxy_warp_lists.data(), use_contiguous_mapping);

    cudaMemcpy(d_warp_to_proxy, h_warp_to_proxy.data(),
               sizeof(uint32_t) * config.num_warps, cudaMemcpyHostToDevice);
    cudaMemcpy(d_proxy_to_combiner, h_proxy_to_combiner.data(),
               sizeof(uint32_t) * config.num_proxies, cudaMemcpyHostToDevice);

    // Allocate and copy proxy warp lists to GPU
    ProxyWarpList* d_proxy_warp_lists;
    cudaMallocManaged(&d_proxy_warp_lists, sizeof(ProxyWarpList) * config.num_proxies);

    // Copy the proxy warp lists structure and allocate GPU memory for warp_ids arrays
    for (uint32_t p = 0; p < config.num_proxies; p++) {
        d_proxy_warp_lists[p].count = h_proxy_warp_lists[p].count;
        if (h_proxy_warp_lists[p].count > 0) {
            cudaMallocManaged(&d_proxy_warp_lists[p].warp_ids,
                              sizeof(uint32_t) * h_proxy_warp_lists[p].count);
            cudaMemcpy(d_proxy_warp_lists[p].warp_ids, h_proxy_warp_lists[p].warp_ids,
                       sizeof(uint32_t) * h_proxy_warp_lists[p].count, cudaMemcpyHostToDevice);
        } else {
            d_proxy_warp_lists[p].warp_ids = nullptr;
        }
    }

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
    volatile bool h_stop_flag = false;
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
        d_mgr, d_ring_buffers, d_pub_list,
        d_warp_to_proxy, d_proxy_to_combiner,
        d_payload_buffers, d_payload_write_ptrs,
        d_proxy_warp_lists,
        config, d_metrics, d_stop
    );

    // Wait for test duration
    std::this_thread::sleep_for(std::chrono::milliseconds(config.test_duration_ms + 500));

    *d_stop = true;
    cudaDeviceSynchronize();
    h_stop_flag = true;

    for (auto& t : proxy_threads) {
        t.join();
    }

    // Use precise test duration instead of wall clock time
    double duration_sec = config.test_duration_ms / 1000.0;

    // Print results
    std::vector<WarpMetrics> h_metrics(config.num_warps);
    cudaMemcpy(h_metrics.data(), d_metrics, sizeof(WarpMetrics) * config.num_warps, cudaMemcpyDeviceToHost);

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

    // Cleanup proxy warp lists
    for (uint32_t p = 0; p < config.num_proxies; p++) {
        if (d_proxy_warp_lists[p].warp_ids != nullptr) {
            cudaFree(d_proxy_warp_lists[p].warp_ids);
        }
        if (h_proxy_warp_lists[p].warp_ids != nullptr) {
            delete[] h_proxy_warp_lists[p].warp_ids;
        }
    }
    cudaFree(d_proxy_warp_lists);

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

    // Parse command line arguments
    bool use_contiguous = false;
    if (argc > 1 && std::string(argv[1]) == "contiguous") {
        use_contiguous = true;
        printf("Using CONTIGUOUS mapping (better cache locality)\n");
    } else {
        printf("Using MODULO mapping (default interleaved)\n");
    }

    // Test parameters
    // Throughput test - find saturation point
    std::vector<uint32_t> warp_counts = {64, 128, 256, 512, 1024};
    uint32_t payload_size = 32768;  // Fixed payload size
    uint32_t num_proxies = 4;

    printf("\nWarps | Throughput (Mops/s)\n");
    printf("------|------------------\n");

    for (auto num_warps : warp_counts) {
        run_throughput_test(num_warps, num_proxies, payload_size, use_contiguous);
    }
    return 0;
}