#include "common.hpp"
#include "copy_ring.hpp"
#include "gpu_kernel.cuh"
#include "peer_copy_worker.hpp"
#include "proxy.hpp"
#include "rdma.hpp"
#include "ring_buffer.cuh"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./benchmark_remote <rank> <peer_ip>\n";
    return 1;
  }
  int rank = std::atoi(argv[1]);
  char const* peer_ip = argv[2];

  pin_thread_to_cpu(MAIN_THREAD_CPU_IDX);
#ifdef NUMA_AWARE_SCHEDULING
  int dev = -1;
  cudaGetDevice(&dev);
  printf("About to launch on GPU %d\n", dev);
  int numa_node = gpu_numa_node(dev);
  printf("Rank %d: GPU NUMA node is %d\n", rank, numa_node);
  discover_nics(numa_node);
#endif
  if (!GdrSupportInitOnce()) {
    printf(
        "Error: GPUDirect RDMA module is not loaded. Please load "
        "nvidia_peermem or nv_peer_mem!\n");
    exit(1);
  }

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaCheckErrors("cudaStreamCreate failed");

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("clock rate: %d kHz\n", prop.clockRate);

  DeviceToHostCmdBuffer* rbs;
  cudaHostAlloc(&rbs, sizeof(DeviceToHostCmdBuffer) * kNumThBlocks,
                cudaHostAllocMapped);
  for (int i = 0; i < kNumThBlocks; ++i) {
    rbs[i].head = 0;
    rbs[i].tail = 0;
    for (uint32_t j = 0; j < kQueueSize; ++j) {
      rbs[i].buf[j].cmd = 0;
    }
  }

  size_t total_size = kRemoteBufferSize;
  void* gpu_buffer = nullptr;
  cudaMalloc(&gpu_buffer, total_size);
#ifdef ENABLE_PROXY_CUDA_MEMCPY
  if (rank == 1) {
    for (int d = 0; d < NUM_GPUS; ++d) {
      cudaSetDevice(d);
      cudaMalloc(&per_GPU_device_buf[d], total_size);
      if (per_GPU_device_buf[d] == nullptr) {
        fprintf(stderr, "Failed to allocate GPU buffer on GPU %d\n", d);
        exit(1);
      }
    }
    cudaSetDevice(0);
  }
#endif

#ifndef NUMA_AWARE_SCHEDULING
  RDMAConnectionInfo local_info;
  global_rdma_init(gpu_buffer, total_size, &local_info, rank);
#endif
  std::vector<std::thread> cpu_threads;
  std::vector<CopyRing> g_rings(kNumThBlocks);
  for (int i = 0; i < kNumThBlocks; ++i) {
    if (rank == 0)
      cpu_threads.emplace_back(cpu_proxy, &rbs[i], i, gpu_buffer,
                               kRemoteBufferSize, rank, peer_ip);
    else {
      cpu_threads.emplace_back(remote_cpu_proxy, &rbs[i], i, gpu_buffer,
                               kRemoteBufferSize, rank, peer_ip,
                               std::ref(g_rings[i]));
    }
  }
  if (rank == 0) {
    printf("Waiting for 2 seconds before issuing commands...\n");
    sleep(2);

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t shmem_bytes = kQueueSize * 2 * sizeof(unsigned long long);
    gpu_issue_batched_commands<<<kNumThBlocks, kNumThPerBlock, shmem_bytes,
                                 stream1>>>(rbs);
    cudaCheckErrors("gpu_issue_batched_commands kernel failed");

    cudaStreamSynchronize(stream1);
    cudaCheckErrors("cudaStreamSynchronize failed");
    auto t1 = std::chrono::high_resolution_clock::now();

    printf("Before cpu_threads join\n");
    for (auto& t : cpu_threads) {
      t.join();
    }
    printf("After cpu_threads join\n");

    unsigned int tot_ops = 0;
#ifdef MEASURE_PER_OP_LATENCY
    double total_us = 0;
    unsigned long long tot_cycles = 0;
    printf("\nPer-block avg latency:\n");
    for (int b = 0; b < kNumThBlocks; ++b) {
      double us = (double)rbs[b].cycle_accum * 1000.0 / prop.clockRate /
                  rbs[b].op_count;
      printf("  Block %d : %.3f µs over %lu ops\n", b, us, rbs[b].op_count);
      total_us += us;
      tot_cycles += rbs[b].cycle_accum;
      tot_ops += rbs[b].op_count;
    }
#else
    tot_ops = kNumThBlocks * kIterations;
#endif
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double wall_ms_gpu = (rbs[0].cycle_end - rbs[0].cycle_start) * 1000.0 /
                         prop.clockRate / 1000.0;
    double throughput = (double)(tot_ops) / (wall_ms_gpu * 1000.0);

#ifdef MEASURE_PER_OP_LATENCY
    printf("\nOverall avg GPU-measured latency  : %.3f µs\n",
           (double)tot_cycles * 1000.0 / prop.clockRate / tot_ops);
    printf("Total cycles                      : %llu\n", tot_cycles);
#endif
    printf("Total ops                         : %u\n", tot_ops);
    printf("End-to-end wall-clock time        : %.3f ms\n", wall_ms_gpu);
    printf("Ops Throughput                    : %.2f Mops\n", throughput);
    printf("Total Throughput                  : %.2f Gbps\n",
           throughput * 1e6 * kObjectSize * 8 / 1e9);

    cudaFreeHost(rbs);
    cudaCheckErrors("cudaFreeHost failed");
    cudaStreamDestroy(stream1);
    cudaCheckErrors("cudaStreamDestroy failed");
  } else {
#ifdef ENABLE_PROXY_CUDA_MEMCPY

    int num_copy_engine = kNumThBlocks;
    std::vector<std::thread> copy_threads;
    copy_threads.reserve(num_copy_engine);

    for (int t = 0; t < num_copy_engine; ++t) {
      copy_threads.emplace_back(peer_copy_worker, std::ref(g_rings[t]), t);
    }
    g_run.store(true, std::memory_order_release);
#endif
    int i = 0;
    while (i < 60) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      printf("Rank %d is waiting...\n", rank);
      i++;
    }
    g_progress_run.store(false);

#ifdef ENABLE_PROXY_CUDA_MEMCPY
    g_run.store(false, std::memory_order_release);
    for (auto& th : copy_threads) th.join();
#endif

    exit(0);
  }
  sleep(1);
}