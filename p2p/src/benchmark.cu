#include "common.hpp"
#include "ring_buffer.cuh"
#include "gpu_kernel.cuh"
#include "proxy.hpp"

#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstdlib>

int main() {
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaCheckErrors("cudaStreamCreate failed");

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("clock rate: %d kHz\n", prop.clockRate);

  RingBuffer* rbs;
  cudaHostAlloc(&rbs, sizeof(RingBuffer) * kNumThBlocks, cudaHostAllocMapped);

  for (int i = 0; i < kNumThBlocks; ++i) {
    rbs[i].head = 0;
    rbs[i].tail = 0;
    for (uint32_t j = 0; j < kQueueSize; ++j) {
      rbs[i].buf[j].cmd = 0;  // Initialize the buffer
    }
  }

  // Launch one CPU polling thread per block
  std::vector<std::thread> cpu_threads;
  for (int i = 0; i < kNumThBlocks; ++i) {
    cpu_threads.emplace_back(cpu_consume, &rbs[i], i);
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
  gpu_issue_batched_commands<<<kNumThBlocks, kNumThPerBlock, shmem_bytes,
                               stream1>>>(rbs);
  cudaCheckErrors("gpu_issue_batched_commands kernel failed");

  cudaStreamSynchronize(stream1);
  cudaCheckErrors("cudaStreamSynchronize failed");
  auto t1 = std::chrono::high_resolution_clock::now();


cudaDeviceSynchronize(); 
cudaCheckErrors("cudaDeviceSynchronize failed");

  for (auto& t : cpu_threads) {
    t.join();
  }

#ifdef MEASURE_PER_OP_LATENCY
//   unsigned long long h_cycles[kNumThBlocks];
//   unsigned int h_ops[kNumThBlocks];
//   cudaMemcpyFromSymbol(h_cycles, cycle_accum, sizeof(h_cycles));
//   cudaMemcpyFromSymbol(h_ops, op_count, sizeof(h_ops));
#endif

  unsigned int tot_ops = 0;
#ifdef MEASURE_PER_OP_LATENCY
  double total_us = 0;
  unsigned long long tot_cycles = 0;
  printf("\nPer-block avg latency:\n");
  for (int b = 0; b < kNumThBlocks; ++b) {
    double us = (double)rbs[b].cycle_accum * 1000.0 / prop.clockRate / rbs[b].op_count;
    printf("  Block %d : %.3f µs over %lu ops\n", b, us, rbs[b].op_count);
    total_us += us;
    tot_cycles += rbs[b].cycle_accum;
    tot_ops += rbs[b].op_count;
  }
#else
  tot_ops = kNumThBlocks * kIterations;
#endif
  double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double throughput = (double)(kNumThBlocks * kIterations) / (wall_ms * 1000.0);

#ifdef MEASURE_PER_OP_LATENCY
  printf("\nOverall avg GPU-measured latency  : %.3f µs\n",
         (double)tot_cycles * 1000.0 / prop.clockRate / tot_ops);
  printf("Total cycles                       : %llu\n", tot_cycles);
#endif
  printf("Total ops                          : %u\n", tot_ops);
  printf("End-to-end Wall-clock time        : %.3f ms\n", wall_ms);
  printf("Throughput                        : %.2f Mops/s\n", throughput);

  cudaFreeHost(rbs);
  cudaCheckErrors("cudaFreeHost failed");
  cudaStreamDestroy(stream1);
  cudaCheckErrors("cudaStreamDestroy failed");
}