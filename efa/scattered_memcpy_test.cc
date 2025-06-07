#include "scattered_memcpy.cuh"
#include "transport_config.h"
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <string.h>

#define TEST_RUNS 100

void checkDst(uint8_t* d_dst, uint64_t size) {
  uint8_t* h_dst = (uint8_t*)malloc(size);
  cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);
  for (uint64_t i = 0; i < size; i++) {
    if (h_dst[i] != 1) {
      std::cerr << "Error: h_dst[" << i << "] = " << (int)h_dst[i] << std::endl;
      exit(1);
    }
  }
}

void checkCuda(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

/**
 * Testing results on p4d A100
 * CUDA Scattered Memcpy Performance:
 *      Execution Time: 13.8324 us
 *      Memory Bandwidth: 82.6166 GB/s
 * Async CUDA Scattered Memcpy Performance:
 *      Launch Time: 3.39737 us
 *      Memory Bandwidth: 336.373 GB/s
 *      Polling Time: 0.590113 us
 */

int main() {
  uint32_t num_copies = MAX_COPIES;

  copy_param_t* h_params = new copy_param_t();

  void* dst_buf;
  cudaMalloc(&dst_buf, num_copies * EFA_MTU);

  uint32_t dst_offset = 0;

  // Allocate and initialize GPU memory for copies
  for (int i = 0; i < num_copies; i++) {
    uint64_t len = EFA_MTU - i * 16;
    h_params->len[i] = len;

    cudaMalloc((void**)&h_params->src[i], len);
    h_params->dst[i] = (uint64_t)((uint8_t*)dst_buf + dst_offset);

    dst_offset += len;

    // Initialize source memory with some data
    uint8_t* h_src = new uint8_t[len];
    memset(h_src, 1, len);
    cudaMemcpy((void*)h_params->src[i], h_src, len, cudaMemcpyHostToDevice);
    delete[] h_src;
  }

  uint64_t total_bytes = 0;
  for (int i = 0; i < num_copies; i++) {
    total_bytes += h_params->len[i];
  }

  // Warm up.
  launchScatteredMemcpy(num_copies, h_params);

  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TEST_RUNS; i++) {
    launchScatteredMemcpy(num_copies, h_params);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         end_time - start_time)
                         .count();
  double time_us = duration_ns / 1000.0 / TEST_RUNS;
  double bandwidth_gbps = total_bytes / (time_us * 1e3);

  std::cout << "CUDA Scattered Memcpy Performance:" << std::endl;
  std::cout << "  Execution Time: " << time_us << " us" << std::endl;
  std::cout << "  Memory Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

  checkDst((uint8_t*)dst_buf, total_bytes);

  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream));

  start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TEST_RUNS; i++) {
    launchScatteredMemcpyAsync(num_copies, h_params, stream);
  }
  end_time = std::chrono::high_resolution_clock::now();
  duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                                     start_time)
                    .count();
  time_us = duration_ns / 1000.0 / TEST_RUNS;
  bandwidth_gbps = total_bytes / (time_us * 1e3);

  std::cout << "Async CUDA Scattered Memcpy Performance:" << std::endl;
  std::cout << "  Launch Time: " << time_us << " us" << std::endl;
  std::cout << "  Memory Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

  start_time = std::chrono::high_resolution_clock::now();
  int finished_cnt = 0, poll_cnt = 0;
  while (finished_cnt < TEST_RUNS) {
    finished_cnt += pollScatteredMemcpy(stream);
    poll_cnt++;
  }
  end_time = std::chrono::high_resolution_clock::now();
  duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                                     start_time)
                    .count();
  time_us = duration_ns / 1000.0 / poll_cnt;
  std::cout << "  Polling Time: " << time_us << " us" << std::endl;

  checkDst((uint8_t*)dst_buf, total_bytes);

  for (int i = 0; i < num_copies; i++) {
    cudaFree((void*)h_params->src[i]);
    cudaFree((void*)dst_buf);
  }

  return 0;
}
