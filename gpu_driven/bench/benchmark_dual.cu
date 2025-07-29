#include "bench_utils.hpp"
#include "gpu_kernel.cuh"
#include "peer_copy_worker.hpp"
#include "proxy.hpp"
#include "rdma.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./benchmark_dual <rank> <peer_ip>\n";
    return 1;
  }
  int const rank = std::atoi(argv[1]);
  char const* peer_ip = argv[2];

  pin_thread_to_cpu(MAIN_THREAD_CPU_IDX);
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // Common CUDA + rbs setup
  BenchEnv env;
  init_env(env);

  // RDMA-visible buffer
  const size_t total_size = kRemoteBufferSize;
  void* gpu_buffer = nullptr;
#ifdef USE_GRACE_HOPPER
  cudaMallocHost(&gpu_buffer, total_size);
#else
  cudaMalloc(&gpu_buffer, total_size);
#endif
  cudaCheckErrors("gpu_buffer allocation failed");

#ifdef ENABLE_PROXY_CUDA_MEMCPY
  // Optional per-GPU staging on BOTH ranks in duplex (enable if you need it)
  for (int d = 0; d < NUM_GPUS; ++d) {
    cudaSetDevice(d);
    void* buf = nullptr;
    cudaMalloc(&buf, total_size);
    cudaCheckErrors("cudaMalloc per_GPU_device_buf failed");
    per_GPU_device_buf[d] = buf;
  }
  cudaSetDevice(0);
#endif

  // One CopyRingBuffer per block (required by dual mode on both ranks)
  std::vector<CopyRingBuffer> rings(env.blocks);

  // Launch one dual proxy thread per block
  std::vector<std::thread> cpu_threads;
  cpu_threads.reserve(env.blocks);
  for (int i = 0; i < env.blocks; ++i) {
    cpu_threads.emplace_back([&, i]() {
      Proxy p{make_cfg(env, i, rank, peer_ip,
                       /*gpu_buffer*/ gpu_buffer,
                       /*total_size*/ total_size,
                       /*ring*/ &rings[i],
                       /*pin_thread*/ true)};
      p.run_dual();  // single thread does both TX and RX
    });
  }

#ifdef ENABLE_PROXY_CUDA_MEMCPY
  // Optional copy engines that consume CopyRingBuffer tasks (both ranks)
  std::vector<std::thread> copy_threads;
  copy_threads.reserve(env.blocks);
  for (int t = 0; t < env.blocks; ++t) {
    copy_threads.emplace_back(peer_copy_worker, std::ref(rings[t]), t);
  }
  g_run.store(true, std::memory_order_release);
#endif
  std::printf("[rank %d] Waiting 2s before issuing commands...\n", rank);
  ::sleep(2);

  // Issue commands from GPU on BOTH ranks in duplex
  auto t0 = std::chrono::high_resolution_clock::now();
  gpu_issue_batched_commands<<<env.blocks, kNumThPerBlock, shmem_bytes_remote(),
                               env.stream>>>(env.rbs);
  cudaCheckErrors("gpu_issue_batched_commands kernel failed");
  cudaStreamSynchronize(env.stream);
  cudaCheckErrors("cudaStreamSynchronize failed");
  auto t1 = std::chrono::high_resolution_clock::now();

  // Reporting
  print_block_latencies(env);
  const Stats s = compute_stats(env, t0, t1);
  print_summary(env, s);
  ::sleep(30);

  // Join proxy threads
  for (auto& t : cpu_threads) t.join();

#ifdef ENABLE_PROXY_CUDA_MEMCPY
  g_run.store(false, std::memory_order_release);
  for (auto& th : copy_threads) th.join();
#endif

  // Cleanup
  destroy_env(env);
#ifdef USE_GRACE_HOPPER
  cudaFreeHost(gpu_buffer);
#else
  cudaFree(gpu_buffer);
#endif
  cudaCheckErrors("free gpu_buffer failed");

  return 0;
}