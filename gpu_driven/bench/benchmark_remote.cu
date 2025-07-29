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
    std::cerr << "Usage: ./benchmark_remote <rank> <peer_ip>\n";
    return 1;
  }
  int const rank = std::atoi(argv[1]);
  char const* peer_ip = argv[2];

  pin_thread_to_cpu(MAIN_THREAD_CPU_IDX);

  BenchEnv env;
  init_env(env);
  const size_t total_size = kRemoteBufferSize;
  void* gpu_buffer = nullptr;
#ifdef USE_GRACE_HOPPER
  cudaMallocHost(&gpu_buffer, total_size);
#else
  cudaMalloc(&gpu_buffer, total_size);
#endif
  cudaCheckErrors("gpu_buffer allocation failed");

#ifdef ENABLE_PROXY_CUDA_MEMCPY
  if (rank == 1) {
    for (int d = 0; d < NUM_GPUS; ++d) {
      cudaSetDevice(d);
      void* buf = nullptr;
      cudaMalloc(&buf, total_size);
      cudaCheckErrors("cudaMalloc per_GPU_device_buf failed");
      per_GPU_device_buf[d] = buf;
    }
    cudaSetDevice(0);
  }
#endif
  std::vector<CopyRingBuffer> rings(env.blocks);
  std::vector<std::thread> cpu_threads;
  cpu_threads.reserve(env.blocks);
  for (int i = 0; i < env.blocks; ++i) {
    cpu_threads.emplace_back([&, i]() {
      Proxy p{make_cfg(env, i, rank, peer_ip,
                       /*gpu_buffer*/ gpu_buffer,
                       /*total_size*/ total_size,
                       /*ring*/ (rank == 0) ? nullptr : &rings[i],
                       /*pin_thread*/ true)};
      if (rank == 0)
        p.run_sender();
      else
        p.run_remote();
    });
  }

  if (rank == 0) {
    std::printf("Waiting for 2 seconds before issuing commands...\n");
    ::sleep(2);
    auto t0 = std::chrono::high_resolution_clock::now();
    const size_t shmem_bytes = kQueueSize * 2 * sizeof(unsigned long long);
    gpu_issue_batched_commands<<<env.blocks, kNumThPerBlock, shmem_bytes,
                                 env.stream>>>(env.rbs);
    cudaCheckErrors("gpu_issue_batched_commands kernel failed");
    cudaStreamSynchronize(env.stream);
    cudaCheckErrors("cudaStreamSynchronize failed");
    auto t1 = std::chrono::high_resolution_clock::now();

    for (auto& t : cpu_threads) t.join();
    print_block_latencies(env);
    const Stats s = compute_stats(env, t0, t1);
    print_summary(env, s);
    destroy_env(env);
#ifdef USE_GRACE_HOPPER
    cudaFreeHost(gpu_buffer);
#else
    cudaFree(gpu_buffer);
#endif
    cudaCheckErrors("free gpu_buffer failed");

  } else {
#ifdef ENABLE_PROXY_CUDA_MEMCPY
    int const num_copy_engine = env.blocks;
    std::vector<std::thread> copy_threads;
    copy_threads.reserve(num_copy_engine);
    for (int t = 0; t < num_copy_engine; ++t) {
      copy_threads.emplace_back(peer_copy_worker, std::ref(rings[t]), t);
    }
    g_run.store(true, std::memory_order_release);
#endif
    for (int i = 0; i < 60; ++i) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::printf("Rank %d is waiting...\n", rank);
    }
    g_progress_run.store(false);

#ifdef ENABLE_PROXY_CUDA_MEMCPY
    g_run.store(false, std::memory_order_release);
    for (auto& th : copy_threads) th.join();
#endif
    destroy_env(env);
#ifdef USE_GRACE_HOPPER
    cudaFreeHost(gpu_buffer);
#else
    cudaFree(gpu_buffer);
#endif
    cudaCheckErrors("free gpu_buffer failed");
    return 0;
  }

  ::sleep(1);
  return 0;
}