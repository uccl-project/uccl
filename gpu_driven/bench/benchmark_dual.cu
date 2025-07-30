#include "bench_utils.hpp"
#include "gpu_kernel.cuh"
#include "peer_copy_worker.hpp"
#include "proxy.hpp"
#include "rdma.hpp"
#include <chrono>
#include <cstdlib>
#include <functional>
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
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // Common CUDA + ring buffers for GPU producer
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

  // Create proxies (one per block), and launch one proxy thread per block
  std::vector<std::thread> cpu_threads;
  std::vector<std::unique_ptr<Proxy>> proxies;
  cpu_threads.reserve(env.blocks);
  proxies.reserve(env.blocks);

  for (int i = 0; i < env.blocks; ++i) {
    auto cfg = make_cfg(env, i, rank, peer_ip,
                        /*gpu_buffer*/ gpu_buffer,
                        /*total_size*/ total_size,
                        /*pin_thread*/ true);

    auto proxy = std::make_unique<Proxy>(std::move(cfg));

    // Start the proxy thread; capture raw pointer to ensure lifetime via
    // proxies[]
    cpu_threads.emplace_back([rank, p = proxy.get()]() {
      if (rank == 0) {
        p->run_sender();
      } else {
        p->run_remote();
      }
    });

    proxies.emplace_back(std::move(proxy));
  }

#ifdef ENABLE_PROXY_CUDA_MEMCPY
  // Remote peer-copy workers consume CopyRingBuffer tasks
  // Only needed on the receiving rank
  std::vector<std::thread> workers;
  std::vector<PeerWorkerCtx> worker_ctx;
  PeerCopyShared shared;

  if (rank != 0) {
    shared.src_device = 0;  // choose your source GPU
    workers.reserve(env.blocks);
    worker_ctx.resize(env.blocks);

    for (int i = 0; i < env.blocks; ++i) {
      workers.emplace_back(peer_copy_worker, std::ref(shared),
                           std::ref(worker_ctx[i]), std::ref(proxies[i]->ring),
                           i);
    }
  }
#endif

  if (rank == 0) {
    std::printf("Waiting for 2 seconds before issuing commands...\n");
    ::sleep(2);

    // Launch the GPU kernel that produces commands into env.rbs
    auto t0 = std::chrono::high_resolution_clock::now();
    const size_t shmem_bytes = kQueueSize * 2 * sizeof(unsigned long long);
    gpu_issue_batched_commands<<<env.blocks, kNumThPerBlock, shmem_bytes,
                                 env.stream>>>(env.rbs);
    cudaCheckErrors("gpu_issue_batched_commands kernel failed");
    cudaStreamSynchronize(env.stream);
    cudaCheckErrors("cudaStreamSynchronize failed");
    auto t1 = std::chrono::high_resolution_clock::now();

    // Join proxy threads
    for (auto& t : cpu_threads) t.join();

    // Report
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
    // Remote rank: run until you decide to stop the progress loops.
    // Here we wait a while and then signal shutdown cleanly.
    for (int i = 0; i < 60; ++i) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::printf("Rank %d is waiting...\n", rank);
    }

    // Stop proxy progress loops
    for (int i = 0; i < env.blocks; ++i) {
      proxies[i]->set_progress_run(false);
    }

    // Join proxy threads
    for (auto& t : cpu_threads) t.join();

#ifdef ENABLE_PROXY_CUDA_MEMCPY
    // Stop peer-copy workers and join
    shared.run.store(false, std::memory_order_release);
    for (auto& th : workers) th.join();
#endif

    destroy_env(env);
#ifdef USE_GRACE_HOPPER
    cudaFreeHost(gpu_buffer);
#else
    cudaFree(gpu_buffer);
#endif
    cudaCheckErrors("free gpu_buffer failed");
  }

  ::sleep(1);
  return 0;
}