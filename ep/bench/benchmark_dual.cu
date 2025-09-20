#include "bench_utils.hpp"
#include "gpu_kernel.cuh"
#include "peer_copy_worker.hpp"
#include "proxy.hpp"
#include "rdma.hpp"
#include "util/gpu_rt.h"
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: CUDA_MODULE_LOADING=EAGER ./benchmark_dual <rank> "
                 "<peer_ip>\n";
    return 1;
  }
  int const rank = std::atoi(argv[1]);
  char const* peer_ip = argv[2];

  pin_thread_to_cpu(MAIN_THREAD_CPU_IDX);
  GPU_RT_CHECK(gpuSetDeviceFlags(gpuDeviceMapHost));

  // Common CUDA + ring-buffers setup
  BenchEnv env;
  init_env(env);

  // RDMA-visible buffer
  size_t const total_size = kRemoteBufferSize;
  void* gpu_buffer = nullptr;
#ifdef USE_GRACE_HOPPER
  GPU_RT_CHECK(gpuHostAlloc(&gpu_buffer, total_size, 0));
#else
  GPU_RT_CHECK(gpuMalloc(&gpu_buffer, total_size));
#endif

  // Build Proxies (one per block), each bound to its ring buffer
  std::vector<std::unique_ptr<Proxy>> proxies;
  proxies.reserve(env.blocks);
  for (int i = 0; i < env.blocks; ++i) {
    Proxy::Config cfg{};
    cfg.rb = &env.rbs[i];  // ring for this block
    cfg.thread_idx = i;
    cfg.gpu_buffer = gpu_buffer;  // RDMA-visible region
    cfg.total_size = total_size;
    cfg.rank = rank;
    cfg.peer_ip = peer_ip;
    cfg.pin_thread = true;

    proxies.emplace_back(std::make_unique<Proxy>(cfg));
  }

  // Launch one dual proxy thread per block (each does both TX and RX)
  std::vector<std::thread> cpu_threads;
  cpu_threads.reserve(env.blocks);
  for (int i = 0; i < env.blocks; ++i) {
    cpu_threads.emplace_back([&, i]() { proxies[i]->run_dual(); });
  }
  PeerCopyShared shared;
  shared.src_device = 0;
  std::vector<PeerWorkerCtx> worker_ctx(env.blocks);
  std::vector<std::thread> workers;
  workers.reserve(env.blocks);

  for (int i = 0; i < env.blocks; ++i) {
    workers.emplace_back(peer_copy_worker, std::ref(shared),
                         std::ref(worker_ctx[i]), std::ref(proxies[i]->ring),
                         i);
  }

  std::printf("[rank %d] Waiting 2s before issuing commands...\n", rank);
  ::sleep(2);  // give both ranks time to bring QPs up

  // Issue commands from GPU on BOTH ranks in duplex
  auto t0 = std::chrono::high_resolution_clock::now();
  gpu_issue_batched_commands<<<env.blocks, kNumThPerBlock, shmem_bytes_remote(),
                               env.stream>>>(env.rbs);
  GPU_RT_CHECK_ERRORS("gpu_issue_batched_commands kernel failed");
  GPU_RT_CHECK(gpuStreamSynchronize(env.stream));
  auto t1 = std::chrono::high_resolution_clock::now();

  // Reporting
  print_block_latencies(env);
  Stats const s = compute_stats(env, t0, t1);
  print_summary(env, s);

  // Sleep
  ::sleep(2);
  for (int i = 0; i < env.blocks; ++i) {
    proxies[i]->set_progress_run(false);
  }
  // Join proxy threads (each exits after consuming kIterations)
  for (auto& t : cpu_threads) t.join();
  std::printf("proxy threads joined\n");

  // Stop copy workers and join
  shared.run.store(false, std::memory_order_release);
  for (auto& th : workers) th.join();
  std::printf("copy threads joined\n");

  // Cleanup
  destroy_env(env);
#ifdef USE_GRACE_HOPPER
  GPU_RT_CHECK(gpuFreeHost(gpu_buffer));
#else
  GPU_RT_CHECK(gpuFree(gpu_buffer));
#endif

  return 0;
}