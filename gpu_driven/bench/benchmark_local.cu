#include "bench_utils.hpp"
#include "gpu_kernel.cuh"
#include "proxy.hpp"
#include <thread>

int main(int argc, char** argv) {
  if (argc > 1) {
    std::fprintf(stderr, "Usage: ./benchmark_local\n");
    return 1;
  }
  BenchEnv env;
  init_env(env);
  std::vector<std::thread> threads;
  threads.reserve(env.blocks);
  for (int i = 0; i < env.blocks; ++i) {
    threads.emplace_back([&, i]() {
      Proxy p{make_cfg(env, i, /*rank*/ 0, /*peer_ip*/ nullptr)};
      p.run_local();
    });
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  const size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
  gpu_issue_batched_commands<<<env.blocks, kNumThPerBlock, shmem_bytes,
                               env.stream>>>(env.rbs);
  cudaCheckErrors("gpu_issue_batched_commands failed");
  cudaStreamSynchronize(env.stream);
  cudaCheckErrors("cudaStreamSynchronize failed");
  auto t1 = std::chrono::high_resolution_clock::now();

  for (auto& t : threads) t.join();

  print_block_latencies(env);
  const Stats s = compute_stats(env, t0, t1);
  print_summary(env, s);

  destroy_env(env);
  return 0;
}