#include "bench_kernel.cuh"
#include "bench_utils.hpp"
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
 hipLaunchKernelGGL(( gpu_issue_batched_commands), dim3(env.blocks), dim3(kNumThPerBlock), shmem_bytes,
                               env.stream, env.rbs);
  GPU_RT_CHECK_ERRORS("gpu_issue_batched_commands failed");
  GPU_RT_CHECK(gpuStreamSynchronize(env.stream));
  auto t1 = std::chrono::high_resolution_clock::now();

  for (auto& t : threads) t.join();

  print_block_latencies(env);
  const Stats s = compute_stats(env, t0, t1);
  print_summary(env, s);

  destroy_env(env);
  return 0;
}