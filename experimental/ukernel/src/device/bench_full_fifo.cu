#include "persistent.h"
#include "task.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <cuda_runtime.h>

using namespace UKernel;

static inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void print_latency(std::vector<uint64_t>& v) {
  std::sort(v.begin(), v.end());
  auto p = [&](double q) {
    size_t i = size_t(q * v.size());
    if (i >= v.size()) i = v.size() - 1;
    return v[i] / 1e3;
  };

  printf("Latency (us): min %.2f | p50 %.2f | p90 %.2f | p99 %.2f | max %.2f\n",
         v.front() / 1e3, p(0.5), p(0.9), p(0.99), v.back() / 1e3);
}

int main() {
  int device;
  gpuGetDevice(&device);
  int sm_count;
  gpuDeviceGetAttribute(&sm_count, gpuDevAttrMultiProcessorCount, device);

  printf("Number of SMs on the current GPU: %d\n", sm_count);

  constexpr int fifo_cap = 1024;
  int warmup = 1000;
  int latency_iters = 10000 * sm_count;
  constexpr int throughput_iters = 100'000;

  printf("FIFO benchmark via PersistentKernel\n");

  TaskManager::instance().init(1, 1);

  PersistentKernelConfig cfg;
  cfg.numBlocks = sm_count;
  cfg.threadsPerBlock = 64;
  cfg.fifoCapacity = fifo_cap;

  PersistentKernel<Task> kernel(cfg);
  kernel.launch();

  // warmup
  for (uint32_t b = 0; b < cfg.numBlocks; b++) {
    for (int i = 0; i < warmup; ++i) {
      kernel.submit(Task(TaskType::BenchNop, DataType::Fp32, b, 0));
    }
  }

  while (true) {
    bool all_done = true;
    for (int block_id = 0; block_id < sm_count; ++block_id) {
      if (!kernel.is_done(block_id, warmup - 1)) {
        all_done = false;
        break;
      }
    }
    if (all_done) break;
  }

  printf("Warmup done.\n");

  // latency
  std::vector<uint64_t> lat;
  lat.reserve(latency_iters);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, sm_count - 1);

  for (int i = 0; i < latency_iters; ++i) {
    uint64_t t0 = now_ns();
    uint32_t test_block_id = dis(gen);
    uint64_t id = kernel.submit(
        Task(TaskType::BenchNop, DataType::Fp32, test_block_id, 0));
    kernel.is_done(test_block_id, id);
    uint64_t t1 = now_ns();
    lat.push_back(t1 - t0);
  }

  print_latency(lat);

  // throughput
  uint64_t t0 = now_ns();

  std::vector<uint64_t> ids(sm_count, 0);
  for (int i = 0; i < throughput_iters; ++i) {
    uint32_t test_block_id = i % sm_count;
    uint64_t id = kernel.submit(
        Task(TaskType::BenchNop, DataType::Fp32, test_block_id, 0));

    ids[test_block_id] = id;
  }

  for (int block_id = 0; block_id < sm_count; ++block_id) {
    kernel.is_done(block_id, ids[block_id]);
  }

  uint64_t t1 = now_ns();
  double sec = (t1 - t0) * 1e-9;

  printf("Throughput: %.2f K tasks/s\n", throughput_iters / sec / 1e3);

  kernel.stop();
  printf("Done.\n");
  return 0;
}
