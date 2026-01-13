#include "persistent.h"
#include "task.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace eccl;

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
  constexpr int fifo_cap = 1024;
  constexpr int warmup = 1000;
  constexpr int latency_iters = 10000;
  constexpr int throughput_iters = 1'000'000;

  printf("FIFO benchmark via PersistentKernel\n");

  TaskManager::instance().init(1, 1);

  PersistentKernelConfig cfg;
  cfg.numBlocks = 1;
  cfg.threadsPerBlock = 64;
  cfg.fifoCapacity = fifo_cap;

  PersistentKernel<Task> kernel(cfg);
  kernel.launch();

  // warmup
  for (int i = 0; i < warmup; ++i) {
    kernel.submit(Task(TaskType::BenchNop, DataType::Fp32, 0));
  }

  while (!kernel.is_done(warmup - 1)) {
  }

  printf("Warmup done.\n");

  // latency
  std::vector<uint64_t> lat;
  lat.reserve(latency_iters);

  for (int i = 0; i < latency_iters; ++i) {
    uint64_t t0 = now_ns();
    uint64_t id = kernel.submit(Task(TaskType::BenchNop, DataType::Fp32, 0));
    kernel.is_done(id);
    uint64_t t1 = now_ns();
    lat.push_back(t1 - t0);
  }

  print_latency(lat);

  // throughput
  uint64_t t0 = now_ns();
  uint64_t first = kernel.submit(Task(TaskType::BenchNop, DataType::Fp32, 0));

  for (int i = 1; i < throughput_iters; ++i) {
    kernel.submit(Task(TaskType::BenchNop, DataType::Fp32, 0));
  }

  while (!kernel.is_done(first + throughput_iters - 1)) {
  }

  uint64_t t1 = now_ns();
  double sec = (t1 - t0) * 1e-9;

  printf("Throughput: %.2f M tasks/s\n", throughput_iters / sec / 1e6);

  kernel.stop();
  printf("Done.\n");
  return 0;
}
