#include "persistent.h"
#include "task.h"
#include "benchmarks/bench_support.h"
#include <cstdio>
#include <vector>

int main() {
  constexpr int fifo_cap = 1024;
  constexpr int warmup = 1000;
  constexpr int latency_iters = 10000;
  constexpr int throughput_iters = 100'000;

  printf("FIFO benchmark via PersistentKernel\n");

  UKernel::Device::TaskManager::instance().init(1);

  UKernel::Device::PersistentKernelConfig cfg;
  cfg.numBlocks = 1;
  cfg.threadsPerBlock = 64;
  cfg.fifoCapacity = fifo_cap;

  uint32_t test_block_id = 0;

  UKernel::Device::PersistentKernel<UKernel::Device::Task> kernel(cfg);
  kernel.launch();

  // warmup
  for (int i = 0; i < warmup; ++i) {
    kernel.submit(UKernel::Device::Task(UKernel::Device::TaskType::BenchNop,
                                         UKernel::Device::DataType::Fp32,
                                         test_block_id, 0));
  }

  while (!kernel.is_done(test_block_id, warmup - 1)) {
  }

  printf("Warmup done.\n");

  // latency
  std::vector<uint64_t> lat;
  lat.reserve(latency_iters);

  for (int i = 0; i < latency_iters; ++i) {
    uint64_t t0 = now_ns();
    uint64_t id = kernel.submit(UKernel::Device::Task(
        UKernel::Device::TaskType::BenchNop, UKernel::Device::DataType::Fp32,
        test_block_id, 0));
    kernel.is_done(test_block_id, id);
    uint64_t t1 = now_ns();
    lat.push_back(t1 - t0);
  }

  print_latency(lat);

  // throughput
  uint64_t t0 = now_ns();
  uint64_t first = kernel.submit(UKernel::Device::Task(
      UKernel::Device::TaskType::BenchNop, UKernel::Device::DataType::Fp32,
      test_block_id, 0));

  for (int i = 1; i < throughput_iters; ++i) {
    kernel.submit(UKernel::Device::Task(UKernel::Device::TaskType::BenchNop,
                                         UKernel::Device::DataType::Fp32,
                                         test_block_id, 0));
  }

  while (!kernel.is_done(test_block_id, first + throughput_iters - 1)) {
  }

  uint64_t t1 = now_ns();
  double sec = (t1 - t0) * 1e-9;

  printf("Throughput: %.2f K tasks/s\n", throughput_iters / sec / 1e3);

  kernel.stop();
  printf("Done.\n");
  return 0;
}
