#include "benchmarks/bench_support.h"
#include "worker.h"
#include <cstdio>
#include <thread>
#include <vector>

namespace {

uint64_t enqueue_until_accepted(UKernel::Device::WorkerPool& pool,
                                UKernel::Device::Task const& task,
                                uint32_t fifo_id) {
  while (true) {
    uint64_t task_id = pool.enqueue(task, fifo_id);
    if (task_id != UKernel::Device::WorkerPool::kInvalidTaskId) {
      return task_id;
    }
    std::this_thread::yield();
  }
}

void wait_until_done(UKernel::Device::WorkerPool& pool, uint64_t task_id,
                     uint32_t fifo_id) {
  while (!pool.is_done(task_id, fifo_id)) {
    std::this_thread::yield();
  }
}

}  // namespace

int main() {
  constexpr int fifo_cap = 1024;
  constexpr int warmup = 1000;
  constexpr int latency_iters = 10000;
  constexpr int throughput_iters = 100'000;

  printf("FIFO benchmark via WorkerPool\n");

  UKernel::Device::TaskManager::instance().init(1);

  UKernel::Device::WorkerPool::Config cfg;
  cfg.numMaxWorkers = 1;
  cfg.threadsPerBlock = 64;
  cfg.fifoCapacity = fifo_cap;

  UKernel::Device::WorkerPool pool(cfg);
  pool.createWorker(0, 1);
  pool.waitWorker(0);

  uint32_t test_block_id = 0;
  uint64_t warmup_last_id = 0;

  for (int i = 0; i < warmup; ++i) {
    UKernel::Device::Task task(UKernel::Device::TaskType::BenchNop,
                               UKernel::Device::DataType::Fp32,
                               test_block_id, 0);
    warmup_last_id = enqueue_until_accepted(pool, task, 0);
  }

  wait_until_done(pool, warmup_last_id, 0);
  printf("Warmup done.\n");

  std::vector<uint64_t> lat;
  lat.reserve(latency_iters);

  for (int i = 0; i < latency_iters; ++i) {
    uint64_t t0 = now_ns();
    UKernel::Device::Task task(UKernel::Device::TaskType::BenchNop,
                               UKernel::Device::DataType::Fp32,
                               test_block_id, 0);
    uint64_t id = enqueue_until_accepted(pool, task, 0);
    wait_until_done(pool, id, 0);
    uint64_t t1 = now_ns();
    lat.push_back(t1 - t0);
  }

  print_latency(lat);

  uint64_t t0 = now_ns();
  uint64_t last_id = 0;
  for (int i = 0; i < throughput_iters; ++i) {
    UKernel::Device::Task task(UKernel::Device::TaskType::BenchNop,
                               UKernel::Device::DataType::Fp32,
                               test_block_id, 0);
    uint64_t id = enqueue_until_accepted(pool, task, 0);
    last_id = id;
  }

  wait_until_done(pool, last_id, 0);

  uint64_t t1 = now_ns();
  double sec = (t1 - t0) * 1e-9;

  printf("Throughput: %.2f K tasks/s\n", throughput_iters / sec / 1e3);

  pool.shutdown_all();
  printf("Done.\n");
  return 0;
}
