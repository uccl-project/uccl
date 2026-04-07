#include "benchmarks/bench_support.h"
#include "gpu_rt.h"
#include "worker.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

using namespace UKernel::Device;

namespace {

uint64_t enqueue_until_accepted(WorkerPool& pool, Task const& task,
                                uint32_t fifo_id) {
  while (true) {
    uint64_t task_id = pool.enqueue(task, fifo_id);
    if (task_id != WorkerPool::kInvalidTaskId) {
      return task_id;
    }
    std::this_thread::yield();
  }
}

void wait_until_done(WorkerPool& pool, uint64_t task_id, uint32_t fifo_id) {
  while (!pool.is_done(task_id, fifo_id)) {
    std::this_thread::yield();
  }
}

}  // namespace

int main(int argc, char** argv) {
  int device;
  gpuGetDevice(&device);
  int sm_count;
  gpuDeviceGetAttribute(&sm_count, gpuDevAttrMultiProcessorCount, device);

  printf("Number of SMs on the current GPU: %d\n", sm_count);

  constexpr int kDefaultWorkers = 64;
  int num_workers = std::min(sm_count, kDefaultWorkers);
  if (argc >= 2) {
    num_workers = std::max(1, std::min(sm_count, atoi(argv[1])));
  }

  constexpr int fifo_cap = 1024;
  int warmup = 1000;
  int latency_iters = 1000 * num_workers;
  constexpr int throughput_iters = 10'000;

  printf("FIFO benchmark via WorkerPool\n");
  printf("Launching %d persistent workers\n", num_workers);

  TaskManager::instance().init(1);

  WorkerPool::Config cfg;
  cfg.numMaxWorkers = num_workers;
  cfg.threadsPerBlock = 64;
  cfg.fifoCapacity = fifo_cap;

  WorkerPool pool(cfg);

  for (int i = 0; i < num_workers; ++i) {
    if (!pool.createWorker(i, 1)) {
      fprintf(stderr, "failed to create worker for fifo %d\n", i);
      return 2;
    }
  }
  for (int i = 0; i < num_workers; ++i) {
    pool.waitWorker(i);
  }

  std::vector<uint64_t> warmup_last_ids(num_workers, 0);
  for (uint32_t b = 0; b < static_cast<uint32_t>(num_workers); b++) {
    for (int i = 0; i < warmup; ++i) {
      Task t(TaskType::BenchNop, DataType::Fp32, b, 0);
      warmup_last_ids[b] = enqueue_until_accepted(pool, t, b);
    }
  }

  for (int fifo_id = 0; fifo_id < num_workers; ++fifo_id) {
    wait_until_done(pool, warmup_last_ids[fifo_id], fifo_id);
  }

  printf("Warmup done.\n");

  std::vector<uint64_t> lat;
  lat.reserve(latency_iters);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, num_workers - 1);

  for (int i = 0; i < latency_iters; ++i) {
    uint64_t t0 = now_ns();
    uint32_t test_fifo_id = dis(gen);
    Task t(TaskType::BenchNop, DataType::Fp32, 0, 0);
    uint64_t id = enqueue_until_accepted(pool, t, test_fifo_id);
    wait_until_done(pool, id, test_fifo_id);
    uint64_t t1 = now_ns();
    lat.push_back(t1 - t0);
  }

  print_latency(lat);

  uint64_t t0 = now_ns();

  std::vector<uint64_t> last_ids(num_workers, 0);
  for (int i = 0; i < throughput_iters; ++i) {
    uint32_t fifo_id = i % num_workers;
    Task t(TaskType::BenchNop, DataType::Fp32, 0, 0);
    uint64_t id = enqueue_until_accepted(pool, t, fifo_id);
    last_ids[fifo_id] = id;
  }

  for (int fifo_id = 0; fifo_id < num_workers; ++fifo_id) {
    wait_until_done(pool, last_ids[fifo_id], fifo_id);
  }

  uint64_t t1 = now_ns();
  double sec = (t1 - t0) * 1e-9;

  printf("Throughput: %.2f K tasks/s\n", throughput_iters / sec / 1e3);

  pool.shutdown_all();
  printf("Done.\n");
  return 0;
}
