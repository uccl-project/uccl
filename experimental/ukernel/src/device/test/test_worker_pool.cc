#include "../worker.h"
#include "test_support.h"
#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {

void fill_data(float* ptr, size_t n, float base, float step) {
  for (size_t i = 0; i < n; ++i) {
    ptr[i] = base + step * static_cast<float>(i);
  }
}

bool verify_data(float const* ptr, size_t n, float base, float step) {
  for (size_t i = 0; i < n; ++i) {
    float expected = base + step * static_cast<float>(i);
    if (!UKernel::Device::Testing::feq(ptr[i], expected)) {
      std::cerr << "  [MISMATCH] i=" << i << " got=" << ptr[i]
                << " exp=" << expected << "\n";
      return false;
    }
  }
  return true;
}

uint64_t submit_copy(UKernel::Device::WorkerPool& pool, void* dst,
                     void const* src, size_t bytes,
                     UKernel::Device::DataType dtype, uint32_t fifo_id) {
  UKernel::Device::TaskArgs args{};
  args.src = const_cast<void*>(src);
  args.src2 = nullptr;
  args.dst = dst;
  args.bytes = bytes;
  args.src_rank = 0;
  args.dst_rank = 0;
  args.src_device = 0;
  args.dst_device = 0;
  args.redType = UKernel::Device::ReduceType::None;
  args.flags = 0;

  auto task = UKernel::Device::TaskManager::instance().create_task(
      args, UKernel::Device::TaskType::CollCopy, dtype, fifo_id);
  return pool.enqueue(task, fifo_id);
}

}  // namespace

void test_worker_lifecycle() {
  std::cout << "\n=== Test: Worker Lifecycle ===\n";

  UKernel::Device::WorkerPool::Config config;
  config.numMaxWorkers = 4;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 8;

  UKernel::Device::WorkerPool pool(config);

  if (pool.num_fifos() != config.numMaxWorkers) {
    std::cerr << "FAIL: num_fifos mismatch\n";
    return;
  }

  if (!pool.createWorker(0, 1)) {
    std::cerr << "FAIL: createWorker(0) failed\n";
    return;
  }

  if (!pool.createWorker(1, 1)) {
    std::cerr << "FAIL: createWorker(1) failed\n";
    return;
  }

  if (pool.createWorker(0, 1)) {
    std::cerr << "FAIL: duplicate createWorker should fail\n";
    return;
  }

  pool.waitWorker(0);
  pool.waitWorker(1);

  if (!pool.pollWorker(0)) {
    std::cerr << "FAIL: pollWorker(0) should be ready\n";
    return;
  }

  if (!pool.pollWorker(1)) {
    std::cerr << "FAIL: pollWorker(1) should be ready\n";
    return;
  }

  pool.destroyWorker(0);
  pool.destroyWorker(1);

  std::cout << "  PASSED\n";
}

void test_single_task() {
  std::cout << "\n=== Test: Single Task ===\n";

  constexpr size_t N = 256;

  float *d_dst = nullptr, *d_src = nullptr;
  UKernel::Device::Testing::ck(
      gpuMalloc(&d_dst, N * sizeof(float)), "malloc d_dst");
  UKernel::Device::Testing::ck(
      gpuMalloc(&d_src, N * sizeof(float)), "malloc d_src");

  std::vector<float> h_src(N), h_dst(N);
  fill_data(h_src.data(), N, 1.0f, 0.5f);

  UKernel::Device::Testing::ck(
      gpuMemcpy(d_src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice),
      "H2D");

  UKernel::Device::WorkerPool::Config config;
  config.numMaxWorkers = 2;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 4;
  UKernel::Device::WorkerPool pool(config);

  pool.createWorker(0, 1);
  pool.waitWorker(0);

  uint64_t task_id =
      submit_copy(pool, d_dst, d_src, N * sizeof(float),
                  UKernel::Device::DataType::Fp32, 0);

  while (!pool.is_done(task_id, 0)) {
  }

  UKernel::Device::Testing::ck(
      gpuMemcpy(h_dst.data(), d_dst, N * sizeof(float), gpuMemcpyDeviceToHost),
      "D2H");

  if (!verify_data(h_dst.data(), N, 1.0f, 0.5f)) {
    std::cerr << "FAIL: data mismatch\n";
  } else {
    std::cout << "  PASSED\n";
  }

  gpuFree(d_dst);
  gpuFree(d_src);
}

void test_multiple_fifos() {
  std::cout << "\n=== Test: Multiple FIFOs ===\n";

  constexpr size_t N = 128;

  float *d_dst0 = nullptr, *d_src0 = nullptr;
  float *d_dst1 = nullptr, *d_src1 = nullptr;
  UKernel::Device::Testing::ck(gpuMalloc(&d_dst0, N * sizeof(float)), "malloc");
  UKernel::Device::Testing::ck(gpuMalloc(&d_src0, N * sizeof(float)), "malloc");
  UKernel::Device::Testing::ck(gpuMalloc(&d_dst1, N * sizeof(float)), "malloc");
  UKernel::Device::Testing::ck(gpuMalloc(&d_src1, N * sizeof(float)), "malloc");

  std::vector<float> h_src0(N), h_src1(N), h_dst0(N), h_dst1(N);
  fill_data(h_src0.data(), N, 10.0f, 1.0f);
  fill_data(h_src1.data(), N, 20.0f, 2.0f);

  gpuMemcpy(d_src0, h_src0.data(), N * sizeof(float), gpuMemcpyHostToDevice);
  gpuMemcpy(d_src1, h_src1.data(), N * sizeof(float), gpuMemcpyHostToDevice);

  UKernel::Device::WorkerPool::Config config;
  config.numMaxWorkers = 4;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 8;
  UKernel::Device::WorkerPool pool(config);

  pool.createWorker(0, 1);
  pool.createWorker(1, 1);
  pool.waitWorker(0);
  pool.waitWorker(1);

  uint64_t id0 =
      submit_copy(pool, d_dst0, d_src0, N * sizeof(float),
                  UKernel::Device::DataType::Fp32, 0);
  uint64_t id1 =
      submit_copy(pool, d_dst1, d_src1, N * sizeof(float),
                  UKernel::Device::DataType::Fp32, 1);

  while (!pool.is_done(id0, 0) || !pool.is_done(id1, 1)) {
  }

  gpuMemcpy(h_dst0.data(), d_dst0, N * sizeof(float), gpuMemcpyDeviceToHost);
  gpuMemcpy(h_dst1.data(), d_dst1, N * sizeof(float), gpuMemcpyDeviceToHost);

  bool ok = true;
  if (!verify_data(h_dst0.data(), N, 10.0f, 1.0f)) ok = false;
  if (!verify_data(h_dst1.data(), N, 20.0f, 2.0f)) ok = false;

  if (ok) std::cout << "  PASSED\n";

  gpuFree(d_dst0);
  gpuFree(d_src0);
  gpuFree(d_dst1);
  gpuFree(d_src1);
}

void test_repeated_tasks() {
  std::cout << "\n=== Test: Repeated Tasks ===\n";

  constexpr size_t N = 64;
  constexpr int ITERATIONS = 10;

  float *d_dst = nullptr, *d_src = nullptr;
  UKernel::Device::Testing::ck(gpuMalloc(&d_dst, N * sizeof(float)), "malloc");
  UKernel::Device::Testing::ck(gpuMalloc(&d_src, N * sizeof(float)), "malloc");

  std::vector<float> h_src(N), h_dst(N);
  fill_data(h_src.data(), N, 1.0f, 0.1f);

  gpuMemcpy(d_src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice);

  UKernel::Device::WorkerPool::Config config;
  config.numMaxWorkers = 1;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 16;
  UKernel::Device::WorkerPool pool(config);

  pool.createWorker(0, 1);
  pool.waitWorker(0);

  for (int i = 0; i < ITERATIONS; ++i) {
    gpuMemset(d_dst, 0, N * sizeof(float));

    uint64_t id =
        submit_copy(pool, d_dst, d_src, N * sizeof(float),
                    UKernel::Device::DataType::Fp32, 0);

    while (!pool.is_done(id, 0)) {
    }

    gpuMemcpy(h_dst.data(), d_dst, N * sizeof(float), gpuMemcpyDeviceToHost);

    if (!verify_data(h_dst.data(), N, 1.0f, 0.1f)) {
      std::cerr << "FAIL: iteration " << i << "\n";
      gpuFree(d_dst);
      gpuFree(d_src);
      return;
    }
  }

  std::cout << "  PASSED\n";

  gpuFree(d_dst);
  gpuFree(d_src);
}

void test_shutdown_and_restart() {
  std::cout << "\n=== Test: Shutdown and Restart ===\n";

  float *d_dst = nullptr, *d_src = nullptr;
  constexpr size_t N = 64;
  UKernel::Device::Testing::ck(gpuMalloc(&d_dst, N * sizeof(float)), "malloc");
  UKernel::Device::Testing::ck(gpuMalloc(&d_src, N * sizeof(float)), "malloc");

  std::vector<float> h_src(N), h_dst(N);
  fill_data(h_src.data(), N, 5.0f, 0.2f);
  gpuMemcpy(d_src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice);

  UKernel::Device::WorkerPool::Config config;
  config.numMaxWorkers = 2;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 4;
  UKernel::Device::WorkerPool pool(config);

  pool.createWorker(0, 1);
  pool.waitWorker(0);

  uint64_t id = submit_copy(pool, d_dst, d_src, N * sizeof(float),
                             UKernel::Device::DataType::Fp32, 0);
  while (!pool.is_done(id, 0)) {
  }

  pool.shutdown_all();

  gpuMemcpy(h_dst.data(), d_dst, N * sizeof(float), gpuMemcpyDeviceToHost);
  if (!verify_data(h_dst.data(), N, 5.0f, 0.2f)) {
    std::cerr << "FAIL: data after shutdown\n";
    gpuFree(d_dst);
    gpuFree(d_src);
    return;
  }

  pool.createWorker(1, 1);
  pool.waitWorker(1);

  gpuMemset(d_dst, 0, N * sizeof(float));
  id = submit_copy(pool, d_dst, d_src, N * sizeof(float),
                   UKernel::Device::DataType::Fp32, 1);
  while (!pool.is_done(id, 1)) {
  }

  gpuMemcpy(h_dst.data(), d_dst, N * sizeof(float), gpuMemcpyDeviceToHost);
  if (!verify_data(h_dst.data(), N, 5.0f, 0.2f)) {
    std::cerr << "FAIL: data after restart\n";
  } else {
    std::cout << "  PASSED\n";
  }

  gpuFree(d_dst);
  gpuFree(d_src);
}

void test_invalid_operations() {
  std::cout << "\n=== Test: Invalid Operations ===\n";

  UKernel::Device::WorkerPool::Config config;
  config.numMaxWorkers = 2;
  config.fifoCapacity = 4;
  UKernel::Device::WorkerPool pool(config);

  if (pool.createWorker(100, 1)) {
    std::cerr << "FAIL: createWorker with invalid fifoId should fail\n";
    return;
  }

  if (pool.pollWorker(100)) {
    std::cerr << "FAIL: pollWorker with invalid fifoId should return false\n";
    return;
  }

  if (pool.is_done(0, 100)) {
    std::cerr << "FAIL: is_done with invalid fifoId should return true\n";
    return;
  }

  UKernel::Device::TaskArgs args{};
  args.src = nullptr;
  args.src2 = nullptr;
  args.dst = nullptr;
  args.bytes = 64;
  args.src_rank = 0;
  args.dst_rank = 0;
  args.src_device = 0;
  args.dst_device = 0;
  args.redType = UKernel::Device::ReduceType::None;
  args.flags = 0;
  auto task = UKernel::Device::TaskManager::instance().create_task(
      args, UKernel::Device::TaskType::CollCopy,
      UKernel::Device::DataType::Fp32, 0);

  if (pool.enqueue(task, 0) != 0) {
    std::cerr << "FAIL: enqueue without worker should fail\n";
    return;
  }

  std::cout << "  PASSED\n";
}

void test_resource_cleanup() {
  std::cout << "\n=== Test: Resource Cleanup ===\n";

  {
    UKernel::Device::WorkerPool::Config config;
    config.numMaxWorkers = 4;
    config.threadsPerBlock = 64;
    config.fifoCapacity = 8;
    UKernel::Device::WorkerPool pool(config);

    pool.createWorker(0, 1);
    pool.createWorker(1, 1);
    pool.waitWorker(0);
    pool.waitWorker(1);

    pool.destroyWorker(0);
    pool.destroyWorker(1);
  }

  {
    UKernel::Device::WorkerPool::Config config;
    config.numMaxWorkers = 2;
    UKernel::Device::WorkerPool pool(config);
    pool.createWorker(0, 1);
    pool.waitWorker(0);
    pool.shutdown_all();
  }

  std::cout << "  PASSED\n";
}

void test_concurrent_enqueue() {
  std::cout << "\n=== Test: Concurrent Enqueue ===\n";

  constexpr size_t N = 128;
  constexpr int TASKS_PER_FIFO = 5;

  float *d_dst = nullptr, *d_src = nullptr;
  UKernel::Device::Testing::ck(gpuMalloc(&d_dst, N * sizeof(float)), "malloc");
  UKernel::Device::Testing::ck(gpuMalloc(&d_src, N * sizeof(float)), "malloc");

  std::vector<float> h_src(N);
  fill_data(h_src.data(), N, 3.0f, 0.3f);
  gpuMemcpy(d_src, h_src.data(), N * sizeof(float), gpuMemcpyHostToDevice);

  UKernel::Device::WorkerPool::Config config;
  config.numMaxWorkers = 2;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 32;
  UKernel::Device::WorkerPool pool(config);

  pool.createWorker(0, 1);
  pool.createWorker(1, 1);
  pool.waitWorker(0);
  pool.waitWorker(1);

  std::vector<uint64_t> task_ids;

  for (int i = 0; i < TASKS_PER_FIFO; ++i) {
    gpuMemset(d_dst, 0, N * sizeof(float));
    uint64_t id = submit_copy(pool, d_dst, d_src, N * sizeof(float),
                              UKernel::Device::DataType::Fp32, 0);
    task_ids.push_back(id);

    gpuMemset(d_dst, 0, N * sizeof(float));
    id = submit_copy(pool, d_dst, d_src, N * sizeof(float),
                     UKernel::Device::DataType::Fp32, 1);
    task_ids.push_back(id);
  }

  for (uint64_t id : task_ids) {
    uint32_t fifo_id = id % 2;
    while (!pool.is_done(id, fifo_id)) {
    }
  }

  std::cout << "  PASSED\n";

  gpuFree(d_dst);
  gpuFree(d_src);
}

int main() {
  std::cout << "===== WorkerPool Test Suite =====\n";

  UKernel::Device::TaskManager::instance().init(1024);

  test_worker_lifecycle();
  test_single_task();
  test_multiple_fifos();
  test_repeated_tasks();
  test_shutdown_and_restart();
  test_invalid_operations();
  test_resource_cleanup();
  test_concurrent_enqueue();

  std::cout << "\n===== All Tests Passed =====\n";
  return 0;
}