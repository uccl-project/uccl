#include "../worker.h"
#include "test_support.h"
#include <iostream>
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

int main() {
  std::cout << "===== Multi-Block Test =====\n";

  UKernel::Device::TaskManager::instance().init(1024);

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
  config.numMaxWorkers = 1;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 4;
  UKernel::Device::WorkerPool pool(config);

  pool.createWorker(0, 4);
  
  std::cout << "Waiting for worker ready...\n";
  pool.waitWorker(0);

  std::cout << "Created worker with 4 blocks\n";

  uint64_t task_id =
      submit_copy(pool, d_dst, d_src, N * sizeof(float),
                  UKernel::Device::DataType::Fp32, 0);

  while (!pool.is_done(task_id, 0)) {
  }

  std::cout << "Task done\n";

  UKernel::Device::Testing::ck(
      gpuMemcpy(h_dst.data(), d_dst, N * sizeof(float), gpuMemcpyDeviceToHost),
      "D2H");

  if (!verify_data(h_dst.data(), N, 1.0f, 0.5f)) {
    std::cout << "FAILED\n";
    return 1;
  }

  std::cout << "PASSED\n";

  gpuFree(d_dst);
  gpuFree(d_src);

  return 0;
}