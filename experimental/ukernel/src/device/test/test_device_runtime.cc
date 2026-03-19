#include "../worker.h"
#include "test_support.h"
#include <iostream>
#include <vector>

#define N 1024

uint64_t submit_copy_task(
    UKernel::Device::WorkerPool& pool,
    void* dst, void const* src, uint64_t bytes,
    UKernel::Device::DataType dtype, uint32_t fifo_id) {
  UKernel::Device::TaskArgs h{};
  h.src = const_cast<void*>(src);
  h.src2 = nullptr;
  h.dst = dst;
  h.bytes = bytes;
  h.src_rank = 0;
  h.dst_rank = 0;
  h.src_device = 0;
  h.dst_device = 0;
  h.redType = UKernel::Device::ReduceType::None;
  h.flags = 0;

  UKernel::Device::Task t =
      UKernel::Device::TaskManager::instance().create_task(
          h, UKernel::Device::TaskType::CollCopy, dtype, fifo_id);

  return pool.enqueue(t, fifo_id);
}

uint64_t submit_reduce_task(
    UKernel::Device::WorkerPool& pool,
    void* dst, void const* src, uint64_t bytes,
    UKernel::Device::DataType dtype, UKernel::Device::ReduceType redop,
    uint32_t fifo_id) {
  UKernel::Device::TaskArgs h{};
  h.src = const_cast<void*>(src);
  h.src2 = nullptr;
  h.dst = dst;
  h.bytes = bytes;
  h.src_rank = 0;
  h.dst_rank = 0;
  h.src_device = 0;
  h.dst_device = 0;
  h.redType = redop;
  h.flags = 0;

  UKernel::Device::Task t =
      UKernel::Device::TaskManager::instance().create_task(
          h, UKernel::Device::TaskType::CollReduce, dtype, fifo_id);

  return pool.enqueue(t, fifo_id);
}

int main() {
  using UKernel::Device::Testing::ck;
  using UKernel::Device::Testing::feq;
  using UKernel::Device::Testing::fill;

  UKernel::Device::TaskManager::instance().init(1024);

  UKernel::Device::WorkerPool::Config config;
  config.numMaxWorkers = 8;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 16;

  float *dst_copy = nullptr, *src_copy = nullptr;
  float *dst_reduce = nullptr, *src_reduce = nullptr;

  ck(gpuMalloc(&dst_copy, N * sizeof(float)), "gpuMalloc dst_copy");
  ck(gpuMalloc(&src_copy, N * sizeof(float)), "gpuMalloc src_copy");
  ck(gpuMalloc(&dst_reduce, N * sizeof(float)), "gpuMalloc dst_reduce");
  ck(gpuMalloc(&src_reduce, N * sizeof(float)), "gpuMalloc src_reduce");

  std::vector<float> h_src_copy(N), h_dst_copy(N, 0.0f);
  std::vector<float> h_dst0(N), h_src_red(N), h_dst1(N, 0.0f);

  fill(h_src_copy, 1.25f, 0.5f);
  fill(h_dst0, 2.0f, 0.25f);
  fill(h_src_red, -1.0f, 0.125f);

  ck(gpuMemcpy(src_copy, h_src_copy.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D src_copy");
  ck(gpuMemset(dst_copy, 0, N * sizeof(float)), "memset dst_copy");

  ck(gpuMemcpy(dst_reduce, h_dst0.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D dst_reduce");
  ck(gpuMemcpy(src_reduce, h_src_red.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D src_reduce");

  UKernel::Device::WorkerPool pool(config);
  
  pool.createWorker(0, 1);
  pool.createWorker(1, 1);
  pool.waitWorker(0);
  pool.waitWorker(1);
  std::cout << "WorkerPool started with 2 workers (both single-SM).\n";

  uint64_t id = submit_copy_task(pool, dst_copy, src_copy, N * sizeof(float),
                                 UKernel::Device::DataType::Fp32, 0);
  while (!pool.is_done(id, 0)) {
  }
  std::cout << "COPY DONE\n";

  id = submit_reduce_task(pool, dst_reduce, src_reduce, N * sizeof(float),
                         UKernel::Device::DataType::Fp32,
                         UKernel::Device::ReduceType::Sum, 1);
  while (!pool.is_done(id, 1)) {
  }
  std::cout << "REDUCE DONE\n";

  id = submit_copy_task(pool, dst_copy, src_copy, N * sizeof(float),
                        UKernel::Device::DataType::Fp32, 0);
  while (!pool.is_done(id, 0)) {
  }
  std::cout << "COPY 2 DONE\n";

  pool.shutdown_all();
  std::cout << "WorkerPool shutdown.\n";

  ck(gpuMemcpy(h_dst_copy.data(), dst_copy, N * sizeof(float),
               gpuMemcpyDeviceToHost),
     "D2H dst_copy");
  ck(gpuMemcpy(h_dst1.data(), dst_reduce, N * sizeof(float),
               gpuMemcpyDeviceToHost),
     "D2H dst_reduce");

  {
    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
      if (!feq(h_dst_copy[i], h_src_copy[i])) {
        if (bad < 8)
          std::cerr << "[COPY MISMATCH] i=" << i << " got=" << h_dst_copy[i]
                    << " exp=" << h_src_copy[i] << "\n";
        ++bad;
      }
    }
    if (bad == 0) {
      std::cout << "COPY PASSED\n";
    } else {
      std::cout << "COPY FAILED mismatches=" << bad << "/" << N << "\n";
    }
  }

  {
    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
      float expected = h_dst0[i] + h_src_red[i];
      if (!feq(h_dst1[i], expected)) {
        if (bad < 8)
          std::cerr << "[REDUCE MISMATCH] i=" << i << " got=" << h_dst1[i]
                    << " exp=" << expected << "\n";
        ++bad;
      }
    }
    if (bad == 0) {
      std::cout << "REDUCE PASSED\n";
    } else {
      std::cout << "REDUCE FAILED mismatches=" << bad << "/" << N << "\n";
    }
  }

  ck(gpuFree(dst_copy), "free");
  ck(gpuFree(src_copy), "free");
  ck(gpuFree(dst_reduce), "free");
  ck(gpuFree(src_reduce), "free");

  return 0;
}