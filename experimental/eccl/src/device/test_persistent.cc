#include "c2d_fifo.h"
#include "gpu_rt.h"
#include "operator.h"
#include "persistent.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define N 1024
using namespace eccl;

static void CK(gpuError_t e, char const* msg) {
  if (e != gpuSuccess) {
    std::cerr << "CUDA error: " << msg << ": " << gpuGetErrorString(e) << "\n";
    std::exit(1);
  }
}

static bool feq(float a, float b, float rtol = 1e-5f, float atol = 1e-6f) {
  float diff = std::fabs(a - b);
  return diff <= (atol + rtol * std::fabs(b));
}

static void fill(std::vector<float>& v, float base, float step) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = base + step * (float)i;
}

uint64_t submit_copy_task(eccl::PersistentKernel<eccl::Task>& kernel, void* dst,
                          void const* src, uint64_t bytes, eccl::DataType dtype,
                          uint32_t block_id) {
  eccl::CollArgs h{};
  h.src = const_cast<void*>(src);
  h.src2 = nullptr;
  h.dst = dst;
  h.bytes = static_cast<uint32_t>(bytes);
  h.redType = eccl::ReduceType::None;

  eccl::Task t = eccl::TaskManager::instance().create_coll_task(
      h, eccl::TaskType::CollCopy, dtype, block_id);

  return kernel.submit(t);
}

uint64_t submit_reduce_task(eccl::PersistentKernel<eccl::Task>& kernel,
                            void* dst, void const* src, uint64_t bytes,
                            eccl::DataType dtype, eccl::ReduceType redop,
                            uint32_t block_id) {
  eccl::CollArgs h{};
  h.src = const_cast<void*>(src);
  h.src2 = nullptr;
  h.dst = dst;
  h.bytes = static_cast<uint32_t>(bytes);
  h.redType = redop;

  eccl::Task t = eccl::TaskManager::instance().create_coll_task(
      h, eccl::TaskType::CollReduce, dtype, block_id);

  return kernel.submit(t);
}

int main() {
  eccl::TaskManager::instance().init(1024, 256);

  PersistentKernelConfig config;
  config.numBlocks = 3;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 16;
  config.smemSize = 0;

  uint32_t test_block_id = 0;
  uint32_t test_block_id_2 = 1;

  float *dst_copy = nullptr, *src_copy = nullptr;
  float *dst_reduce = nullptr, *src_reduce = nullptr;

  CK(gpuMalloc(&dst_copy, N * sizeof(float)), "gpuMalloc dst_copy");
  CK(gpuMalloc(&src_copy, N * sizeof(float)), "gpuMalloc src_copy");
  CK(gpuMalloc(&dst_reduce, N * sizeof(float)), "gpuMalloc dst_reduce");
  CK(gpuMalloc(&src_reduce, N * sizeof(float)), "gpuMalloc src_reduce");

  std::vector<float> h_src_copy(N), h_dst_copy(N, 0.0f);
  std::vector<float> h_dst0(N), h_src_red(N), h_dst1(N, 0.0f);

  fill(h_src_copy, 1.25f, 0.5f);
  fill(h_dst0, 2.0f, 0.25f);
  fill(h_src_red, -1.0f, 0.125f);

  CK(gpuMemcpy(src_copy, h_src_copy.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D src_copy");
  CK(gpuMemset(dst_copy, 0, N * sizeof(float)), "memset dst_copy");

  CK(gpuMemcpy(dst_reduce, h_dst0.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D dst_reduce");
  CK(gpuMemcpy(src_reduce, h_src_red.data(), N * sizeof(float),
               gpuMemcpyHostToDevice),
     "H2D src_reduce");

  PersistentKernel<eccl::Task> kernel(config);
  kernel.launch();
  std::cout << "Persistent kernel launched.\n";

  uint64_t id = submit_copy_task(kernel, dst_copy, src_copy, N * sizeof(float),
                                 DataType::Fp32, test_block_id);

  while (!kernel.is_done(test_block_id, id)) {
  }
  std::cout << "COPY DONE\n";

  id = submit_reduce_task(kernel, dst_reduce, src_reduce, N * sizeof(float),
                          DataType::Fp32, ReduceType::Sum, test_block_id_2);

  while (!kernel.is_done(test_block_id_2, id)) {
  }
  std::cout << "REDUCE DONE\n";

  id = submit_copy_task(kernel, dst_copy, src_copy, N * sizeof(float),
                        DataType::Fp32, test_block_id);

  while (!kernel.is_done(test_block_id, id)) {
  }
  std::cout << "COPY2 DONE\n";

  kernel.stop();
  std::cout << "Stop signal sent.\n";

  CK(gpuMemcpy(h_dst_copy.data(), dst_copy, N * sizeof(float),
               gpuMemcpyDeviceToHost),
     "D2H dst_copy");
  CK(gpuMemcpy(h_dst1.data(), dst_reduce, N * sizeof(float),
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
    if (bad) {
      std::cerr << "COPY FAILED mismatches=" << bad << "/" << N << "\n";
      return 2;
    }
    std::cout << "COPY PASSED\n";
  }

  {
    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
      float exp = h_dst0[i] + h_src_red[i];
      if (!feq(h_dst1[i], exp)) {
        if (bad < 8)
          std::cerr << "[REDUCE MISMATCH] i=" << i << " got=" << h_dst1[i]
                    << " exp=" << exp << "\n";
        ++bad;
      }
    }
    if (bad) {
      std::cerr << "REDUCE FAILED mismatches=" << bad << "/" << N << "\n";
      return 3;
    }
    std::cout << "REDUCE PASSED\n";
  }

  CK(gpuFree(dst_copy), "gpuFree dst_copy");
  CK(gpuFree(src_copy), "gpuFree src_copy");
  CK(gpuFree(dst_reduce), "gpuFree dst_reduce");
  CK(gpuFree(src_reduce), "gpuFree src_reduce");

  return 0;
}
