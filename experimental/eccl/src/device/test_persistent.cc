#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#include "persistent.h"
#include "operator.h"
#include "c2d_fifo.h"

#define N 1024
using namespace eccl;

static void CK(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(e) << "\n";
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

uint64_t submit_copy_task(PersistentKernel<OpTask>& kernel,
                          void* dst, const void* src,
                          uint64_t bytes, uint64_t wpt) {
  OpTask task(reinterpret_cast<uint64_t>(src),
              reinterpret_cast<uint64_t>(dst),
              bytes,
              OpTaskCopy, OpDataFp32, OpRedSum,
              wpt);
  return kernel.submit(task);
}

uint64_t submit_reduce_task(PersistentKernel<OpTask>& kernel,
                            void* dst, const void* src,
                            uint64_t elem_count, OpDataType dtype,
                            OpRedType redop, uint64_t wpt) {
  uint64_t elem_size = 0;
  switch (dtype) {
    case OpDataFp32: elem_size = 4; break;
    case OpDataFp16: elem_size = 2; break;
    case OpDataFp8:  elem_size = 1; break;
    default:         elem_size = 4; break;
  }

  OpTask task(reinterpret_cast<uint64_t>(src),
              reinterpret_cast<uint64_t>(dst),
              elem_count * elem_size,
              OpTaskReduce, dtype, redop,
              wpt);
  return kernel.submit(task);
}

int main() {
  PersistentKernelConfig config;
  config.numBlocks = 1;
  config.threadsPerBlock = 64;
  config.fifoCapacity = 16;
  config.smemSize = 0;

  float *dst_copy = nullptr, *src_copy = nullptr;
  float *dst_reduce = nullptr, *src_reduce = nullptr;

  CK(cudaMalloc(&dst_copy,   N * sizeof(float)), "cudaMalloc dst_copy");
  CK(cudaMalloc(&src_copy,   N * sizeof(float)), "cudaMalloc src_copy");
  CK(cudaMalloc(&dst_reduce, N * sizeof(float)), "cudaMalloc dst_reduce");
  CK(cudaMalloc(&src_reduce, N * sizeof(float)), "cudaMalloc src_reduce");

  std::vector<float> h_src_copy(N), h_dst_copy(N, 0.0f);
  std::vector<float> h_dst0(N), h_src_red(N), h_dst1(N, 0.0f);

  fill(h_src_copy, 1.25f, 0.5f);
  fill(h_dst0, 2.0f, 0.25f);
  fill(h_src_red, -1.0f, 0.125f);

  CK(cudaMemcpy(src_copy, h_src_copy.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D src_copy");
  CK(cudaMemset(dst_copy, 0, N * sizeof(float)), "memset dst_copy");

  CK(cudaMemcpy(dst_reduce, h_dst0.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D dst_reduce");
  CK(cudaMemcpy(src_reduce, h_src_red.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D src_reduce");

  PersistentKernel<OpTask> kernel(config);
  kernel.launch();
  std::cout << "Persistent kernel launched.\n";

  uint64_t id = submit_copy_task(kernel, dst_copy, src_copy, N * sizeof(float), 15);
  while (!kernel.is_done(id)) {}
  std::cout << "COPY DONE\n";

  id = submit_reduce_task(kernel, dst_reduce, src_reduce, (uint64_t)N, OpDataFp32, OpRedSum, 3);
  while (!kernel.is_done(id)) {}
  std::cout << "REDUCE DONE\n";

  kernel.stop();
  std::cout << "Stop signal sent.\n";

  CK(cudaMemcpy(h_dst_copy.data(), dst_copy, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H dst_copy");
  CK(cudaMemcpy(h_dst1.data(), dst_reduce, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H dst_reduce");

  {
    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
      if (!feq(h_dst_copy[i], h_src_copy[i])) {
        if (bad < 8) std::cerr << "[COPY MISMATCH] i=" << i << " got=" << h_dst_copy[i] << " exp=" << h_src_copy[i] << "\n";
        ++bad;
      }
    }
    if (bad) { std::cerr << "COPY FAILED mismatches=" << bad << "/" << N << "\n"; return 2; }
    std::cout << "COPY PASSED\n";
  }

  {
    size_t bad = 0;
    for (int i = 0; i < N; ++i) {
      float exp = h_dst0[i] + h_src_red[i];
      if (!feq(h_dst1[i], exp)) {
        if (bad < 8) std::cerr << "[REDUCE MISMATCH] i=" << i << " got=" << h_dst1[i] << " exp=" << exp << "\n";
        ++bad;
      }
    }
    if (bad) { std::cerr << "REDUCE FAILED mismatches=" << bad << "/" << N << "\n"; return 3; }
    std::cout << "REDUCE PASSED\n";
  }

  CK(cudaFree(dst_copy), "cudaFree dst_copy");
  CK(cudaFree(src_copy), "cudaFree src_copy");
  CK(cudaFree(dst_reduce), "cudaFree dst_reduce");
  CK(cudaFree(src_reduce), "cudaFree src_reduce");

  return 0;
}
