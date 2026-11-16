// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <sched.h>

namespace uccl_lite {
constexpr size_t guardBytes = 4096;
inline void checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
    std::exit(1);
  }
}
// Each rank owns four physical cores, on its GPU's NUMA node. All subsequently
// created CUDA/NCCL/MSCCL++ workers inherit the same mask on both backends.
inline void pinRank(int rank) {
  cpu_set_t allowed, selected;
  CPU_ZERO(&allowed);
  CPU_ZERO(&selected);
  if (sched_getaffinity(0, sizeof(allowed), &allowed)) std::exit(2);
  char const* placement = std::getenv("UCCL_LITE_PLACEMENT");
  int base = (rank / 4) * 96 + (rank % 4) * 4;
  int width = 4;
  if (placement && std::string(placement) == "single") width = 1;
  if (placement && std::string(placement) == "legacy") {
    base = (rank / 4) * 84 + rank % 4;
    width = 1;
  }
  char const* cpus = std::getenv("UCCL_LITE_CPU_BASES");
  if (cpus) {
    std::stringstream list(cpus);
    std::string item;
    for (int i = 0; i <= rank; ++i)
      if (!std::getline(list, item, ',')) std::exit(2);
    base = std::stoi(item);
  }
  for (int cpu = base; cpu < base + width; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) {
      std::fprintf(stderr, "CPU %d outside allowed mask\n", cpu);
      std::exit(2);
    }
    CPU_SET(cpu, &selected);
  }
  if (sched_setaffinity(0, sizeof(selected), &selected)) std::exit(2);
  std::printf("# rank=%d cpus=%d-%d placement=%s\n", rank, base,
              base + width - 1, placement ? placement : "local-four");
  std::fflush(stdout);
}

__host__ __device__ inline int pattern(size_t j, int rank, int seed) {
  // Signed, changing, exactly representable data; different reduction groups
  // have different sums, making routing/offset mistakes observable.
  return int((j * 17 + (j / 97) * 13 + rank * 23 + seed * 31) % 127) - 63;
}
__global__ void initialize(float* data, size_t count, int rank, int seed) {
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < count;
       j += size_t(gridDim.x) * blockDim.x)
    data[j] = ldexpf(float(pattern(j, rank, seed)), -100);
}
__global__ void verifyOutput(float const* data, size_t count, int ranks,
                             int seed, int calls, unsigned long long* errors) {
  unsigned long long bad = 0;
  int growth = ranks == 8 ? 3 : 2;
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < count;
       j += size_t(gridDim.x) * blockDim.x) {
    int sum = 0;
    for (int r = 0; r < ranks; ++r) sum += pattern(j, r, seed);
    float expected = ldexpf(float(sum), -100 + growth * (calls - 1));
    if (!isfinite(data[j]) || data[j] != expected) ++bad;
  }
  if (bad) atomicAdd(errors, bad);
}
inline unsigned long long checkOutput(float* data, size_t count, int ranks,
                                      int seed, int calls, cudaStream_t stream,
                                      unsigned long long* errors) {
  // Finish communication before checker runtime/module operations.
  checkCuda(cudaStreamSynchronize(stream));
  checkCuda(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
  verifyOutput<<<256, 256, 0, stream>>>(data, count, ranks, seed, calls,
                                        errors);
  unsigned long long result;
  checkCuda(cudaMemcpyAsync(&result, errors, sizeof(result),
                            cudaMemcpyDeviceToHost, stream));
  checkCuda(cudaStreamSynchronize(stream));
  return result;
}
inline void setGuard(char* base, size_t bytes, cudaStream_t stream) {
  checkCuda(cudaMemsetAsync(base - guardBytes, 0xa5, guardBytes, stream));
  checkCuda(cudaMemsetAsync(base + bytes, 0xa5, guardBytes, stream));
}
inline unsigned long long checkGuard(char* base, size_t bytes) {
  std::array<unsigned char, guardBytes> guard;
  unsigned long long bad = 0;
  for (char* p : {base - guardBytes, base + bytes}) {
    checkCuda(cudaMemcpy(guard.data(), p, guardBytes, cudaMemcpyDeviceToHost));
    for (auto v : guard) bad += v != 0xa5;
  }
  return bad;
}
}  // namespace uccl_lite
