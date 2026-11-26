#include "nccl_fusion.h"

#include <cuda_bf16.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int ranks = 8, shardCount = 4096, repeats = 5, warmups = 20,
              iterations = 60;
struct Results {
  double us[2][repeats][ranks];
};
uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return uint16_t(bits >> 16);
}
void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
    std::_Exit(2);
  }
}
void check(ncclResult_t result) {
  if (result != ncclSuccess) {
    std::fprintf(stderr, "NCCL: %s\n", ncclGetErrorString(result));
    std::_Exit(3);
  }
}
__global__ void addBf16(__nv_bfloat16* output,
                        __nv_bfloat16 const* first,
                        __nv_bfloat16 const* second, int count) {
  for (int i = int(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int(gridDim.x) * blockDim.x)
    output[i] = __float2bfloat16_rn(__bfloat162float(first[i]) +
                                    __bfloat162float(second[i]));
}
void rankRun(int rank, ncclUniqueId id, Results* results) {
  check(cudaSetDevice(rank));
  ncclComm_t comm = nullptr;
  check(ncclCommInitRank(&comm, ranks, id, rank));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  constexpr int inputCount = shardCount * ranks;
  std::vector<uint16_t> first(inputCount), second(inputCount);
  for (int i = 0; i < inputCount; ++i) {
    first[i] = bf16(float((i * 13 + rank * 17) % 127 - 63) / 32.f);
    second[i] = bf16(float((i * 7 + rank * 23) % 113 - 56) / 64.f);
  }
  void *a = nullptr, *b = nullptr, *sum = nullptr, *output = nullptr;
  check(cudaMalloc(&a, inputCount * 2));
  check(cudaMalloc(&b, inputCount * 2));
  check(cudaMalloc(&sum, inputCount * 2));
  check(cudaMalloc(&output, shardCount * 2));
  check(cudaMemcpyAsync(a, first.data(), inputCount * 2,
                        cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(b, second.data(), inputCount * 2,
                        cudaMemcpyHostToDevice, stream));
  check(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&stop));
  auto enqueue = [&](int method) {
    if (method == 0) {
      addBf16<<<32, 256, 0, stream>>>(
          static_cast<__nv_bfloat16*>(sum),
          static_cast<__nv_bfloat16 const*>(a),
          static_cast<__nv_bfloat16 const*>(b), inputCount);
      check(cudaGetLastError());
      check(ncclReduceScatter(sum, output, shardCount, ncclBfloat16, ncclSum,
                              comm, stream));
    } else {
      check(uccl_lite_nccl_reduce_scatter_sum2(a, b, output, shardCount, comm,
                                                stream));
    }
  };
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (int order = 0; order < 2; ++order) {
      int method = (repeat + order) % 2;
      for (int i = 0; i < warmups; ++i) enqueue(method);
      check(cudaStreamSynchronize(stream));
      check(cudaEventRecord(start, stream));
      for (int i = 0; i < iterations; ++i) enqueue(method);
      check(cudaEventRecord(stop, stream));
      check(cudaEventSynchronize(stop));
      float ms = 0.f;
      check(cudaEventElapsedTime(&ms, start, stop));
      results->us[method][repeat][rank] = 1000. * ms / iterations;
    }
  }
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(stop));
  check(cudaFree(a));
  check(cudaFree(b));
  check(cudaFree(sum));
  check(cudaFree(output));
  check(cudaStreamDestroy(stream));
  check(ncclCommDestroy(comm));
}
}  // namespace

int main() {
  auto* results = static_cast<Results*>(
      mmap(nullptr, sizeof(Results), PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (results == MAP_FAILED) return 1;
  ncclUniqueId id;
  check(ncclGetUniqueId(&id));
  std::vector<pid_t> children;
  for (int rank = 0; rank < ranks; ++rank) {
    pid_t pid = fork();
    if (pid == 0) {
      rankRun(rank, id, results);
      std::_Exit(0);
    }
    if (pid < 0) return 1;
    children.push_back(pid);
  }
  for (pid_t pid : children) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) return 1;
  }
  for (int repeat = 0; repeat < repeats; ++repeat)
    for (int method = 0; method < 2; ++method) {
      double worst = 0;
      for (int rank = 0; rank < ranks; ++rank)
        worst = std::max(worst, results->us[method][repeat][rank]);
      std::printf("method=%s repeat=%d worst_rank_us=%.3f\n",
                  method ? "sum2_packet_rs" : "add_packet_rs", repeat, worst);
    }
  munmap(results, sizeof(Results));
}
