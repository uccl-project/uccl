#include "nccl_fusion.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return uint16_t(bits >> 16);
}
float fp32(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
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
void rankRun(int rank, ncclUniqueId id) {
  check(cudaSetDevice(rank));
  ncclComm_t comm = nullptr;
  check(ncclCommInitRank(&comm, 8, id, rank));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (size_t count : {size_t(128), size_t(4096), size_t(128)}) {
    size_t inputCount = count * 8;
    std::vector<uint16_t> first(inputCount), second(inputCount),
        combined(inputCount), fused(count), reference(count);
    for (size_t i = 0; i < inputCount; ++i) {
      first[i] = bf16(float(int((i * 13 + rank * 17) % 127) - 63) / 32.f);
      second[i] = bf16(float(int((i * 7 + rank * 23) % 113) - 56) / 64.f);
      combined[i] = bf16(fp32(first[i]) + fp32(second[i]));
    }
    void *firstGpu = nullptr, *secondGpu = nullptr, *combinedGpu = nullptr;
    void *fusedGpu = nullptr, *referenceGpu = nullptr;
    check(cudaMalloc(&firstGpu, inputCount * 2));
    check(cudaMalloc(&secondGpu, inputCount * 2));
    check(cudaMalloc(&combinedGpu, inputCount * 2));
    check(cudaMalloc(&fusedGpu, count * 2));
    check(cudaMalloc(&referenceGpu, count * 2));
    check(cudaMemcpyAsync(firstGpu, first.data(), inputCount * 2,
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(secondGpu, second.data(), inputCount * 2,
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(combinedGpu, combined.data(), inputCount * 2,
                          cudaMemcpyHostToDevice, stream));
    check(uccl_lite_nccl_reduce_scatter_sum2(
        firstGpu, secondGpu, fusedGpu, count, comm, stream));
    check(ncclReduceScatter(combinedGpu, referenceGpu, count, ncclBfloat16,
                            ncclSum, comm, stream));
    check(cudaMemcpyAsync(fused.data(), fusedGpu, count * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(reference.data(), referenceGpu, count * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < count; ++i)
      if (fused[i] != reference[i]) {
        std::fprintf(stderr,
                     "RS sum2 mismatch rank=%d count=%zu i=%zu fused=%04x reference=%04x\n",
                     rank, count, i, fused[i], reference[i]);
        std::_Exit(4);
      }
    check(cudaFree(firstGpu));
    check(cudaFree(secondGpu));
    check(cudaFree(combinedGpu));
    check(cudaFree(fusedGpu));
    check(cudaFree(referenceGpu));
  }
  check(cudaStreamDestroy(stream));
  check(ncclCommDestroy(comm));
}
}  // namespace

int main() {
  ncclUniqueId id;
  check(ncclGetUniqueId(&id));
  std::vector<pid_t> children;
  for (int rank = 0; rank < 8; ++rank) {
    pid_t pid = fork();
    if (pid == 0) {
      rankRun(rank, id);
      std::_Exit(0);
    }
    if (pid < 0) return 1;
    children.push_back(pid);
  }
  bool passed = true;
  for (pid_t pid : children) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) {
      std::fprintf(stderr, "child %d status=%d signal=%d exit=%d\n", pid,
                   status, WIFSIGNALED(status) ? WTERMSIG(status) : 0,
                   WIFEXITED(status) ? WEXITSTATUS(status) : -1);
      passed = false;
    }
  }
  std::puts(passed ? "UCCL Lite RS sum2 8-process check passed" :
                     "UCCL Lite RS sum2 8-process check failed");
  return passed ? 0 : 1;
}
