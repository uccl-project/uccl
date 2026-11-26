#include "nccl_fusion.h"

#include <cublas_v2.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
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
void check(cublasStatus_t result) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS: %d\n", int(result));
    std::_Exit(4);
  }
}
void rankRun(int rank, ncclUniqueId id) {
  constexpr int k = 4096, n = 128, ranks = 8;
  check(cudaSetDevice(rank));
  ncclComm_t comm = nullptr;
  check(ncclCommInitRank(&comm, ranks, id, rank));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t blas;
  check(cublasCreate(&blas));
  check(cublasSetStream(blas, stream));
  std::vector<uint16_t> hostInput(k), hostWeight(n * k),
      fusedGather(ranks * k), referenceGather(ranks * k),
      fusedLogits(ranks * n), referenceLogits(ranks * n), rsOutput(k);
  for (int i = 0; i < k; ++i)
    hostInput[i] = bf16(float(rank + 1) * float(i % 17 - 8) / 128.f);
  for (int i = 0; i < n * k; ++i)
    hostWeight[i] = bf16(float((i / k) % 19 - 9) *
                         float(i % 23 - 11) / 512.f);
  void *input = nullptr, *weight = nullptr, *gathered = nullptr;
  void *gatheredRef = nullptr, *logits = nullptr, *logitsRef = nullptr;
  void* rsOutputGpu = nullptr;
  check(cudaMalloc(&input, k * 2));
  check(cudaMalloc(&weight, n * k * 2));
  check(cudaMalloc(&gathered, ranks * k * 2));
  check(cudaMalloc(&gatheredRef, ranks * k * 2));
  check(cudaMalloc(&logits, ranks * n * 2));
  check(cudaMalloc(&logitsRef, ranks * n * 2));
  check(cudaMalloc(&rsOutputGpu, k * 2));
  check(cudaMemcpyAsync(input, hostInput.data(), k * 2,
                        cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(weight, hostWeight.data(), n * k * 2,
                        cudaMemcpyHostToDevice, stream));
  for (int iteration = 0; iteration < 3; ++iteration) {
    check(uccl_lite_nccl_all_gather_gemm_128x4096(
        input, gathered, weight, logits, comm, stream));
    check(ncclAllGather(input, gatheredRef, k, ncclBfloat16, comm, stream));
    float alpha = 1.f, beta = 0.f;
    check(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, ranks, k, &alpha,
                       weight, CUDA_R_16BF, k, gatheredRef, CUDA_R_16BF, k,
                       &beta, logitsRef, CUDA_R_16BF, n,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    check(uccl_lite_nccl_reduce_scatter_sum2(
        gathered, gatheredRef, rsOutputGpu, k, comm, stream));
    check(cudaMemcpyAsync(fusedGather.data(), gathered, ranks * k * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(referenceGather.data(), gatheredRef, ranks * k * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(fusedLogits.data(), logits, ranks * n * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(referenceLogits.data(), logitsRef, ranks * n * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(rsOutput.data(), rsOutputGpu, k * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (int i = 0; i < ranks * k; ++i)
      if (fusedGather[i] != referenceGather[i]) {
        std::fprintf(stderr, "AG mismatch rank=%d i=%d\n", rank, i);
        std::_Exit(5);
      }
    float largest = 0.f;
    int differing = 0;
    for (int i = 0; i < ranks * n; ++i) {
      differing += fusedLogits[i] != referenceLogits[i];
      largest = std::fmax(largest,
                          std::fabs(fp32(fusedLogits[i]) -
                                    fp32(referenceLogits[i])));
    }
    if (rank == 0 && iteration == 0)
      std::fprintf(stderr, "AG+GEMM logits differing=%d/%d max_abs=%f\n",
                   differing, ranks * n, largest);
    if (largest > 0.02f) {
      std::fprintf(stderr, "AG+GEMM mismatch rank=%d largest=%f\n", rank,
                   largest);
      std::_Exit(6);
    }
    for (int i = 0; i < k; ++i) {
      uint16_t source =
          bf16(float(rank + 1) * float(i % 17 - 8) / 128.f);
      if (rsOutput[i] != bf16(16.f * fp32(source))) {
        std::fprintf(stderr, "AG+GEMM+RS mismatch rank=%d i=%d\n", rank, i);
        std::_Exit(7);
      }
    }
  }
  check(cublasDestroy(blas));
  check(cudaStreamDestroy(stream));
  check(cudaFree(input));
  check(cudaFree(weight));
  check(cudaFree(gathered));
  check(cudaFree(gatheredRef));
  check(cudaFree(logits));
  check(cudaFree(logitsRef));
  check(cudaFree(rsOutputGpu));
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
  std::puts(passed ? "UCCL Lite AG+GEMM 8-process check passed" :
                     "UCCL Lite AG+GEMM 8-process check failed");
  return passed ? 0 : 1;
}
