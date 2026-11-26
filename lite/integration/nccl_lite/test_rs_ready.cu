#include "uccl_lite/multiprocess_collectives.h"

#include <cuda_bf16.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int ranks = 8, shardCount = 4096, inputCount = ranks * shardCount;
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
void check(int result) {
  if (result != 0) {
    std::fprintf(stderr, "UCCL: %s\n", uccl_lite_mp_last_error());
    std::_Exit(3);
  }
}
__global__ void produceRows(__nv_bfloat16* output, uint32_t* ready, int rank) {
  int row = int(blockIdx.x);
  for (int i = int(threadIdx.x); i < shardCount; i += blockDim.x) {
    float value = float(rank + 1) * float(1 + row + i % 4) / 16.f;
    output[row * shardCount + i] = __float2bfloat16_rn(value);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < row * 500; ++i) __nanosleep(64);
    asm volatile("st.release.sys.global.u32 [%0], %1;"
                 :
                 : "l"(ready + row), "r"(1)
                 : "memory");
  }
}
void rankRun(int rank, std::string const& id) {
  check(cudaSetDevice(rank));
  char const* gpuOnly = std::getenv("UCCL_LITE_READY_TEST_GPU_ONLY");
  void* handle = gpuOnly && std::strcmp(gpuOnly, "1") == 0
                     ? uccl_lite_mp_create_rs_ready(id.c_str(), ranks, rank)
                     : uccl_lite_mp_create(id.c_str(), ranks, rank, 32UL << 20);
  if (!handle) {
    std::fprintf(stderr, "UCCL create: %s\n", uccl_lite_mp_last_error());
    std::_Exit(4);
  }
  cudaStream_t producer, transport;
  check(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
  check(cudaStreamCreateWithFlags(&transport, cudaStreamNonBlocking));
  void *first = nullptr, *second = nullptr, *combined = nullptr;
  void *output = nullptr, *reference = nullptr, *ready = nullptr;
  check(cudaMalloc(&first, inputCount * 2));
  check(cudaMalloc(&second, inputCount * 2));
  check(cudaMalloc(&combined, inputCount * 2));
  check(cudaMalloc(&output, shardCount * 2));
  check(cudaMalloc(&reference, shardCount * 2));
  check(cudaMalloc(&ready, ranks * sizeof(uint32_t)));
  std::vector<uint16_t> hostSecond(inputCount), hostCombined(inputCount),
      hostOutput(shardCount), hostReference(shardCount);
  for (int row = 0; row < ranks; ++row)
    for (int i = 0; i < shardCount; ++i) {
      int offset = row * shardCount + i;
      uint16_t a = bf16(float(rank + 1) * float(1 + row + i % 4) / 16.f);
      uint16_t b = bf16(float(rank + row + i % 5 - 4) / 32.f);
      hostSecond[offset] = b;
      hostCombined[offset] = bf16(fp32(a) + fp32(b));
    }
  check(cudaMemcpyAsync(second, hostSecond.data(), inputCount * 2,
                        cudaMemcpyHostToDevice, producer));
  check(cudaMemcpyAsync(combined, hostCombined.data(), inputCount * 2,
                        cudaMemcpyHostToDevice, producer));
  // Preload the producer kernel before a resident consumer waits on its flag.
  produceRows<<<ranks, 256, 0, producer>>>(
      static_cast<__nv_bfloat16*>(first), static_cast<uint32_t*>(ready), rank);
  check(cudaGetLastError());
  check(cudaStreamSynchronize(producer));
  for (int iteration = 0; iteration < 3; ++iteration) {
    check(cudaMemsetAsync(ready, 0, ranks * sizeof(uint32_t), producer));
    check(cudaStreamSynchronize(producer));
    check(uccl_lite_mp_enqueue_rs_ready_sum2(
        handle, first, second, output, static_cast<uint32_t const*>(ready),
        transport));
    produceRows<<<ranks, 256, 0, producer>>>(
        static_cast<__nv_bfloat16*>(first), static_cast<uint32_t*>(ready),
        rank);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(transport));
    check(cudaStreamSynchronize(producer));
    check(uccl_lite_mp_enqueue(handle, 2, combined, reference, shardCount,
                               transport));
    check(cudaStreamSynchronize(transport));
    check(cudaMemcpy(hostOutput.data(), output, shardCount * 2,
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hostReference.data(), reference, shardCount * 2,
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < shardCount; ++i)
      if (hostOutput[i] != hostReference[i]) {
        std::fprintf(stderr,
                     "ready RS mismatch rank=%d iteration=%d i=%d got=%04x ref=%04x\n",
                     rank, iteration, i, hostOutput[i], hostReference[i]);
        std::_Exit(5);
      }
  }
  check(cudaFree(first));
  check(cudaFree(second));
  check(cudaFree(combined));
  check(cudaFree(output));
  check(cudaFree(reference));
  check(cudaFree(ready));
  check(cudaStreamDestroy(producer));
  check(cudaStreamDestroy(transport));
  check(uccl_lite_mp_destroy(handle));
}
}  // namespace

int main() {
  std::string id = "ready-rs-test-" + std::to_string(getpid());
  std::vector<pid_t> children;
  for (int rank = 0; rank < ranks; ++rank) {
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
  std::puts(passed ? "UCCL Lite ready RS 8-process check passed" :
                     "UCCL Lite ready RS 8-process check failed");
  return passed ? 0 : 1;
}
