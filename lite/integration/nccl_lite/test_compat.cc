#include <nccl.h>
#include <cuda_runtime_api.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return uint16_t(bits >> 16);
}
void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
    std::exit(2);
  }
}
void check(ncclResult_t result) {
  if (result != ncclSuccess) {
    std::fprintf(stderr, "NCCL: %s\n", ncclGetErrorString(result));
    std::exit(3);
  }
}
void rankRun(int rank, ncclUniqueId id) {
  check(cudaSetDevice(rank));
  ncclComm_t comm = nullptr;
  check(ncclCommInitRank(&comm, 8, id, rank));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (size_t count :
       {size_t(8), size_t(128), size_t(1009), size_t(16384), size_t(131072)}) {
    std::vector<uint16_t> hostInput(count);
    for (size_t i = 0; i < count; ++i)
      hostInput[i] = bf16(float(rank + 1 + i % 4));
    std::vector<uint16_t> hostOutput(count * 8);
    void *input = nullptr, *output = nullptr;
    check(cudaMalloc(&input, count * 2));
    check(cudaMalloc(&output, count * 8 * 2));
    check(cudaMemcpyAsync(input, hostInput.data(), count * 2,
                          cudaMemcpyHostToDevice, stream));
    check(ncclAllGather(input, output, count, ncclBfloat16, comm, stream));
    check(cudaMemcpyAsync(hostOutput.data(), output, count * 8 * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (int r = 0; r < 8; ++r)
      for (size_t i = 0; i < count; ++i)
        if (hostOutput[size_t(r) * count + i] !=
            bf16(float(r + 1 + i % 4))) {
          std::fprintf(stderr, "AG mismatch rank=%d count=%zu source=%d i=%zu\n",
                       rank, count, r, i);
          std::exit(4);
        }
    check(cudaFree(input));
    check(cudaFree(output));

    // Distinct exact BF16 values per destination check RS shard placement.
    hostInput.resize(count * 8);
    for (int destination = 0; destination < 8; ++destination)
      for (size_t i = 0; i < count; ++i)
        hostInput[size_t(destination) * count + i] =
            bf16(float(rank + 1 + destination * 8));
    hostOutput.assign(count, 0);
    check(cudaMalloc(&input, count * 8 * 2));
    check(cudaMalloc(&output, count * 2));
    check(cudaMemcpyAsync(input, hostInput.data(), count * 8 * 2,
                          cudaMemcpyHostToDevice, stream));
    check(ncclReduceScatter(input, output, count, ncclBfloat16, ncclSum,
                            comm, stream));
    check(cudaMemcpyAsync(hostOutput.data(), output, count * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < count; ++i)
      if (hostOutput[i] != bf16(float(36 + rank * 64))) {
        std::fprintf(stderr, "RS mismatch rank=%d count=%zu i=%zu got=%04x\n",
                     rank, count, i, hostOutput[i]);
        std::exit(5);
      }
    check(cudaFree(input));
    check(cudaFree(output));

    hostInput.resize(count);
    for (size_t i = 0; i < count; ++i)
      hostInput[i] = bf16(float(rank + 1 + i % 4));
    hostOutput.assign(count, 0);
    check(cudaMalloc(&input, count * 2));
    check(cudaMalloc(&output, count * 2));
    check(cudaMemcpyAsync(input, hostInput.data(), count * 2,
                          cudaMemcpyHostToDevice, stream));
    check(ncclAllReduce(input, output, count, ncclBfloat16, ncclSum, comm,
                        stream));
    check(cudaMemcpyAsync(hostOutput.data(), output, count * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < count; ++i)
      if (hostOutput[i] != bf16(float(36 + 8 * (i % 4)))) {
        std::fprintf(stderr, "AR mismatch rank=%d count=%zu i=%zu\n", rank,
                     count, i);
        std::exit(6);
      }

    check(cudaMemcpyAsync(input, hostInput.data(), count * 2,
                          cudaMemcpyHostToDevice, stream));
    check(ncclAllReduce(input, input, count, ncclBfloat16, ncclSum, comm,
                        stream));
    check(cudaMemcpyAsync(hostOutput.data(), input, count * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < count; ++i)
      if (hostOutput[i] != bf16(float(36 + 8 * (i % 4)))) {
        std::fprintf(stderr, "in-place AR mismatch rank=%d count=%zu i=%zu\n",
                     rank, count, i);
        std::exit(7);
      }
    check(cudaFree(input));
    check(cudaFree(output));
  }
  // Queue mixed collectives before synchronizing. Input and result packet
  // credits must preserve every prior generation across operation changes.
  constexpr int burst = 9;
  void* burstInput[burst]{};
  void* burstOutput[burst]{};
  for (int step = 0; step < burst; ++step) {
    int operation = step % 3;
    size_t agCount = step == 3 ? size_t(2097152) : size_t(262144);
    size_t rsCount = step == 4   ? size_t(1048576)
                     : step == 7 ? size_t(128)
                                 : size_t(262144);
    size_t arCount = step == 5 ? size_t(8388608) : size_t(524288);
    size_t count = operation == 0   ? agCount
                   : operation == 1 ? rsCount * 8
                                    : arCount;
    size_t outputCount = operation == 0   ? agCount * 8
                         : operation == 1 ? rsCount
                                          : arCount;
    std::vector<uint16_t> values(count, bf16(float(rank + 1 + step)));
    if (operation == 1)
      for (int destination = 0; destination < 8; ++destination)
        for (size_t i = 0; i < rsCount; ++i)
          values[size_t(destination) * rsCount + i] =
              bf16(float(rank + 1 + step + destination * 8));
    check(cudaMalloc(&burstInput[step], count * 2));
    check(cudaMalloc(&burstOutput[step], outputCount * 2));
    check(cudaMemcpyAsync(burstInput[step], values.data(), count * 2,
                          cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  for (int step = 0; step < burst; ++step) {
    size_t agCount = step == 3 ? size_t(2097152) : size_t(262144);
    size_t rsCount = step == 4   ? size_t(1048576)
                     : step == 7 ? size_t(128)
                                 : size_t(262144);
    size_t arCount = step == 5 ? size_t(8388608) : size_t(524288);
    if (step % 3 == 0)
      check(ncclAllGather(burstInput[step], burstOutput[step], agCount,
                          ncclBfloat16, comm, stream));
    else if (step % 3 == 1)
      check(ncclReduceScatter(burstInput[step], burstOutput[step], rsCount,
                              ncclBfloat16, ncclSum, comm, stream));
    else
      check(ncclAllReduce(burstInput[step], burstOutput[step], arCount,
                          ncclBfloat16, ncclSum, comm, stream));
  }
  check(cudaStreamSynchronize(stream));
  for (int step = 0; step < burst; ++step) {
    int operation = step % 3;
    size_t agCount = step == 3 ? size_t(2097152) : size_t(262144);
    size_t rsCount = step == 4   ? size_t(1048576)
                     : step == 7 ? size_t(128)
                                 : size_t(262144);
    size_t arCount = step == 5 ? size_t(8388608) : size_t(524288);
    size_t count = operation == 0   ? agCount * 8
                   : operation == 1 ? rsCount
                                    : arCount;
    std::vector<uint16_t> actual(count);
    check(cudaMemcpy(actual.data(), burstOutput[step], count * 2,
                     cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
      int source = operation == 0 ? int(i / agCount) : -1;
      float expected = operation == 0   ? float(source + 1 + step)
                       : operation == 1 ? float(36 + 8 * step + rank * 64)
                                        : float(36 + 8 * step);
      if (actual[i] != bf16(expected)) {
        std::fprintf(stderr, "async burst mismatch rank=%d step=%d i=%zu\n",
                     rank, step, i);
        std::exit(8);
      }
    }
    check(cudaFree(burstInput[step]));
    check(cudaFree(burstOutput[step]));
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
  std::puts(passed ? "UCCL Lite NCCL AG/RS/AR 8-process check passed" :
                     "UCCL Lite NCCL AG/RS/AR 8-process check failed");
  return passed ? 0 : 1;
}
