// SPDX-License-Identifier: Apache-2.0
// Match the original PCIeCCL FP32/in-place C++ batch timing contract.
#include <mscclpp/core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

constexpr int kRanks = 8;

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t result = (call);                                      \
    if (result != cudaSuccess) {                                      \
      std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result)); \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

#define NCCL_CHECK(call)                                              \
  do {                                                                \
    ncclResult_t result = (call);                                     \
    if (result != ncclSuccess) {                                      \
      std::fprintf(stderr, "NCCL: %s\n", ncclGetErrorString(result)); \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

// Deliberately preserve the original harness's CPU pinning, including its
// original 84-core boundary, to avoid changing placement between backends.
void pinRankToLocalCpu(int rank) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
  int lo = rank < 4 ? 0 : 84;
  int hi = rank < 4 ? 84 : 168;
  int preferred = lo + (rank & 3);
  for (int offset = 0; offset < hi - lo; ++offset) {
    int cpu = lo + ((preferred - lo + offset) % (hi - lo));
    if (CPU_ISSET(cpu, &allowed)) {
      cpu_set_t one;
      CPU_ZERO(&one);
      CPU_SET(cpu, &one);
      sched_setaffinity(0, sizeof(one), &one);
      return;
    }
  }
}

void runRank(int rank, mscclpp::UniqueId id, size_t minBytes, size_t maxBytes,
             int warmups, int iterations) {
  pinRankToLocalCpu(rank);
  CUDA_CHECK(cudaSetDevice(rank));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, kRanks);
  bootstrap->initialize(id);
  std::array<ncclUniqueId, kRanks> ids{};
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&ids[0]));
  bootstrap->allGather(ids.data(), sizeof(ncclUniqueId));
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, kRanks, ids[0], rank));
  float* buffer;
  CUDA_CHECK(cudaMalloc(&buffer, maxBytes));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::vector<float> input, output;
  auto fill = [&](size_t count) {
    input.resize(count);
    for (size_t j = 0; j < count; ++j)
      input[j] =
          static_cast<float>(rank + 1) * static_cast<float>(1 + (j % 97));
    CUDA_CHECK(cudaMemcpyAsync(buffer, input.data(), count * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  auto launch = [&](size_t count) {
    NCCL_CHECK(
        ncclAllReduce(buffer, buffer, count, ncclFloat, ncclSum, comm, stream));
    CUDA_CHECK(cudaGetLastError());
  };
  if (rank == 0) {
    int version;
    NCCL_CHECK(ncclGetVersion(&version));
    std::printf("# NCCL %d FP32 in-place SUM, eight ranks\n", version);
    std::printf(
        "# warmups=%d iterations=%d C++ batch host timer, slowest rank\n",
        warmups, iterations);
    std::printf("# bytes count mode us algbw busbw maxrelerr\n");
  }
  for (size_t bytes = minBytes; bytes <= maxBytes; bytes *= 2) {
    size_t count = bytes / sizeof(float);
    fill(count);
    bootstrap->barrier();
    launch(count);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    output.resize(count);
    CUDA_CHECK(
        cudaMemcpy(output.data(), buffer, bytes, cudaMemcpyDeviceToHost));
    double error = 0;
    for (size_t j = 0; j < count; ++j) {
      double expected = 36.0 * static_cast<double>(1 + (j % 97));
      double relative =
          std::fabs(static_cast<double>(output[j]) - expected) / expected;
      if (!std::isfinite(relative)) relative = INFINITY;
      error = std::max(error, relative);
    }
    bootstrap->barrier();
    fill(count);
    for (int i = 0; i < warmups; ++i) launch(count);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bootstrap->barrier();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) launch(count);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();
    double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::array<double, kRanks> times{}, errors{};
    times[rank] = us;
    errors[rank] = error;
    bootstrap->allGather(times.data(), sizeof(double));
    bootstrap->allGather(errors.data(), sizeof(double));
    double worst = *std::max_element(times.begin(), times.end());
    double worstError = *std::max_element(errors.begin(), errors.end());
    if (rank == 0) {
      double algbw = static_cast<double>(bytes) / (worst * 1000);
      std::printf("%zu %zu sm %.6f %.6f %.6f %.1e\n", bytes, count, worst,
                  algbw, algbw * 1.75, worstError);
      std::fflush(stdout);
    }
    if (worstError > 1e-5) std::exit(2);
    bootstrap->barrier();
    if (bytes > maxBytes / 2) break;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  bootstrap->barrier();
  NCCL_CHECK(ncclCommDestroy(comm));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(buffer));
}

int main(int argc, char** argv) {
  size_t minBytes = 1024, maxBytes = 1048576;
  int warmups = 10, iterations = 20;
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    if (++i == argc) return 2;
    if (option == "-b")
      minBytes = std::strtoull(argv[i], nullptr, 0);
    else if (option == "-e")
      maxBytes = std::strtoull(argv[i], nullptr, 0);
    else if (option == "-w")
      warmups = std::atoi(argv[i]);
    else if (option == "-n")
      iterations = std::atoi(argv[i]);
    else
      return 2;
  }
  if (!minBytes || maxBytes < minBytes || minBytes % 4 || maxBytes % 4 ||
      warmups < 0 || iterations < 1)
    return 2;
  auto id = mscclpp::TcpBootstrap::createUniqueId();
  std::vector<pid_t> children;
  for (int rank = 1; rank < kRanks; ++rank) {
    pid_t pid = fork();
    if (pid == 0) {
      runRank(rank, id, minBytes, maxBytes, warmups, iterations);
      _exit(0);
    }
    if (pid < 0) return 1;
    children.push_back(pid);
  }
  runRank(0, id, minBytes, maxBytes, warmups, iterations);
  int result = 0;
  for (pid_t pid : children) {
    int status;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) result = 1;
  }
  return result;
}
