// Raw sender-initiated CE routing diagnostic, not a collective admission gate.
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <unistd.h>

namespace uccl_lite {
void checked(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
struct Ring {
  int rank[4];
  char const* name;
};
struct Device {
  int rank = -1;
  float *input = nullptr, *output = nullptr;
  unsigned* wrong = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, end = nullptr;
  ~Device() {
    if (rank < 0) return;
    cudaSetDevice(rank);
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
    if (stream) cudaStreamDestroy(stream);
    cudaFree(input);
    cudaFree(output);
    cudaFree(wrong);
  }
};
__global__ void fillInput(float* input, size_t count, int rank,
                          int generation) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    input[i] = float(rank + 1 + generation * 4) * float(1 + i % 97);
}
__global__ void checkOutput(float const* output, size_t shard, int source,
                            int prevPos, Ring ring, int generation,
                            unsigned* wrong) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < 4 * shard;
       i += size_t(gridDim.x) * blockDim.x) {
    bool mismatch;
    if (i >= 3 * shard)
      mismatch = __float_as_uint(output[i]) != 0xffffffffU;
    else {
      int step = i / shard;
      size_t offset = i % shard;
      int owner = ring.rank[(prevPos - step + 3) % 4];
      float expected = float(source + 1 + generation * 4) *
                       float(1 + (size_t(owner) * shard + offset) % 97);
      mismatch = output[i] != expected;
    }
    if (mismatch) atomicAdd(wrong, 1);
  }
}
void synchronize(std::array<Device, 4>& devices) {
  for (auto& d : devices) {
    checked(cudaSetDevice(d.rank));
    checked(cudaStreamSynchronize(d.stream));
  }
}
}  // namespace uccl_lite

int main() {
  using namespace uccl_lite;
  std::array<Device, 4> devices;
  try {
    int count;
    checked(cudaGetDeviceCount(&count));
    if (count != 4) throw std::runtime_error("requires four GPUs");
    constexpr size_t bytes = 1UL << 30, shard = bytes / 16;
    constexpr size_t chunk = (8UL << 20) / 4;
    constexpr int warmups = 5, iterations = 10, rounds = 3;
    Ring rings[] = {{{0, 1, 3, 2}, "gray-forward"},
                    {{0, 2, 3, 1}, "gray-reverse"},
                    {{0, 1, 2, 3}, "linear-forward"},
                    {{0, 3, 2, 1}, "linear-reverse"}};
    char host[256]{};
    gethostname(host, sizeof(host) - 1);
    std::printf(
        "# UCCL Lite raw CE topology diagnostic: host=%s dtype=FP32 "
        "allocation=1GiB/GPU chunk=8MiB copies=3/4GiB/GPU/call "
        "warmups=5 iterations=10 rounds=3 scope=max-GPU events; "
        "preparation/checks excluded; no reduction or IPC handoff; "
        "fixed routes, no per-size variant selection\n",
        host);
    for (int r = 0; r < 4; ++r) {
      checked(cudaSetDevice(r));
      cudaDeviceProp property{};
      checked(cudaGetDeviceProperties(&property, r));
      if (std::string(property.name) != "NVIDIA L4")
        throw std::runtime_error("requires L4 testbed");
      std::printf("# gpu=%d name=%s pci=%04x:%02x:%02x.0\n", r, property.name,
                  property.pciDomainID, property.pciBusID,
                  property.pciDeviceID);
      for (int p = 0; p < 4; ++p) {
        if (p == r) continue;
        auto status = cudaDeviceEnablePeerAccess(p, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled)
          cudaGetLastError();
        else
          checked(status);
      }
      auto& d = devices[r];
      d.rank = r;
      checked(cudaMalloc(&d.input, bytes));
      checked(cudaMalloc(&d.output, bytes));
      checked(cudaMalloc(&d.wrong, sizeof(unsigned)));
      checked(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
      checked(cudaEventCreate(&d.start));
      checked(cudaEventCreate(&d.end));
    }
    for (int round = 0; round < rounds; ++round) {
      for (int order = 0; order < 4; ++order) {
        int which = round == 1 ? 3 - order : (order + (round == 2 ? 1 : 0)) % 4;
        Ring ring = rings[which];
        double total = 0;
        for (int iteration = 0; iteration < warmups + iterations; ++iteration) {
          int generation = 10 + round * (warmups + iterations) + iteration;
          for (auto& d : devices) {
            checked(cudaSetDevice(d.rank));
            fillInput<<<256, 256, 0, d.stream>>>(d.input, bytes / 4, d.rank,
                                                 generation);
            checked(cudaMemsetAsync(d.output, 0xff, bytes, d.stream));
            checked(cudaGetLastError());
          }
          synchronize(devices);
          for (auto& d : devices) {
            checked(cudaSetDevice(d.rank));
            checked(cudaEventRecord(d.start, d.stream));
          }
          for (int step = 0; step < 3; ++step)
            for (size_t offset = 0; offset < shard; offset += chunk)
              for (int pos = 0; pos < 4; ++pos) {
                int r = ring.rank[pos], next = ring.rank[(pos + 1) % 4];
                int owner = ring.rank[(pos - step + 3) % 4];
                checked(cudaSetDevice(r));
                checked(cudaMemcpyPeerAsync(
                    devices[next].output + step * shard + offset, next,
                    devices[r].input + owner * shard + offset, r, chunk * 4,
                    devices[r].stream));
              }
          for (auto& d : devices) {
            checked(cudaSetDevice(d.rank));
            checked(cudaEventRecord(d.end, d.stream));
          }
          synchronize(devices);
          double maximum = 0;
          for (int pos = 0; pos < 4; ++pos) {
            int r = ring.rank[pos], prevPos = (pos + 3) % 4;
            auto& d = devices[r];
            checked(cudaSetDevice(r));
            float elapsed;
            checked(cudaEventElapsedTime(&elapsed, d.start, d.end));
            maximum = std::max(maximum, double(elapsed) * 1000);
            checked(cudaMemsetAsync(d.wrong, 0, sizeof(unsigned), d.stream));
            checkOutput<<<256, 256, 0, d.stream>>>(d.output, shard,
                                                   ring.rank[prevPos], prevPos,
                                                   ring, generation, d.wrong);
            checked(cudaGetLastError());
            unsigned wrong;
            checked(cudaMemcpyAsync(&wrong, d.wrong, sizeof(wrong),
                                    cudaMemcpyDeviceToHost, d.stream));
            checked(cudaStreamSynchronize(d.stream));
            if (wrong)
              throw std::runtime_error("full output/guard check failed");
          }
          if (iteration >= warmups) {
            total += maximum;
            std::printf("sample,%s,%d,%d,%.6f\n", ring.name, round,
                        iteration - warmups, maximum);
          }
        }
        std::printf("mean,%s,%d,%.6f\n", ring.name, round, total / iterations);
        std::fflush(stdout);
      }
    }
    std::puts("Full output and untouched-quarter checks passed for all calls.");
  } catch (std::exception const& error) {
    for (auto& d : devices)
      if (d.rank >= 0) {
        cudaSetDevice(d.rank);
        cudaDeviceSynchronize();
      }
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
