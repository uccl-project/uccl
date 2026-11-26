// Compare BF16 CPU reduction after CPU cache warming, LLC eviction, and
// GPU writes into the same NUMA-local registered host buffers.
#include "cpu_executor.hpp"
#include "memory.hpp"
#include <cuda_runtime.h>
#include <immintrin.h>
#include <sched.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using uccl_lite::cpu_router::CpuExecutor;
using uccl_lite::cpu_router::MemorySpan;
using uccl_lite::cpu_router::RegisteredMemory;

namespace {
constexpr int kSources = 4;
constexpr size_t kMaxBytes = 8UL << 20;
volatile uint64_t touched = 0;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void pinCpu(int cpu) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask))
    throw std::runtime_error("CPU pinning failed");
}
void flush(void const* ptr, size_t bytes) {
  auto* p = static_cast<char const*>(ptr);
  for (size_t offset = 0; offset < bytes; offset += 64)
    _mm_clflush(p + offset);
  _mm_mfence();
}
void warm(void const* ptr, size_t bytes) {
  auto* p = static_cast<uint64_t const volatile*>(ptr);
  uint64_t sum = 0;
  for (size_t offset = 0; offset < bytes / sizeof(uint64_t); offset += 8)
    sum += p[offset];
  touched += sum;
}
uint16_t bf16(int rank) {
  static constexpr uint16_t values[] = {0x3f80, 0x4000, 0x4040, 0x4080};
  return values[rank];
}
__global__ void fill16(uint4* dest, size_t words, uint32_t pair) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += size_t(gridDim.x) * blockDim.x)
    dest[i] = make_uint4(pair, pair, pair, pair);
  __threadfence_system();
}
struct Gpu {
  int device;
  void* source = nullptr;
  void* mapped = nullptr;
  cudaStream_t stream = nullptr;
};
enum class Mode {
  Hot, Cold, SmWrite, SmWriteFlushed, CopyEngine, CopyEngineFlushed
};
char const* name(Mode mode) {
  switch (mode) {
    case Mode::Hot: return "cpu_hot";
    case Mode::Cold: return "cpu_flushed";
    case Mode::SmWrite: return "gpu_sm_write";
    case Mode::SmWriteFlushed: return "gpu_sm_write_flushed";
    case Mode::CopyEngine: return "gpu_copy_engine";
    case Mode::CopyEngineFlushed: return "gpu_copy_engine_flushed";
  }
  return "unknown";
}
double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int trials = 21, warmups = 5, node = 0, cpu = 0;
    for (int i = 1; i < argc; ++i) {
      if (i + 1 == argc) throw std::invalid_argument("missing option value");
      std::string option = argv[i], value = argv[++i];
      if (option == "--trials") trials = std::stoi(value);
      else if (option == "--warmups") warmups = std::stoi(value);
      else if (option == "--node") node = std::stoi(value);
      else if (option == "--cpu") cpu = std::stoi(value);
      else throw std::invalid_argument("unknown option: " + option);
    }
    if (trials < 3 || warmups < 0 || node < 0 || cpu < 0)
      throw std::invalid_argument("invalid probe options");
    pinCpu(cpu);
    std::array<RegisteredMemory, kSources> host;
    for (auto& buffer : host) buffer = RegisteredMemory::pinned(kMaxBytes, node);
    for (int rank = 0; rank < kSources; ++rank)
      std::fill_n(static_cast<uint16_t*>(host[rank].data()), kMaxBytes / 2,
                  bf16(rank));
    RegisteredMemory result;
    result.allocate(kMaxBytes, node, false);
    std::array<Gpu, kSources> gpu;
    for (int rank = 0; rank < kSources; ++rank) {
      auto& g = gpu[rank];
      g.device = rank + node * kSources;
      check(cudaSetDevice(g.device));
      check(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
      check(cudaMalloc(&g.source, kMaxBytes));
      g.mapped = host[rank].span().gpuAddress(g.device);
      uint32_t pattern = bf16(rank) | (uint32_t(bf16(rank)) << 16);
      fill16<<<256, 256, 0, g.stream>>>(static_cast<uint4*>(g.source),
                                        kMaxBytes / 16, pattern);
      check(cudaGetLastError());
      check(cudaStreamSynchronize(g.stream));
    }
    std::cout << "# node=" << node << " cpu=" << cpu
              << " gpu_sources=" << node * 4 << '-' << node * 4 + 3
              << " trials=" << trials << " warmups=" << warmups
              << " target=CpuExecutor::sumBf16\n";
    std::cout << "bytes_per_source,condition,median_us,min_us,max_us\n";
    for (size_t bytes : {size_t(16) << 10, size_t(64) << 10,
                         size_t(256) << 10, size_t(512) << 10,
                         size_t(2) << 20, size_t(8) << 20}) {
      std::array<MemorySpan, kSources> sourceSpans;
      for (int rank = 0; rank < kSources; ++rank)
        sourceSpans[rank] = host[rank].span().subspan(0, bytes);
      auto output = result.span().subspan(0, bytes);
      std::array<std::vector<double>, 6> timing;
      for (int repetition = 0; repetition < trials + warmups; ++repetition) {
        for (int rotation = 0; rotation < 6; ++rotation) {
          auto mode = Mode((repetition + rotation) % 6);
          flush(result.data(), bytes);
          if (mode == Mode::Hot) {
            for (auto const& buffer : host) warm(buffer.data(), bytes);
          } else {
            for (auto const& buffer : host) flush(buffer.data(), bytes);
            if (mode != Mode::Cold) {
              for (int rank = 0; rank < kSources; ++rank) {
                auto& g = gpu[rank];
                check(cudaSetDevice(g.device));
                if (mode == Mode::SmWrite || mode == Mode::SmWriteFlushed) {
                  uint32_t pattern = bf16(rank) |
                                     (uint32_t(bf16(rank)) << 16);
                  fill16<<<256, 256, 0, g.stream>>>(
                      static_cast<uint4*>(g.mapped), bytes / 16, pattern);
                  check(cudaGetLastError());
                } else {
                  check(cudaMemcpyAsync(host[rank].data(), g.source, bytes,
                                        cudaMemcpyDeviceToHost, g.stream));
                }
              }
              for (auto& g : gpu) {
                check(cudaSetDevice(g.device));
                check(cudaStreamSynchronize(g.stream));
              }
              if (mode == Mode::SmWriteFlushed ||
                  mode == Mode::CopyEngineFlushed)
                for (auto const& buffer : host) flush(buffer.data(), bytes);
            }
          }
          auto start = std::chrono::steady_clock::now();
          CpuExecutor::sumBf16(sourceSpans.data(), kSources, output);
          auto end = std::chrono::steady_clock::now();
          auto* values = static_cast<uint16_t const*>(result.data());
          for (size_t i = 0; i < bytes / 2; ++i)
            if (values[i] != 0x4120)
              throw std::runtime_error("BF16 reduction output mismatch");
          if (repetition >= warmups)
            timing[int(mode)].push_back(
                std::chrono::duration<double, std::micro>(end - start).count());
        }
      }
      for (int index = 0; index < 6; ++index) {
        auto& samples = timing[index];
        std::cout << bytes << ',' << name(Mode(index)) << ','
                  << std::fixed << std::setprecision(3) << median(samples)
                  << ',' << *std::min_element(samples.begin(), samples.end())
                  << ',' << *std::max_element(samples.begin(), samples.end())
                  << '\n';
      }
      std::cout.flush();
    }
    for (auto& g : gpu) {
      check(cudaSetDevice(g.device));
      check(cudaFree(g.source));
      check(cudaStreamDestroy(g.stream));
    }
    std::cout << "# correctness: PASS\n";
  } catch (std::exception const& error) {
    std::cerr << "probe error: " << error.what() << '\n';
    return 1;
  }
}
