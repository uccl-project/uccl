// Four-GPU direct peer AllGather and ReduceScatter benchmark.
// Every destination owns its output; source tensors stay immutable until all
// peer copies and the destination reduction have completed.
#include <cuda_runtime.h>
#include <nccl.h>
#include <mscclpp/packet_device.hpp>
#ifdef UCCL_LITE_PACKET_ROUTER
#include "uccl_lite/cpu_router/packet_pipeline.hpp"
#endif
#include <linux/mempolicy.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
constexpr int kRanks = 4;
constexpr size_t kCompactGatherBytes = 256UL << 10;
constexpr size_t kCompactScatterBytes = 64UL << 10;
bool directSelfReduce(size_t fullBytes) {
  return fullBytes == (256UL << 10) ||
         (fullBytes >= (1UL << 20) && fullBytes <= (8UL << 20)) ||
         fullBytes >= (32UL << 20);
}
void cudaCheck(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void ncclCheck(ncclResult_t status) {
  if (status != ncclSuccess) throw std::runtime_error(ncclGetErrorString(status));
}
struct Config {
  size_t begin = 1UL << 10, end = 1UL << 26;
  size_t pullMaxBytes = 32UL << 10;
  size_t pullVecMaxBytes = 0;
  size_t pushSerialMaxBytes = 2UL << 20;
  size_t rsPushSerialMaxBytes = 8UL << 20;
  size_t rsLlMaxBytes = 8UL << 10;
  size_t rsFenceMaxBytes = 1UL << 20;
  size_t routerChunkBytes = 4UL << 20;
  size_t packetRouterMinBytes = 4UL << 20;
  int warmups = 5, iterations = 10, repeats = 3;
  bool routerAg = false;
  bool packetRouterAg = false;
  bool packetRouterGpuProgress = false;
  bool packetRouterPerSource = false;
  bool rsOnly = false;
  bool routerLocalNuma = true;
};
Config parse(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    if (option == "--router-ag") {
      cfg.routerAg = true;
      continue;
    }
    if (option == "--packet-router-ag") {
      cfg.packetRouterAg = true;
      continue;
    }
    if (option == "--packet-router-gpu-progress") {
      cfg.packetRouterGpuProgress = true;
      continue;
    }
    if (option == "--packet-router-per-source") {
      cfg.packetRouterGpuProgress = true;
      cfg.packetRouterPerSource = true;
      continue;
    }
    if (option == "--rs-only") {
      cfg.rsOnly = true;
      continue;
    }
    if (option == "--help") {
      std::puts("Usage: uccl_lite_peer_ag_rs [-b BYTES] [-e BYTES] "
                "[-w WARMUPS] [-n ITERATIONS] [-r REPEATS] "
                "[--pull-max-bytes BYTES] [--pull-vec-max-bytes BYTES] "
                "[--push-serial-max-bytes BYTES] "
                "[--rs-push-serial-max-bytes BYTES] "
                "[--rs-ll-max-bytes BYTES] "
                "[--rs-fence-max-bytes BYTES] "
                "[--router-ag] [--packet-router-ag] [--rs-only] "
                "[--packet-router-min-bytes BYTES] "
                "[--packet-router-gpu-progress|--packet-router-per-source] "
                "[--router-chunk-bytes BYTES] "
                "[--router-host-placement local|cuda]\n"
                "BYTES is full AllGather output / ReduceScatter input per "
                "rank; requires four peer-accessible GPUs.");
      std::exit(0);
    }
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    char* value = argv[++i];
    if (option == "-b") cfg.begin = std::strtoull(value, nullptr, 0);
    else if (option == "-e") cfg.end = std::strtoull(value, nullptr, 0);
    else if (option == "-w") cfg.warmups = std::atoi(value);
    else if (option == "-n") cfg.iterations = std::atoi(value);
    else if (option == "-r") cfg.repeats = std::atoi(value);
    else if (option == "--pull-max-bytes")
      cfg.pullMaxBytes = std::strtoull(value, nullptr, 0);
    else if (option == "--pull-vec-max-bytes")
      cfg.pullVecMaxBytes = std::strtoull(value, nullptr, 0);
    else if (option == "--push-serial-max-bytes")
      cfg.pushSerialMaxBytes = std::strtoull(value, nullptr, 0);
    else if (option == "--rs-push-serial-max-bytes")
      cfg.rsPushSerialMaxBytes = std::strtoull(value, nullptr, 0);
    else if (option == "--rs-ll-max-bytes")
      cfg.rsLlMaxBytes = std::strtoull(value, nullptr, 0);
    else if (option == "--rs-fence-max-bytes")
      cfg.rsFenceMaxBytes = std::strtoull(value, nullptr, 0);
    else if (option == "--router-chunk-bytes")
      cfg.routerChunkBytes = std::strtoull(value, nullptr, 0);
    else if (option == "--packet-router-min-bytes")
      cfg.packetRouterMinBytes = std::strtoull(value, nullptr, 0);
    else if (option == "--router-host-placement") {
      std::string placement = value;
      if (placement == "local") cfg.routerLocalNuma = true;
      else if (placement == "cuda") cfg.routerLocalNuma = false;
      else throw std::invalid_argument("invalid router host placement");
    }
    else throw std::invalid_argument("unknown option: " + option);
  }
  if (cfg.begin < 16 || cfg.begin > cfg.end || cfg.begin % 16 ||
      cfg.end % 16 || cfg.routerChunkBytes < 16 ||
      cfg.routerChunkBytes % 16 || cfg.packetRouterMinBytes % 16 ||
      cfg.warmups < 0 || cfg.iterations < 1 ||
      cfg.repeats < 1)
    throw std::invalid_argument("invalid size or iteration settings");
  if (cfg.routerAg && cfg.rsOnly)
    throw std::invalid_argument("--router-ag and --rs-only conflict");
  if (cfg.packetRouterAg && (cfg.routerAg || cfg.rsOnly))
    throw std::invalid_argument("--packet-router-ag requires AllGather only");
  if (cfg.packetRouterGpuProgress && !cfg.packetRouterAg)
    throw std::invalid_argument("packet-router progress requires --packet-router-ag");
#ifndef UCCL_LITE_PACKET_ROUTER
  if (cfg.packetRouterAg)
    throw std::invalid_argument("build peer-ag-rs-packet for --packet-router-ag");
#endif
  return cfg;
}
struct Device {
  int rank;
  float *input = nullptr, *output = nullptr, *scratch = nullptr,
        *reference = nullptr;
  mscclpp::LL16Packet* packets = nullptr;
  uint32_t* flags = nullptr;
  unsigned int* wrong = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, end = nullptr, transferDone = nullptr,
              reduceDone = nullptr, transferAllDone = nullptr;
  Device(int r, size_t bytes, size_t llBytes, bool needFlags) : rank(r) {
    cudaCheck(cudaSetDevice(rank));
    cudaCheck(cudaMalloc(&input, bytes));
    cudaCheck(cudaMalloc(&output, bytes));
    cudaCheck(cudaMalloc(&scratch, bytes));
    if (llBytes) {
      cudaCheck(cudaMalloc(&packets, 2 * llBytes));
      cudaCheck(cudaMemset(packets, 0, 2 * llBytes));
    }
    if (needFlags) {
      cudaCheck(cudaMalloc(&flags, kRanks * sizeof(uint32_t)));
      cudaCheck(cudaMemset(flags, 0, kRanks * sizeof(uint32_t)));
    }
    cudaCheck(cudaMalloc(&reference, bytes));
    cudaCheck(cudaMalloc(&wrong, sizeof(unsigned int)));
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventCreateWithFlags(&transferDone, cudaEventDisableTiming));
    cudaCheck(cudaEventCreateWithFlags(&transferAllDone, cudaEventDisableTiming));
    cudaCheck(cudaEventCreateWithFlags(&reduceDone, cudaEventDisableTiming));
  }
  ~Device() {
    cudaSetDevice(rank);
    if (transferAllDone) cudaEventDestroy(transferAllDone);
    if (reduceDone) cudaEventDestroy(reduceDone);
    if (transferDone) cudaEventDestroy(transferDone);
    if (end) cudaEventDestroy(end);
    if (start) cudaEventDestroy(start);
    if (stream) cudaStreamDestroy(stream);
    if (reference) cudaFree(reference);
    if (wrong) cudaFree(wrong);
    if (scratch) cudaFree(scratch);
    if (packets) cudaFree(packets);
    if (flags) cudaFree(flags);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
  }
};
__global__ void sumFour(float const* source, float const* localInput,
                        float* output, size_t shard, int destination,
                        bool directSelf) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= shard) return;
  float value = 0;
  for (int rank = 0; rank < kRanks; ++rank)
    value += directSelf && rank == destination
                 ? localInput[size_t(destination) * shard + index]
                 : source[size_t(rank) * shard + index];
  output[index] = value;
}
__global__ void fillInput(float* input, size_t elements, int rank,
                          int generation) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  float rankValue = float(rank + 1 + generation * kRanks);
  for (; index < elements; index += stride)
    input[index] = rankValue * float(1 + index % 97);
}
__global__ void checkOutput(float const* output, unsigned int* wrong,
                            size_t elements, size_t shard, int rank,
                            int generation, bool gather) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; index < elements; index += stride) {
    float want = gather
                     ? float(index / shard + 1 + generation * kRanks) *
                           float(1 + (index % shard) % 97)
                     : float(10 + generation * kRanks * kRanks) *
                           float(1 + (size_t(rank) * shard + index) % 97);
    if (!(fabsf(output[index] - want) <= 1e-5f)) atomicAdd(wrong, 1);
  }
}
__global__ void pullGather(float const* in0, float const* in1,
                           float const* in2, float const* in3, float* output,
                           size_t shard) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= shard) return;
  int source = blockIdx.y;
  float const* input = source == 0 ? in0 : source == 1 ? in1 : source == 2 ? in2 : in3;
  output[size_t(source) * shard + index] = input[index];
}
__global__ void pullVecGather(float const* in0, float const* in1,
                              float const* in2, float const* in3,
                              float* output, size_t shard) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int source = blockIdx.y;
  float const* input = source == 0 ? in0 : source == 1 ? in1 : source == 2 ? in2 : in3;
  size_t vectors = shard / 4;
  if (index < vectors) {
    auto* out = reinterpret_cast<uint4*>(output + size_t(source) * shard);
    auto const* in = reinterpret_cast<uint4 const*>(input);
    out[index] = in[index];
  }
  if (index == 0)
    for (size_t tail = vectors * 4; tail < shard; ++tail)
      output[size_t(source) * shard + tail] = input[tail];
}
__global__ void pushSerialGather(float const* input, float* out0,
                                 float* out1, float* out2, float* out3,
                                 size_t shard, int source) {
  float* targets[4] = {out0, out1, out2, out3};
  for (int phase = 0; phase < kRanks; ++phase) {
    int dest = (source + phase) % kRanks;
    float* output = targets[dest] + size_t(source) * shard;
    if (shard % 4 == 0) {
      auto const* in = reinterpret_cast<uint4 const*>(input);
      auto* out = reinterpret_cast<uint4*>(output);
      for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x)
        out[i] = in[i];
    } else
      for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
        output[i] = input[i];
    __syncthreads();
  }
}
__global__ void pushSerialScatter(float const* input, float* scratch0,
                                  float* scratch1, float* scratch2,
                                  float* scratch3, size_t shard, int source,
                                  bool directSelf) {
  float* targets[4] = {scratch0, scratch1, scratch2, scratch3};
  for (int phase = 0; phase < kRanks; ++phase) {
    int dest = (source + phase) % kRanks;
    if (!(directSelf && dest == source)) {
      float const* from = input + size_t(dest) * shard;
      float* to = targets[dest] + size_t(source) * shard;
      if (shard % 4 == 0) {
        auto const* in = reinterpret_cast<uint4 const*>(from);
        auto* out = reinterpret_cast<uint4*>(to);
        for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x)
          out[i] = in[i];
      } else
        for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
          to[i] = from[i];
    }
    __syncthreads();
  }
}
__global__ void scatterPacketLL(float const* input, float* output,
                                mscclpp::LL16Packet* packets0,
                                mscclpp::LL16Packet* packets1,
                                mscclpp::LL16Packet* packets2,
                                mscclpp::LL16Packet* packets3,
                                size_t shard, int rank, uint32_t flag) {
  mscclpp::LL16Packet* packets[4] = {packets0, packets1, packets2,
                                     packets3};
  size_t units = shard / 2;
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    for (int phase = 1; phase < kRanks; ++phase) {
      int dest = (rank + phase) % kRanks;
      float2 value = reinterpret_cast<float2 const*>(
          input + size_t(dest) * shard)[i];
      packets[dest][size_t(rank) * units + i].write(
          make_uint2(__float_as_uint(value.x), __float_as_uint(value.y)), flag);
    }
    float2 sum = reinterpret_cast<float2 const*>(
        input + size_t(rank) * shard)[i];
    for (int source = 0; source < kRanks; ++source) {
      if (source == rank) continue;
      uint2 value = packets[rank][size_t(source) * units + i].read(flag);
      sum.x += __uint_as_float(value.x);
      sum.y += __uint_as_float(value.y);
    }
    reinterpret_cast<float2*>(output)[i] = sum;
  }
}
__device__ void releaseSystemFlag(uint32_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::
                   "l"(address), "r"(value) : "memory");
}
__device__ uint32_t acquireSystemFlag(uint32_t const* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" :
                   "=r"(value) : "l"(address) : "memory");
  return value;
}
__global__ void scatterFenceFused(float const* input, float* output,
                                  float* scratch0, float* scratch1,
                                  float* scratch2, float* scratch3,
                                  uint32_t* flags0, uint32_t* flags1,
                                  uint32_t* flags2, uint32_t* flags3,
                                  size_t shard, int rank, uint32_t flag) {
  float* scratches[4] = {scratch0, scratch1, scratch2, scratch3};
  uint32_t* flags[4] = {flags0, flags1, flags2, flags3};
  for (int phase = 1; phase < kRanks; ++phase) {
    int dest = (rank + phase) % kRanks;
    float const* from = input + size_t(dest) * shard;
    float* to = scratches[dest] + size_t(rank) * shard;
    if (shard % 4 == 0) {
      auto const* in = reinterpret_cast<uint4 const*>(from);
      auto* out = reinterpret_cast<uint4*>(to);
      for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x)
        out[i] = in[i];
    } else
      for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
        to[i] = from[i];
    __syncthreads();
  }
  // Every thread publishes its own peer stores before one leader signals.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int dest = 0; dest < kRanks; ++dest)
      if (dest != rank) releaseSystemFlag(flags[dest] + rank, flag);
    for (int source = 0; source < kRanks; ++source) {
      if (source == rank) continue;
      while (acquireSystemFlag(flags[rank] + source) != flag)
        __nanosleep(64);
    }
  }
  __syncthreads();
  if (shard % 4 == 0) {
    auto const* own = reinterpret_cast<float4 const*>(
        input + size_t(rank) * shard);
    auto const* peer = reinterpret_cast<float4 const*>(scratches[rank]);
    auto* out = reinterpret_cast<float4*>(output);
    for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x) {
      float4 value = own[i];
      for (int source = 0; source < kRanks; ++source) {
        if (source == rank) continue;
        float4 remote = peer[size_t(source) * (shard / 4) + i];
        value.x += remote.x;
        value.y += remote.y;
        value.z += remote.z;
        value.w += remote.w;
      }
      out[i] = value;
    }
  } else
    for (size_t i = threadIdx.x; i < shard; i += blockDim.x) {
      float value = input[size_t(rank) * shard + i];
      for (int source = 0; source < kRanks; ++source)
        if (source != rank)
          value += scratches[rank][size_t(source) * shard + i];
      output[i] = value;
    }
}
__global__ void pullScatter(float const* in0, float const* in1,
                            float const* in2, float const* in3, float* output,
                            size_t shard, int destination) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= shard) return;
  size_t offset = size_t(destination) * shard + index;
  float value = in0[offset];
  value += in1[offset];
  value += in2[offset];
  value += in3[offset];
  output[index] = value;
}
enum class Operation { Gather, Scatter };
enum class Backend { Peer, Pull, Nccl, Router, PacketRouter, PullVec, PushSerial,
                     ScatterLL, ScatterFence };
char const* opName(Operation op) {
  return op == Operation::Gather ? "all_gather" : "reduce_scatter";
}
char const* backendName(Backend backend) {
  if (backend == Backend::Peer) return "peer";
  if (backend == Backend::Pull) return "pull";
  if (backend == Backend::PullVec) return "pull-vec";
  if (backend == Backend::PushSerial) return "push-serial";
  if (backend == Backend::PacketRouter) return "packet-router";
  if (backend == Backend::ScatterLL) return "scatter-ll";
  if (backend == Backend::ScatterFence) return "scatter-fence";
  return backend == Backend::Router ? "cpu-router" : "nccl";
}
struct HostRouter {
  size_t chunkBytes;
  bool localNuma;
  std::array<std::array<char*, 2>, kRanks> stage{};
  std::array<int, kRanks> hostNode{};
  std::array<cudaStream_t, kRanks> uploads{};
  std::array<std::array<cudaEvent_t, 2>, kRanks> ready{};
  std::array<std::array<cudaEvent_t, 2>, kRanks> consumed{};
  static int gpuNumaNode(int rank) {
    cudaDeviceProp prop{};
    cudaCheck(cudaGetDeviceProperties(&prop, rank));
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node",
                  prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    FILE* file = std::fopen(path, "r");
    if (!file) throw std::runtime_error("GPU PCI NUMA node unavailable");
    int node = -1;
    int parsed = std::fscanf(file, "%d", &node);
    std::fclose(file);
    if (parsed != 1 || node < 0 || node >= 8 * int(sizeof(unsigned long)))
      throw std::runtime_error("invalid GPU PCI NUMA node");
    return node;
  }
  explicit HostRouter(size_t bytes, bool placeLocal)
      : chunkBytes(bytes), localNuma(placeLocal) {
    for (int rank = 0; rank < kRanks; ++rank) {
      cudaCheck(cudaSetDevice(rank));
      hostNode[rank] = placeLocal ? gpuNumaNode(rank) : -1;
      cudaCheck(cudaStreamCreateWithFlags(&uploads[rank], cudaStreamNonBlocking));
      for (int slot = 0; slot < 2; ++slot) {
        if (placeLocal) {
          void* address = mmap(nullptr, chunkBytes, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
          if (address == MAP_FAILED)
            throw std::runtime_error("router host mmap failed");
          unsigned long mask = 1UL << hostNode[rank];
          if (syscall(SYS_mbind, address, chunkBytes, MPOL_BIND, &mask,
                      8 * sizeof(mask), 0)) {
            munmap(address, chunkBytes);
            throw std::runtime_error("router host mbind failed");
          }
          std::memset(address, 0, chunkBytes);
          auto status = cudaHostRegister(address, chunkBytes,
                                         cudaHostRegisterPortable);
          if (status != cudaSuccess) {
            munmap(address, chunkBytes);
            cudaCheck(status);
          }
          stage[rank][slot] = static_cast<char*>(address);
        } else
          cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&stage[rank][slot]),
                                  chunkBytes, cudaHostAllocPortable));
        cudaCheck(cudaEventCreateWithFlags(&ready[rank][slot],
                                           cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&consumed[rank][slot],
                                           cudaEventDisableTiming));
      }
    }
  }
  ~HostRouter() {
    for (int rank = 0; rank < kRanks; ++rank) {
      cudaSetDevice(rank);
      for (int slot = 0; slot < 2; ++slot) {
        if (ready[rank][slot]) cudaEventDestroy(ready[rank][slot]);
        if (consumed[rank][slot]) cudaEventDestroy(consumed[rank][slot]);
        if (stage[rank][slot]) {
          if (localNuma) {
            cudaHostUnregister(stage[rank][slot]);
            munmap(stage[rank][slot], chunkBytes);
          } else
            cudaFreeHost(stage[rank][slot]);
        }
      }
      if (uploads[rank]) cudaStreamDestroy(uploads[rank]);
    }
  }
  void launch(std::array<Device*, kRanks> const& devices, size_t shard) {
    size_t shardBytes = shard * sizeof(float);
    for (size_t offset = 0, piece = 0; offset < shardBytes;
         offset += chunkBytes, ++piece) {
      int slot = int(piece & 1);
      size_t count = std::min(chunkBytes, shardBytes - offset);
      for (int source = 0; source < kRanks; ++source) {
        cudaCheck(cudaSetDevice(source));
        // A slot cannot be overwritten until all four destination DMA reads
        // from its previous chunk have completed.
        if (piece >= 2)
          for (int dest = 0; dest < kRanks; ++dest)
            cudaCheck(cudaStreamWaitEvent(uploads[source],
                                          consumed[dest][slot], 0));
        cudaCheck(cudaMemcpyAsync(stage[source][slot],
                                  reinterpret_cast<char*>(devices[source]->input) +
                                      offset,
                                  count, cudaMemcpyDeviceToHost,
                                  uploads[source]));
        cudaCheck(cudaEventRecord(ready[source][slot], uploads[source]));
      }
      for (int dest = 0; dest < kRanks; ++dest) {
        cudaCheck(cudaSetDevice(dest));
        // Each GPU reads all source shards through host memory. Two slots
        // overlap the next uploads with the current fan-out.
        for (int source = 0; source < kRanks; ++source) {
          cudaCheck(cudaStreamWaitEvent(devices[dest]->stream,
                                        ready[source][slot], 0));
          cudaCheck(cudaMemcpyAsync(
              reinterpret_cast<char*>(devices[dest]->output) +
                  size_t(source) * shardBytes + offset,
              stage[source][slot], count, cudaMemcpyHostToDevice,
              devices[dest]->stream));
        }
        cudaCheck(cudaEventRecord(consumed[dest][slot], devices[dest]->stream));
      }
    }
  }
};
void launchPeer(std::array<Device*, kRanks> const& devices, Operation op,
                size_t shard) {
  size_t bytes = shard * sizeof(float);
  size_t fullBytes = bytes * kRanks;
  bool compact = op == Operation::Gather
                     ? fullBytes <= kCompactGatherBytes
                     : fullBytes <= kCompactScatterBytes;
  bool directSelf = op == Operation::Scatter && directSelfReduce(fullBytes);
  for (int source = 0; source < kRanks; ++source) {
    auto& s = *devices[source];
    cudaCheck(cudaSetDevice(source));
    // The destination may still be reducing the previous generation. Retain
    // its scratch until that reader has completed.
    if (op == Operation::Scatter)
      for (int dest = 0; dest < kRanks; ++dest)
        cudaCheck(cudaStreamWaitEvent(s.stream, devices[dest]->reduceDone, 0));
    for (int phase = 0; phase < kRanks; ++phase) {
      int dest = (source + phase) % kRanks;
      float* target = op == Operation::Gather
                          ? devices[dest]->output + size_t(source) * shard
                          : devices[dest]->scratch + size_t(source) * shard;
      float const* input = op == Operation::Gather
                               ? s.input
                               : s.input + size_t(dest) * shard;
      if (source == dest) {
        if (!directSelf)
          cudaCheck(cudaMemcpyAsync(target, input, bytes,
                                    cudaMemcpyDeviceToDevice, s.stream));
      }
      else
        cudaCheck(cudaMemcpyPeerAsync(target, dest, input, source, bytes,
                                      s.stream));
    }
    cudaCheck(cudaEventRecord(s.transferDone, s.stream));
  }
  if (compact) {
    // One leader publishes all source completions, avoiding a four-way event
    // fan-out on every destination for small messages.
    cudaCheck(cudaSetDevice(0));
    for (int source = 1; source < kRanks; ++source)
      cudaCheck(cudaStreamWaitEvent(devices[0]->stream,
                                    devices[source]->transferDone, 0));
    cudaCheck(cudaEventRecord(devices[0]->transferAllDone,
                              devices[0]->stream));
  }
  for (int dest = 0; dest < kRanks; ++dest) {
    auto& d = *devices[dest];
    cudaCheck(cudaSetDevice(dest));
    if (compact) {
      if (dest != 0)
        cudaCheck(cudaStreamWaitEvent(d.stream,
                                      devices[0]->transferAllDone, 0));
    } else
      for (int source = 0; source < kRanks; ++source)
        cudaCheck(cudaStreamWaitEvent(d.stream,
                                      devices[source]->transferDone, 0));
    if (op == Operation::Scatter)
      sumFour<<<int((shard + 255) / 256), 256, 0, d.stream>>>(
          d.scratch, d.input, d.output, shard, dest, directSelf);
    if (op == Operation::Scatter)
      cudaCheck(cudaEventRecord(d.reduceDone, d.stream));
    cudaCheck(cudaGetLastError());
  }
}
void launchPull(std::array<Device*, kRanks> const& devices, Operation op,
                size_t shard) {
  // Inputs are immutable and fully published by fill() before this launch.
  // A caller using other streams must supply the same input-ready boundary.
  for (int destination = 0; destination < kRanks; ++destination) {
    auto& d = *devices[destination];
    cudaCheck(cudaSetDevice(destination));
    if (op == Operation::Gather)
      pullGather<<<dim3(int((shard + 255) / 256), kRanks), 256, 0,
                   d.stream>>>(
          devices[0]->input, devices[1]->input, devices[2]->input,
          devices[3]->input, d.output, shard);
    else
      pullScatter<<<int((shard + 255) / 256), 256, 0, d.stream>>>(
          devices[0]->input, devices[1]->input, devices[2]->input,
          devices[3]->input, d.output, shard, destination);
    cudaCheck(cudaGetLastError());
  }
}
void launchPullVec(std::array<Device*, kRanks> const& devices, size_t shard) {
  if (shard % 4) {
    launchPull(devices, Operation::Gather, shard);
    return;
  }
  for (int destination = 0; destination < kRanks; ++destination) {
    auto& d = *devices[destination];
    cudaCheck(cudaSetDevice(destination));
    pullVecGather<<<dim3(int(((shard + 3) / 4 + 127) / 128), kRanks), 128,
                    0, d.stream>>>(devices[0]->input, devices[1]->input,
                                    devices[2]->input, devices[3]->input,
                                    d.output, shard);
    cudaCheck(cudaGetLastError());
  }
}
void launchPushSerial(std::array<Device*, kRanks> const& devices,
                      Operation op, size_t shard) {
  size_t fullBytes = shard * kRanks * sizeof(float);
  int threads = fullBytes == (64UL << 10) ? 128 : 256;
  bool directSelf = op == Operation::Scatter && directSelfReduce(fullBytes);
  for (int source = 0; source < kRanks; ++source) {
    auto& d = *devices[source];
    cudaCheck(cudaSetDevice(source));
    if (op == Operation::Scatter)
      for (int dest = 0; dest < kRanks; ++dest)
        cudaCheck(cudaStreamWaitEvent(d.stream, devices[dest]->reduceDone, 0));
    if (op == Operation::Gather)
      pushSerialGather<<<1, threads, 0, d.stream>>>(
          d.input, devices[0]->output, devices[1]->output,
          devices[2]->output, devices[3]->output, shard, source);
    else
      pushSerialScatter<<<1, threads, 0, d.stream>>>(
          d.input, devices[0]->scratch, devices[1]->scratch,
          devices[2]->scratch, devices[3]->scratch, shard, source,
          directSelf);
    cudaCheck(cudaGetLastError());
    if (op == Operation::Scatter)
      cudaCheck(cudaEventRecord(d.transferDone, d.stream));
  }
  if (op == Operation::Scatter) {
    for (int dest = 0; dest < kRanks; ++dest) {
      auto& d = *devices[dest];
      cudaCheck(cudaSetDevice(dest));
      for (int source = 0; source < kRanks; ++source)
        cudaCheck(cudaStreamWaitEvent(d.stream,
                                      devices[source]->transferDone, 0));
      sumFour<<<int((shard + 255) / 256), 256, 0, d.stream>>>(
          d.scratch, d.input, d.output, shard, dest, directSelf);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaEventRecord(d.reduceDone, d.stream));
    }
  }
}
void launchScatterLL(std::array<Device*, kRanks> const& devices,
                     size_t shard) {
  static uint32_t nextFlag = 1;
  uint32_t flag = nextFlag++;
  if (!flag) throw std::runtime_error("scatter packet flag exhausted");
  for (int rank = 0; rank < kRanks; ++rank) {
    auto& d = *devices[rank];
    cudaCheck(cudaSetDevice(rank));
    scatterPacketLL<<<1, 256, 0, d.stream>>>(
        d.input, d.output, devices[0]->packets, devices[1]->packets,
        devices[2]->packets, devices[3]->packets, shard, rank, flag);
    cudaCheck(cudaGetLastError());
  }
}
void launchScatterFence(std::array<Device*, kRanks> const& devices,
                        size_t shard) {
  static uint32_t nextFlag = 1;
  uint32_t flag = nextFlag++;
  if (!flag) throw std::runtime_error("scatter fence flag exhausted");
  for (int rank = 0; rank < kRanks; ++rank) {
    auto& d = *devices[rank];
    cudaCheck(cudaSetDevice(rank));
    scatterFenceFused<<<1, 256, 0, d.stream>>>(
        d.input, d.output, devices[0]->scratch, devices[1]->scratch,
        devices[2]->scratch, devices[3]->scratch, devices[0]->flags,
        devices[1]->flags, devices[2]->flags, devices[3]->flags,
        shard, rank, flag);
    cudaCheck(cudaGetLastError());
  }
}
void launchNccl(std::array<Device*, kRanks> const& devices,
                std::array<ncclComm_t, kRanks> const& comms, Operation op,
                size_t shard) {
  ncclCheck(ncclGroupStart());
  for (int rank = 0; rank < kRanks; ++rank) {
    auto& d = *devices[rank];
    cudaCheck(cudaSetDevice(rank));
    if (op == Operation::Gather)
      ncclCheck(ncclAllGather(d.input, d.reference, shard, ncclFloat,
                              comms[rank], d.stream));
    else
      ncclCheck(ncclReduceScatter(d.input, d.reference, shard, ncclFloat,
                                  ncclSum, comms[rank], d.stream));
  }
  ncclCheck(ncclGroupEnd());
}
#ifdef UCCL_LITE_PACKET_ROUTER
struct PacketRouter {
  std::array<uccl_lite::cpu_router::PacketPort, kRanks> ports{};
  uccl_lite::cpu_router::PacketPipeline pipeline;
  static uccl_lite::cpu_router::PacketPipelineOptions options(Config const& cfg) {
    uccl_lite::cpu_router::PacketPipelineOptions result;
    result.maxTensorBytes = cfg.end;
    result.packetBytes = 4UL << 20;
    result.workersPerNode = 8;
    result.transportBlocks = 248;
    if (cfg.packetRouterGpuProgress) {
      result.reduction = uccl_lite::cpu_router::ReductionPlacement::Gpu;
      result.gpuProgress = true;
    }
    result.agPerSourceProgress = cfg.packetRouterPerSource;
    return result;
  }
  PacketRouter(std::array<Device*, kRanks> const& devices, Config const& cfg)
      : pipeline(options(cfg)) {
    for (int rank = 0; rank < kRanks; ++rank)
      ports[rank] = {devices[rank]->input, devices[rank]->output,
                     devices[rank]->stream, rank};
    pipeline.prepare(ports);
  }
  void launch(size_t shard) {
    // AllGather is a byte-for-byte copy: two BF16 storage units carry one FP32
    // element, and the FP32 output checker validates the original bit pattern.
    pipeline.enqueue(uccl_lite::cpu_router::PacketProgram::allGather(
                         shard * sizeof(float) / sizeof(uint16_t), kRanks),
                     ports);
  }
};
#else
struct PacketRouter {};
#endif
void launch(std::array<Device*, kRanks> const& devices,
            std::array<ncclComm_t, kRanks> const& comms, Operation op,
            Backend backend, size_t shard, HostRouter* router,
            PacketRouter* packetRouter) {
  if (backend == Backend::Peer) launchPeer(devices, op, shard);
  else if (backend == Backend::Pull) launchPull(devices, op, shard);
  else if (backend == Backend::PullVec) launchPullVec(devices, shard);
  else if (backend == Backend::PushSerial) launchPushSerial(devices, op, shard);
  else if (backend == Backend::ScatterLL) launchScatterLL(devices, shard);
  else if (backend == Backend::ScatterFence) launchScatterFence(devices, shard);
  else if (backend == Backend::Router) router->launch(devices, shard);
  else if (backend == Backend::PacketRouter) {
#ifdef UCCL_LITE_PACKET_ROUTER
    if (!packetRouter) throw std::invalid_argument("packet router not prepared");
    packetRouter->launch(shard);
#else
    throw std::invalid_argument("packet router not linked");
#endif
  }
  else launchNccl(devices, comms, op, shard);
}
void synchronize(std::array<Device*, kRanks> const& devices) {
  for (auto* d : devices) {
    cudaCheck(cudaSetDevice(d->rank));
    cudaCheck(cudaStreamSynchronize(d->stream));
  }
}
void fill(std::array<Device*, kRanks> const& devices, Operation op,
          size_t shard, int generation) {
  size_t elements = shard * (op == Operation::Gather ? 1 : kRanks);
  for (auto* d : devices) {
    cudaCheck(cudaSetDevice(d->rank));
    fillInput<<<256, 256, 0, d->stream>>>(d->input, elements, d->rank,
                                         generation);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemsetAsync(d->output, 0xff,
                              shard * kRanks * sizeof(float), d->stream));
    cudaCheck(cudaMemsetAsync(d->reference, 0xff,
                              shard * kRanks * sizeof(float), d->stream));
    cudaCheck(cudaDeviceSynchronize());
  }
}
void verify(std::array<Device*, kRanks> const& devices, Operation op,
            Backend backend, size_t shard, int generation) {
  size_t elements = shard * (op == Operation::Gather ? kRanks : 1);
  for (auto* d : devices) {
    cudaCheck(cudaSetDevice(d->rank));
    cudaCheck(cudaMemsetAsync(d->wrong, 0, sizeof(unsigned int), d->stream));
    checkOutput<<<256, 256, 0, d->stream>>>(
        backend == Backend::Nccl ? d->reference : d->output, d->wrong,
        elements, shard, d->rank, generation, op == Operation::Gather);
    cudaCheck(cudaGetLastError());
    unsigned int wrong = 0;
    cudaCheck(cudaMemcpyAsync(&wrong, d->wrong, sizeof(unsigned int),
                              cudaMemcpyDeviceToHost, d->stream));
    cudaCheck(cudaStreamSynchronize(d->stream));
    if (wrong) {
      std::fprintf(stderr, "%s %s rank=%d wrong=%u\n", opName(op),
                   backendName(backend), d->rank, wrong);
      throw std::runtime_error("collective output mismatch");
    }
  }
}
double measure(std::array<Device*, kRanks> const& devices,
               std::array<ncclComm_t, kRanks> const& comms, Operation op,
               Backend backend, size_t shard, Config const& cfg, int repeat,
               HostRouter* router, PacketRouter* packetRouter) {
  double timedSum = 0;
  double hostSum = 0;
  for (int n = 0; n < cfg.warmups + cfg.iterations; ++n) {
    int generation = 10 + repeat * (cfg.warmups + cfg.iterations) + n;
    fill(devices, op, shard, generation);
    for (auto* d : devices) {
      cudaCheck(cudaSetDevice(d->rank));
      cudaCheck(cudaEventRecord(d->start, d->stream));
    }
    auto hostStart = std::chrono::steady_clock::now();
    launch(devices, comms, op, backend, shard, router, packetRouter);
    for (auto* d : devices) {
      cudaCheck(cudaSetDevice(d->rank));
      cudaCheck(cudaEventRecord(d->end, d->stream));
    }
    synchronize(devices);
    auto hostEnd = std::chrono::steady_clock::now();
    double slowest = 0;
    for (auto* d : devices) {
      cudaCheck(cudaSetDevice(d->rank));
      float ms = 0;
      cudaCheck(cudaEventElapsedTime(&ms, d->start, d->end));
      slowest = std::max(slowest, double(ms) * 1000);
    }
    verify(devices, op, backend, shard, generation);
    if (n >= cfg.warmups) {
      timedSum += slowest;
      double hostUs = std::chrono::duration<double, std::micro>(hostEnd -
                                                                 hostStart)
                          .count();
      hostSum += hostUs;
      std::printf("# sample,%s,%zu,%s,%d,%d,%.3f\n", opName(op),
                  shard * kRanks * sizeof(float), backendName(backend),
                  repeat + 1, n - cfg.warmups + 1, slowest);
      if (cfg.routerAg)
        std::printf("# host_sample,%s,%zu,%s,%d,%d,%.3f\n", opName(op),
                    shard * kRanks * sizeof(float), backendName(backend),
                    repeat + 1, n - cfg.warmups + 1, hostUs);
    }
  }
  if (cfg.routerAg)
    std::printf("# host_run_mean,%s,%zu,%s,%d,%.3f\n", opName(op),
                shard * kRanks * sizeof(float), backendName(backend),
                repeat + 1, hostSum / cfg.iterations);
  return timedSum / cfg.iterations;
}
double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  try {
    auto cfg = parse(argc, argv);
    int available = 0;
    cudaCheck(cudaGetDeviceCount(&available));
    if (available < kRanks)
      throw std::runtime_error("four GPUs are required");
    for (int rank = 0; rank < kRanks; ++rank) {
      for (int peer = 0; peer < kRanks; ++peer) {
        if (rank == peer) continue;
        int accessible = 0;
        cudaCheck(cudaDeviceCanAccessPeer(&accessible, rank, peer));
        if (!accessible) throw std::runtime_error("GPU peer access required");
        cudaCheck(cudaSetDevice(rank));
        auto result = cudaDeviceEnablePeerAccess(peer, 0);
        if (result == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else cudaCheck(result);
      }
    }
    std::array<std::unique_ptr<Device>, kRanks> owned;
    std::array<Device*, kRanks> devices{};
    for (int rank = 0; rank < kRanks; ++rank) {
      owned[rank] = std::make_unique<Device>(
          rank, cfg.end, std::min(cfg.end, cfg.rsLlMaxBytes),
          cfg.rsFenceMaxBytes != 0);
      devices[rank] = owned[rank].get();
    }
    std::array<ncclComm_t, kRanks> comms{};
    int ranks[kRanks] = {0, 1, 2, 3};
    ncclCheck(ncclCommInitAll(comms.data(), kRanks, ranks));
    int version = 0;
    ncclCheck(ncclGetVersion(&version));
    std::unique_ptr<HostRouter> router;
    if (cfg.routerAg)
      router = std::make_unique<HostRouter>(cfg.routerChunkBytes,
                                            cfg.routerLocalNuma);
    std::unique_ptr<PacketRouter> packetRouter;
#ifdef UCCL_LITE_PACKET_ROUTER
    if (cfg.packetRouterAg)
      packetRouter = std::make_unique<PacketRouter>(devices, cfg);
#endif
    std::printf("# four L4 peer AG/RS, float32; NCCL version=%d; "
                "warmups=%d iterations=%d repeats=%d\n",
                version, cfg.warmups, cfg.iterations, cfg.repeats);
    std::puts("# full bytes per rank; slowest-rank per-call CUDA-event microseconds; "
              "median of run-level means; every warmup and timed output checked");
    std::printf("# direct peer-read path eligible through %zu bytes\n",
                cfg.pullMaxBytes);
    std::printf("# compact peer completion through AG %zu B, RS %zu B\n",
                kCompactGatherBytes, kCompactScatterBytes);
    std::puts("# RS direct self contribution at 256 KiB, 1-8 MiB, and >=32 MiB");
    if (cfg.routerAg)
      std::printf("# cpu-router AG: pinned host, two slots/source, chunk=%zu B; "
                  "placement=%s nodes=%d,%d,%d,%d; "
                  "host_sample includes launch and completion\n",
                  cfg.routerChunkBytes,
                  cfg.routerLocalNuma ? "local" : "cuda",
                  router->hostNode[0], router->hostNode[1],
                  router->hostNode[2], router->hostNode[3]);
    if (cfg.packetRouterAg)
      std::printf("# packet-router AG: FP32 bitwise copy, 8 workers/NUMA, "
                  "transport-block cap=248, packet cap=4194304 B, "
                  "candidate from %zu B; progress=%s\n",
                  cfg.packetRouterMinBytes,
                  cfg.packetRouterPerSource ? "gpu-per-source"
                      : cfg.packetRouterGpuProgress ? "gpu-all-ready" : "cpu");
    std::puts("collective,bytes,backend,median_us,min_us,max_us");
    for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2) {
      size_t shard = bytes / sizeof(float) / kRanks;
      for (auto op : {Operation::Gather, Operation::Scatter}) {
        if (cfg.rsOnly && op == Operation::Gather) continue;
        if ((cfg.routerAg || cfg.packetRouterAg) && op == Operation::Scatter)
          continue;
        bool packetEligible = cfg.packetRouterAg &&
                              bytes >= cfg.packetRouterMinBytes;
        bool pullEligible = bytes <= cfg.pullMaxBytes;
        bool vecEligible = op == Operation::Gather &&
                           bytes <= cfg.pullVecMaxBytes;
        bool serialEligible = op == Operation::Gather
                                  ? bytes <= cfg.pushSerialMaxBytes
                                  : bytes <= cfg.rsPushSerialMaxBytes;
        bool llEligible = op == Operation::Scatter && shard % 2 == 0 &&
                          bytes <= cfg.rsLlMaxBytes;
        bool fenceEligible = op == Operation::Scatter &&
                             bytes <= cfg.rsFenceMaxBytes;
        std::vector<Backend> backends{Backend::Peer};
        if (pullEligible) backends.push_back(Backend::Pull);
        if (vecEligible) backends.push_back(Backend::PullVec);
        if (serialEligible) backends.push_back(Backend::PushSerial);
        if (llEligible) backends.push_back(Backend::ScatterLL);
        if (fenceEligible) backends.push_back(Backend::ScatterFence);
        backends.push_back(Backend::Nccl);
        if (cfg.routerAg) backends.push_back(Backend::Router);
        if (packetEligible) backends.push_back(Backend::PacketRouter);
        for (int generation = 0; generation < 2; ++generation) {
          fill(devices, op, shard, generation);
          for (auto backend : backends) {
            launch(devices, comms, op, backend, shard, router.get(),
                   packetRouter.get());
            synchronize(devices);
            verify(devices, op, backend, shard, generation);
          }
        }
        std::array<std::vector<double>, 9> samples;
        for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
          std::vector<Backend> order;
          if (pullEligible) {
            std::array<Backend, 3> latin{Backend::Peer, Backend::Pull,
                                         Backend::Nccl};
            for (int offset = 0; offset < 3; ++offset)
              order.push_back(latin[(repeat + offset) % 3]);
          } else if (repeat % 2 == 0)
            order = {Backend::Peer, Backend::Nccl};
          else
            order = {Backend::Nccl, Backend::Peer};
          if (cfg.routerAg) {
            order.push_back(Backend::Router);
            std::rotate(order.begin(), order.begin() + repeat % order.size(),
                        order.end());
          }
          if (packetEligible) {
            order.push_back(Backend::PacketRouter);
            std::rotate(order.begin(), order.begin() + repeat % order.size(),
                        order.end());
          }
          if (vecEligible) {
            order.push_back(Backend::PullVec);
            std::rotate(order.begin(), order.begin() + repeat % order.size(),
                        order.end());
          }
          if (serialEligible) {
            order.push_back(Backend::PushSerial);
            std::rotate(order.begin(), order.begin() + repeat % order.size(),
                        order.end());
          }
          if (llEligible) {
            order.push_back(Backend::ScatterLL);
            std::rotate(order.begin(), order.begin() + repeat % order.size(),
                        order.end());
          }
          if (fenceEligible) {
            order.push_back(Backend::ScatterFence);
            std::rotate(order.begin(), order.begin() + repeat % order.size(),
                        order.end());
          }
          for (auto backend : order)
            samples[static_cast<int>(backend)].push_back(
                measure(devices, comms, op, backend, shard, cfg, repeat,
                        router.get(), packetRouter.get()));
        }
        fill(devices, op, shard, 2);
        for (auto backend : backends) {
          launch(devices, comms, op, backend, shard, router.get(),
                 packetRouter.get());
          synchronize(devices);
          verify(devices, op, backend, shard, 2);
        }
        for (auto backend : backends) {
          auto const& run = samples[static_cast<int>(backend)];
          auto range = std::minmax_element(run.begin(), run.end());
          std::printf("%s,%zu,%s,%.3f,%.3f,%.3f\n", opName(op), bytes,
                      backendName(backend), median(run), *range.first,
                      *range.second);
        }
        Backend best = Backend::Peer;
        if (pullEligible && median(samples[static_cast<int>(Backend::Pull)]) <
                                median(samples[static_cast<int>(best)]))
          best = Backend::Pull;
        if (vecEligible && median(samples[static_cast<int>(Backend::PullVec)]) <
                               median(samples[static_cast<int>(best)]))
          best = Backend::PullVec;
        if (serialEligible &&
            median(samples[static_cast<int>(Backend::PushSerial)]) <
                median(samples[static_cast<int>(best)]))
          best = Backend::PushSerial;
        if (llEligible &&
            median(samples[static_cast<int>(Backend::ScatterLL)]) <
                median(samples[static_cast<int>(best)]))
          best = Backend::ScatterLL;
        if (fenceEligible &&
            median(samples[static_cast<int>(Backend::ScatterFence)]) <
                median(samples[static_cast<int>(best)]))
          best = Backend::ScatterFence;
        char const* selected =
            best == Backend::Pull ? "pull"
                : best == Backend::PullVec ? "pull-vec"
                : best == Backend::PushSerial ? "push-serial"
                : best == Backend::ScatterLL ? "scatter-ll"
                : best == Backend::ScatterFence ? "scatter-fence"
                : op == Operation::Scatter && directSelfReduce(bytes)
                      ? "peer-noself"
                : ((op == Operation::Gather && bytes <= kCompactGatherBytes) ||
                   (op == Operation::Scatter && bytes <= kCompactScatterBytes))
                      ? "peer-compact"
                      : "peer";
        std::printf("# selected,%s,%zu,%s\n", opName(op), bytes, selected);
        std::fflush(stdout);
      }
      if (bytes > cfg.end / 2) break;
    }
    for (auto comm : comms) ncclCheck(ncclCommDestroy(comm));
  } catch (std::exception const& error) {
    std::fprintf(stderr, "peer AG/RS: %s\n", error.what());
    return 1;
  }
}
