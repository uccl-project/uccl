// Eight-L4 BF16 All-to-All: project-owned NUMA host staging and local peer
// copies against an explicit NCCL send/recv control. One process owns all GPUs.
#include <linux/mempolicy.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
constexpr int kRanks = 8;
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void check(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(status));
}
int nodeOf(int device) {
  char bus[32];
  check(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));
  std::ifstream input(std::string("/sys/bus/pci/devices/") + bus +
                      "/numa_node");
  int node = -1;
  input >> node;
  if (!input || node < 0 || node >= 64)
    throw std::runtime_error("GPU NUMA node unavailable");
  return node;
}
void* pinnedNuma(size_t bytes, int node) {
  void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) throw std::runtime_error("host staging mmap failed");
  unsigned long mask = 1UL << node;
  if (syscall(SYS_mbind, ptr, bytes, MPOL_BIND, &mask, 64, 0)) {
    munmap(ptr, bytes);
    throw std::runtime_error("host staging mbind failed");
  }
  std::memset(ptr, 0, bytes);
  auto status = cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
  if (status != cudaSuccess) {
    munmap(ptr, bytes);
    check(status);
  }
  return ptr;
}
__global__ void fill(__nv_bfloat16* data, size_t count, size_t shard,
                     int rank) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    int dest = int(i / shard);
    data[i] = __float2bfloat16(
        float((rank + 1) * 16 + (dest + 1) * 2 + (i % shard) % 2));
  }
}
__global__ void verify(__nv_bfloat16 const* data, size_t count, size_t shard,
                       int rank, int* errors) {
  bool bad = false;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    int source = int(i / shard);
    auto expected = __float2bfloat16(
        float((source + 1) * 16 + (rank + 1) * 2 + (i % shard) % 2));
    if (__bfloat162float(data[i]) != __bfloat162float(expected)) bad = true;
  }
  if (bad) atomicExch(errors, 1);
}
struct Config {
  size_t begin = 1UL << 20, end = 1UL << 30;
  int warmup = 5, iterations = 10, repeats = 3;
  std::string samples;
};
enum class Backend { Hybrid, Host, Nccl };
char const* name(Backend backend) {
  switch (backend) {
    case Backend::Hybrid:
      return "uccl-hybrid";
    case Backend::Host:
      return "uccl-host";
    case Backend::Nccl:
      return "nccl";
  }
  return "unknown";
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  auto n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
struct Device {
  int id;
  __nv_bfloat16 *input = nullptr, *output = nullptr;
  int* errors = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  Device(int device, size_t bytes) : id(device) {
    check(cudaSetDevice(id));
    check(cudaMalloc(&input, bytes));
    check(cudaMalloc(&output, bytes));
    check(cudaMalloc(&errors, sizeof(int)));
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
  }
  Device(Device const&) = delete;
  Device& operator=(Device const&) = delete;
  ~Device() {
    cudaSetDevice(id);
    if (stream) cudaStreamDestroy(stream);
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (errors) cudaFree(errors);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
  }
};
struct Bench {
  size_t maxBytes;
  std::array<int, kRanks> nodes{};
  std::vector<std::unique_ptr<Device>> devices;
  std::array<void*, kRanks> stage{};
  std::array<ncclComm_t, kRanks> comms{};
  explicit Bench(size_t maximum) : maxBytes(maximum) {
    for (int rank = 0; rank < kRanks; ++rank) nodes[rank] = nodeOf(rank);
    for (int rank = 0; rank < kRanks; ++rank)
      if (nodes[rank] != nodes[rank < 4 ? 0 : 4])
        throw std::invalid_argument("expected four GPUs per NUMA group");
    if (nodes[0] == nodes[4])
      throw std::invalid_argument("expected two GPU NUMA groups");
    for (int dest = 0; dest < kRanks; ++dest) {
      check(cudaSetDevice(dest));
      for (int source = 0; source < kRanks; ++source) {
        if (source == dest || nodes[source] != nodes[dest]) continue;
        int canAccess = 0;
        check(cudaDeviceCanAccessPeer(&canAccess, dest, source));
        if (!canAccess)
          throw std::invalid_argument("local CUDA peer unavailable");
        auto status = cudaDeviceEnablePeerAccess(source, 0);
        if (status != cudaSuccess &&
            status != cudaErrorPeerAccessAlreadyEnabled)
          check(status);
      }
    }
    for (int rank = 0; rank < kRanks; ++rank)
      devices.emplace_back(std::make_unique<Device>(rank, maxBytes));
    for (int rank = 0; rank < kRanks; ++rank)
      stage[rank] = pinnedNuma(maxBytes / kRanks, nodes[rank]);
    int ids[kRanks] = {0, 1, 2, 3, 4, 5, 6, 7};
    check(ncclCommInitAll(comms.data(), kRanks, ids));
  }
  ~Bench() {
    for (auto comm : comms)
      if (comm) ncclCommDestroy(comm);
    for (void* ptr : stage)
      if (ptr) {
        cudaHostUnregister(ptr);
        munmap(ptr, maxBytes / kRanks);
      }
  }
  void syncAll() {
    for (auto const& device : devices) {
      check(cudaSetDevice(device->id));
      check(cudaStreamSynchronize(device->stream));
    }
  }
  void local(size_t bytes, bool hybrid) {
    size_t shard = bytes / kRanks;
    for (int k = 0; k < kRanks; ++k) {
      for (int source = 0; source < kRanks; ++source) {
        int dest = (source + k) % kRanks;
        auto& from = *devices[source];
        auto& to = *devices[dest];
        auto* input =
            reinterpret_cast<char*>(from.input) + size_t(dest) * shard;
        auto* output =
            reinterpret_cast<char*>(to.output) + size_t(source) * shard;
        if (source == dest) {
          check(cudaSetDevice(dest));
          check(cudaMemcpyAsync(output, input, shard, cudaMemcpyDeviceToDevice,
                                to.stream));
        } else if (hybrid && nodes[source] == nodes[dest]) {
          check(cudaSetDevice(dest));
          check(cudaMemcpyPeerAsync(output, dest, input, source, shard,
                                    to.stream));
        } else {
          check(cudaSetDevice(source));
          check(cudaMemcpyAsync(stage[source], input, shard,
                                cudaMemcpyDeviceToHost, from.stream));
        }
      }
      syncAll();
      if (k == 0) continue;
      for (int source = 0; source < kRanks; ++source) {
        int dest = (source + k) % kRanks;
        if (source == dest || (hybrid && nodes[source] == nodes[dest]))
          continue;
        auto& to = *devices[dest];
        auto* output =
            reinterpret_cast<char*>(to.output) + size_t(source) * shard;
        check(cudaSetDevice(dest));
        check(cudaMemcpyAsync(output, stage[source], shard,
                              cudaMemcpyHostToDevice, to.stream));
      }
      syncAll();
    }
  }
  void nccl(size_t bytes) {
    size_t shard = bytes / kRanks;
    check(ncclGroupStart());
    for (int rank = 0; rank < kRanks; ++rank) {
      auto& device = *devices[rank];
      check(cudaSetDevice(rank));
      for (int peer = 0; peer < kRanks; ++peer) {
        if (peer == rank) continue;
        auto* send =
            reinterpret_cast<char*>(device.input) + size_t(peer) * shard;
        auto* recv =
            reinterpret_cast<char*>(device.output) + size_t(peer) * shard;
        check(ncclSend(send, shard / 2, ncclBfloat16, peer, comms[rank],
                       device.stream));
        check(ncclRecv(recv, shard / 2, ncclBfloat16, peer, comms[rank],
                       device.stream));
      }
    }
    check(ncclGroupEnd());
    for (int rank = 0; rank < kRanks; ++rank) {
      auto& device = *devices[rank];
      auto* send = reinterpret_cast<char*>(device.input) + size_t(rank) * shard;
      auto* recv =
          reinterpret_cast<char*>(device.output) + size_t(rank) * shard;
      check(cudaSetDevice(rank));
      check(cudaMemcpyAsync(recv, send, shard, cudaMemcpyDeviceToDevice,
                            device.stream));
    }
  }
  double run(Backend backend, size_t bytes) {
    size_t count = bytes / 2, shard = count / kRanks;
    for (auto const& device : devices) {
      check(cudaSetDevice(device->id));
      fill<<<64, 256, 0, device->stream>>>(device->input, count, shard,
                                           device->id);
      check(cudaGetLastError());
      check(cudaMemsetAsync(device->errors, 0, sizeof(int), device->stream));
      check(cudaEventRecord(device->start, device->stream));
    }
    if (backend == Backend::Nccl)
      nccl(bytes);
    else
      local(bytes, backend == Backend::Hybrid);
    for (auto const& device : devices) {
      check(cudaSetDevice(device->id));
      check(cudaEventRecord(device->stop, device->stream));
      verify<<<64, 256, 0, device->stream>>>(device->output, count, shard,
                                             device->id, device->errors);
      check(cudaGetLastError());
    }
    double worst = 0;
    for (auto const& device : devices) {
      check(cudaSetDevice(device->id));
      check(cudaStreamSynchronize(device->stream));
      float ms = 0;
      check(cudaEventElapsedTime(&ms, device->start, device->stop));
      worst = std::max(worst, double(ms) * 1000);
      int errors = 0;
      check(cudaMemcpy(&errors, device->errors, sizeof(int),
                       cudaMemcpyDeviceToHost));
      if (errors) throw std::runtime_error("All-to-All output mismatch");
    }
    return worst;
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    Config config;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto next = [&]() {
        if (++i >= argc) throw std::invalid_argument("missing option value");
        return std::string(argv[i]);
      };
      if (arg == "--begin")
        config.begin = std::stoull(next());
      else if (arg == "--end")
        config.end = std::stoull(next());
      else if (arg == "--warmup")
        config.warmup = std::stoi(next());
      else if (arg == "--iterations")
        config.iterations = std::stoi(next());
      else if (arg == "--repeats")
        config.repeats = std::stoi(next());
      else if (arg == "--samples")
        config.samples = next();
      else if (arg == "--help") {
        std::puts(
            "Eight-L4 BF16 All-to-All: --begin BYTES --end BYTES "
            "--warmup N --iterations N --repeats N --samples NEW_CSV");
        return 0;
      } else
        throw std::invalid_argument("unknown option: " + arg);
    }
    if (config.begin < 128 || config.begin % 16 || config.begin % kRanks ||
        config.end < config.begin || config.end > (1UL << 30) ||
        config.end % 16 || config.end % kRanks || config.warmup < 1 ||
        config.iterations < 1 || config.repeats < 3)
      throw std::invalid_argument("invalid benchmark options");
    std::ofstream samples;
    if (!config.samples.empty()) {
      if (std::filesystem::exists(config.samples))
        throw std::invalid_argument("sample output already exists");
      samples.open(config.samples);
      if (!samples) throw std::runtime_error("cannot create sample output");
      samples << "bytes,backend,repeat,iteration,slowest_rank_us\n";
    }
    Bench bench(config.end);
    int ncclVersion = 0;
    check(ncclGetVersion(&ncclVersion));
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver));
    check(cudaRuntimeGetVersion(&runtime));
    std::printf(
        "# GPUs=8 dtype=BF16 topology=two-four-GPU-NUMA-groups "
        "placement=out-of-place\n");
    std::printf(
        "# CUDA=%d driver=%d NCCL=%d warmup=%d iterations=%d "
        "repeats=%d\n",
        runtime, driver, ncclVersion, config.warmup, config.iterations,
        config.repeats);
    std::puts(
        "# timing=slowest-rank CUDA events; fill and full-output "
        "verification outside timing; both UCCL paths eager");
    std::puts("bytes,backend,median_us,min_us,max_us");
    for (size_t bytes = config.begin; bytes <= config.end; bytes *= 2) {
      std::array<std::vector<double>, 3> results;
      for (int repeat = 0; repeat < config.repeats; ++repeat) {
        std::array<Backend, 3> order =
            repeat % 2 ? std::array<Backend, 3>{Backend::Nccl, Backend::Hybrid,
                                                Backend::Host}
                       : std::array<Backend, 3>{Backend::Host, Backend::Hybrid,
                                                Backend::Nccl};
        for (auto backend : order) {
          for (int warmup = 0; warmup < config.warmup; ++warmup)
            bench.run(backend, bytes);
          std::vector<double> times;
          for (int iteration = 0; iteration < config.iterations; ++iteration) {
            double us = bench.run(backend, bytes);
            times.push_back(us);
            if (samples)
              samples << bytes << ',' << name(backend) << ',' << repeat << ','
                      << iteration << ',' << us << '\n';
          }
          results[int(backend)].push_back(median(times));
        }
      }
      for (auto backend : {Backend::Hybrid, Backend::Host, Backend::Nccl}) {
        auto const& runs = results[int(backend)];
        std::printf("%zu,%s,%.3f,%.3f,%.3f\n", bytes, name(backend),
                    median(runs), *std::min_element(runs.begin(), runs.end()),
                    *std::max_element(runs.begin(), runs.end()));
      }
      std::fflush(stdout);
    }
    std::puts("# correctness: PASS (every invocation, all eight ranks)");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
