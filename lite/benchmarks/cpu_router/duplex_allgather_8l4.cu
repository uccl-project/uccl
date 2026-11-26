// Eight-GPU BF16 AllGather: NUMA-local double-buffered host staging versus
// the packet CPU router and NCCL. One process owns all devices.
#include "uccl_lite/cpu_router/memory.hpp"
#include "uccl_lite/cpu_router/packet_pipeline.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <nccl.h>

using namespace uccl_lite::cpu_router;
namespace {
constexpr int kRanks = 8;
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void check(ncclResult_t code) {
  if (code != ncclSuccess) throw std::runtime_error(ncclGetErrorString(code));
}
__global__ void fill(__nv_bfloat16* data, size_t count, int rank) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    data[i] = __float2bfloat16(float((rank + 1) * (1 + i % 7)) / 64.f);
}
__global__ void verify(__nv_bfloat16 const* data, size_t count,
                       size_t contribution, int* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    int source = int(i / contribution);
    size_t offset = i % contribution;
    float expected = float((source + 1) * (1 + offset % 7)) / 64.f;
    if (__bfloat162float(data[i]) != expected) atomicAdd(errors, 1);
  }
}
struct Rank {
  RegisteredMemory input, output, errors;
  cudaStream_t stream{};
  cudaEvent_t start{}, stop{};
  ncclComm_t nccl{};
};
struct Duplex {
  size_t chunkBytes;
  bool selfDirect, localPeer;
  std::array<std::array<RegisteredMemory, 2>, kRanks> stage;
  std::array<cudaStream_t, kRanks> upload{};
  std::array<std::array<cudaEvent_t, 2>, kRanks> ready{};
  std::array<std::array<cudaEvent_t, 2>, kRanks> consumed{};
  explicit Duplex(size_t chunk, bool self, bool peer)
      : chunkBytes(chunk), selfDirect(self || peer), localPeer(peer) {
    for (int r = 0; r < kRanks; ++r) {
      check(cudaSetDevice(r));
      check(cudaStreamCreateWithFlags(&upload[r], cudaStreamNonBlocking));
      int node = gpuNumaNode(r);
      if (node != r / 4)
        throw std::runtime_error("expected four GPUs per NUMA group");
      if (localPeer)
        for (int peer = (r / 4) * 4; peer < (r / 4 + 1) * 4; ++peer)
          if (peer != r) {
            int accessible = 0;
            check(cudaDeviceCanAccessPeer(&accessible, r, peer));
            if (!accessible)
              throw std::runtime_error("local GPU peer access unavailable");
          }
      for (int slot = 0; slot < 2; ++slot) {
        stage[r][slot] = RegisteredMemory::pinned(chunkBytes, node);
        check(cudaEventCreateWithFlags(&ready[r][slot],
                                       cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&consumed[r][slot],
                                       cudaEventDisableTiming));
      }
    }
  }
  ~Duplex() {
    for (int r = 0; r < kRanks; ++r) {
      cudaSetDevice(r);
      for (int slot = 0; slot < 2; ++slot) {
        if (ready[r][slot]) cudaEventDestroy(ready[r][slot]);
        if (consumed[r][slot]) cudaEventDestroy(consumed[r][slot]);
      }
      if (upload[r]) cudaStreamDestroy(upload[r]);
    }
  }
  void enqueue(std::vector<Rank> const& ranks, size_t contributionBytes) {
    for (int r = 0; r < kRanks; ++r) {
      check(cudaSetDevice(r));
      check(cudaStreamWaitEvent(upload[r], ranks[r].start, 0));
    }
    for (size_t offset = 0, piece = 0; offset < contributionBytes;
         offset += chunkBytes, ++piece) {
      int slot = int(piece & 1);
      size_t count = std::min(chunkBytes, contributionBytes - offset);
      for (int source = 0; source < kRanks; ++source) {
        check(cudaSetDevice(source));
        if (piece >= 2)
          for (int dest = 0; dest < kRanks; ++dest)
            if (!(localPeer && source / 4 == dest / 4) &&
                !(selfDirect && source == dest))
              check(cudaStreamWaitEvent(upload[source], consumed[dest][slot],
                                        0));
        check(cudaMemcpyAsync(stage[source][slot].data(),
                              static_cast<char*>(ranks[source].input.data()) +
                                  offset,
                              count, cudaMemcpyDeviceToHost, upload[source]));
        check(cudaEventRecord(ready[source][slot], upload[source]));
      }
      for (int dest = 0; dest < kRanks; ++dest) {
        check(cudaSetDevice(dest));
        // Consume local NUMA sources first, then remote ones.
        for (int index = 0; index < kRanks; ++index) {
          int source = (dest / 4) * 4 + index;
          if (source >= kRanks) source -= kRanks;
          void* target = static_cast<char*>(ranks[dest].output.data()) +
                         size_t(source) * contributionBytes + offset;
          void* input = static_cast<char*>(ranks[source].input.data()) +
                        offset;
          if (source == dest && selfDirect)
            check(cudaMemcpyAsync(target, input, count,
                                  cudaMemcpyDeviceToDevice,
                                  ranks[dest].stream));
          else if (localPeer && source / 4 == dest / 4) {
            check(cudaStreamWaitEvent(ranks[dest].stream,
                                      ranks[source].start, 0));
            check(cudaMemcpyPeerAsync(target, dest, input, source, count,
                                      ranks[dest].stream));
          } else {
            check(cudaStreamWaitEvent(ranks[dest].stream,
                                      ready[source][slot], 0));
            check(cudaMemcpyAsync(target, stage[source][slot].data(), count,
                                  cudaMemcpyHostToDevice,
                                  ranks[dest].stream));
          }
        }
        check(cudaEventRecord(consumed[dest][slot], ranks[dest].stream));
      }
    }
  }
};
struct Config {
  size_t begin = 1UL << 20, end = 1UL << 30, chunk = 8UL << 20;
  int warmup = 3, iterations = 10, repeats = 3;
  int method = -1;
  bool selfDirect = true, localPeer = false, reverse = false;
  std::string samples;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--reverse") {
        cfg.reverse = true;
        continue;
      }
      if (i + 1 >= argc) throw std::invalid_argument("missing option value");
      std::string option = argv[i++];
      if (option == "--begin") cfg.begin = std::stoull(argv[i]);
      else if (option == "--end") cfg.end = std::stoull(argv[i]);
      else if (option == "--chunk-bytes") cfg.chunk = std::stoull(argv[i]);
      else if (option == "--warmup") cfg.warmup = std::stoi(argv[i]);
      else if (option == "--iterations") cfg.iterations = std::stoi(argv[i]);
      else if (option == "--repeats") cfg.repeats = std::stoi(argv[i]);
      else if (option == "--samples") cfg.samples = argv[i];
      else if (option == "--method") {
        std::string method = argv[i];
        if (method == "nccl") cfg.method = 0;
        else if (method == "packet") cfg.method = 1;
        else if (method == "duplex") cfg.method = 2;
        else throw std::invalid_argument("invalid method");
      } else if (option == "--self-direct") cfg.selfDirect = std::stoi(argv[i]);
      else if (option == "--local-peer") cfg.localPeer = std::stoi(argv[i]);
      else throw std::invalid_argument("unknown option: " + option);
    }
    if (cfg.begin < 128 || cfg.end < cfg.begin || cfg.end > (1UL << 30) ||
        cfg.begin % (kRanks * 2) || cfg.end % (kRanks * 2) ||
        cfg.chunk < 4096 || cfg.chunk % 4096 || cfg.warmup < 1 ||
        cfg.iterations < 1 || cfg.repeats < 1)
      throw std::invalid_argument("invalid benchmark configuration");
    int available = 0;
    check(cudaGetDeviceCount(&available));
    if (available < kRanks) throw std::runtime_error("eight GPUs required");
    std::vector<Rank> ranks(kRanks);
    std::vector<int> devices(kRanks);
    std::vector<PacketPort> ports(kRanks);
    for (int r = 0; r < kRanks; ++r) {
      devices[r] = r;
      check(cudaSetDevice(r));
      auto& x = ranks[r];
      x.input = RegisteredMemory::gpu(cfg.end / kRanks, r);
      x.output = RegisteredMemory::gpu(cfg.end, r);
      x.errors = RegisteredMemory::gpu(sizeof(int), r);
      check(cudaStreamCreateWithFlags(&x.stream, cudaStreamNonBlocking));
      check(cudaEventCreate(&x.start));
      check(cudaEventCreate(&x.stop));
      ports[r] = {x.input.data(), x.output.data(), x.stream, r};
    }
    std::vector<ncclComm_t> comms(kRanks);
    check(ncclCommInitAll(comms.data(), kRanks, devices.data()));
    for (int r = 0; r < kRanks; ++r) ranks[r].nccl = comms[r];
    PacketPipelineOptions options;
    options.maxTensorBytes = cfg.end;
    PacketPipeline router(options);
    router.prepare(ports);
    Duplex duplex(cfg.chunk, cfg.selfDirect, cfg.localPeer);
    std::ofstream samples;
    if (!cfg.samples.empty()) {
      if (std::filesystem::exists(cfg.samples))
        throw std::runtime_error("samples file already exists");
      samples.open(cfg.samples);
      if (!samples) throw std::runtime_error("cannot create samples file");
      samples << "bytes,method,repeat,iteration,slowest_rank_us\n";
    }
    auto sync = [&] {
      for (int r = 0; r < kRanks; ++r) {
        check(cudaSetDevice(r));
        check(cudaStreamSynchronize(ranks[r].stream));
      }
    };
    auto run = [&](int method, size_t bytes) {
      size_t contribution = bytes / kRanks / 2;
      for (int r = 0; r < kRanks; ++r) {
        check(cudaSetDevice(r));
        fill<<<32, 256, 0, ranks[r].stream>>>(
            static_cast<__nv_bfloat16*>(ranks[r].input.data()), contribution,
            r);
        check(cudaEventRecord(ranks[r].start, ranks[r].stream));
      }
      if (method == 0) {
        check(ncclGroupStart());
        for (int r = 0; r < kRanks; ++r)
          check(ncclAllGather(ranks[r].input.data(), ranks[r].output.data(),
                              contribution, ncclBfloat16, ranks[r].nccl,
                              ranks[r].stream));
        check(ncclGroupEnd());
      } else if (method == 1) {
        router.enqueue(PacketProgram::allGather(contribution, kRanks), ports);
      } else {
        duplex.enqueue(ranks, bytes / kRanks);
      }
      for (int r = 0; r < kRanks; ++r) {
        check(cudaSetDevice(r));
        check(cudaEventRecord(ranks[r].stop, ranks[r].stream));
        check(cudaMemsetAsync(ranks[r].errors.data(), 0, sizeof(int),
                              ranks[r].stream));
        verify<<<32, 256, 0, ranks[r].stream>>>(
            static_cast<__nv_bfloat16 const*>(ranks[r].output.data()),
            bytes / 2, contribution,
            static_cast<int*>(ranks[r].errors.data()));
      }
      sync();
      double worst = 0;
      for (int r = 0; r < kRanks; ++r) {
        check(cudaSetDevice(r));
        int errors = 0;
        check(cudaMemcpy(&errors, ranks[r].errors.data(), sizeof(int),
                         cudaMemcpyDeviceToHost));
        if (errors)
          throw std::runtime_error("wrong output on rank " +
                                   std::to_string(r) + ": " +
                                   std::to_string(errors));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, ranks[r].start, ranks[r].stop));
        worst = std::max(worst, double(ms) * 1000);
      }
      return worst;
    };
    char const* names[] = {"nccl", "packet-router", "duplex-stage"};
    std::vector<size_t> sizes;
    for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2)
      sizes.push_back(bytes);
    if (cfg.reverse) std::reverse(sizes.begin(), sizes.end());
    for (size_t bytes : sizes) {
      std::vector<double> timings[3];
      for (int round = 0; round < cfg.warmup + cfg.repeats; ++round) {
        for (int order = 0; order < 3; ++order) {
          int method = (round + order) % 3;
          if (cfg.method >= 0 && method != cfg.method) continue;
          int count = round < cfg.warmup ? 1 : cfg.iterations;
          for (int iter = 0; iter < count; ++iter) {
            double us = run(method, bytes);
            if (round >= cfg.warmup) {
              timings[method].push_back(us);
              if (samples)
                samples << bytes << ',' << names[method] << ','
                        << round - cfg.warmup << ',' << iter << ',' << us
                        << '\n';
            }
          }
        }
      }
      for (int method = 0; method < 3; ++method) {
        auto& v = timings[method];
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        std::printf("bytes=%zu method=%s min_us=%.3f median_us=%.3f max_us=%.3f samples=%zu\n",
                    bytes, names[method], v.front(), v[v.size() / 2],
                    v.back(), v.size());
      }
    }
    for (int r = 0; r < kRanks; ++r) {
      check(cudaSetDevice(r));
      check(ncclCommDestroy(ranks[r].nccl));
    }
    std::puts("# correctness: PASS (every invocation, all ranks)");
  } catch (std::exception const& error) {
    std::fprintf(stderr, "duplex_allgather_8l4: %s\n", error.what());
    return 1;
  }
}
