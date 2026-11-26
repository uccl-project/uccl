// Independent host-double oracle for the opt-in GPU reduction arithmetic.
#include "cpu_router.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>

using namespace uccl_lite::cpu_router;
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int ranks = 4;
    bool interleaveScatter = false;
    bool gpuProgress = false;
    bool allEncodings = false;
    bool cpuReduce = false, batchCpuJobs = false, directScatterResults = false;
    int gpuLaunchWorkers = 0;
    std::vector<int> devices;
    int transportVectorBytes = 16;
    size_t packetBytes = 512UL << 10, customElements = 0;
    size_t singleCtaBytes = 0;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--gpus" && i + 1 < argc)
        ranks = std::stoi(argv[++i]);
      else if (std::string(argv[i]) == "--devices" && i + 1 < argc) {
        std::stringstream list(argv[++i]);
        std::string device;
        while (std::getline(list, device, ','))
          devices.push_back(std::stoi(device));
      } else if (std::string(argv[i]) == "--cpu-reduce")
        cpuReduce = true;
      else if (std::string(argv[i]) == "--cpu-interleave-scatter") {
        cpuReduce = true;
        interleaveScatter = true;
      } else if (std::string(argv[i]) == "--batch-cpu-jobs")
        batchCpuJobs = true;
      else if (std::string(argv[i]) == "--gpu-launch-workers-per-node" &&
               i + 1 < argc)
        gpuLaunchWorkers = std::stoi(argv[++i]);
      else if (std::string(argv[i]) == "--direct-scatter-results")
        directScatterResults = true;
      else if (std::string(argv[i]) == "--interleave-scatter")
        interleaveScatter = true;
      else if (std::string(argv[i]) == "--gpu-progress")
        gpuProgress = true;
      else if (std::string(argv[i]) == "--all-encodings")
        allEncodings = true;
      else if (std::string(argv[i]) == "--packet-bytes" && i + 1 < argc)
        packetBytes = std::stoull(argv[++i]);
      else if (std::string(argv[i]) == "--transport-vector-bytes" &&
               i + 1 < argc)
        transportVectorBytes = std::stoi(argv[++i]);
      else if (std::string(argv[i]) == "--single-cta-bytes" && i + 1 < argc)
        singleCtaBytes = std::stoull(argv[++i]);
      else if (std::string(argv[i]) == "--elements" && i + 1 < argc) {
        customElements = std::stoull(argv[++i]);
        if (!customElements || customElements > (1UL << 24))
          throw std::invalid_argument("invalid element count");
      } else
        throw std::invalid_argument("expected --gpus N");
    }
    if (!devices.empty())
      ranks = int(devices.size());
    else
      for (int r = 0; r < ranks; ++r) devices.push_back(r);
    if (ranks < 1 || ranks > (cpuReduce ? 8 : 4))
      throw std::invalid_argument(
          "oracle supports up to eight CPU-reduction or four GPU-reduction "
          "ranks");
    if (cpuReduce && allEncodings)
      throw std::invalid_argument("full-range oracle is GPU-only");
    if (allEncodings && ranks > 2)
      throw std::invalid_argument("full-range oracle uses at most two sources");
    std::vector<int> numa(ranks), nodes;
    if (cpuReduce)
      for (int r = 0; r < ranks; ++r) {
        numa[r] = gpuNumaNode(devices[r]);
        if (std::find(nodes.begin(), nodes.end(), numa[r]) == nodes.end())
          nodes.push_back(numa[r]);
      }
    std::vector<size_t> counts{1UL,   3UL,    8UL,     17UL,
                               257UL, 4096UL, 65537UL, 262144UL};
    if (customElements) counts = {customElements};
    size_t maxCount = *std::max_element(counts.begin(), counts.end()) * ranks,
           guards = transportVectorBytes == 32 ? 16 : 8;
    std::vector<uint16_t*> input(ranks), output(ranks);
    std::vector<cudaStream_t> streams(ranks);
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices[r]));
      check(cudaMalloc(&input[r], (maxCount + guards * 2) * 2));
      check(cudaMalloc(&output[r], (maxCount + guards * 2) * 2));
      check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
    }
    {
      CpuRouterOptions options;
      options.maxTensorBytes = maxCount * 2;
      options.reduction =
          cpuReduce ? ReductionPlacement::Cpu : ReductionPlacement::Gpu;
      options.batchCpuJobs = batchCpuJobs;
      options.directScatterResults = directScatterResults;
      options.gpuLaunchWorkersPerNode = gpuLaunchWorkers;
      options.interleaveScatter = interleaveScatter;
      options.gpuProgress = gpuProgress;
      options.packetBytes = packetBytes;
      options.transportVectorBytes = transportVectorBytes;
      options.singleCtaBytes = singleCtaBytes;
      CpuRouter router(options);
      std::mt19937 rng(260940093);
      size_t checked = 0;
      int cases = 0;
      for (size_t count : counts)
        for (bool scatter : {false, true})
          for (bool inPlace : {false, true})
            for (bool masked : {false, true}) {
              if (masked && ranks == 1) continue;
              auto plan = scatter ? PacketProgram::reduceScatter(count, ranks)
                                  : PacketProgram::allReduce(count, ranks);
              int root = ranks / 2;
              if (masked) {
                plan.sources &= ~(1U << root);
                plan.destinations = 1U << root;
              }
              // Alternate aligned vector and deliberately unaligned scalar
              // paths.
              size_t offset = count % 8 ? 3 : guards;
              size_t extent = plan.inputElements + guards * 2;
              std::vector<std::vector<uint16_t>> host(
                  ranks, std::vector<uint16_t>(extent));
              std::vector<PacketPort> ports;
              for (int r = 0; r < ranks; ++r) {
                for (size_t i = 0; i < host[r].size(); ++i) {
                  auto& bits = host[r][i];
                  if (allEncodings) {
                    bits = uint16_t(i + r * 193);
                    continue;
                  }
                  unsigned v = rng();
                  // Bounded finite BF16 values with cancellation and varied
                  // exponents. Sum of <=4 terms is exactly representable in
                  // FP32, allowing bitwise comparison to an independent
                  // host-double oracle.
                  bits =
                      uint16_t(((v & 1) << 15) |
                               ((120 + (v >> 1) % (cpuReduce ? 8 : 15)) << 7) |
                               ((v >> 16) & 127));
                }
                check(cudaSetDevice(devices[r]));
                check(cudaMemcpyAsync(input[r], host[r].data(), extent * 2,
                                      cudaMemcpyHostToDevice, streams[r]));
                check(cudaMemsetAsync(output[r], 0xa5, extent * 2, streams[r]));
                auto* out = (inPlace ? input[r] : output[r]) + offset;
                if (inPlace && scatter) out += r * count;
                ports.push_back(
                    {plan.sources & (1U << r) ? input[r] + offset : nullptr,
                     plan.destinations & (1U << r) ? out : nullptr, streams[r],
                     devices[r]});
              }
              router.enqueue(plan, ports);
              for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(devices[r]));
                check(cudaStreamSynchronize(streams[r]));
                std::vector<uint16_t> actual(extent),
                    expected = inPlace ? host[r]
                                       : std::vector<uint16_t>(extent, 0xa5a5);
                if (plan.destinations & (1U << r)) {
                  for (size_t i = 0; i < count; ++i) {
                    size_t index = offset + (scatter ? r * count : 0) + i;
                    double sum = 0;
                    for (int peer = 0; peer < ranks; ++peer)
                      if (plan.sources & (1U << peer))
                        sum += double(__bfloat162float(
                            __ushort_as_bfloat16(host[peer][index])));
                    if (cpuReduce) {
                      // CPU placement rounds each NUMA partial, then the merged
                      // packet, then the retained local RS contribution.
                      float merged = 0;
                      for (int node : nodes) {
                        double partial = 0;
                        for (int peer = 0; peer < ranks; ++peer)
                          if (numa[peer] == node &&
                              (plan.sources & (1U << peer)) &&
                              (!scatter || peer != r))
                            partial += double(__bfloat162float(
                                __ushort_as_bfloat16(host[peer][index])));
                        merged += __bfloat162float(
                            __float2bfloat16_rn(float(partial)));
                      }
                      merged = __bfloat162float(__float2bfloat16_rn(merged));
                      if (scatter && (plan.sources & (1U << r)))
                        merged += __bfloat162float(
                            __ushort_as_bfloat16(host[r][index]));
                      sum = merged;
                    }
                    size_t target =
                        offset + (inPlace && scatter ? r * count : 0) + i;
                    expected[target] =
                        __bfloat16_as_ushort(__float2bfloat16_rn(float(sum)));
                  }
                }
                check(cudaMemcpy(actual.data(), inPlace ? input[r] : output[r],
                                 extent * 2, cudaMemcpyDeviceToHost));
                if (actual != expected) {
                  bool matches = allEncodings;
                  if (allEncodings) {
                    size_t first =
                        offset + (inPlace && scatter ? r * count : 0);
                    auto nan = [](uint16_t bits) {
                      return (bits & 0x7fffU) > 0x7f80U;
                    };
                    for (size_t i = 0; i < actual.size(); ++i) {
                      bool result = (plan.destinations & (1U << r)) &&
                                    i >= first && i - first < count;
                      // NaN payloads need not agree across CPU/GPU conversions.
                      // Every guard and non-destination value remains bitwise.
                      if (actual[i] != expected[i] &&
                          !(result && nan(actual[i]) && nan(expected[i]))) {
                        matches = false;
                        break;
                      }
                    }
                  }
                  if (!matches)
                    throw std::runtime_error(
                        "reduction differs from host-double oracle or "
                        "overwrote a guard");
                }
                checked += extent;
              }
              ++cases;
            }
      std::printf(
          "PASS: %d randomized reduction cases, %zu scalar/guard checks; "
          "ranks=%d all_encodings=%d\n",
          cases, checked, ranks, allEncodings);
    }
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(devices[r]));
      check(cudaStreamDestroy(streams[r]));
      check(cudaFree(input[r]));
      check(cudaFree(output[r]));
    }
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
