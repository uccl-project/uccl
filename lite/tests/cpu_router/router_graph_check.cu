#include "cpu_router.hpp"
#include "cpu_router_c.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
static void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void fill(__nv_bfloat16* x, size_t count, float value = 1.0f) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    x[i] = __float2bfloat16(value);
}
__global__ void validate(__nv_bfloat16 const* x, size_t count, float expected,
                         int* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    if (__bfloat162float(x[i]) != expected) atomicAdd(errors, 1);
}
int main(int argc, char** argv) {
  try {
    bool large = false;
    bool gpuReduce = false;
    bool singleNumaReduction = false;
    bool gpuProgress = false;
    bool agPerSource = false;
    bool interleaveScatter = false;
    bool batchCpuJobs = false, directScatterResults = false;
    int smallRsWorkers = 0;
    int gpuLaunchWorkers = 0;
    size_t packetBytes = 512UL << 10;
    size_t minPacketBytes = 128UL << 10;
    size_t singleCtaBytes = 0;
    size_t contributionElements = 64;
    bool customElements = false;
    int ranks = 8;
    int transportThreads = 256;
    int transportBlocks = 64;
    int workersPerNode = 23;
    int transportVectorBytes = 16;
    std::vector<int> devices;
    bool capi = false, legacy = false;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--devices" && i + 1 < argc) {
        std::stringstream list(argv[++i]);
        std::string device;
        while (std::getline(list, device, ','))
          devices.push_back(std::stoi(device));
      } else if (arg == "--gpus" && i + 1 < argc)
        ranks = std::stoi(argv[++i]);
      else if (arg == "--elements" && i + 1 < argc) {
        contributionElements = std::stoull(argv[++i]);
        customElements = true;
      } else if (arg == "--large")
        large = true;
      else if (arg == "--gpu-reduce")
        gpuReduce = true;
      else if (arg == "--cpu-interleave-scatter")
        interleaveScatter = true;
      else if (arg == "--batch-cpu-jobs")
        batchCpuJobs = true;
      else if (arg == "--gpu-launch-workers-per-node" && i + 1 < argc)
        gpuLaunchWorkers = std::stoi(argv[++i]);
      else if (arg == "--direct-scatter-results")
        directScatterResults = true;
      else if (arg == "--small-rs-workers-per-node" && i + 1 < argc)
        smallRsWorkers = std::stoi(argv[++i]);
      else if (arg == "--single-numa-reduction")
        singleNumaReduction = true;
      else if (arg == "--gpu-progress") {
        gpuReduce = true;
        gpuProgress = true;
      } else if (arg == "--ag-per-source") {
        gpuReduce = true;
        gpuProgress = true;
        agPerSource = true;
      } else if (arg == "--interleave-scatter") {
        gpuReduce = true;
        interleaveScatter = true;
      } else if (arg == "--c-api")
        capi = true;
      else if (arg == "--c-api-legacy") {
        capi = true;
        legacy = true;
      } else if (arg == "--packet-bytes" && i + 1 < argc)
        packetBytes = std::stoull(argv[++i]);
      else if (arg == "--min-packet-bytes" && i + 1 < argc)
        minPacketBytes = std::stoull(argv[++i]);
      else if (arg == "--transport-threads" && i + 1 < argc)
        transportThreads = std::stoi(argv[++i]);
      else if (arg == "--transport-blocks" && i + 1 < argc)
        transportBlocks = std::stoi(argv[++i]);
      else if (arg == "--workers-per-node" && i + 1 < argc)
        workersPerNode = std::stoi(argv[++i]);
      else if (arg == "--transport-vector-bytes" && i + 1 < argc)
        transportVectorBytes = std::stoi(argv[++i]);
      else if (arg == "--single-cta-bytes" && i + 1 < argc)
        singleCtaBytes = std::stoull(argv[++i]);
      else
        throw std::invalid_argument("unknown graph check argument");
    }
    if (!devices.empty())
      ranks = int(devices.size());
    else
      for (int rank = 0; rank < ranks; ++rank) devices.push_back(rank);
    if (ranks < 1 || ranks > uccl_lite::cpu_router::kMaxRanks ||
        (legacy && ranks != 8))
      throw std::invalid_argument("invalid graph rank count");
    if (large && !customElements) contributionElements = 1UL << 19;
    if (!contributionElements || contributionElements > (1UL << 29) / ranks)
      throw std::invalid_argument("invalid graph contribution size");
    size_t extent = std::max<size_t>(1024, contributionElements * ranks);
    uccl_lite::cpu_router::CpuRouterOptions options;
    options.interleaveScatter = interleaveScatter;
    options.batchCpuJobs = batchCpuJobs;
    options.directScatterResults = directScatterResults;
    options.gpuLaunchWorkersPerNode = gpuLaunchWorkers;
    options.smallReduceScatterWorkersPerNode = smallRsWorkers;
    options.gpuProgress = gpuProgress;
    options.agPerSourceProgress = agPerSource;
    options.packetBytes = packetBytes;
    options.minPacketBytes = minPacketBytes;
    options.transportThreads = transportThreads;
    options.transportBlocks = transportBlocks;
    options.workersPerNode = workersPerNode;
    options.transportVectorBytes = transportVectorBytes;
    options.singleCtaBytes = singleCtaBytes;
    options.singleNumaReduction = singleNumaReduction;
    if (gpuReduce && capi)
      throw std::invalid_argument(
          "GPU reduction is currently a C++-only option");
    if (capi && (interleaveScatter || batchCpuJobs || directScatterResults ||
                 smallRsWorkers || gpuLaunchWorkers))
      throw std::invalid_argument(
          "CPU routing experiments are C++-only options");
    if (gpuReduce)
      options.reduction = uccl_lite::cpu_router::ReductionPlacement::Gpu;
    if (std::getenv("UCCL_LITE_INLINE_REPLAY")) options.inlineReplay = true;
    if (std::getenv("UCCL_LITE_TRANSPORT") &&
        std::string(std::getenv("UCCL_LITE_TRANSPORT")) == "dma")
      options.backend = uccl_lite::cpu_router::TransportBackend::Dma;
    auto router = std::make_unique<uccl_lite::cpu_router::CpuRouter>(options);
    std::vector<uccl_lite::cpu_router::RouterPort> ports(ranks);
    std::vector<cudaGraph_t> graphs(ranks);
    std::vector<cudaGraphExec_t> executable(ranks);
    std::vector<int*> errors(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
      check(cudaSetDevice(devices[rank]));
      void *input, *output;
      check(cudaMalloc(&input, extent * 2));
      check(cudaMalloc(&output, extent * 2));
      cudaStream_t stream;
      check(cudaStreamCreate(&stream));
      check(cudaMalloc(&errors[rank], sizeof(int)));
      check(cudaMemset(errors[rank], 0, sizeof(int)));
      ports[rank] = {input, output, stream, devices[rank]};
      fill<<<4, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(input), extent);
    }
    auto cRouter = std::unique_ptr<void, decltype(&uccl_lite_router_destroy)>(
        capi ? uccl_lite_router_create() : nullptr, &uccl_lite_router_destroy);
    if (capi && !cRouter)
      throw std::runtime_error(uccl_lite_router_last_error());
    auto enqueue = [&](uccl_lite::cpu_router::PacketProgram const& plan,
                       int operation) {
      if (!capi) {
        router->enqueue(plan, ports);
        return;
      }
      std::vector<const void*> input(ranks);
      std::vector<void*> output(ranks), streams(ranks);
      for (int rank = 0; rank < ranks; ++rank) {
        input[rank] = ports[rank].input;
        output[rank] = ports[rank].output;
        streams[rank] = ports[rank].stream;
      }
      int status = legacy ? uccl_lite_router_enqueue(
                                cRouter.get(), operation, contributionElements,
                                input.data(), output.data(), streams.data(),
                                devices.data())
                          : uccl_lite_router_enqueue_n(
                                cRouter.get(), operation, contributionElements,
                                ranks, input.data(), output.data(),
                                streams.data(), devices.data());
      if (status) throw std::runtime_error(uccl_lite_router_last_error());
    };
    auto control = [&](int action) {
      if (capi) {
        if (uccl_lite_router_graph_control(cRouter.get(), action))
          throw std::runtime_error(uccl_lite_router_last_error());
      } else if (action == 0)
        router->beginCapture();
      else if (action == 1)
        router->endCapture();
      else if (action == 2)
        router->beginReplay();
      else
        router->endReplay();
    };
    for (int op = 0; op < 3; ++op) {
      auto plan = op == 0 ? uccl_lite::cpu_router::RouterPlan::allReduce(
                                contributionElements, ranks)
                  : op == 1 ? uccl_lite::cpu_router::RouterPlan::allGather(
                                  contributionElements, ranks)
                            : uccl_lite::cpu_router::RouterPlan::reduceScatter(
                                  contributionElements, ranks);
      enqueue(plan, op);
    }
    for (int rank = 0; rank < ranks; ++rank) {
      auto& p = ports[rank];
      check(cudaSetDevice(p.device));
      check(cudaStreamSynchronize(p.stream));
    }
    control(0);
    for (int rank = 0; rank < ranks; ++rank) {
      auto& p = ports[rank];
      check(cudaSetDevice(p.device));
      check(cudaStreamBeginCapture(p.stream, cudaStreamCaptureModeRelaxed));
    }
    for (int call = 0; call < 200; ++call) {
      int op = call % 3;
      float value = float(call % 5 + 1);
      auto plan = op == 0 ? uccl_lite::cpu_router::RouterPlan::allReduce(
                                contributionElements, ranks)
                  : op == 1 ? uccl_lite::cpu_router::RouterPlan::allGather(
                                  contributionElements, ranks)
                            : uccl_lite::cpu_router::RouterPlan::reduceScatter(
                                  contributionElements, ranks);
      for (int rank = 0; rank < ranks; ++rank) {
        auto& p = ports[rank];
        check(cudaSetDevice(p.device));
        fill<<<4, 256, 0, p.stream>>>(
            static_cast<__nv_bfloat16*>(const_cast<void*>(p.input)), extent,
            value);
      }
      enqueue(plan, op);
      for (int rank = 0; rank < ranks; ++rank) {
        auto& p = ports[rank];
        check(cudaSetDevice(p.device));
        validate<<<4, 256, 0, p.stream>>>(
            static_cast<__nv_bfloat16*>(p.output),
            op == 1 ? contributionElements * ranks : contributionElements,
            (op == 1 ? 1.0f : float(ranks)) * value, errors[rank]);
      }
    }
    for (int rank = 0; rank < ranks; ++rank) {
      auto& p = ports[rank];
      check(cudaSetDevice(p.device));
      check(cudaStreamEndCapture(p.stream, &graphs[rank]));
      check(cudaGraphInstantiate(&executable[rank], graphs[rank], 0));
    }
    control(1);
    for (int replay = 0; replay < 5; ++replay) {
      control(2);
      for (int rank = 0; rank < ranks; ++rank) {
        auto& p = ports[rank];
        check(cudaSetDevice(p.device));
        check(cudaGraphLaunch(executable[rank], p.stream));
      }
      for (int rank = 0; rank < ranks; ++rank) {
        auto& p = ports[rank];
        check(cudaSetDevice(p.device));
        check(cudaStreamSynchronize(p.stream));
      }
      control(3);
    }
    int wrong = 0;
    if (ranks == 1 && !capi) {
      for (int call = 0; call < 150; ++call) {
        uccl_lite::cpu_router::PacketPipeline::Binding binding(1);
        auto input =
            uccl_lite::cpu_router::RegisteredMemory::gpu(128, devices[0]);
        binding.inputs[0] = input.span();
        binding.outputs[0] = uccl_lite::cpu_router::MemorySpan::borrowedGpu(
            ports[0].output, 128, devices[0]);
        binding.devices[0] = devices[0];
        binding.streams[0] = ports[0].stream;
        fill<<<1, 64, 0, ports[0].stream>>>(
            static_cast<__nv_bfloat16*>(input.data()), 64);
        router->enqueue(uccl_lite::cpu_router::PacketProgram::allReduce(64, 1),
                        binding);
        validate<<<1, 64, 0, ports[0].stream>>>(
            static_cast<__nv_bfloat16 const*>(ports[0].output), 64, 1.f,
            errors[0]);
      }
    }
    for (int rank = 0; rank < ranks; ++rank) {
      auto& p = ports[rank];
      check(cudaSetDevice(p.device));
      int local;
      check(cudaMemcpy(&local, errors[rank], sizeof(int),
                       cudaMemcpyDeviceToHost));
      wrong += local;
      check(cudaGraphExecDestroy(executable[rank]));
      check(cudaGraphDestroy(graphs[rank]));
    }
    cRouter.reset();
    router.reset();
    for (int rank = 0; rank < ranks; ++rank) {
      auto& p = ports[rank];
      check(cudaSetDevice(p.device));
      check(cudaFree(const_cast<void*>(p.input)));
      check(cudaFree(p.output));
      check(cudaFree(errors[rank]));
      check(cudaStreamDestroy(p.stream));
    }
    if (wrong) throw std::runtime_error("graph validation failed");
    std::printf(
        "PASS: 1000 captured mixed collectives, producer/consumer "
        "dependencies, five replays and "
        "128-entry ring wrap; GPUs=%d C_API=%s\n",
        ranks, capi ? (legacy ? "legacy" : "count-aware") : "false");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
