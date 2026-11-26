#include "cpu_router.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>

static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
enum Kind { AR, AG, RS, BC, RD };
__device__ uint16_t value(int rank, size_t index, int step) {
  float v = (rank & 1 ? -1.0f : 1.0f) * (rank + 1) * (1 + index % 7) / 64.0f +
            (step % 5) / 16.0f;
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}
__global__ void producer(uint16_t* input, size_t n, int rank, int step,
                         bool delayed) {
  if (delayed && threadIdx.x == 0) {
    auto start = clock64();
    while (clock64() - start < 200000) __nanosleep(64);
  }
  __syncthreads();
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    input[i] = value(rank, i, step);
}
__global__ void consumer(uint16_t const* output, size_t n, int rank, int step,
                         Kind kind, int* errors, int ranks) {
  size_t count = kind == AG ? n * ranks : n;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    uint16_t expected;
    if (kind == AG)
      expected = value(i / n, i % n, step);
    else if (kind == BC)
      expected = value(ranks - 1, i, step);
    else {
      size_t index = kind == RS ? size_t(rank) * n + i : i;
      float sum = 0;
      for (int peer = 0; peer < ranks; ++peer)
        sum += __bfloat162float(__ushort_as_bfloat16(value(peer, index, step)));
      expected = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
    }
    if (output[i] != expected) atomicAdd(errors, 1);
  }
}
int main(int argc, char** argv) {
  try {
    bool small = false;
    bool gpuReduce = false;
    bool singleNumaReduction = false;
    bool gpuProgress = false;
    bool agPerSource = false;
    bool interleaveScatter = false;
    bool batchCpuJobs = false, directScatterResults = false;
    int smallRsWorkers = 0;
    int gpuLaunchWorkers = 0;
    int delayRank = -1;
    size_t packetBytes = 512UL << 10;
    size_t minPacketBytes = 128UL << 10;
    size_t singleCtaBytes = 0;
    int ranks = 8;
    int transportThreads = 256;
    int transportBlocks = 64;
    int workersPerNode = 23;
    int transportVectorBytes = 16;
    std::vector<int> devices;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--devices" && i + 1 < argc) {
        std::stringstream list(argv[++i]);
        std::string device;
        while (std::getline(list, device, ','))
          devices.push_back(std::stoi(device));
      } else if (arg == "--gpus" && i + 1 < argc)
        ranks = std::stoi(argv[++i]);
      else if (arg == "small" || arg == "--small")
        small = true;
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
      } else if (arg == "--delay-rank" && i + 1 < argc)
        delayRank = std::stoi(argv[++i]);
      else if (arg == "--packet-bytes" && i + 1 < argc)
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
        throw std::invalid_argument("unknown router check option");
    }
    if (!devices.empty())
      ranks = int(devices.size());
    else
      for (int rank = 0; rank < ranks; ++rank) devices.push_back(rank);
    if (ranks < 1 || ranks > uccl_lite::cpu_router::kMaxRanks ||
        delayRank < -1 || delayRank >= ranks)
      throw std::invalid_argument("invalid rank count");
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
    if (gpuReduce)
      options.reduction = uccl_lite::cpu_router::ReductionPlacement::Gpu;
    options.maxTensorBytes = 32UL << 20;
    if (std::getenv("UCCL_LITE_TRANSPORT") &&
        std::string(std::getenv("UCCL_LITE_TRANSPORT")) == "dma")
      options.backend = uccl_lite::cpu_router::TransportBackend::Dma;
    uccl_lite::cpu_router::CpuRouter router(options);
    std::vector<uint16_t*> inputs(ranks), outputs(ranks);
    std::vector<int*> errors(ranks);
    std::vector<cudaStream_t> streams(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
      check(cudaSetDevice(devices[rank]));
      check(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
      check(cudaMalloc(&inputs[rank], options.maxTensorBytes + 32));
      check(cudaMalloc(&outputs[rank], options.maxTensorBytes + 32));
      check(cudaMalloc(&errors[rank], sizeof(int)));
      check(cudaMemsetAsync(errors[rank], 0, sizeof(int), streams[rank]));
    }
    std::vector<size_t> counts =
        small ? std::vector<size_t>{1, 3, 8, 17, 127, 512}
              : std::vector<size_t>{1,    3,    8,     17,     127,    512,
                                    4095, 4096, 32771, 262147, 1048583};
    int total = 0;
    auto run = [&](Kind kind, size_t n, bool inPlace, int step) {
      uccl_lite::cpu_router::RouterPlan plan =
          kind == AR   ? uccl_lite::cpu_router::RouterPlan::allReduce(n, ranks)
          : kind == AG ? uccl_lite::cpu_router::RouterPlan::allGather(n, ranks)
          : kind == RS
              ? uccl_lite::cpu_router::RouterPlan::reduceScatter(n, ranks)
          : kind == BC
              ? uccl_lite::cpu_router::RouterPlan::broadcast(n, ranks - 1,
                                                             ranks)
              : uccl_lite::cpu_router::RouterPlan::reduce(n, ranks / 2, ranks);
      std::vector<uccl_lite::cpu_router::RouterPort> ports(ranks);
      for (int rank = 0; rank < ranks; ++rank) {
        check(cudaSetDevice(devices[rank]));
        uint16_t* out = outputs[rank] + (step & 1);
        uint16_t* input = inPlace ? out : inputs[rank] + (step & 1);
        if (inPlace && kind == AG) input += rank * n;
        if (inPlace && kind == RS) out += rank * n;
        bool source = kind != BC || rank == ranks - 1,
             destination = kind != RD || rank == ranks / 2;
        if (source)
          producer<<<64, 256, 0, streams[rank]>>>(
              input, plan.inputElements, rank, step, rank == delayRank);
        ports[rank] = {source ? input : nullptr, destination ? out : nullptr,
                       streams[rank], devices[rank]};
      }
      router.enqueue(plan, ports);
      for (int rank = 0; rank < ranks; ++rank) {
        check(cudaSetDevice(devices[rank]));
        if (kind != RD || rank == ranks / 2)
          consumer<<<64, 256, 0, streams[rank]>>>(
              static_cast<const uint16_t*>(ports[rank].output), n, rank, step,
              kind, errors[rank], ranks);
        check(cudaGetLastError());
      }
      ++total;
    };
    for (int pass = 0; pass < (small ? 2 : 4); ++pass)
      for (size_t index = 0; index < counts.size(); ++index)
        for (Kind kind : {AR, AG, RS, BC, RD})
          run(kind, counts[pass & 1 ? counts.size() - 1 - index : index],
              (pass + index) & 1, total);
    if (!small)
      for (int i = 0; i < 512; ++i) run(Kind(i % 3), 32768, i & 1, total);
    for (int rank = 0; rank < ranks; ++rank) {
      check(cudaSetDevice(devices[rank]));
      int bad;
      check(cudaMemcpyAsync(&bad, errors[rank], sizeof(int),
                            cudaMemcpyDeviceToHost, streams[rank]));
      check(cudaStreamSynchronize(streams[rank]));
      if (bad)
        throw std::runtime_error("router GPU consumer found incorrect values");
    }
    std::printf(
        "BF16 router: %d mixed producer/collective/consumer cases passed, "
        "including concatenate, "
        "sum, scatter, broadcast and reduce masks; GPUs=%d\n",
        total, ranks);
    for (int rank = 0; rank < ranks; ++rank) {
      check(cudaSetDevice(devices[rank]));
      check(cudaFree(inputs[rank]));
      check(cudaFree(outputs[rank]));
      check(cudaFree(errors[rank]));
      check(cudaStreamDestroy(streams[rank]));
    }
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
