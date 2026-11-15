// Eight-GPU BF16 dense fusion: resident GEMM/CPU router versus separate
// cuBLAS/CPU router and cuBLAS/NCCL for AG+GEMM, GEMM+AR, and GEMM+RS.
#include "../cpu_router/compact_transport.cuh"
#include "uccl_lite/cpu_router/memory.hpp"
#include "uccl_lite/cpu_router/packet_pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <nccl.h>

using namespace uccl_lite::cpu_router;

static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void check(cublasStatus_t e) {
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS error " + std::to_string(int(e)));
}
static void check(ncclResult_t e) {
  if (e != ncclSuccess) throw std::runtime_error(ncclGetErrorString(e));
}

__global__ void fillA(__nv_bfloat16* p, size_t count, int k, int rowBase,
                      int source) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    p[i] = __float2bfloat16(float(source + 1) *
                            float(1 + ((rowBase + int(i / k)) % 4)) / 16.f);
}
__global__ void fillB(__nv_bfloat16* p, size_t count, int k, int rank) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    p[i] = __float2bfloat16(float(rank + 1) *
                            float(1 + ((i / k) % 4)) / 16.f);
}
__global__ void verify(__nv_bfloat16 const* p, size_t count, float scale,
                       int n, int localM, int rank, bool gather, bool scatter,
                       int* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    size_t global = scatter ? size_t(rank) * count + i : i;
    int row = int(global / n), col = int(global % n);
    float factor = gather ? float(row / localM + 1) * float(rank + 1) : 36.f;
    float expected = scale * factor * float(1 + row % 4) *
                     float(1 + col % 4);
    float actual = __bfloat162float(p[i]);
    if (!isfinite(actual) || fabsf(actual - expected) > expected * 0.02f)
      atomicAdd(errors, 1);
  }
}

struct Rank {
  RegisteredMemory a, gathered, b, partial, output, errors;
  cudaStream_t stream{};
  cudaEvent_t start{}, stop{};
  cublasHandle_t blas{};
  ncclComm_t nccl{};
};

int main(int argc, char** argv) {
  try {
    int m = 512, n = 4096, k = 4096, warmup = 2, iterations = 5,
        repeats = 3;
    int packetKib = 0, tileM = 0, tileN = 0, tileK = 0, stages = 0,
        transportBlocks = 64, fusedTransportCrew = 0, workersPerNode = 23,
        scatterWorkersPerNode = 16, transportGrainKib = 16, gemmMGroup = 0,
        selectedMethod = -1;
    bool gather = false, scatter = false, batchCpuJobs = false;
    std::string routerProfile = "historical";
    std::string samplePath;
    for (int i = 1; i < argc; ++i) {
      if (i + 1 >= argc) throw std::runtime_error("missing option value");
      std::string arg = argv[i++];
      if (arg == "--op") {
        std::string op = argv[i];
        if (op != "gemm-ar" && op != "gemm-rs" && op != "ag-gemm")
          throw std::runtime_error("--op must be gemm-ar, gemm-rs, or ag-gemm");
        gather = op == "ag-gemm";
        scatter = op == "gemm-rs";
        continue;
      }
      if (arg == "--samples") {
        samplePath = argv[i];
        continue;
      }
      if (arg == "--router-profile") {
        routerProfile = argv[i];
        if (routerProfile != "historical" && routerProfile != "current")
          throw std::runtime_error("invalid --router-profile");
        continue;
      }
      if (arg == "--method") {
        std::string method = argv[i];
        if (method == "nccl") selectedMethod = 0;
        else if (method == "router") selectedMethod = 1;
        else if (method == "fused") selectedMethod = 2;
        else if (method == "cublas-only") selectedMethod = 3;
        else if (method == "resident-only") selectedMethod = 4;
        else throw std::runtime_error("invalid --method");
        continue;
      }
      int value = std::atoi(argv[i]);
      if (arg == "--m") m = value;
      else if (arg == "--n") n = value;
      else if (arg == "--k") k = value;
      else if (arg == "--warmup") warmup = value;
      else if (arg == "--iterations") iterations = value;
      else if (arg == "--repeats") repeats = value;
      else if (arg == "--packet-kib") packetKib = value;
      else if (arg == "--tile-m") tileM = value;
      else if (arg == "--tile-n") tileN = value;
      else if (arg == "--tile-k") tileK = value;
      else if (arg == "--stages") stages = value;
      else if (arg == "--transport-blocks") transportBlocks = value;
      else if (arg == "--fused-transport-crew") fusedTransportCrew = value;
      else if (arg == "--workers-per-node") workersPerNode = value;
      else if (arg == "--scatter-workers-per-node")
        scatterWorkersPerNode = value;
      else if (arg == "--transport-grain-kib") transportGrainKib = value;
      else if (arg == "--gemm-m-group") gemmMGroup = value;
      else if (arg == "--batch-cpu-jobs")
        batchCpuJobs = value != 0;
      else throw std::runtime_error("unknown option " + arg);
    }
    if (m <= 0 || n <= 0 || k <= 0 || n % 8 || k % 8 ||
        (gather && m % 8) || warmup < 0 ||
        iterations < 1 || repeats < 1)
      throw std::runtime_error("invalid shape or repetition count");
    if (!packetKib) packetKib = gather || scatter ? 512 : 128;
    if (gather && m >= 512 && k >= 4096 && !tileM && !tileN &&
        !tileK && !stages) {
      tileM = 32;
      tileN = 128;
      tileK = 64;
      stages = 2;
    }
    if (packetKib < 128 || transportGrainKib < 1)
      throw std::runtime_error("invalid packet size");
    constexpr int ranks = 8;
    int available = 0;
    check(cudaGetDeviceCount(&available));
    if (available < ranks) throw std::runtime_error("eight GPUs required");
    char const* operation = gather    ? "ag-gemm"
                            : scatter ? "gemm-rs"
                                      : "gemm-ar";
    size_t aCount = size_t(gather ? m / ranks : m) * k,
           bCount = size_t(n) * k,
           cCount = size_t(m) * n,
           outputCount = scatter ? cCount / ranks : cCount;
    std::vector<Rank> rank(ranks);
    std::vector<int> devices(ranks);
    std::vector<PacketPort> ports(ranks);
    PacketPipeline::Binding binding(ranks);
    std::vector<MemorySpan> a(ranks), gatheredA(ranks), b(ranks),
        partial(ranks), output(ranks);
    for (int r = 0; r < ranks; ++r) {
      devices[r] = r;
      check(cudaSetDevice(r));
      auto& x = rank[r];
      x.a = RegisteredMemory::gpu(aCount * 2, r);
      if (gather) x.gathered = RegisteredMemory::gpu(size_t(m) * k * 2, r);
      x.b = RegisteredMemory::gpu(bCount * 2, r);
      x.partial = RegisteredMemory::gpu(cCount * 2, r);
      x.output = RegisteredMemory::gpu(cCount * 2, r);
      x.errors = RegisteredMemory::gpu(sizeof(int), r);
      check(cudaStreamCreateWithFlags(&x.stream, cudaStreamNonBlocking));
      check(cudaEventCreate(&x.start));
      check(cudaEventCreate(&x.stop));
      check(cublasCreate(&x.blas));
      check(cublasSetStream(x.blas, x.stream));
      fillA<<<32, 256, 0, x.stream>>>(
          static_cast<__nv_bfloat16*>(x.a.data()), aCount, k,
          gather ? r * (m / ranks) : 0, gather ? r : 0);
      fillB<<<32, 256, 0, x.stream>>>(
          static_cast<__nv_bfloat16*>(x.b.data()), bCount, k, r);
      a[r] = x.a.span();
      if (gather) gatheredA[r] = x.gathered.span();
      b[r] = x.b.span();
      partial[r] = x.partial.span();
      output[r] = x.output.span();
      ports[r] = {gather ? x.a.data() : x.partial.data(),
                  gather ? x.gathered.data() : x.output.data(), x.stream, r};
      binding.inputs[r] = gather ? x.a.span() : x.partial.span();
      binding.outputs[r] = gather ? x.gathered.span() : x.output.span();
      binding.streams[r] = x.stream;
      binding.devices[r] = r;
    }
    for (auto& x : rank) check(cudaStreamSynchronize(x.stream));
    std::vector<ncclComm_t> comms(ranks);
    check(ncclCommInitAll(comms.data(), ranks, devices.data()));
    for (int r = 0; r < ranks; ++r) rank[r].nccl = comms[r];
    PacketPipelineOptions options;
    options.packetBytes = size_t(packetKib) << 10;
    options.reduction = ReductionPlacement::Cpu;
    options.gemmTileM = tileM;
    options.gemmTileN = tileN;
    options.gemmTileK = tileK;
    options.gemmStages = stages;
    options.transportBlocks = transportBlocks;
    options.fusedTransportCrew = fusedTransportCrew;
    options.workersPerNode = workersPerNode;
    options.scatterWorkersPerNode = scatterWorkersPerNode;
    options.transportGrainBytes = size_t(transportGrainKib) << 10;
    options.gemmMGroup = gemmMGroup;
    options.batchCpuJobs = batchCpuJobs;
    auto router = std::make_unique<PacketPipeline>(options);
    router->prepare(ports);
    if (gather)
      router->prepareAllGatherGemm(m / ranks, n, k, a, gatheredA, b, output);
    else
      router->prepareGemm(m, n, k, a, b, partial,
                          scatter ? PacketLayout::Scatter
                                  : PacketLayout::Replicate);
    PacketProgram plan = gather
                             ? PacketProgram::allGather(aCount, ranks)
                             : scatter
                             ? PacketProgram::reduceScatter(outputCount, ranks)
                             : PacketProgram::allReduce(cCount, ranks);

    // Benchmark-only control. Keep resident GEMM preparation/occupancy and its
    // CPU reduction contract independent of the standalone collective schedule.
    // "current" follows compact_collectives' explicit benchmark schedule; it
    // does not install a production routing policy.
    std::unique_ptr<PacketPipeline> separateRouter;
    std::unique_ptr<uccl_lite::compact_experiment::Transport> mini;
    if (routerProfile == "current") {
      size_t bytes = (gather ? size_t(m) * k : cCount) * 2;
      PacketPipelineOptions standalone;
      standalone.packetBytes = 512UL << 10;
      standalone.batchCpuJobs = true;
      standalone.gpuLaunchWorkersPerNode = 4;
      if (gather) {
        standalone.reduction = ReductionPlacement::Gpu;
        standalone.gpuProgress = true;
        standalone.agPerSourceProgress = true;
        if (bytes >= (4UL << 20)) standalone.packetBytes = 2UL << 20;
      } else if (scatter) {
        if (bytes <= (4UL << 20) && bytes >= (256UL << 10))
          standalone.packetBytes = 128UL << 10;
        if (bytes <= (64UL << 10) || bytes == (128UL << 10)) {
          standalone.reduction = ReductionPlacement::Gpu;
          standalone.gpuProgress = true;
        } else if (bytes >= (1UL << 20) && bytes <= (8UL << 20)) {
          standalone.reduction = ReductionPlacement::Gpu;
          standalone.gpuProgress = true;
          standalone.hostRingReduceScatter = true;
          standalone.packetBytes = std::max(size_t(128UL << 10), bytes / ranks);
          standalone.transportBlocks = 16;
        } else if (bytes <= (1UL << 20)) {
          standalone.scatterWorkersPerNode = 4;
        }
      } else {
        if (bytes <= (4UL << 20)) standalone.packetBytes = 128UL << 10;
        if (bytes <= 8192) {
          standalone.reduction = ReductionPlacement::Gpu;
          standalone.gpuProgress = true;
        } else if (bytes <= (4UL << 20)) {
          standalone.workersPerNode = 4;
          standalone.transportBlocks = 16;
        }
      }
      if (gather && bytes >= (256UL << 10) && bytes <= (1UL << 20))
        mini = std::make_unique<uccl_lite::compact_experiment::Transport>(
            ports, bytes, 16);
      else {
        separateRouter = std::make_unique<PacketPipeline>(standalone);
        separateRouter->prepare(ports);
      }
    }
    std::printf("# router_profile=%s fused_batch_cpu_jobs=%d\n",
                routerProfile.c_str(), int(batchCpuJobs));

    auto sync = [&] {
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(r));
        check(cudaStreamSynchronize(rank[r].stream));
      }
    };
    auto gemm = [&] {
      float alpha = 1.f, beta = 0.f;
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(r));
        auto& x = rank[r];
        check(cublasGemmEx(x.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                           &alpha, x.b.data(), CUDA_R_16BF, k,
                           gather ? x.gathered.data() : x.a.data(),
                           CUDA_R_16BF, k,
                           &beta,
                           gather ? x.output.data() : x.partial.data(),
                           CUDA_R_16BF, n,
                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    };
    auto run = [&](int method) {
      // Reject stale output and unwritten/NaN values. Poisoning and its stream
      // completion are outside complete-operation timing.
      if (method < 3) {
        for (int r = 0; r < ranks; ++r) {
          check(cudaSetDevice(r));
          check(cudaMemsetAsync(rank[r].output.data(), 0xff, outputCount * 2,
                                rank[r].stream));
        }
        sync();
      }
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(r));
        check(cudaEventRecord(rank[r].start, rank[r].stream));
      }
      if (method == 0) {
        if (gather) {
          check(ncclGroupStart());
          for (int r = 0; r < ranks; ++r)
            check(ncclAllGather(rank[r].a.data(), rank[r].gathered.data(),
                                aCount, ncclBfloat16, rank[r].nccl,
                                rank[r].stream));
          check(ncclGroupEnd());
        }
        gemm();
        if (!gather) {
          check(ncclGroupStart());
          for (int r = 0; r < ranks; ++r)
            if (scatter)
              check(ncclReduceScatter(rank[r].partial.data(),
                                      rank[r].output.data(), outputCount,
                                      ncclBfloat16, ncclSum, rank[r].nccl,
                                      rank[r].stream));
            else
              check(ncclAllReduce(rank[r].partial.data(),
                                  rank[r].output.data(), cCount,
                                  ncclBfloat16, ncclSum, rank[r].nccl,
                                  rank[r].stream));
          check(ncclGroupEnd());
        }
      } else if (method == 1) {
        if (!gather) gemm();
        if (mini)
          mini->gather(aCount, ports);
        else if (separateRouter)
          separateRouter->enqueue(plan, binding);
        else
          router->enqueue(plan, binding);
        if (gather) gemm();
      } else if (method == 2) {
        router->enqueueGemm(plan, binding);
      } else if (method == 3) {
        gemm();
      } else {
        router->enqueueGemmProducerOnly(binding.streams);
      }
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(r));
        check(cudaEventRecord(rank[r].stop, rank[r].stream));
      }
      sync();
      float worst = 0.f;
      for (int r = 0; r < ranks; ++r) {
        float ms = 0.f;
        check(cudaEventElapsedTime(&ms, rank[r].start, rank[r].stop));
        worst = std::max(worst, ms);
      }
      if (method >= 3) return double(worst);
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(r));
        check(cudaMemsetAsync(rank[r].errors.data(), 0, sizeof(int),
                              rank[r].stream));
        verify<<<32, 256, 0, rank[r].stream>>>(
            static_cast<__nv_bfloat16 const*>(rank[r].output.data()), outputCount,
            float(k) / 256.f, n, m / ranks, r, gather, scatter,
            static_cast<int*>(rank[r].errors.data()));
      }
      sync();
      for (int r = 0; r < ranks; ++r) {
        check(cudaSetDevice(r));
        int errors = 0;
        check(cudaMemcpy(&errors, rank[r].errors.data(), sizeof(int),
                         cudaMemcpyDeviceToHost));
        if (errors)
          throw std::runtime_error("wrong result at rank " + std::to_string(r) +
                                   ": " + std::to_string(errors));
      }
      return double(worst);
    };
    char const* names[] = {"cublas+nccl", "cublas+router", "fused+router",
                           "cublas-only", "resident-only"};
    std::vector<double> samples[5];
    std::ofstream sampleFile;
    if (!samplePath.empty()) {
      std::ifstream existing(samplePath);
      bool hasContent = existing.peek() != std::ifstream::traits_type::eof();
      sampleFile.open(samplePath, std::ios::app);
      if (!sampleFile) throw std::runtime_error("cannot open sample file");
      if (!hasContent)
        sampleFile << "op,m,n,k,repeat,iteration,method,latency_ms\n";
    }
    for (int round = 0; round < warmup + repeats; ++round) {
      for (int order = 0; order < (selectedMethod >= 3 ? 5 : 3); ++order) {
        int method = (round + order) % (selectedMethod >= 3 ? 5 : 3);
        if (selectedMethod >= 0 && method != selectedMethod) continue;
        for (int iter = 0; iter < (round < warmup ? 1 : iterations); ++iter) {
          double ms = run(method);
          if (round >= warmup) {
            samples[method].push_back(ms);
            if (sampleFile)
              sampleFile << operation << ',' << m << ','
                         << n << ',' << k << ',' << round - warmup << ','
                         << iter << ',' << names[method] << ',' << ms << '\n';
          }
        }
      }
    }
    for (int method = 0; method < 5; ++method) {
      auto& v = samples[method];
      if (v.empty()) continue;
      std::sort(v.begin(), v.end());
      std::printf("op=%s shape=%dx%dx%d method=%s packet_kib=%d tile=%dx%dx%d stages=%d blocks=%d workers=%d mgroup=%d min_ms=%.4f median_ms=%.4f max_ms=%.4f samples=%zu\n",
                  operation, m, n, k, names[method], packetKib,
                  tileM, tileN, tileK, stages, transportBlocks, workersPerNode,
                  gemmMGroup,
                  v.front(), v[v.size() / 2],
                  v.back(), v.size());
    }
    mini.reset();
    separateRouter.reset();
    router.reset();
    for (int r = 0; r < ranks; ++r) {
      check(cudaSetDevice(r));
      check(ncclCommDestroy(rank[r].nccl));
      check(cublasDestroy(rank[r].blas));
      check(cudaEventDestroy(rank[r].start));
      check(cudaEventDestroy(rank[r].stop));
      check(cudaStreamDestroy(rank[r].stream));
    }
  } catch (std::exception const& e) {
    std::fprintf(stderr, "router_fusion_8l4: %s\n", e.what());
    return 1;
  }
}
