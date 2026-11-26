// Standalone BF16 AR/AG/RS dispatcher and paired local calibration. The
// project-owned two-NUMA collective is the default fallback. NCCL is an
// explicit external comparison; the optional router module does not link it.
#include "nccl_identity.hpp"
#include "uccl_lite/collective_dispatch.hpp"
#include "uccl_lite/local_collectives.hpp"
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <nccl.h>
#include <sched.h>

using namespace uccl_lite;
extern char** environ;
namespace {
void cudaCheck(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void ncclCheck(ncclResult_t e) {
  if (e != ncclSuccess) throw std::runtime_error(ncclGetErrorString(e));
}
__global__ void fill(__nv_bfloat16* data, size_t count, int rank) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    data[i] = __float2bfloat16(float((rank + 1) * (1 + i % 7)) / 64.0f);
}
__global__ void verify(__nv_bfloat16 const* data, size_t count, int rank,
                       int ranks, int operation, size_t contribution,
                       int* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    size_t index = operation == 2 ? rank * contribution + i : i;
    float expected =
        operation == 1
            ? float((i / contribution + 1) * (1 + (i % contribution) % 7)) /
                  64.0f
            : float(ranks * (ranks + 1) / 2 * (1 + index % 7)) / 64.0f;
    if (__bfloat162float(data[i]) != expected) atomicAdd(errors, 1);
  }
}
struct Config {
  std::vector<int> devices;
  int gpus = 8, warmup = 3, iterations = 10, repeats = 3;
  size_t begin = 1024, end = 1 << 20;
  size_t packetBytes = 512UL << 10;
  size_t minPacketBytes = 128UL << 10;
  size_t largeAllGatherPacketBytes = 0;
  size_t smallAllReducePacketBytes = 128UL << 10;
  size_t smallReduceScatterPacketBytes = 128UL << 10;
  int workersPerNode = 23, transportBlocks = 64;
  int transportThreads = 256, scatterWorkersPerNode = 16;
  int smallReduceScatterWorkersPerNode = 0;
  bool batchCpuJobs = false, directScatterResults = false;
  int gpuLaunchWorkersPerNode = 0;
  int transportVectorBytes = 16;
  int mainCpu = -1;
  size_t transportGrainBytes = 16UL << 10, grainElements = 8192;
  size_t singleCtaBytes = 0;
  bool smallReduceScatterGpuProgress = true;
  size_t smallReduceScatterGpuProgressMaxBytes = 0;
  bool inPlace = false;
  bool gpuReduce = false;
  bool gpuProgress = false;
  bool agPerSourceProgress = false;
  bool singleNumaReduction = false;
  bool interleaveScatter = false;
  bool shmOnly = false;
  bool reverse = false;
  bool routerOnly = false;
  std::string operation = "all", fallback = "local", profile, calibrate,
              samples;
  RouterMode mode = RouterMode::Auto;
};
struct Buffers {
  std::vector<void*> inputs, outputs;
  std::vector<int*> errors;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> start, stop;
  std::vector<int> devices;
  Buffers(std::vector<int> const& ids, size_t bytes) : devices(ids) {
    for (int id : ids) {
      cudaCheck(cudaSetDevice(id));
      void *in, *out;
      int* err;
      cudaStream_t stream;
      cudaEvent_t a, b;
      cudaCheck(cudaMalloc(&in, bytes));
      cudaCheck(cudaMalloc(&out, bytes));
      cudaCheck(cudaMalloc(&err, sizeof(int)));
      cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      cudaCheck(cudaEventCreate(&a));
      cudaCheck(cudaEventCreate(&b));
      inputs.push_back(in);
      outputs.push_back(out);
      errors.push_back(err);
      streams.push_back(stream);
      start.push_back(a);
      stop.push_back(b);
    }
  }
  ~Buffers() {
    for (size_t r = 0; r < devices.size(); ++r) {
      cudaSetDevice(devices[r]);
      cudaStreamSynchronize(streams[r]);
      cudaFree(inputs[r]);
      cudaFree(outputs[r]);
      cudaFree(errors[r]);
      cudaEventDestroy(start[r]);
      cudaEventDestroy(stop[r]);
      cudaStreamDestroy(streams[r]);
    }
  }
  CollectiveCall call(Collective op, size_t bytes, bool inPlace) {
    CollectiveCall result;
    result.collective = op;
    result.inPlace = inPlace;
    result.count =
        bytes / 2 / (op == Collective::AllReduce ? 1 : devices.size());
    for (size_t rank = 0; rank < devices.size(); ++rank) {
      auto* base =
          static_cast<__nv_bfloat16*>(inPlace ? outputs[rank] : inputs[rank]);
      auto* out = static_cast<__nv_bfloat16*>(outputs[rank]);
      auto* in = base;
      if (inPlace && op == Collective::AllGather) in += rank * result.count;
      if (inPlace && op == Collective::ReduceScatter)
        out += rank * result.count;
      result.ports.push_back({in, out, streams[rank], devices[rank]});
    }
    return result;
  }
};
TimingRange summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  return {samples.front(),
          n % 2 ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2,
          samples.back()};
}
double run(CollectiveDispatcher& dispatch, CollectiveCall const& call,
           Buffers& buffers, int iterations, bool timed,
           std::ostream* samples = nullptr, char const* backend = "",
           int repeat = -1) {
  static unsigned idleUs = std::getenv("UCCL_LITE_BENCH_IDLE_US") ? std::stoul(std::getenv("UCCL_LITE_BENCH_IDLE_US")) : 0;
  std::vector<double> latencies;
  int ranks = int(call.ports.size());
  size_t inputCount =
      call.count * (call.collective == Collective::ReduceScatter ? ranks : 1);
  size_t outputCount =
      call.count * (call.collective == Collective::AllGather ? ranks : 1);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (idleUs) std::this_thread::sleep_for(std::chrono::microseconds(idleUs));
    for (int rank = 0; rank < ranks; ++rank) {
      auto const& p = call.ports[rank];
      cudaCheck(cudaSetDevice(p.device));
      fill<<<32, 256, 0, p.stream>>>(
          static_cast<__nv_bfloat16*>(const_cast<void*>(p.input)), inputCount,
          rank);
      cudaCheck(
          cudaMemsetAsync(buffers.errors[rank], 0, sizeof(int), p.stream));
      cudaCheck(cudaEventRecord(buffers.start[rank], p.stream));
    }
    dispatch.enqueue(call);
    for (int rank = 0; rank < ranks; ++rank) {
      auto const& p = call.ports[rank];
      cudaCheck(cudaSetDevice(p.device));
      cudaCheck(cudaEventRecord(buffers.stop[rank], p.stream));
      verify<<<32, 256, 0, p.stream>>>(
          static_cast<__nv_bfloat16 const*>(p.output), outputCount, rank, ranks,
          int(call.collective), call.count, buffers.errors[rank]);
      cudaCheck(cudaGetLastError());
    }
    double worst = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      cudaCheck(cudaSetDevice(call.ports[rank].device));
      cudaCheck(cudaStreamSynchronize(call.ports[rank].stream));
      int errors = 0;
      cudaCheck(cudaMemcpy(&errors, buffers.errors[rank], sizeof(int),
                           cudaMemcpyDeviceToHost));
      if (errors)
        throw std::runtime_error("collective correctness failed on rank " +
                                 std::to_string(rank));
      float ms;
      cudaCheck(
          cudaEventElapsedTime(&ms, buffers.start[rank], buffers.stop[rank]));
      worst = std::max(worst, double(ms) * 1000);
    }
    if (timed) {
      latencies.push_back(worst);
      if (samples)
        *samples << collectiveName(call.collective) << ','
                 << call.workload().bytes << ',' << call.inPlace << ','
                 << backend << ',' << repeat << ',' << iteration << ',' << worst
                 << '\n';
    }
  }
  return timed ? summarize(latencies).median : 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto next = [&]() {
        if (++i >= argc) throw std::invalid_argument("missing option value");
        return std::string(argv[i]);
      };
      if (arg == "--gpus")
        cfg.gpus = std::stoi(next());
      else if (arg == "--devices") {
        std::istringstream list(next());
        std::string d;
        while (std::getline(list, d, ',')) cfg.devices.push_back(std::stoi(d));
      } else if (arg == "--begin")
        cfg.begin = std::stoull(next());
      else if (arg == "--end")
        cfg.end = std::stoull(next());
      else if (arg == "--packet-bytes")
        cfg.packetBytes = std::stoull(next());
      else if (arg == "--min-packet-bytes")
        cfg.minPacketBytes = std::stoull(next());
      else if (arg == "--large-ag-packet-bytes")
        cfg.largeAllGatherPacketBytes = std::stoull(next());
      else if (arg == "--small-ar-packet-bytes")
        cfg.smallAllReducePacketBytes = std::stoull(next());
      else if (arg == "--small-rs-packet-bytes")
        cfg.smallReduceScatterPacketBytes = std::stoull(next());
      else if (arg == "--workers-per-node")
        cfg.workersPerNode = std::stoi(next());
      else if (arg == "--transport-blocks")
        cfg.transportBlocks = std::stoi(next());
      else if (arg == "--transport-threads")
        cfg.transportThreads = std::stoi(next());
      else if (arg == "--transport-vector-bytes")
        cfg.transportVectorBytes = std::stoi(next());
      else if (arg == "--main-cpu")
        cfg.mainCpu = std::stoi(next());
      else if (arg == "--transport-grain-bytes")
        cfg.transportGrainBytes = std::stoull(next());
      else if (arg == "--single-cta-bytes")
        cfg.singleCtaBytes = std::stoull(next());
      else if (arg == "--cpu-grain-elements")
        cfg.grainElements = std::stoull(next());
      else if (arg == "--scatter-workers-per-node")
        cfg.scatterWorkersPerNode = std::stoi(next());
      else if (arg == "--small-rs-workers-per-node")
        cfg.smallReduceScatterWorkersPerNode = std::stoi(next());
      else if (arg == "--batch-cpu-jobs")
        cfg.batchCpuJobs = true;
      else if (arg == "--direct-scatter-results")
        cfg.directScatterResults = true;
      else if (arg == "--gpu-launch-workers-per-node")
        cfg.gpuLaunchWorkersPerNode = std::stoi(next());
      else if (arg == "--cpu-interleave-scatter")
        cfg.interleaveScatter = true;
      else if (arg == "--no-small-rs-gpu-progress")
        cfg.smallReduceScatterGpuProgress = false;
      else if (arg == "--small-rs-gpu-progress-max-bytes")
        cfg.smallReduceScatterGpuProgressMaxBytes = std::stoull(next());
      else if (arg == "--single-numa-reduction")
        cfg.singleNumaReduction = true;
      else if (arg == "--op")
        cfg.operation = next();
      else if (arg == "--fallback")
        cfg.fallback = next();
      else if (arg == "--backend")
        cfg.mode = parseRouterMode(next());
      else if (arg == "--router-only")
        cfg.routerOnly = true;
      else if (arg == "--profile")
        cfg.profile = next();
      else if (arg == "--calibrate")
        cfg.calibrate = next();
      else if (arg == "--samples")
        cfg.samples = next();
      else if (arg == "--in-place")
        cfg.inPlace = true;
      else if (arg == "--shm-only")
        cfg.shmOnly = true;
      else if (arg == "--reverse")
        cfg.reverse = true;
      else if (arg == "--gpu-reduce")
        cfg.gpuReduce = true;
      else if (arg == "--gpu-progress") {
        cfg.gpuReduce = true;
        cfg.gpuProgress = true;
      } else if (arg == "--ag-per-source") {
        cfg.gpuReduce = true;
        cfg.gpuProgress = true;
        cfg.agPerSourceProgress = true;
      } else if (arg == "--interleave-scatter") {
        cfg.gpuReduce = true;
        cfg.interleaveScatter = true;
      } else if (arg == "--warmup")
        cfg.warmup = std::stoi(next());
      else if (arg == "--iterations")
        cfg.iterations = std::stoi(next());
      else if (arg == "--repeats")
        cfg.repeats = std::stoi(next());
      else if (arg == "--help") {
        std::puts(
            "UCCL Lite BF16 collectives (one process owns all selected GPUs)\n"
            "  --gpus N | --devices 0,1,...\n  --op "
            "all|allreduce|allgather|reducescatter\n"
            "  --backend auto|off|force  --fallback local|nccl (default: "
            "local)\n"
            "  --router-only: force UCCL router at every size; do not "
            "initialize NCCL or a fallback\n"
            "  --shm-only: require NCCL SHM transport, disable all logical "
            "GPU peer links, and reject the P2P local fallback\n"
            "  --reverse: sweep message sizes from large to small\n"
            "  local requires two peer-accessible four-GPU NUMA groups\n"
            "  --begin BYTES --end BYTES --in-place\n"
            "  --profile FILE  --calibrate NEW_FILE\n"
            "  --samples NEW_CSV (checked per-invocation timing records)\n"
            "  --gpu-reduce (experimental destination-GPU Sum arithmetic)\n"
            "  --gpu-progress (GPU upload/credit handoff; implies GPU "
            "reduction)\n"
            "  --interleave-scatter (GPU reduction with all-destination packet "
            "stripes)\n"
            "  --packet-bytes BYTES (upper bound; default 524288, minimum "
            "131072)\n"
            "  --min-packet-bytes BYTES (lower geometry bound; default "
            "131072)\n"
            "  --large-ag-packet-bytes BYTES (AllGather >=4 MiB; 0 disables)\n"
            "  --small-ar-packet-bytes BYTES (default 131072 through 4 MiB; "
            "0 disables)\n"
            "  --small-rs-packet-bytes BYTES (default 131072 for "
            "256 KiB to 4 MiB; 0 disables)\n"
            "  --workers-per-node N --transport-blocks N (CPU router "
            "tuning)\n"
            "  --transport-threads N (32..1024, multiple of 32)\n"
            "  --transport-vector-bytes 16|32 (native 256-bit on Blackwell)\n"
            "  --main-cpu N (pin submission after topology/thread "
            "initialization)\n"
            "  --transport-grain-bytes N --cpu-grain-elements N\n"
            "  --scatter-workers-per-node N (0 uses all workers)\n"
            "  --small-rs-workers-per-node N (CPU RS 256 KiB--1 MiB; 0 "
            "preserves the regular cap)\n"
            "  --batch-cpu-jobs (publish RS jobs before waking workers)\n"
            "  --direct-scatter-results (CPU merge directly into result)\n"
            "  --gpu-launch-workers-per-node N (eager GPU submission lanes; "
            "0 uses caller)\n"
            "  --cpu-interleave-scatter (CPU reduction, destination stripes)\n"
            "  --ag-per-source (GPU progress with per-source AG readiness)\n"
            "  --single-numa-reduction (direct CPU result packets on one NUMA "
            "node)\n"
            "  --single-cta-bytes N (one upload/consume CTA through N bytes)\n"
            "  --no-small-rs-gpu-progress (disable 128-KiB RS GPU progress)\n"
            "  --small-rs-gpu-progress-max-bytes N (GPU progress through N "
            "bytes, up to 64 KiB)\n"
            "  --warmup N --iterations N --repeats N (at least 3)\n"
            "Auto keeps the selected fallback unless matching validated "
            "evidence shows a stable "
            "gain.\n"
            "Calibration alternates forced CPU-router and the selected "
            "fallback; checks every "
            "invocation.");
        return 0;
      } else
        throw std::invalid_argument("unknown option: " + arg);
    }
    if (cfg.devices.empty())
      for (int r = 0; r < cfg.gpus; ++r) cfg.devices.push_back(r);
    if (cfg.fallback != "local" && cfg.fallback != "nccl")
      throw std::invalid_argument("fallback must be local or nccl");
    if (cfg.routerOnly) {
      if (cfg.fallback != "local" || cfg.shmOnly || !cfg.profile.empty() ||
          !cfg.calibrate.empty() || cfg.mode == RouterMode::Off)
        throw std::invalid_argument(
            "--router-only cannot use a fallback, profile, calibration or "
            "disabled router");
      cfg.mode = RouterMode::Force;
    }
    if (cfg.shmOnly) {
      if (cfg.devices != std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7})
        throw std::invalid_argument(
            "--shm-only requires devices 0,1,2,3,4,5,6,7");
      auto required = [](char const* key, char const* value) {
        auto* current = std::getenv(key);
        if (!current || std::string(current) != value)
          throw std::invalid_argument(std::string("--shm-only requires ") +
                                      key + "=" + value);
      };
      required("NCCL_P2P_DISABLE", "1");
      required("NCCL_SHM_DISABLE", "0");
      required("NCCL_IB_DISABLE", "1");
      required("NCCL_NET", "Socket");
      if (cfg.fallback != "nccl")
        throw std::invalid_argument(
            "--shm-only requires the NCCL control, not the P2P local path");
      if (cfg.calibrate.empty() && cfg.mode == RouterMode::Auto)
        throw std::invalid_argument(
            "--shm-only requires --calibrate or explicit --backend force/off");
    }
    if (cfg.devices.empty() || cfg.devices.size() > 32 || cfg.begin < 128 ||
        cfg.end < cfg.begin || cfg.end > (1UL << 30) ||
        cfg.begin % (cfg.devices.size() * 2) || cfg.warmup < 1 ||
        cfg.iterations < 1 || cfg.repeats < 3 ||
        cfg.packetBytes < (128UL << 10) || cfg.packetBytes > (32UL << 20) ||
        cfg.packetBytes % 16 || cfg.minPacketBytes < 16 ||
        cfg.minPacketBytes > cfg.packetBytes || cfg.minPacketBytes % 16 ||
        cfg.workersPerNode < 1 || cfg.workersPerNode > 63 ||
        cfg.transportBlocks < 2 ||
        cfg.transportBlocks > cpu_router::kMaxTransportBlocks ||
        cfg.transportThreads < 32 || cfg.transportThreads > 1024 ||
        cfg.transportThreads % 32 || cfg.transportGrainBytes < 16 ||
        (cfg.transportVectorBytes != 16 && cfg.transportVectorBytes != 32) ||
        cfg.mainCpu < -1 || cfg.mainCpu >= CPU_SETSIZE || !cfg.grainElements ||
        cfg.scatterWorkersPerNode < 0 || cfg.scatterWorkersPerNode > 63 ||
        cfg.smallReduceScatterWorkersPerNode < 0 ||
        cfg.smallReduceScatterWorkersPerNode > 63 ||
        cfg.gpuLaunchWorkersPerNode < 0 || cfg.gpuLaunchWorkersPerNode > 32 ||
        (cfg.largeAllGatherPacketBytes &&
         (cfg.largeAllGatherPacketBytes < cfg.packetBytes ||
          cfg.largeAllGatherPacketBytes > (32UL << 20) ||
          cfg.largeAllGatherPacketBytes % 16)) ||
        (cfg.smallAllReducePacketBytes &&
         (cfg.smallAllReducePacketBytes < (128UL << 10) ||
          cfg.smallAllReducePacketBytes > cfg.packetBytes ||
          cfg.smallAllReducePacketBytes % 16)) ||
        (cfg.smallReduceScatterPacketBytes &&
         (cfg.smallReduceScatterPacketBytes < (128UL << 10) ||
          cfg.smallReduceScatterPacketBytes > cfg.packetBytes ||
          cfg.smallReduceScatterPacketBytes % 16)))
      throw std::invalid_argument(
          "invalid device count, size, or iteration settings");
    if (!cfg.calibrate.empty() && std::filesystem::exists(cfg.calibrate))
      throw std::invalid_argument(
          "calibration output exists; choose a new file");
    std::ofstream sampleOutput;
    if (!cfg.samples.empty()) {
      if (std::filesystem::exists(cfg.samples) ||
          (!cfg.calibrate.empty() &&
           std::filesystem::absolute(cfg.samples).lexically_normal() ==
               std::filesystem::absolute(cfg.calibrate).lexically_normal()))
        throw std::invalid_argument(
            "sample output must be a new, distinct file");
      sampleOutput.exceptions(std::ios::failbit | std::ios::badbit);
      sampleOutput.open(cfg.samples);
      sampleOutput.precision(10);
      sampleOutput << "collective,bytes,in_place,backend,repeat,iteration,"
                      "slowest_rank_us\n";
    }
    auto topology = inspectTopology(cfg.devices);
    if (cfg.mainCpu >= 0) {
      bool permitted = false;
      for (auto const& group : topology.permittedCpuIds)
        permitted |= std::find(group.second.begin(), group.second.end(),
                               cfg.mainCpu) != group.second.end();
      if (!permitted)
        throw std::invalid_argument(
            "main CPU is not permitted on a selected GPU NUMA node");
      topology.identity += ";benchmark-main-cpu=" + std::to_string(cfg.mainCpu);
    }
    if (cfg.shmOnly) {
      int physicalLinks = int(std::count(topology.peerAccess.begin(),
                                         topology.peerAccess.end(), true));
      std::fill(topology.peerAccess.begin(), topology.peerAccess.end(), false);
      topology.identity += ";logical-no-p2p-shm-v1";
      std::fprintf(stderr,
                   "# 8xL4 SHM: %d physically accessible directed peer "
                   "links disabled by benchmark policy; NCCL P2P disabled "
                   "and SHM enabled\n",
                   physicalLinks);
    }
    std::unique_ptr<LocalCollectives> local;
    if (cfg.routerOnly) {
      topology.identity += ";benchmark-router-only-v1";
    } else if (cfg.fallback == "local") {
      topology.identity += ";benchmark-local-fallback-v1";
      local = std::make_unique<LocalCollectives>(topology, cfg.end);
    } else {
      // Invalidate pre-fingerprint NCCL profiles even when no overrides exist.
      topology.identity += ";benchmark-nccl-config-v1";
    }
    BackendPolicy policy;
    policy.mode = cfg.mode;
    if (cfg.routerOnly)
      policy.fallbackIdentity = "uccl-lite-router-only:no-fallback-v1";
    else if (local)
      policy.fallbackIdentity = LocalCollectives::identity();
    else {
      int ncclVersion;
      ncclCheck(ncclGetVersion(&ncclVersion));
      policy.fallbackIdentity = benchmark::ncclFallbackIdentity(
          ncclVersion, benchmark::ncclConfiguration(environ));
    }
    if (!cfg.profile.empty()) {
      auto local = loadRouterProfile(cfg.profile);
      policy.measurements.insert(policy.measurements.end(), local.begin(),
                                 local.end());
    }
    std::vector<ncclComm_t> comms;
    if (!local && !cfg.routerOnly) {
      comms.resize(cfg.devices.size());
      ncclCheck(
          ncclCommInitAll(comms.data(), int(comms.size()), cfg.devices.data()));
    }
    char const* fallbackName = cfg.routerOnly ? "unavailable"
                               : local        ? "uccl-lite-local"
                                              : "nccl";
    CollectiveDispatcher::Fallback fallback = [&](CollectiveCall const& call) {
      if (cfg.routerOnly)
        throw std::runtime_error("router-only benchmark invoked fallback");
      if (local) {
        local->enqueue(call);
        return;
      }
      ncclCheck(ncclGroupStart());
      for (size_t r = 0; r < call.ports.size(); ++r) {
        const auto& p = call.ports[r];
        cudaCheck(cudaSetDevice(p.device));
        if (call.collective == Collective::AllReduce)
          ncclCheck(ncclAllReduce(p.input, p.output, call.count, ncclBfloat16,
                                  ncclSum, comms[r], p.stream));
        else if (call.collective == Collective::AllGather)
          ncclCheck(ncclAllGather(p.input, p.output, call.count, ncclBfloat16,
                                  comms[r], p.stream));
        else
          ncclCheck(ncclReduceScatter(p.input, p.output, call.count,
                                      ncclBfloat16, ncclSum, comms[r],
                                      p.stream));
      }
      ncclCheck(ncclGroupEnd());
    };
    std::vector<Measurement> calibrated;
    {
      Buffers buffers(cfg.devices, cfg.end);
      cpu_router::CpuRouterOptions options;
      options.maxTensorBytes = cfg.end;
      options.packetBytes = cfg.packetBytes;
      options.minPacketBytes = cfg.minPacketBytes;
      options.largeAllGatherPacketBytes = cfg.largeAllGatherPacketBytes;
      options.workersPerNode = cfg.workersPerNode;
      options.transportBlocks = cfg.transportBlocks;
      options.transportThreads = cfg.transportThreads;
      options.transportVectorBytes = cfg.transportVectorBytes;
      options.transportGrainBytes = cfg.transportGrainBytes;
      options.singleCtaBytes = cfg.singleCtaBytes;
      options.grainElements = cfg.grainElements;
      options.scatterWorkersPerNode = cfg.scatterWorkersPerNode;
      options.smallReduceScatterWorkersPerNode =
          cfg.smallReduceScatterWorkersPerNode;
      options.batchCpuJobs = cfg.batchCpuJobs;
      options.directScatterResults = cfg.directScatterResults;
      options.gpuLaunchWorkersPerNode = cfg.gpuLaunchWorkersPerNode;
      options.smallAllReducePacketBytes = cfg.smallAllReducePacketBytes;
      options.smallReduceScatterPacketBytes = cfg.smallReduceScatterPacketBytes;
      options.smallReduceScatterGpuProgress = cfg.smallReduceScatterGpuProgress;
      options.smallReduceScatterGpuProgressMaxBytes =
          cfg.smallReduceScatterGpuProgressMaxBytes;
      if (cfg.devices.size() == 8 && !cfg.gpuReduce &&
          cfg.workersPerNode == 23 && cfg.transportBlocks == 64) {
        options.smallAllReduceWorkersPerNode = 4;
        options.smallAllReduceTransportBlocks = 16;
      }
      if (cfg.shmOnly && !cfg.gpuReduce && cfg.workersPerNode == 23 &&
          cfg.transportBlocks == 64)
        options.smallReduceScatterGpuProgress64K = true;
      options.reduction = cfg.gpuReduce ? cpu_router::ReductionPlacement::Gpu
                                        : cpu_router::ReductionPlacement::Cpu;
      options.interleaveScatter = cfg.interleaveScatter;
      options.gpuProgress = cfg.gpuProgress;
      options.agPerSourceProgress = cfg.agPerSourceProgress;
      options.singleNumaReduction = cfg.singleNumaReduction;
      char const* routerName =
          cfg.interleaveScatter && !cfg.gpuReduce ? "cpu-router-cpu-striped"
          : cfg.interleaveScatter                 ? "cpu-router-gpu-striped"
          : cfg.gpuReduce                         ? "cpu-router-gpu-reduce"
                                                  : "cpu-router";
      if (cfg.gpuProgress)
        routerName = cfg.interleaveScatter ? "gpu-progress-striped"
                                           : "gpu-progress-flat";
      if (cfg.agPerSourceProgress) routerName = "gpu-progress-per-source";
      topology.identity += std::string(";cadence-probe-us=") + (std::getenv("UCCL_LITE_BENCH_IDLE_US") ? std::getenv("UCCL_LITE_BENCH_IDLE_US") : "0");
      CollectiveDispatcher selected(topology, policy, fallback, options);
      BackendPolicy forced = policy, off = policy;
      forced.mode = RouterMode::Force;
      off.mode = RouterMode::Off;
      std::unique_ptr<CollectiveDispatcher> router, baseline;
      if (!cfg.calibrate.empty()) {
        router = std::make_unique<CollectiveDispatcher>(topology, forced,
                                                        fallback, options);
        baseline = std::make_unique<CollectiveDispatcher>(topology, off,
                                                          fallback, options);
      }
      if (cfg.mainCpu >= 0) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cfg.mainCpu, &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask))
          throw std::runtime_error("cannot pin submission CPU");
      }
      std::vector<Collective> operations =
          cfg.operation == "all"
              ? std::vector<Collective>{Collective::AllReduce,
                                        Collective::AllGather,
                                        Collective::ReduceScatter}
              : std::vector<Collective>{parseCollective(cfg.operation)};
      std::vector<size_t> sizes;
      for (size_t bytes = cfg.begin; bytes <= cfg.end; bytes *= 2)
        sizes.push_back(bytes);
      if (cfg.reverse) std::reverse(sizes.begin(), sizes.end());
      std::printf(
          "# GPUs=%zu dtype=BF16 fallback=%s; no hidden calibration; checks "
          "outside timed region\n",
          cfg.devices.size(), policy.fallbackIdentity.c_str());
      std::printf("# topology=%s\n", topology.identity.c_str());
      std::puts(
          "collective,bytes,in_place,backend,median_us,min_us,max_us,speedup,"
          "dispatcher_router_initialized");
      for (auto op : operations)
        for (size_t bytes : sizes) {
          auto call = buffers.call(op, bytes, cfg.inPlace);
          char const* activeRouterName =
              !cfg.gpuReduce && !cfg.interleaveScatter &&
                      cfg.smallReduceScatterGpuProgress &&
                      op == Collective::ReduceScatter &&
                      (bytes == (128UL << 10) ||
                       (cfg.smallReduceScatterGpuProgressMaxBytes &&
                        bytes <= cfg.smallReduceScatterGpuProgressMaxBytes) ||
                       (options.smallReduceScatterGpuProgress64K &&
                        bytes == (64UL << 10)))
                  ? "gpu-progress-small-rs"
                  : routerName;
          auto decision = selected.prepare(call);
          if (local && (!decision.cpuRouter || baseline)) local->prepare();
          std::fprintf(stderr, "%s %zu: %s\n", collectiveName(op), bytes,
                       decision.reason.c_str());
          if (router) {
            router->prepare(call);
            baseline->prepare(call);
            run(*baseline, call, buffers, cfg.warmup, false);
            run(*router, call, buffers, cfg.warmup, false);
            std::vector<double> cpuSamples, baselineSamples;
            auto measure = [&](CollectiveDispatcher& dispatcher,
                               char const* backend, int repeat) {
              return run(dispatcher, call, buffers, cfg.iterations, true,
                         cfg.samples.empty() ? nullptr : &sampleOutput, backend,
                         repeat);
            };
            for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
              if (repeat % 2 == 0) {
                baselineSamples.push_back(
                    measure(*baseline, fallbackName, repeat));
                cpuSamples.push_back(
                    measure(*router, activeRouterName, repeat));
              } else {
                cpuSamples.push_back(
                    measure(*router, activeRouterName, repeat));
                baselineSamples.push_back(
                    measure(*baseline, fallbackName, repeat));
              }
            }
            Measurement m{router->topology().identity,
                          policy.fallbackIdentity,
                          call.workload(),
                          summarize(cpuSamples),
                          summarize(baselineSamples),
                          cfg.repeats,
                          true};
            calibrated.push_back(m);
            BackendPolicy measured = policy;
            measured.mode = RouterMode::Auto;
            measured.measurements = {m};
            auto verdict = measured.select(router->topology(), call.workload());
            for (int backend = 0; backend < 2; ++backend) {
              auto t = backend ? m.router : m.fallback;
              std::printf("%s,%zu,%d,%s,%.3f,%.3f,%.3f,%.4f,%d\n",
                          collectiveName(op), bytes, cfg.inPlace,
                          backend ? activeRouterName : "fallback", t.median,
                          t.minimum, t.maximum, m.fallback.median / t.median,
                          backend ? router->routerInitialized()
                                  : baseline->routerInitialized());
            }
            std::fprintf(stderr, "calibrated auto=%s: %s\n",
                         verdict.cpuRouter ? "cpu-router" : "fallback",
                         verdict.reason.c_str());
          } else {
            run(selected, call, buffers, cfg.warmup, false);
            std::vector<double> samples;
            for (int repeat = 0; repeat < cfg.repeats; ++repeat)
              samples.push_back(
                  run(selected, call, buffers, cfg.iterations, true,
                      cfg.samples.empty() ? nullptr : &sampleOutput,
                      decision.cpuRouter ? activeRouterName : fallbackName,
                      repeat));
            auto t = summarize(samples);
            std::printf("%s,%zu,%d,%s,%.3f,%.3f,%.3f,%.4f,%d\n",
                        collectiveName(op), bytes, cfg.inPlace,
                        decision.cpuRouter ? activeRouterName : "fallback",
                        t.median, t.minimum, t.maximum,
                        decision.measuredSpeedup, selected.routerInitialized());
          }
        }
    }
    if (!calibrated.empty()) saveRouterProfile(cfg.calibrate, calibrated);
    if (sampleOutput.is_open()) sampleOutput.close();
    for (auto comm : comms) ncclCheck(ncclCommDestroy(comm));
    std::puts("# correctness: PASS (every invocation, all ranks)");
  } catch (std::exception const& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
