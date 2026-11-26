#pragma once
#include "channel.hpp"
#include "router_core.hpp"
#include "uccl_lite/device_port.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace uccl_lite::cpu_router {
// Rank masks use uint32_t; capacities are bounds, not the active world size.
inline constexpr int kMaxRanks = 32;
// The runtime additionally caps cooperative grids at measured CUDA occupancy.
inline constexpr int kMaxTransportBlocks = 1024;
// CUDA NUMA hint, PCI sysfs fallback, then the first online host node.
int gpuNumaNode(int device);
enum class PacketTransform : uint32_t { Sum, Copy };
enum class PacketLayout : uint32_t { Replicate, Concatenate, Scatter };
// CPU routing/ownership stays shared; only the Sum arithmetic is relocated.
// GPU mode is an opt-in SM-transport ablation, not an automatic policy.
enum class ReductionPlacement : uint32_t { Cpu, Gpu };

#ifdef __CUDACC__
#define UCCL_LITE_HD __host__ __device__
#else
#define UCCL_LITE_HD
#endif
struct PacketUpload {
  size_t inputOffset = 0, packetOffset = 0, elements = 0;
};
struct PacketRoute {
  size_t sourceOffset = 0, outputOffset = 0, elements = 0, retainedOffset = 0;
  int sourceRank = -1;
  bool local = false, addRetained = false;
};
struct PacketUploads {
  int count = 0;
  PacketUpload ranges[2];
};
struct PacketFrame {
  size_t begin = 0, elements = 0;
  int uploads = 0, routes = 0;
  PacketUpload upload[2];
  PacketRoute route[kMaxRanks];
};
// BF16 packets. The transport is independent of the reduction/copy operator
// and output placement. Masks permit reduce-to-root and broadcast plans too.
struct PacketProgram {
  PacketTransform transform;
  PacketLayout layout;
  size_t inputElements;
  uint32_t sources = 255;
  uint32_t destinations = 255;
  int ranks = 8;
  static PacketProgram allReduce(size_t elements, int ranks = 8);
  static PacketProgram allGather(size_t contributionElements, int ranks = 8);
  static PacketProgram reduceScatter(size_t shardElements, int ranks = 8);
  static PacketProgram broadcast(size_t elements, int root, int ranks = 8);
  static PacketProgram reduce(size_t elements, int root, int ranks = 8);
  // These scalar routes are shared by host DMA emission and device channels.
  UCCL_LITE_HD PacketUploads uploadRanges(int rank, size_t tile,
                                          size_t extent) const {
    PacketUploads u;
    size_t begin = tile * extent,
           n = extent < inputElements - begin ? extent : inputElements - begin,
           end = begin + n;
    if (!(sources & (1U << rank))) return u;
    if (layout != PacketLayout::Scatter)
      u.ranges[u.count++] = {begin, 0, n};
    else {
      size_t shard = inputElements / ranks, ownLo = size_t(rank) * shard,
             ownHi = ownLo + shard;
      size_t before = end < ownLo ? end : ownLo;
      if (before > begin) u.ranges[u.count++] = {begin, 0, before - begin};
      size_t after = begin > ownHi ? begin : ownHi;
      if (after < end)
        u.ranges[u.count++] = {after, after - begin, end - after};
    }
    return u;
  }
  UCCL_LITE_HD PacketRoute outputRoute(int rank, size_t tile, size_t extent,
                                       int source = 0) const {
    PacketRoute r;
    size_t begin = tile * extent,
           n = extent < inputElements - begin ? extent : inputElements - begin,
           end = begin + n;
    if (!(destinations & (1U << rank))) return r;
    if (layout == PacketLayout::Concatenate)
      return {source == rank ? begin : 0,
              size_t(source) * inputElements + begin,
              n,
              0,
              source,
              source == rank,
              false};
    if (layout == PacketLayout::Scatter) {
      size_t shard = inputElements / ranks,
             lo = begin > size_t(rank) * shard ? begin : size_t(rank) * shard,
             hi = end < size_t(rank + 1) * shard ? end
                                                 : size_t(rank + 1) * shard;
      if (lo < hi)
        return {lo - begin, lo - size_t(rank) * shard,    hi - lo, lo, -1,
                false,      (sources & (1U << rank)) != 0};
      return r;
    }
    if (transform == PacketTransform::Copy) {
      source = 0;
      while (!(sources & (1U << source))) ++source;
      return {source == rank ? begin : 0,
              begin,
              n,
              0,
              source,
              source == rank,
              false};
    }
    return {0, begin, n, 0, -1, false, false};
  }
  UCCL_LITE_HD PacketFrame frame(int rank, size_t tile, size_t extent) const {
    PacketFrame f;
    f.begin = tile * extent;
    f.elements =
        extent < inputElements - f.begin ? extent : inputElements - f.begin;
    auto ranges = uploadRanges(rank, tile, extent);
    f.uploads = ranges.count;
    for (int i = 0; i < ranges.count; ++i) f.upload[i] = ranges.ranges[i];
    int count = layout == PacketLayout::Concatenate ? ranks : 1;
    for (int i = 0; i < count; ++i) {
      auto route = outputRoute(rank, tile, extent, i);
      if (route.elements) f.route[f.routes++] = route;
    }
    return f;
  }
};
struct PacketPipelineOptions {
  TransportBackend backend = TransportBackend::GpuKernel;
  size_t maxTensorBytes = 1UL << 30;
  size_t packetBytes = 512UL << 10;
  size_t minPacketBytes = 128UL << 10;
  // Adaptive dispatcher only: optional packet size for AllGather outputs
  // of at least 4 MiB. A nonzero explicit size takes priority over the
  // automatic target below.
  size_t largeAllGatherPacketBytes = 0;
  // Adaptive dispatcher only: target for AllGather outputs of at least 4 MiB.
  // Never reduces packetBytes; zero disables automatic enlargement.
  size_t autoLargeAllGatherPacketBytes = 8UL << 20;
  // The adaptive dispatcher prepares a smaller router for AllReduce through
  // 4 MiB. Zero disables this choice; direct PacketPipeline users keep
  // packetBytes unchanged.
  size_t smallAllReducePacketBytes = 128UL << 10;
  // Adaptive dispatcher only: smaller ReduceScatter packets from 256 KiB
  // through 4 MiB.
  // Zero keeps packetBytes for every ReduceScatter call.
  size_t smallReduceScatterPacketBytes = 128UL << 10;
  // Optional compact worker/grid choice for 128–256 KiB AllReduce.
  // Zero preserves the regular worker/grid choice.
  int smallAllReduceWorkersPerNode = 0;
  int smallAllReduceTransportBlocks = 0;
  // Adaptive dispatcher only: the 128-KiB ReduceScatter case uses GPU
  // reduction/progress. Direct PacketPipeline users keep their chosen mode.
  bool smallReduceScatterGpuProgress = true;
  // Optional 64-KiB ReduceScatter GPU progress; the usual 128-KiB choice
  // remains enabled by smallReduceScatterGpuProgress.
  bool smallReduceScatterGpuProgress64K = false;
  // Adaptive dispatcher only: optionally use GPU reduction/progress for all
  // complete ReduceScatter inputs through this limit (128 B--64 KiB). Zero
  // keeps the existing exact-size choices. Requires separate calibration.
  size_t smallReduceScatterGpuProgressMaxBytes = 0;
  // Adaptive dispatcher only: optional GPU progress through these complete
  // tensor byte limits (16 B--1 MiB). AG uses per-source readiness and complete
  // output bytes per rank.
  // Zero preserves CPU progress. Each choice needs separate calibration;
  // GPU AllReduce rereads every source and can lose beyond tiny messages.
  size_t smallAllReduceGpuProgressMaxBytes = 0;
  size_t smallAllGatherGpuProgressMaxBytes = 0;
  int transportBlocks = 64;
  // Fused GEMM only: cap each upload/download crew. Zero selects by shape.
  int fusedTransportCrew = 0;
  int transportThreads = 256;
  // 32 uses Blackwell 256-bit loads/stores when all ranges are aligned;
  // earlier architectures use two 128-bit accesses. Ownership flags stay u32.
  int transportVectorBytes = 16;
  size_t transportGrainBytes = 16UL << 10;
  // Optional complete-tensor byte limit for serial upload/consume in one CTA.
  // Useful for latency-sensitive packets; zero retains independent crews.
  size_t singleCtaBytes = 0;
  int workersPerNode = 23;
  size_t grainElements = 8192;
  // Optional group-major affinity, groups ordered by first device occurrence.
  // Empty discovers NUMA-local online CPUs and reserves one CPU per group.
  std::vector<int> workerCpus;
  // Optional physical NUMA node per rank. Empty queries CUDA then PCI sysfs.
  std::vector<int> numaNodes;
  // -1 selects an unused NUMA-local CPU where possible.
  int replayCpu = -1;
  // Publish <=128 captured commands before launching their GPU graphs. Larger
  // captures use the replay helper so publication cannot exhaust GPU credits.
  bool inlineReplay = false;
  // Low-occupancy persistent transport alongside an independent producer.
  // Caller must establish input readiness with CUDA-visible dependencies.
  bool concurrentProducer = false;
  // Independent GEMM tile override: (0,0) chooses by shape; explicit pairs are
  // (32,128), (64,128), (128,128), (128,256), (256,128).
  // 32x128 needs K/stages=64/2; 128x128 needs K/stages=32/3.
  // packetBytes remains independent.
  int gemmTileM = 0;
  int gemmTileN = 0;
  // Optional matched K/stage pipeline: (0,0) or (32,4), and (64,2).
  // The 128x128 tile instead uses (32,3) for two-CTA residency.
  // Both use the same shared-memory footprint for a selected M/N tile.
  int gemmTileK = 0;
  int gemmStages = 0;
  // 0 retains the cache-size policy; a positive value groups adjacent M tiles.
  int gemmMGroup = 0;
  // ReduceScatter final results are owned by destinations. Cap active CPU lanes
  // separately from the executor pool to avoid waking every AllReduce worker.
  // Zero uses the full workersPerNode limit; positive values are upper bounds.
  int scatterWorkersPerNode = 16;
  // Optional cap for CPU ReduceScatter with 256 KiB--1 MiB complete inputs.
  // Zero preserves scatterWorkersPerNode; requires its own calibration.
  int smallReduceScatterWorkersPerNode = 0;
  // Publish every active CPU lane before waking workers, for all layouts.
  bool batchCpuJobs = false;
  // Explicit CPU merge experiment, independent of GPU progress.
  bool directScatterResults = false;
  // Optional eager SM graph-launch lanes per NUMA group. Zero submits on the
  // caller; positive values reserve spare NUMA-local CPUs at prepare. Requires
  // explicit streams, including cudaStreamLegacy; nullptr and
  // cudaStreamPerThread can be thread relative under per-thread defaults.
  // Launch lanes spin for up to 2 ms after each task before sleeping.
  // Capture, replay, DMA and resident/concurrent producers keep caller
  // launches.
  int gpuLaunchWorkersPerNode = 0;
  // Restrict launch discovery to these CPUs when nonempty. The dispatcher
  // supplies its inspected allowance even if the caller later pins itself.
  // Direct pipeline discovery otherwise uses the cgroup's permitted CPUs.
  std::vector<int> gpuLaunchAllowedCpus;
  // On one host NUMA node, reduce directly into the pinned result packet.
  // Result-reader credits precede the CPU writes; upload credits follow them.
  // Multiple NUMA nodes retain the staged partial/merge schedule.
  bool singleNumaReduction = false;
  ReductionPlacement reduction = ReductionPlacement::Cpu;
  // Each packet carries one stripe for every destination
  // instead of draining a complete destination shard before the next one.
  // Keeps all destination PCIe links active; source/destination layout at the
  // public API is unchanged. CPU interleaving requires standalone SM transport;
  // neither placement supports resident GEMM or flat producer-event stripes.
  bool interleaveScatter = false;
  // GPU reduction only. GPU leaders observe upload/reader generations directly;
  // no CPU packet-routing executor is created. Host command publication
  // remains.
  bool gpuProgress = false;
  // AllGather Copy with GPU progress: each source publishes its own readiness
  // to a dedicated consumer-block group, which can copy before other sources
  // upload. Reader credits still wait for every group before slot reuse.
  bool agPerSourceProgress = false;
  // Explicit standalone BF16 ReduceScatter ring through mapped host buffers.
  // Requires GPU reduction/progress, aligned out-of-place full-rank shards,
  // shard bytes <= packetBytes, and a fixed resident grid of at most 128 CTAs.
  // No topology/size auto-selection: callers need separate calibration.
  // Each hop rounds its partial to BF16 (unlike FP32 flat GPU reduction).
  bool hostRingReduceScatter = false;
};

// One process, 1..kMaxRanks ports in arbitrary device/NUMA order. All GPU
// transfers go through mapped pinned packets. No NCCL collective implementation
// is called by this core.
class PacketPipeline : public RouterCore {
 public:
  explicit PacketPipeline(PacketPipelineOptions options = {});
  ~PacketPipeline();
  PacketPipeline(PacketPipeline const&) = delete;
  PacketPipeline& operator=(PacketPipeline const&) = delete;
  void prepare(std::vector<PacketPort> const& ports);
  void enqueue(PacketProgram const& plan, std::vector<PacketPort> const& ports);
  struct Binding {
    explicit Binding(size_t ranks = 8)
        : inputs(ranks),
          outputs(ranks),
          streams(ranks),
          devices(ranks),
          producerEvents(ranks) {}
    std::vector<MemorySpan> inputs, outputs;
    std::vector<cudaStream_t> streams;
    std::vector<int> devices;
    // Optional event per contiguous input stripe. Each persistent epoch only
    // touches packets whose intersecting stripes have completed. Events outlive
    // captured graphs and this binding. One CPU job spans all epochs.
    std::vector<std::vector<cudaEvent_t>> producerEvents;
    size_t producerElements = 0;
  };
  void enqueue(PacketProgram const& program, Binding const& binding);
  // Resident CUTLASS producer and CPU-router transport in one cooperative grid.
  // A fixed shape, one set per rank of BF16 A[M,K], B[N,K], partial[M,N]
  // operands.
  void prepareGemm(int m, int n, int k, std::vector<MemorySpan> const& a,
                   std::vector<MemorySpan> const& b,
                   std::vector<MemorySpan> const& partial,
                   PacketLayout layout = PacketLayout::Replicate);
  // AllGather local A[localM,K] into gatheredA[localM*ranks,K], with
  // resident workers consuming completed input row tiles in the same grid.
  void prepareAllGatherGemm(int localM, int n, int k,
                            std::vector<MemorySpan> const& localA,
                            std::vector<MemorySpan> const& gatheredA,
                            std::vector<MemorySpan> const& b,
                            std::vector<MemorySpan> const& output);
  void enqueueGemm(PacketProgram const& program, Binding const& binding);
  void enqueueGemmProducerOnly(std::vector<cudaStream_t> const& streams);
  // Source compatibility for callers with fixed std::array port sets.
  template <size_t N>
  void prepare(std::array<PacketPort, N> const& ports) {
    prepare(std::vector<PacketPort>(ports.begin(), ports.end()));
  }
  template <size_t N>
  void enqueue(PacketProgram const& plan,
               std::array<PacketPort, N> const& ports) {
    enqueue(plan, std::vector<PacketPort>(ports.begin(), ports.end()));
  }
  template <size_t N>
  void prepareGemm(int m, int n, int k, std::array<MemorySpan, N> const& a,
                   std::array<MemorySpan, N> const& b,
                   std::array<MemorySpan, N> const& partial) {
    prepareGemm(m, n, k, std::vector<MemorySpan>(a.begin(), a.end()),
                std::vector<MemorySpan>(b.begin(), b.end()),
                std::vector<MemorySpan>(partial.begin(), partial.end()));
  }
  template <size_t N>
  void enqueueGemmProducerOnly(std::array<cudaStream_t, N> const& streams) {
    enqueueGemmProducerOnly(
        std::vector<cudaStream_t>(streams.begin(), streams.end()));
  }
  // Record an externally captured program. GPU graphs belong to the caller.
  // beginReplay publishes fresh jobs asynchronously; launch all GPU graphs,
  // synchronize their streams, then call endReplay before reusing the program.
  void beginCapture();
  void endCapture();
  void beginReplay();
  void endReplay();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
using RouterPort = PacketPort;
using RouterTransform = PacketTransform;
using RouterLayout = PacketLayout;
using RouterPlan = PacketProgram;
using CpuRouterOptions = PacketPipelineOptions;
using CpuRouter = PacketPipeline;
}  // namespace uccl_lite::cpu_router
