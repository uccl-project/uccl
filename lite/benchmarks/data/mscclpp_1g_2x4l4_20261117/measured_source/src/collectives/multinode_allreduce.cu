// Two-node, four-GPU-per-node hierarchical AllReduce for PCIe-only GPUs.
//
// Schedule (H1):
//   1. topology-aware four-GPU ring ReduceScatter inside each node;
//   2. four parallel, rail-local IB exchanges between equal local ranks;
//   3. reduce the remote shard and ring AllGather inside each node.
//
// H2 pipelines the three stages by splitting each rank shard into pieces. While
// the IB channel moves piece p, the local copy engines work on other pieces. It
// is intended for medium/large messages; H1 avoids the extra triggers at small
// sizes.  Both algorithms move the minimum N bytes in each direction between
// two nodes, spread over the two HCAs without routing GPU traffic over UPI.

#include "uccl_lite/multinode_host_gather.cuh"
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <mscclpp/utils.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>


#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e_ = (cmd);                                              \
    if (e_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e_));                              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                    \
  } while (0)

#define MPICHECK(cmd)                                                         \
  do {                                                                        \
    int e_ = (cmd);                                                           \
    if (e_ != MPI_SUCCESS) {                                                  \
      char s_[MPI_MAX_ERROR_STRING];                                          \
      int n_ = 0;                                                             \
      MPI_Error_string(e_, s_, &n_);                                          \
      std::fprintf(stderr, "%s:%d MPI error: %.*s\n", __FILE__, __LINE__, n_, \
                   s_);                                                       \
      MPI_Abort(MPI_COMM_WORLD, e_);                                          \
    }                                                                         \
  } while (0)

#include "../multinode/device_collectives.cuh"
using namespace uccl_lite::multinode_kernels;

__global__ void nullKernel() {}

#include "../multinode/oneshot_allreduce.cuh"

struct Config {
  bool arOneShot = false;
  bool hostOneShot = false;
  bool allGather = false;
  bool reduceScatter = false;
  bool arAuto = false;
  bool hostGather = false;
  int hostRepeatCheck = 0;
  size_t hostPieceBytes = 256UL << 10;
  bool agRing = false;
  bool agAuto = false;
  bool agSerialLocal = false;
  int agPipeline =
      0;  // 1: SM, 2: CE, 3: CE relay, 4: matched CE, 5: matched SM.
  bool agDualRdma = false;
  int agFirstMask = 2;
  size_t agPieceBytes = 64UL << 10;
  int agDepth = 2;
  bool rsRing = false;
  bool rsAuto = false;
  bool rsSerialLocal = false;
  bool rsPullLocal = false;
  bool rsOverlapLocal = false;
  bool rsFastPack = false;
  bool rsDirectRing = false;
  int rsPieces = 1;
  size_t rsTargetPieceBytes = 0;
  bool gpuTiming = false;
  bool checkEach = false;
  size_t minBytes = 1UL << 10;
  size_t maxBytes = 1UL << 30;
  int warmup = 10;
  int iters = 20;
  int blocks = 24;
  int rsFlatBlocks = 24;
  int threads = 512;
  size_t pipelineMinBytes = 4UL << 20;
  size_t pieceBytes = 2UL << 20;
  int pipelineDistance = 2;
  bool flatAg = false;
  bool arSerialLocal = false;
  bool check = true;
  std::array<int, kLocalRanks> railMap{0, 0, 1, 1};
};

static constexpr size_t kSerialArMaxBytes = 1UL << 20;
// On the measured two-node L4 topology, avoid the host phase handoff for
// small AR messages. Larger messages retain the explicit hybrid path.
static constexpr size_t kHybridSerialArMaxBytes = 1UL << 18;
static constexpr size_t kSerialAgMaxBytes = 2UL << 20;
static constexpr size_t kSerialRsMaxBytes = 1UL << 20;

static bool autoPackedRs(size_t bytes) {
  return bytes >= (1UL << 20) && bytes <= (16UL << 20);
}

// Skip input packing where a single direct-ring piece amortizes the local
// handshakes. Kept separate from hybrid AR's measured RS selection.
static bool mediumDirectRs(size_t bytes) {
  return bytes >= (8UL << 20) && bytes <= (16UL << 20);
}

static bool autoDirectRs(size_t bytes) {
  return bytes == 1024 || bytes >= (32UL << 20);
}

static std::array<int, kLocalRanks> parseRailMap(std::string const& text) {
  std::array<int, kLocalRanks> out{};
  std::stringstream ss(text);
  std::string item;
  for (int i = 0; i < kLocalRanks; i++) {
    if (!std::getline(ss, item, ','))
      throw std::runtime_error("rail map needs four entries");
    out[i] = std::stoi(item);
  }
  if (std::getline(ss, item, ','))
    throw std::runtime_error("rail map needs four entries");
  return out;
}

static mscclpp::Transport ibTransport(int index) {
  if (index < 0 || index >= 8)
    throw std::runtime_error("IB rail index must be in [0,7]");
  return static_cast<mscclpp::Transport>(
      static_cast<int>(mscclpp::Transport::IB0) + index);
}

static int ringPosition(int localRank) {
  constexpr int pos[kLocalRanks] = {0, 1, 3, 2};
  return pos[localRank];
}

static int ringSuccessor(int localRank) {
  constexpr int succ[kLocalRanks] = {1, 3, 0, 2};
  return succ[localRank];
}

static int ringPredecessor(int localRank) {
  constexpr int pred[kLocalRanks] = {2, 0, 3, 1};
  return pred[localRank];
}

int main(int argc, char** argv) {
  // UCX is selected by the explicit hybrid runner; route each process to
  // its NUMA-local HCA before MPI creates its endpoints.
  bool hostRequested = false;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--host-gather" && std::atoi(argv[i + 1]))
      hostRequested = true;
  if (hostRequested) {
    char const* local = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int gpu = local ? std::atoi(local) : 0;
    if (!std::getenv("UCX_NET_DEVICES"))
      setenv("UCX_NET_DEVICES", gpu < 2 ? "mlx5_0:1" : "mlx5_1:1", 1);
    if (!std::getenv("UCX_TLS"))
      setenv("UCX_TLS", "rc_mlx5,ud_mlx5,sm,self", 1);
  }
  MPICHECK(MPI_Init(&argc, &argv));
  int rank = 0, world = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
  if (world != kWorld) {
    if (rank == 0)
      std::fprintf(stderr,
                   "requires exactly %d MPI ranks (two nodes x four GPUs)\n",
                   kWorld);
    MPI_Finalize();
    return 2;
  }

  Config cfg;
  try {
    for (int i = 1; i < argc; i++) {
      std::string key = argv[i];
      auto next = [&]() -> char const* {
        if (++i >= argc) throw std::runtime_error("missing value for " + key);
        return argv[i];
      };
      if (key == "-b")
        cfg.minBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-e")
        cfg.maxBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-w")
        cfg.warmup = std::atoi(next());
      else if (key == "-n")
        cfg.iters = std::atoi(next());
      else if (key == "-B")
        cfg.blocks = std::atoi(next());
      else if (key == "-T")
        cfg.threads = std::atoi(next());
      else if (key == "-p")
        cfg.pipelineMinBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-k")
        cfg.pieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-d")
        cfg.pipelineDistance = std::atoi(next());
      else if (key == "-a")
        cfg.flatAg = std::atoi(next()) != 0;
      else if (key == "--host-repeat-check")
        cfg.hostRepeatCheck = std::atoi(next());
      else if (key == "--host-gather")
        cfg.hostGather = std::atoi(next()) != 0;
      else if (key == "--host-piece-bytes")
        cfg.hostPieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "--host-oneshot")
        cfg.hostOneShot = std::atoi(next()) != 0;
      else if (key == "--ar-oneshot")
        cfg.arOneShot = std::atoi(next()) != 0;
      else if (key == "--ar-auto")
        cfg.arAuto = std::atoi(next()) != 0;
      else if (key == "--ar-serial-local")
        cfg.arSerialLocal = std::atoi(next()) != 0;
      else if (key == "-c")
        cfg.check = std::atoi(next()) != 0;
      else if (key == "--rail-map")
        cfg.railMap = parseRailMap(next());
      else if (key == "--op") {
        std::string op = next();
        cfg.allGather = op == "allgather";
        cfg.reduceScatter = op == "reducescatter";
        if (op != "allreduce" && !cfg.allGather && !cfg.reduceScatter)
          throw std::runtime_error(
              "--op must be allreduce, allgather, or reducescatter");
      } else if (key == "--ag-ring")
        cfg.agRing = std::atoi(next()) != 0;
      else if (key == "--ag-auto")
        cfg.agAuto = std::atoi(next()) != 0;
      else if (key == "--ag-pipeline") {
        std::string mode = next();
        if (mode != "sm" && mode != "ce" && mode != "ce-pair" &&
            mode != "matched" && mode != "matched-sm")
          throw std::runtime_error(
              "--ag-pipeline must be sm, ce, ce-pair, matched, or matched-sm");
        cfg.agPipeline = mode == "sm"        ? 1
                         : mode == "ce"      ? 2
                         : mode == "ce-pair" ? 3
                         : mode == "matched" ? 4
                                             : 5;
      } else if (key == "--ag-piece-bytes")
        cfg.agPieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "--ag-first-mask")
        cfg.agFirstMask = std::atoi(next());
      else if (key == "--ag-dual-rdma")
        cfg.agDualRdma = std::atoi(next()) != 0;
      else if (key == "--ag-depth")
        cfg.agDepth = std::atoi(next());
      else if (key == "--ag-serial-local")
        cfg.agSerialLocal = std::atoi(next()) != 0;
      else if (key == "--rs-ring")
        cfg.rsRing = std::atoi(next()) != 0;
      else if (key == "--rs-auto")
        cfg.rsAuto = std::atoi(next()) != 0;
      else if (key == "--rs-overlap-local")
        cfg.rsOverlapLocal = std::atoi(next()) != 0;
      else if (key == "--rs-pull-local")
        cfg.rsPullLocal = std::atoi(next()) != 0;
      else if (key == "--rs-serial-local")
        cfg.rsSerialLocal = std::atoi(next()) != 0;
      else if (key == "--rs-flat-blocks")
        cfg.rsFlatBlocks = std::atoi(next());
      else if (key == "--rs-fast-pack")
        cfg.rsFastPack = std::atoi(next()) != 0;
      else if (key == "--rs-direct-ring")
        cfg.rsDirectRing = std::atoi(next()) != 0;
      else if (key == "--rs-pieces")
        cfg.rsPieces = std::atoi(next());
      else if (key == "--rs-piece-bytes")
        cfg.rsTargetPieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "--gpu-timing")
        cfg.gpuTiming = std::atoi(next()) != 0;
      else if (key == "--check-each")
        cfg.checkEach = std::atoi(next()) != 0;
      else
        throw std::runtime_error("unknown argument " + key);
    }
  } catch (std::exception const& e) {
    if (rank == 0) std::fprintf(stderr, "%s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.arAuto && (cfg.allGather || cfg.reduceScatter)) {
    if (rank == 0) std::fprintf(stderr, "--ar-auto requires AllReduce\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.arSerialLocal && (cfg.allGather || cfg.reduceScatter)) {
    if (rank == 0)
      std::fprintf(stderr, "--ar-serial-local requires AllReduce\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if ((cfg.rsPullLocal || cfg.rsOverlapLocal) &&
      (!cfg.reduceScatter || cfg.rsAuto || cfg.rsRing || cfg.rsSerialLocal ||
       cfg.rsFastPack || cfg.rsDirectRing ||
       (cfg.rsPullLocal && cfg.rsOverlapLocal))) {
    if (rank == 0)
      std::fprintf(
          stderr, "overlap/pull requires ReduceScatter and no other RS mode\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agFirstMask != 1 && cfg.agFirstMask != 2) {
    if (rank == 0) std::fprintf(stderr, "--ag-first-mask must be 1 or 2\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agDualRdma && cfg.agPipeline != 1 && cfg.agPipeline != 2) {
    if (rank == 0)
      std::fprintf(stderr, "--ag-dual-rdma requires --ag-pipeline sm or ce\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agPipeline &&
      (!cfg.allGather || cfg.agAuto || cfg.agRing || cfg.agSerialLocal ||
       cfg.agPieceBytes < 512 || cfg.agPieceBytes % 16 || cfg.agDepth < 1 ||
       cfg.agDepth > 64 ||
       (cfg.maxBytes / kWorld + cfg.agPieceBytes - 1) / cfg.agPieceBytes >
           64)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "AG pipeline requires exclusive AllGather mode, aligned "
                   "pieces >=512 bytes, <=64 pieces, depth 1..64\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agAuto) cfg.agRing = true;
  if (cfg.rsAuto) {
    cfg.rsRing = true;
    cfg.rsDirectRing = true;
  }
  if (cfg.agRing && !cfg.allGather) {
    if (rank == 0)
      std::fprintf(stderr, "--ag-ring is only valid for --op allgather\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agSerialLocal && !cfg.allGather) {
    if (rank == 0)
      std::fprintf(stderr, "--ag-serial-local requires AllGather\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsSerialLocal && !cfg.reduceScatter) {
    if (rank == 0)
      std::fprintf(stderr, "--rs-serial-local requires ReduceScatter\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsRing && !cfg.reduceScatter) {
    if (rank == 0)
      std::fprintf(stderr,
                   "--rs-ring is only valid for --op reducescatter\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsFastPack && !cfg.rsRing) {
    if (rank == 0)
      std::fprintf(stderr, "--rs-fast-pack requires --rs-ring 1\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsDirectRing && (!cfg.rsRing || cfg.rsFastPack)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "--rs-direct-ring requires --rs-ring 1 and no fast pack\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsPieces < 1 || cfg.rsPieces > 32 ||
      (cfg.rsPieces > 1 && !cfg.rsRing)) {
    if (rank == 0)
      std::fprintf(stderr, "--rs-pieces must be 1..32 and requires ring\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsTargetPieceBytes &&
      (!cfg.rsDirectRing || cfg.rsPieces != 1 ||
       cfg.rsTargetPieceBytes < 512 || cfg.rsTargetPieceBytes % 512)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "--rs-piece-bytes requires direct ring, default "
                   "--rs-pieces, and 512-byte alignment\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.minBytes < 1024 || cfg.maxBytes < cfg.minBytes ||
      cfg.minBytes % 512 || cfg.maxBytes % 512 || cfg.warmup < 0 ||
      cfg.iters < 1 || cfg.blocks < 1 || cfg.rsFlatBlocks < 1 ||
      cfg.threads < 1 ||
      cfg.threads > 1024 || cfg.pieceBytes < 512 ||
      cfg.pipelineDistance < 1) {
    if (rank == 0)
      std::fprintf(stderr, "invalid size, launch, or iteration settings\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  if (cfg.arOneShot &&
      (cfg.allGather || cfg.reduceScatter || cfg.maxBytes > (1UL << 20))) {
    if (rank == 0)
      std::fprintf(stderr, "--ar-oneshot requires AllReduce <= 1 MiB\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  if (cfg.hostOneShot &&
      (!cfg.hostGather || cfg.allGather || cfg.reduceScatter || cfg.arOneShot ||
       cfg.maxBytes > (1UL << 20))) {
    if (rank == 0)
      std::fprintf(stderr,
                   "--host-oneshot requires --host-gather 1, "
                   "exclusive AllReduce <= 1 MiB\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  bool hybridRs = cfg.hostGather && !cfg.allGather && !cfg.hostOneShot;
  if (cfg.hostRepeatCheck < 0 || cfg.hostRepeatCheck > 8 ||
      (cfg.hostRepeatCheck &&
       ((!hybridRs && !cfg.hostOneShot) || cfg.reduceScatter))) {
    if (rank == 0)
      std::fprintf(
          stderr,
          "--host-repeat-check requires hybrid AllReduce and count 1..8\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.hostGather) {
    if (cfg.reduceScatter || cfg.gpuTiming || cfg.arAuto || cfg.agAuto ||
        cfg.agPipeline || cfg.agRing || cfg.agSerialLocal ||
        cfg.arSerialLocal || cfg.rsAuto || cfg.rsRing || cfg.rsSerialLocal ||
        cfg.rsPullLocal || cfg.rsOverlapLocal || cfg.maxBytes > (1UL << 30) ||
        cfg.hostPieceBytes < 128 || cfg.hostPieceBytes > (16UL << 20) ||
        cfg.hostPieceBytes % 128) {
      if (rank == 0)
        std::fprintf(stderr,
                     "--host-gather requires exclusive eager AR/AG, host "
                     "timing and aligned host pieces\n");
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (hybridRs) {
      cfg.rsAuto = cfg.rsRing = cfg.rsDirectRing = true;
      cfg.rsTargetPieceBytes = 8UL << 20;
      cfg.blocks = 96;
      cfg.threads = 512;
    }
  }

  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                               MPI_INFO_NULL, &localComm));
  int localRank = 0, localSize = 0;
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  MPICHECK(MPI_Comm_size(localComm, &localSize));
  if (localSize != kLocalRanks) {
    std::fprintf(stderr, "rank %d: expected four ranks per node, got %d\n",
                 rank, localSize);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  CUDACHECK(cudaSetDevice(localRank));
  cudaDeviceProp localProperty{};
  CUDACHECK(cudaGetDeviceProperties(&localProperty, localRank));
  int residentBlocks = 0;
  if (cfg.allGather) {
    if (cfg.agPipeline == 5)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherMatched<false>, cfg.threads,
          0));
    else if (cfg.agPipeline == 4)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherMatched<true>, cfg.threads, 0));
    else if (cfg.agDualRdma && cfg.agPipeline == 1)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<false, false, true>,
          cfg.threads, 0));
    else if (cfg.agDualRdma)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<true, false, true>,
          cfg.threads, 0));
    else if (cfg.agPipeline == 1)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<false>, cfg.threads,
          0));
    else if (cfg.agPipeline == 3)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<true, true>,
          cfg.threads, 0));
    else if (cfg.agPipeline == 2)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<true>, cfg.threads,
          0));
    else if (cfg.agRing)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherRing, cfg.threads, 0));
    else
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGather, cfg.threads, 0));
    if (cfg.agAuto) {
      int flatResidentBlocks = 0;
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &flatResidentBlocks, hierarchicalAllGather, cfg.threads, 0));
      residentBlocks = std::min(residentBlocks, flatResidentBlocks);
    }
  } else if (cfg.reduceScatter || hybridRs) {
    if (cfg.rsPullLocal)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatterOverlap<true>, cfg.threads,
          0));
    else if (cfg.rsOverlapLocal)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatterOverlap<false>, cfg.threads,
          0));
    else if (cfg.rsDirectRing)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatterDirectRing, cfg.threads,
          0));
    else if (cfg.rsRing)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatterRing, cfg.threads, 0));
    else
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatter, cfg.threads, 0));
    if (cfg.rsAuto) {
      int flatResidentBlocks = 0;
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &flatResidentBlocks, hierarchicalReduceScatter, cfg.threads, 0));
      residentBlocks = std::min(residentBlocks, flatResidentBlocks);
      int packedResidentBlocks = 0;
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &packedResidentBlocks, hierarchicalReduceScatterRing, cfg.threads,
          0));
      residentBlocks = std::min(residentBlocks, packedResidentBlocks);
    }
  } else
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &residentBlocks, hierarchicalAllReduce, cfg.threads, 0));
  if (cfg.blocks > residentBlocks * localProperty.multiProcessorCount) {
    std::fprintf(stderr,
                 "rank %d: grid %d blocks exceeds resident limit %d for "
                 "this cooperative kernel\n",
                 rank, cfg.blocks,
                 residentBlocks * localProperty.multiProcessorCount);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  int rail = cfg.railMap[localRank];
  int nIb = mscclpp::getIBDeviceCount();
  if (rail >= nIb) {
    std::fprintf(stderr,
                 "rank %d: rail %d requested but MSCCL++ sees %d IB devices; "
                 "build with MSCCLPP_USE_IB=ON\n",
                 rank, rail, nIb);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
  mscclpp::UniqueId id{};
  if (rank == 0) id = mscclpp::TcpBootstrap::createUniqueId();
  MPICHECK(MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD));
  bootstrap->initialize(id, 120);
  mscclpp::Communicator comm(bootstrap);

  size_t dataBytes = cfg.maxBytes;
  size_t localScratchBytes = dataBytes;
  size_t ibSlotBytes = (dataBytes + kLocalRanks - 1) / kLocalRanks + 4096;
  if (cfg.arOneShot) ibSlotBytes = 2 * dataBytes;
  Layout layout{0, dataBytes, dataBytes + localScratchBytes,
                dataBytes + localScratchBytes + ibSlotBytes, 0};
  size_t bufferBytes = layout.ibScratch + ibSlotBytes;
  if (!hybridRs && cfg.rsRing && (!cfg.rsDirectRing || cfg.rsAuto)) {
    layout.immutableInput = bufferBytes;
    bufferBytes += dataBytes;
  }
  if (bufferBytes >= (1ULL << 32)) {
    if (rank == 0)
      std::fprintf(
          stderr,
          "registered region exceeds PortChannel's 32-bit offset limit\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  char* buffer = nullptr;
  CUDACHECK(cudaMalloc(&buffer, bufferBytes));
  CUDACHECK(cudaMemset(buffer, 0, bufferBytes));

  mscclpp::TransportFlags transports = mscclpp::Transport::CudaIpc;
  transports |= ibTransport(rail);
  auto localMemory = comm.registerMemory(buffer, bufferBytes, transports);

  int nodeBase = rank - localRank;
  int succ = nodeBase + ringSuccessor(localRank);
  int pred = nodeBase + ringPredecessor(localRank);
  int diag = nodeBase + (localRank ^ 3);
  int cross = rank ^ kLocalRanks;
  std::array<int, 4> peers{succ, pred, diag, cross};
  std::array<mscclpp::Transport, 4> peerTransports{
      mscclpp::Transport::CudaIpc, mscclpp::Transport::CudaIpc,
      mscclpp::Transport::CudaIpc, ibTransport(rail)};
  std::array<std::shared_future<mscclpp::Connection>, 4> connectionFutures;
  std::array<std::shared_future<mscclpp::RegisteredMemory>, 4> memoryFutures;
  for (int i = 0; i < 4; i++) {
    connectionFutures[i] =
        comm.connect(peerTransports[i], peers[i], i < 3 ? 100 : 102);
    comm.sendMemory(localMemory, peers[i], i < 3 ? 200 : 202);
    memoryFutures[i] = comm.recvMemory(peers[i], i < 3 ? 200 : 202);
  }

  std::shared_future<mscclpp::Connection> duplicateConnectionFuture;
  std::shared_future<mscclpp::RegisteredMemory> duplicateMemoryFuture;
  if (cfg.agDualRdma) {
    duplicateConnectionFuture = comm.connect(ibTransport(rail), cross ^ 1, 104);
    comm.sendMemory(localMemory, cross ^ 1, 204);
    duplicateMemoryFuture = comm.recvMemory(cross ^ 1, 204);
  }

  std::array<mscclpp::Connection, 4> connections;
  std::array<mscclpp::RegisteredMemory, 4> remoteMemory;
  for (int i = 0; i < 4; i++) {
    connections[i] = connectionFutures[i].get();
    remoteMemory[i] = memoryFutures[i].get();
  }

  // Local DMA and IB must have independent proxy threads/FIFOs; sharing one
  // ProxyService serializes trigger submission and defeats the software
  // pipeline even though the PCIe copy engines and NIC can run concurrently.
  auto localProxy = std::make_shared<mscclpp::ProxyService>();
  auto ibProxy = std::make_shared<mscclpp::ProxyService>();
  std::array<std::shared_future<mscclpp::Semaphore>, 4> semaphoreFutures;
  for (int i = 0; i < 4; i++)
    semaphoreFutures[i] =
        comm.buildSemaphore(connections[i], peers[i], i < 3 ? 300 : 302);
  auto localSendSemaphore = localProxy->addSemaphore(semaphoreFutures[0].get());
  auto localRecvSemaphore = localProxy->addSemaphore(semaphoreFutures[1].get());
  auto localDiagSemaphore = localProxy->addSemaphore(semaphoreFutures[2].get());
  auto ibSemaphore = ibProxy->addSemaphore(semaphoreFutures[3].get());
  auto localLocalMemoryId = localProxy->addMemory(localMemory);
  auto localSendMemoryId = localProxy->addMemory(remoteMemory[0]);
  auto localRecvMemoryId = localProxy->addMemory(remoteMemory[1]);
  auto localDiagMemoryId = localProxy->addMemory(remoteMemory[2]);
  auto ibLocalMemoryId = ibProxy->addMemory(localMemory);
  auto ibRemoteMemoryId = ibProxy->addMemory(remoteMemory[3]);

  // Separate objects are retained because their device handles alias proxy
  // state.
  auto localSendObject = localProxy->portChannel(
      localSendSemaphore, localSendMemoryId, localLocalMemoryId);
  auto localRecvObject = localProxy->portChannel(
      localRecvSemaphore, localRecvMemoryId, localLocalMemoryId);
  auto localDiagObject = localProxy->portChannel(
      localDiagSemaphore, localDiagMemoryId, localLocalMemoryId);
  auto ibObject =
      ibProxy->portChannel(ibSemaphore, ibRemoteMemoryId, ibLocalMemoryId);
  PortChan duplicateIb{};
  std::unique_ptr<mscclpp::PortChannel> duplicateObject;
  if (cfg.agDualRdma) {
    auto connection = duplicateConnectionFuture.get();
    auto memory = duplicateMemoryFuture.get();
    auto semaphore = comm.buildSemaphore(connection, cross ^ 1, 304);
    auto semaphoreId = ibProxy->addSemaphore(semaphore.get());
    auto memoryId = ibProxy->addMemory(memory);
    duplicateObject = std::make_unique<mscclpp::PortChannel>(
        ibProxy->portChannel(semaphoreId, memoryId, ibLocalMemoryId));
    duplicateIb = duplicateObject->deviceHandle();
  }
  PortChan localSend = localSendObject.deviceHandle();
  PortChan localRecv = localRecvObject.deviceHandle();
  LocalAgChans localAg{};
  localAg.c[ringSuccessor(localRank)] = localSendObject.deviceHandle();
  localAg.c[ringPredecessor(localRank)] = localRecvObject.deviceHandle();
  localAg.c[localRank ^ 3] = localDiagObject.deviceHandle();
  LocalDirectChans localDirect{};
  std::array<std::unique_ptr<mscclpp::MemoryChannel>, 3> directObjects;
  bool needsDirectLocal =
      cfg.agPipeline == 1 || cfg.agPipeline >= 4 || cfg.rsOverlapLocal ||
      cfg.rsPullLocal || cfg.agSerialLocal || cfg.rsSerialLocal ||
      cfg.arSerialLocal || cfg.arOneShot ||
      (cfg.agAuto && cfg.allGather && cfg.minBytes <= kSerialAgMaxBytes) ||
      (cfg.rsAuto && (cfg.reduceScatter || hybridRs) &&
       cfg.minBytes <= kSerialRsMaxBytes) ||
      (cfg.arAuto && !cfg.allGather && !cfg.reduceScatter &&
       cfg.minBytes <= kSerialArMaxBytes);
  if (needsDirectLocal) {
    std::array<std::shared_future<mscclpp::Semaphore>, 3> directSemaphores;
    for (int i = 0; i < 3; ++i)
      directSemaphores[i] =
          comm.buildSemaphore(connections[i], peers[i], 320);
    for (int i = 0; i < 3; ++i) {
      directObjects[i] = std::make_unique<mscclpp::MemoryChannel>(
          directSemaphores[i].get(), remoteMemory[i], localMemory, buffer);
      localDirect.c[peers[i] - nodeBase] = directObjects[i]->deviceHandle();
    }
  }
  // MSCCL++ IPC import can leave a handled invalid-handle probe status.
  // Clear only that known setup status before checking our kernel launches.
  auto setupStatus = cudaGetLastError();
  if (setupStatus != cudaSuccess &&
      setupStatus != cudaErrorInvalidResourceHandle)
    CUDACHECK(setupStatus);
  CUDACHECK(cudaDeviceSynchronize());
  PortChan ib = ibObject.deviceHandle();
  if (!cfg.hostOneShot) {
    localProxy->startProxy();
    ibProxy->startProxy();
  }

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t timingStart = nullptr, timingEnd = nullptr;
  if (cfg.gpuTiming) {
    CUDACHECK(cudaEventCreate(&timingStart));
    CUDACHECK(cudaEventCreate(&timingEnd));
  }
  std::vector<float> hostIn, hostOut;

  auto fill = [&](size_t count, int generation) {
    size_t inputCount = cfg.allGather ? count / kWorld : count;
    hostIn.resize(inputCount);
    for (size_t i = 0; i < inputCount; i++) {
      hostIn[i] = (float)(rank + 1 + generation * kWorld) *
                  (float)(1 + i % 97);
    }
    if (cfg.allGather)
      CUDACHECK(cudaMemsetAsync(buffer + layout.data, 0xff,
                                count * sizeof(float), stream));
    size_t inputOffset = cfg.allGather ? layout.ibSend : layout.data;
    bool serialRs = cfg.rsSerialLocal ||
                    (cfg.rsAuto && count * sizeof(float) <=
                                      kSerialRsMaxBytes);
    if (cfg.reduceScatter && !serialRs && cfg.rsRing &&
        (!cfg.rsDirectRing ||
         (cfg.rsAuto && autoPackedRs(count * sizeof(float)) &&
          !mediumDirectRs(count * sizeof(float)))))
      inputOffset = layout.immutableInput;
    CUDACHECK(cudaMemcpyAsync(buffer + inputOffset,
                              hostIn.data(), inputCount * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  };
  auto verify = [&](size_t count, int generation) {
    size_t outputCount = cfg.reduceScatter ? count / kWorld : count;
    char* output = buffer;
    if (cfg.reduceScatter)
      output += layout.ibSend +
                size_t(rank < kLocalRanks ? 0 : 1) * outputCount *
                    sizeof(float);
    hostOut.resize(outputCount);
    CUDACHECK(cudaMemcpy(hostOut.data(), output,
                         outputCount * sizeof(float),
                         cudaMemcpyDeviceToHost));
    double worst = 0.0;
    size_t firstBad = outputCount;
    double rankSum = (double)kWorld * (kWorld + 1) / 2.0 +
                     generation * kWorld * kWorld;
    for (size_t i = 0; i < outputCount; i++) {
      double want = cfg.allGather
                        ? (double)(i / (count / kWorld) + 1 +
                                   generation * kWorld) *
                              (double)(1 + (i % (count / kWorld)) % 97)
                    : cfg.reduceScatter
                        ? rankSum *
                              (double)(1 +
                                       (size_t(rank) * outputCount + i) % 97)
                        : rankSum * (double)(1 + i % 97);
      double error = std::fabs((double)hostOut[i] - want) / want;
      if ((!std::isfinite(error) || error > 1e-5) && firstBad == outputCount)
        firstBad = i;
      worst = std::max(worst, std::isfinite(error) ? error : INFINITY);
    }
    if (firstBad != outputCount) {
      double want = cfg.allGather
                        ? (double)(firstBad / (count / kWorld) + 1 +
                                   generation * kWorld) *
                              (double)(1 + (firstBad % (count / kWorld)) % 97)
                    : cfg.reduceScatter
                        ? rankSum *
                              (double)(1 + (size_t(rank) * outputCount +
                                            firstBad) %
                                           97)
                        : rankSum * (double)(1 + firstBad % 97);
      std::fprintf(
          stderr,
          "rank %d first bad index %zu got %.1f want %.1f relerr %.3e\n", rank,
          firstBad, (double)hostOut[firstBad], want,
          std::fabs((double)hostOut[firstBad] - want) / want);
    }
    return worst;
  };
  auto useAgRing = [&](size_t bytes) {
    return cfg.agAuto ? bytes == 1024 || bytes >= (1UL << 20) : cfg.agRing;
  };
  auto useAgSerial = [&](size_t bytes) {
    return cfg.agSerialLocal ||
           (cfg.agAuto && bytes <= kSerialAgMaxBytes);
  };
  auto useRsSerial = [&](size_t bytes) {
    return cfg.rsSerialLocal ||
           (cfg.rsAuto && bytes <= kSerialRsMaxBytes);
  };
  auto useArSerial = [&](size_t bytes) {
    return cfg.arSerialLocal ||
           (cfg.arAuto && bytes <= kSerialArMaxBytes);
  };
  auto useRsDirectRing = [&](size_t bytes) {
    if (!cfg.rsAuto) return cfg.rsDirectRing;
    return autoDirectRs(bytes) || (cfg.reduceScatter && mediumDirectRs(bytes));
  };
  auto useArFlatGather = [&](size_t bytes) {
    return cfg.arAuto ? bytes <= (1UL << 18) : cfg.flatAg;
  };
  // Two receive slots prevent a faster partner's next call from overwriting
  // data still being consumed. A partner cannot reach call i+2 until it has
  // received call i+1, by which time this rank has finished consuming call i.
  // The allocation already reserves two maximum-size RS shards for IB scratch.
  std::unique_ptr<uccl_lite::host_gather::Gather> hostGather;
  if (cfg.hostGather && (cfg.hostOneShot || cfg.allGather ||
                         cfg.maxBytes > kHybridSerialArMaxBytes))
    hostGather = std::make_unique<uccl_lite::host_gather::Gather>(
        rank, localComm, cfg.hostOneShot ? cfg.maxBytes : cfg.maxBytes / kWorld,
        cfg.hostPieceBytes);
  unsigned overlapSequence = 0;
  unsigned agSequence = 0;
  unsigned oneShotSequence = 0;
  auto launch = [&](size_t count, size_t bytes) {
    CUDACHECK(cudaSetDevice(localRank));
    if (hostGather) uccl_lite::host_gather::clearMpiProbeStatus();
    if (cfg.hostOneShot) {
      hostGather->run(buffer + layout.data, buffer + layout.data, bytes, true);
      return;
    }
    if (cfg.arOneShot) {
      uccl_lite::multinode_experiment::oneShotAllReduce<<<1, 256, 0, stream>>>(
          localDirect, ib, buffer, count, layout, localRank,
          size_t(oneShotSequence++ & 1) * cfg.maxBytes);
      CUDACHECK(cudaGetLastError());
      return;
    }
    if (cfg.allGather) {
      if (hostGather) {
        hostGather->run(buffer + layout.ibSend, buffer + layout.data,
                        bytes / kWorld);
        return;
      }
      if (cfg.agPipeline) {
        size_t slot = agSequence++ & 1;
        size_t receiveOffset =
            layout.ibScratch + slot * (cfg.maxBytes / kWorld);
        size_t relayHalfBytes = cfg.maxBytes / kWorld;
        size_t relayOffset = layout.localScratch + slot * 2 * relayHalfBytes;
        if (cfg.agPipeline == 5)
          hierarchicalAllGatherMatched<false>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, reinterpret_cast<float*>(buffer),
                  count, layout, rank, localRank, receiveOffset,
                  cfg.agPieceBytes, cfg.agDepth, cfg.agFirstMask);
        else if (cfg.agPipeline == 4)
          hierarchicalAllGatherMatched<true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, reinterpret_cast<float*>(buffer),
                  count, layout, rank, localRank, receiveOffset,
                  cfg.agPieceBytes, cfg.agDepth, cfg.agFirstMask);
        else if (cfg.agDualRdma && cfg.agPipeline == 1)
          hierarchicalAllGatherPipeline<false, false, true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
        else if (cfg.agDualRdma)
          hierarchicalAllGatherPipeline<true, false, true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
        else if (cfg.agPipeline == 3)
          hierarchicalAllGatherPipeline<true, true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
        else if (cfg.agPipeline == 2)
          hierarchicalAllGatherPipeline<true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
        else
          hierarchicalAllGatherPipeline<false>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
      } else if (useAgSerial(bytes))
        hierarchicalAllGatherSerial<<<1, 256, 0, stream>>>(
            localDirect, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, localRank);
      else if (useAgRing(bytes))
        hierarchicalAllGatherRing<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localSend, localRecv, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, localRank, ringPosition(localRank));
      else
        hierarchicalAllGather<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localAg, ib, reinterpret_cast<float*>(buffer), count, layout,
            rank, localRank);
      return;
    }
    if (hybridRs && bytes <= kHybridSerialArMaxBytes) {
      hierarchicalAllReduceSerial<<<1, 256, 0, stream>>>(
          localDirect, ib, reinterpret_cast<float*>(buffer), count, layout,
          localRank, ringPosition(localRank));
      return;
    }
    if (cfg.reduceScatter || hybridRs) {
      if (cfg.rsPullLocal)
        hierarchicalReduceScatterOverlap<true>
            <<<cfg.blocks, cfg.threads, 0, stream>>>(
                localDirect, ib, reinterpret_cast<float*>(buffer), count,
                layout, rank, localRank,
                layout.ibScratch +
                    size_t(overlapSequence++ & 1) * (cfg.maxBytes / kWorld));
      else if (cfg.rsOverlapLocal)
        hierarchicalReduceScatterOverlap<false>
            <<<cfg.blocks, cfg.threads, 0, stream>>>(
                localDirect, ib, reinterpret_cast<float*>(buffer), count,
                layout, rank, localRank,
                layout.ibScratch +
                    size_t(overlapSequence++ & 1) * (cfg.maxBytes / kWorld));
      else if (useRsSerial(bytes))
        hierarchicalReduceScatterSerial<<<1, 256, 0, stream>>>(
            localDirect, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, localRank);
      else if (useRsDirectRing(bytes)) {
        int availablePieces =
            int(std::max<size_t>(1, count / kWorld / 32));
        int requestedPieces = cfg.rsTargetPieceBytes
                                  ? int(std::max<size_t>(
                                        2, (bytes / kWorld +
                                            cfg.rsTargetPieceBytes - 1) /
                                               cfg.rsTargetPieceBytes))
                                  : cfg.rsPieces;
        if (cfg.rsAuto && cfg.reduceScatter && mediumDirectRs(bytes))
          requestedPieces = 1;
        int pieces = std::min(requestedPieces, availablePieces);
        hierarchicalReduceScatterDirectRing<<<cfg.blocks, cfg.threads, 0,
                                              stream>>>(
            localSend, localRecv, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, ringPosition(localRank), pieces);
      } else if (cfg.rsRing && (!cfg.rsAuto || autoPackedRs(bytes))) {
        int availablePieces =
            int(std::max<size_t>(1, count / kWorld / 32));
        int pieces = cfg.rsAuto ? 1 : std::min(cfg.rsPieces, availablePieces);
        bool fastPack = cfg.rsAuto || cfg.rsFastPack;
        if (fastPack) {
          size_t shard = count / kWorld;
          int blocks = int(std::min<size_t>(128, (shard + 255) / 256));
          packReduceScatterRing<<<dim3(blocks, kWorld), 256, 0, stream>>>(
              reinterpret_cast<float*>(
                  buffer + (hybridRs ? layout.data : layout.immutableInput)),
              reinterpret_cast<float*>(buffer + layout.localScratch), shard);
          if (hybridRs) {
            // Packed RS receives into data. Every GPU must finish reading its
            // in-place AR input before any peer can overwrite that scratch.
            CUDACHECK(cudaGetLastError());
            CUDACHECK(cudaStreamSynchronize(stream));
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
          }
        }
        hierarchicalReduceScatterRing<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localSend, localRecv, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, ringPosition(localRank), fastPack, pieces);
      } else
        hierarchicalReduceScatter<<<cfg.rsAuto ? cfg.rsFlatBlocks : cfg.blocks,
                                    cfg.threads,
                                    0, stream>>>(
            localAg, ib, reinterpret_cast<float*>(buffer), count, layout,
            rank, localRank);
      if (hybridRs) {
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaStreamSynchronize(stream));
        hostGather->run(buffer + layout.ibSend +
                            size_t(rank / kLocalRanks) * (bytes / kWorld),
                        buffer + layout.data, bytes / kWorld);
      }
      return;
    }
    size_t shardBytes = (bytes + kLocalRanks - 1) / kLocalRanks;
    int pieces =
        bytes >= cfg.pipelineMinBytes
            ? (int)std::max<size_t>(
                  2, std::min<size_t>(32, (shardBytes + cfg.pieceBytes - 1) /
                                              cfg.pieceBytes))
            : 1;
    int distance =
        pieces == 1 ? 1 : std::max(1, std::min(cfg.pipelineDistance, pieces));
    if (useArSerial(bytes))
      hierarchicalAllReduceSerial<<<1, 256, 0, stream>>>(
          localDirect, ib, reinterpret_cast<float*>(buffer), count, layout,
          localRank, ringPosition(localRank));
    else
      hierarchicalAllReduce<<<cfg.blocks, cfg.threads, 0, stream>>>(
          localSend, localRecv, localAg, ib, reinterpret_cast<float*>(buffer),
          count, layout, localRank, ringPosition(localRank), pieces, distance,
          useArFlatGather(bytes));
  };

  char host[MPI_MAX_PROCESSOR_NAME];
  int hostLen = 0;
  MPI_Get_processor_name(host, &hostLen);
  char line[512]{};
  std::snprintf(line, sizeof(line),
                "# rank %d node %.*s gpu %d %s rail %d (%s)\n", rank, hostLen,
                host, localRank, localProperty.name, rail,
                mscclpp::getIBDeviceName(ibTransport(rail)).c_str());
  std::vector<char> lines(rank == 0 ? world * sizeof(line) : 0);
  MPICHECK(MPI_Gather(line, sizeof(line), MPI_BYTE, lines.data(), sizeof(line),
                      MPI_BYTE, 0, MPI_COMM_WORLD));
  if (rank == 0) {
    std::printf("# UCCL Lite two-node hierarchical %s, float32\n",
                cfg.allGather      ? "AllGather"
                : cfg.reduceScatter ? "ReduceScatter"
                                    : "AllReduce");
    std::printf("# timing=%s; slowest of eight ranks\n",
                cfg.gpuTiming ? "CUDA event" : "host enqueue+completion");
    for (int r = 0; r < world; r++)
      std::fputs(lines.data() + r * sizeof(line), stdout);
    if (cfg.hostGather)
      std::printf(
          "# explicit host packet gather; chunk=%zu; slots=3; GPU "
          "reduction=%s; timing includes phase handoff\n",
          cfg.hostPieceBytes,
          hybridRs ? "tuned RS 96x512, 8MiB target"
          : cfg.hostOneShot ? "SM full-input sum" : "none");
    if (cfg.allGather || cfg.reduceScatter)
      std::printf("# local %s, remote-rank IB exchange; rail map "
                  "%d,%d,%d,%d; RS pieces %d\n",
                  cfg.allGather ? (cfg.agSerialLocal ? "serial direct local AG"
                                   : cfg.agAuto ? "adaptive ring/multicast"
                                   : cfg.agRing ? "ring" : "multicast")
                                : (cfg.rsSerialLocal ? "serial direct local reduction"
                                   : cfg.rsAuto ? "adaptive direct/packed/flat reduction"
                                   : cfg.rsDirectRing ? "direct ring reduction"
                                   : cfg.rsRing ? (cfg.rsFastPack
                                                    ? "ring reduction + GPU pack"
                                                    : "ring reduction")
                                              : "flat reduction"),
                  cfg.railMap[0], cfg.railMap[1], cfg.railMap[2],
                  cfg.railMap[3], cfg.rsPieces);
    if (cfg.hostOneShot)
      std::printf(
          "# experimental host one-shot: CE D2H raw inputs; UCX "
          "host RDMA; SM mapped-host reduction; no GPU P2P payload\n");
    else if (cfg.arOneShot)
      std::printf("# experimental one-shot: full-input IB exchange; SM local "
                  "pull/reduction; 1 CTA x 256 threads; 2 banks; no result AG\n");
    else if (cfg.rsTargetPieceBytes && rank == 0)
      std::printf("# RS target piece %zu B per destination shard\n",
                  cfg.rsTargetPieceBytes);
    else
      std::printf(
          "# local ring 0->1->3->2->0; AG %s; rail map %d,%d,%d,%d; "
          "piece %zu B; pipeline >= %zu B\n",
          cfg.arSerialLocal ? "serial direct local AR"
          : cfg.arAuto ? "adaptive flat/ring"
                     : cfg.flatAg ? "flat" : "ring",
          cfg.railMap[0], cfg.railMap[1],
          cfg.railMap[2], cfg.railMap[3], cfg.pieceBytes,
          cfg.pipelineMinBytes);
    std::printf("# %12s %12s %8s %10s %10s %10s %10s\n", "size(B)", "count",
                "pieces", "time(us)", "algbw", "busbw", "maxrelerr");
  }

  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    bytes =
        (bytes + 511) & ~(size_t)511;  // divisible into aligned four-way chunks
    size_t count = bytes / sizeof(float);
    size_t shardBytes = bytes / kLocalRanks;
    int pieces =
        (cfg.reduceScatter && useRsSerial(bytes)) ? 1
        : (cfg.rsRing && (!cfg.rsAuto || useRsDirectRing(bytes)))
            ? std::min(
                  cfg.rsTargetPieceBytes
                      ? int(std::max<size_t>(
                            2, (bytes / kWorld +
                                cfg.rsTargetPieceBytes - 1) /
                                   cfg.rsTargetPieceBytes))
                      : cfg.rsPieces,
                  int(std::max<size_t>(1, count / kWorld / 32)))
        : (cfg.allGather || cfg.reduceScatter) ? 1
        : bytes >= cfg.pipelineMinBytes
            ? (int)std::max<size_t>(
                  2, std::min<size_t>(32, (shardBytes + cfg.pieceBytes - 1) /
                                              cfg.pieceBytes))
            : 1;
    if (cfg.reduceScatter && cfg.rsAuto && mediumDirectRs(bytes)) pieces = 1;
    if (hybridRs) {
      pieces =
          useRsSerial(bytes) || !useRsDirectRing(bytes)
              ? 1
              : std::min(int(std::max<size_t>(
                             2, (bytes / kWorld + cfg.rsTargetPieceBytes - 1) /
                                    cfg.rsTargetPieceBytes)),
                         int(std::max<size_t>(1, count / kWorld / 32)));
    }
    if (rank == 0) {
      char const* selected =
          cfg.hostOneShot  ? "oneshot-host-input-reduction"
          : cfg.arOneShot  ? "oneshot-remote-input-local-pull"
          : cfg.hostGather ? (hybridRs ? (bytes <= kHybridSerialArMaxBytes
                                              ? "serial-local"
                                              : "gpu-rs-host-ag")
                                       : "host-ag-eight-rank")
          : cfg.allGather
              ? (cfg.agPipeline == 5   ? "matched-sm"
                 : cfg.agPipeline == 4 ? "matched-ce"
                 : cfg.agDualRdma
                     ? (cfg.agPipeline == 1 ? "dual-rdma-sm" : "dual-rdma-ce")
                 : cfg.agPipeline == 3 ? "pipeline-ce-pair"
                 : cfg.agPipeline == 2 ? "pipeline-ce"
                 : cfg.agPipeline == 1 ? "pipeline-sm"
                 : useAgSerial(bytes)  ? "serial-local"
                 : useAgRing(bytes)    ? "ring"
                                       : "multicast")
          : cfg.reduceScatter
              ? (cfg.rsOverlapLocal       ? "overlap-local"
                 : cfg.rsPullLocal        ? "pull-local"
                 : useRsSerial(bytes)     ? "serial-local"
                 : useRsDirectRing(bytes) ? "direct-ring"
                 : autoPackedRs(bytes)    ? "packed-ring"
                                          : "flat")
              : (useArSerial(bytes)       ? "serial-local"
                 : useArFlatGather(bytes) ? "flat-local-gather"
                                          : "ring-local-gather");
      std::printf("# selected,%s,%zu,%s\n",
                  cfg.allGather ? "allgather"
                  : cfg.reduceScatter ? "reducescatter" : "allreduce",
                  bytes, selected);
      if (cfg.agPipeline)
        std::printf(
            "# "
            "ag_pipeline,piece_bytes=%zu,pieces=%zu,depth=%d,blocks=%d,threads="
            "%d,slots=2,dual=%d,first_mask=%d\n",
            cfg.agPieceBytes,
            (bytes / kWorld + cfg.agPieceBytes - 1) / cfg.agPieceBytes,
            cfg.agDepth, cfg.blocks, cfg.threads, int(cfg.agDualRdma),
            cfg.agFirstMask);
    }
    double relerr = -1.0;
    if (cfg.hostRepeatCheck) {
      fill(count, 1);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      for (int repeat = 0; repeat < cfg.hostRepeatCheck; ++repeat)
        launch(count, bytes);
      CUDACHECK(cudaStreamSynchronize(stream));
      hostOut.resize(count);
      CUDACHECK(cudaMemcpy(hostOut.data(), buffer + layout.data, bytes,
                           cudaMemcpyDeviceToHost));
      double factor = (double(kWorld) * (kWorld + 1) / 2 + kWorld * kWorld) *
                      std::pow(double(kWorld), cfg.hostRepeatCheck - 1);
      double error = 0;
      for (size_t i = 0; i < count; ++i) {
        double want = factor * (1 + i % 97);
        double e = std::fabs(double(hostOut[i]) - want) / want;
        error = std::max(error, std::isfinite(e) ? e : INFINITY);
      }
      double worst = 0;
      MPICHECK(MPI_Allreduce(&error, &worst, 1, MPI_DOUBLE, MPI_MAX,
                             MPI_COMM_WORLD));
      if (rank == 0)
        std::printf("# host_repeat_check,%zu,%d,%.3e\n", bytes,
                    cfg.hostRepeatCheck, worst);
      if (worst > 1e-5) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (cfg.check) {
      int generations = (cfg.allGather || cfg.reduceScatter) ? 2 : 1;
      for (int generation = 0; generation < generations; ++generation) {
        fill(count, generation);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        launch(count, bytes);
        CUDACHECK(cudaStreamSynchronize(stream));
        relerr = std::max(relerr, verify(count, generation));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
    }

    double us = 0;
    if (cfg.checkEach) {
      double timedSum = 0;
      for (int i = 0; i < cfg.warmup + cfg.iters; ++i) {
        int generation = 10 + i;
        fill(count, generation);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (cfg.gpuTiming) CUDACHECK(cudaEventRecord(timingStart, stream));
        auto begin = std::chrono::steady_clock::now();
        launch(count, bytes);
        if (cfg.gpuTiming) CUDACHECK(cudaEventRecord(timingEnd, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::steady_clock::now();
        double localUs =
            std::chrono::duration<double, std::micro>(end - begin).count();
        if (cfg.gpuTiming) {
          float ms = 0;
          CUDACHECK(cudaEventElapsedTime(&ms, timingStart, timingEnd));
          localUs = double(ms) * 1000;
        }
        double localErr = cfg.check ? verify(count, generation) : -1;
        relerr = std::max(relerr, localErr);
        double callUs = 0, callErr = 0;
        MPICHECK(MPI_Reduce(&localUs, &callUs, 1, MPI_DOUBLE, MPI_MAX, 0,
                            MPI_COMM_WORLD));
        MPICHECK(MPI_Reduce(&localErr, &callErr, 1, MPI_DOUBLE, MPI_MAX, 0,
                            MPI_COMM_WORLD));
        if (rank == 0 && i >= cfg.warmup) {
          timedSum += callUs;
          std::printf("# sample,%s,%zu,%d,%.3f,%.3e\n",
                      cfg.allGather ? "allgather" : cfg.reduceScatter
                                                  ? "reducescatter"
                                                  : "allreduce",
                      bytes, i - cfg.warmup + 1, callUs, callErr);
        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
      if (rank == 0) us = timedSum / cfg.iters;
    } else {
      fill(count, (cfg.allGather || cfg.reduceScatter) ? 1 : 0);
      for (int i = 0; i < cfg.warmup; i++) launch(count, bytes);
      CUDACHECK(cudaStreamSynchronize(stream));
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      if (cfg.gpuTiming) CUDACHECK(cudaEventRecord(timingStart, stream));
      auto begin = std::chrono::steady_clock::now();
      for (int i = 0; i < cfg.iters; i++) launch(count, bytes);
      if (cfg.gpuTiming) CUDACHECK(cudaEventRecord(timingEnd, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
      auto end = std::chrono::steady_clock::now();
      if (cfg.check && (cfg.allGather || cfg.reduceScatter)) {
        relerr = std::max(relerr, verify(count, 1));
        fill(count, 2);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        launch(count, bytes);
        CUDACHECK(cudaStreamSynchronize(stream));
        relerr = std::max(relerr, verify(count, 2));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
      us = std::chrono::duration<double, std::micro>(end - begin).count() /
           cfg.iters;
      if (cfg.gpuTiming) {
        float ms = 0;
        CUDACHECK(cudaEventElapsedTime(&ms, timingStart, timingEnd));
        us = double(ms) * 1000 / cfg.iters;
      }
    }
    double worstUs = 0.0, worstErr = 0.0;
    MPICHECK(
        MPI_Reduce(&us, &worstUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&relerr, &worstErr, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    int failed = rank == 0 && cfg.check && worstErr > 1e-5;
    MPICHECK(MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (failed) MPI_Abort(MPI_COMM_WORLD, 1);
    if (rank == 0) {
      double algbw = (double)bytes / (worstUs * 1e-6) / 1e9;
      double busbw = algbw *
                     ((cfg.allGather || cfg.reduceScatter) ? 1.0 : 2.0) *
                     (kWorld - 1) / kWorld;
      std::printf("  %12zu %12zu %8d %10.2f %10.2f %10.2f %10.1e\n", bytes,
                  count, pieces, worstUs, algbw, busbw, worstErr);
      std::fflush(stdout);
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (bytes > cfg.maxBytes / 2) break;
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  hostGather.reset();
  if (!cfg.hostOneShot) {
    ibProxy->stopProxy();
    localProxy->stopProxy();
  }
  if (timingEnd) CUDACHECK(cudaEventDestroy(timingEnd));
  if (timingStart) CUDACHECK(cudaEventDestroy(timingStart));
  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(buffer));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}
