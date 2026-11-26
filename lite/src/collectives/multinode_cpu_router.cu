// CPU-router variant of the two-node hierarchical AllReduce.
//
// Local communication stays on the NUMA-aware CUDA-IPC ring.  The reduced
// quarter owned by each GPU is staged through pinned host memory and exchanged
// by MPI/UCX over the NUMA-local HCA:
//
//   local RS -> GPU-to-CPU -> UCX/IB -> CPU-to-GPU -> local AG
//
// This version intentionally does not register GPU memory with IB and therefore
// does not require GPUDirect RDMA, nvidia_peermem, or GDRCopy.
// By default the cross-node partials are reduced on the GPU.  Pass `-r cpu` to
// add the two pinned-host partials on the CPU before copying the result back.
// Pass `-m fused --direct-recv 1` to dual-write the final local reduction into
// GPU and pinned host memory, then RDMA-write the host copy to the remote GPU.
// Add `--inband 1` to replace per-piece MPI handshakes with an RDMA-written
// host sequence doorbell and one GPUDirect visibility flush per collective.

#define main uccl_lite_gpudirect_main_disabled
#include "multinode_allreduce.cu"
#undef main

#include <sched.h>

constexpr int kMaxInbandPieces = 256;
constexpr size_t kInbandFlagBytes = 4096;

__device__ void cpuRouterLocalReduceScatterPiece(
    PortChan send, PortChan recv, float* data, float* scratch,
    float* gpuPartial, float* hostPartial, size_t nelem, int pos,
    size_t pieceOff, size_t pieceLen, size_t chunkElems,
    size_t scratchByteOffset, int tid, int nth) {
  const size_t chunkBytes = chunkElems * sizeof(float);
  for (int step = 0; step < kLocalRanks - 1; step++) {
    int cSend = (pos - step + kLocalRanks) % kLocalRanks;
    int cRecv = (cSend - 1 + kLocalRanks) % kLocalRanks;
    size_t sendOff = (size_t)cSend * chunkElems + pieceOff;
    size_t recvOff = (size_t)cRecv * chunkElems + pieceOff;
    size_t sendLen = sendOff >= nelem ? 0 : min(pieceLen, nelem - sendOff);
    size_t recvLen = recvOff >= nelem ? 0 : min(pieceLen, nelem - recvOff);
    publish();
    if (tid == 0)
      send.putWithSignal(scratchByteOffset + (size_t)step * chunkBytes +
                             pieceOff * sizeof(float),
                         sendOff * sizeof(float), sendLen * sizeof(float));
    if (tid == 0) recv.wait();
    gMultiNodeSyncer.sync(gridDim.x);
    if (recvLen) {
      float* incoming = scratch + (size_t)step * chunkElems + pieceOff;
      if (step == kLocalRanks - 2 && hostPartial) {
        size_t n4 = recvLen / 4;
        float4* gpu4 = reinterpret_cast<float4*>(gpuPartial + pieceOff);
        float4* host4 = reinterpret_cast<float4*>(hostPartial + pieceOff);
        float4 const* data4 = reinterpret_cast<float4 const*>(data + recvOff);
        float4 const* incoming4 = reinterpret_cast<float4 const*>(incoming);
        for (size_t i = tid; i < n4; i += nth) {
          float4 x = data4[i], y = incoming4[i];
          x.x += y.x;
          x.y += y.y;
          x.z += y.z;
          x.w += y.w;
          gpu4[i] = x;
          host4[i] = x;
        }
        for (size_t i = n4 * 4 + tid; i < recvLen; i += nth) {
          float value = data[recvOff + i] + incoming[i];
          gpuPartial[pieceOff + i] = value;
          hostPartial[pieceOff + i] = value;
        }
      } else if (step == kLocalRanks - 2) {
        addOut(gpuPartial + pieceOff, data + recvOff, incoming, recvLen, tid,
               nth);
      } else {
        addInto(data + recvOff, incoming, recvLen, tid, nth);
      }
    }
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

__global__ void cpuRouterLocalRs(PortChan localSend, PortChan localRecv,
                                 float* base, size_t nelem, Layout layout,
                                 int ringPos, float* fusedHostPartial) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  float* data =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  float* scratch = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                            layout.localScratch);
  float* localPartial =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.ibSend);
  size_t chunkElems =
      ((nelem + kLocalRanks - 1) / kLocalRanks + 31) & ~(size_t)31;
  size_t ownBase = (size_t)((ringPos + 1) % kLocalRanks) * chunkElems;
  size_t ownLen = ownBase >= nelem ? 0 : min(chunkElems, nelem - ownBase);
  cpuRouterLocalReduceScatterPiece(
      localSend, localRecv, data, scratch, localPartial, fusedHostPartial,
      nelem, ringPos, 0, ownLen, chunkElems, layout.localScratch, tid, nth);
}

__global__ void cpuRouterFinishPiece(PortChan localSend, PortChan localRecv,
                                     float* base, size_t nelem, Layout layout,
                                     int ringPos, size_t pieceOff,
                                     size_t pieceLen,
                                     float const* fusedHostPartial) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  float* data =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  float const* localPartial =
      fusedHostPartial ? fusedHostPartial
                       : reinterpret_cast<float*>(
                             reinterpret_cast<char*>(base) + layout.ibSend);
  float* remotePartial = reinterpret_cast<float*>(
      reinterpret_cast<char*>(base) + layout.ibScratch);
  size_t chunkElems =
      ((nelem + kLocalRanks - 1) / kLocalRanks + 31) & ~(size_t)31;
  size_t ownBase = (size_t)((ringPos + 1) % kLocalRanks) * chunkElems;
  size_t ownLen = ownBase >= nelem ? 0 : min(chunkElems, nelem - ownBase);
  size_t len = pieceOff >= ownLen ? 0 : min(pieceLen, ownLen - pieceOff);
  if (len)
    addOut(data + ownBase + pieceOff, localPartial + pieceOff,
           remotePartial + pieceOff, len, tid, nth);
  gMultiNodeSyncer.sync(gridDim.x);
  localAllGatherPiece(localSend, localRecv, data, nelem, ringPos, pieceOff, len,
                      chunkElems, tid);
}

__global__ void cpuRouterGatherPiece(PortChan localSend, PortChan localRecv,
                                     float* base, size_t nelem, Layout layout,
                                     int ringPos, size_t pieceOff,
                                     size_t pieceLen) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float* data =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  size_t chunkElems =
      ((nelem + kLocalRanks - 1) / kLocalRanks + 31) & ~(size_t)31;
  gMultiNodeSyncer.sync(gridDim.x);
  localAllGatherPiece(localSend, localRecv, data, nelem, ringPos, pieceOff,
                      pieceLen, chunkElems, tid);
}

struct CpuRouterConfig {
  size_t minBytes = 1UL << 10;
  size_t maxBytes = 1UL << 30;
  int warmup = 5;
  int iters = 10;
  int blocks = 24;
  int threads = 512;
  size_t pieceBytes = 4UL << 20;
  bool fusedSmStage = false;
  bool directGpuRecv = false;
  bool inbandSignal = false;
  bool cpuReduce = false;
  bool check = true;
};

static int envInt(char const* name, int fallback) {
  char const* value = std::getenv(name);
  return value ? std::atoi(value) : fallback;
}

static void configureCpuAndRailBeforeMpi() {
  int localRank = envInt("OMPI_COMM_WORLD_LOCAL_RANK", 0);
  int numa = localRank < 2 ? 0 : 1;
  int cpu = numa == 0 ? localRank * 2 : 1 + (localRank - 2) * 2;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  sched_setaffinity(0, sizeof(set), &set);

  // Must be set before MPI_Init so each CPU router chooses its local rail.
  if (!std::getenv("UCX_NET_DEVICES"))
    setenv("UCX_NET_DEVICES", numa == 0 ? "mlx5_0:1" : "mlx5_1:1", 1);
  if (!std::getenv("UCX_TLS")) setenv("UCX_TLS", "rc_mlx5,dc_mlx5,sm,self", 1);
}

int main(int argc, char** argv) {
  configureCpuAndRailBeforeMpi();
  MPICHECK(MPI_Init(&argc, &argv));
  int rank = 0, world = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
  if (world != kWorld) {
    if (rank == 0)
      std::fprintf(stderr, "requires exactly %d MPI ranks\n", kWorld);
    MPI_Finalize();
    return 2;
  }

  CpuRouterConfig cfg;
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
      else if (key == "-k")
        cfg.pieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-m") {
        std::string value = next();
        if (value == "fused")
          cfg.fusedSmStage = true;
        else if (value == "memcpy")
          cfg.fusedSmStage = false;
        else
          throw std::runtime_error("-m must be fused or memcpy");
      } else if (key == "--direct-recv")
        cfg.directGpuRecv = std::atoi(next()) != 0;
      else if (key == "--inband")
        cfg.inbandSignal = std::atoi(next()) != 0;
      else if (key == "-r") {
        std::string value = next();
        if (value == "cpu")
          cfg.cpuReduce = true;
        else if (value == "gpu")
          cfg.cpuReduce = false;
        else
          throw std::runtime_error("-r must be cpu or gpu");
      } else if (key == "-c")
        cfg.check = std::atoi(next()) != 0;
      else
        throw std::runtime_error("unknown argument " + key);
    }
  } catch (std::exception const& error) {
    if (rank == 0) std::fprintf(stderr, "%s\n", error.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                               MPI_INFO_NULL, &localComm));
  int localRank = 0, localSize = 0;
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  MPICHECK(MPI_Comm_size(localComm, &localSize));
  if (localSize != kLocalRanks) MPI_Abort(MPI_COMM_WORLD, 2);
  CUDACHECK(cudaSetDevice(localRank));
  int rail = localRank < 2 ? 0 : 1;
  if (cfg.directGpuRecv && rail >= mscclpp::getIBDeviceCount()) {
    if (rank == 0)
      std::fprintf(stderr, "direct receive requires IB-enabled MSCCL++\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
  mscclpp::UniqueId id{};
  if (rank == 0) id = mscclpp::TcpBootstrap::createUniqueId();
  MPICHECK(MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD));
  bootstrap->initialize(id, 120);
  mscclpp::Communicator comm(bootstrap);

  size_t dataBytes = cfg.maxBytes;
  size_t slotBytes = (dataBytes + kLocalRanks - 1) / kLocalRanks + 4096;
  Layout layout{0, dataBytes, 2 * dataBytes, 2 * dataBytes + slotBytes};
  size_t bufferBytes = layout.ibScratch + slotBytes;
  char* buffer = nullptr;
  CUDACHECK(cudaMalloc(&buffer, bufferBytes));
  CUDACHECK(cudaMemset(buffer, 0, bufferBytes));
  mscclpp::TransportFlags memoryTransports = mscclpp::Transport::CudaIpc;
  if (cfg.directGpuRecv) memoryTransports |= ibTransport(rail);
  auto localMemory = comm.registerMemory(buffer, bufferBytes, memoryTransports);

  int nodeBase = rank - localRank;
  int succ = nodeBase + ringSuccessor(localRank);
  int pred = nodeBase + ringPredecessor(localRank);
  std::array<int, 2> peers{succ, pred};
  std::array<std::shared_future<mscclpp::Connection>, 2> connectionFutures;
  std::array<std::shared_future<mscclpp::RegisteredMemory>, 2> memoryFutures;
  for (int i = 0; i < 2; i++) {
    connectionFutures[i] =
        comm.connect(mscclpp::Transport::CudaIpc, peers[i], 410);
    comm.sendMemory(localMemory, peers[i], 411);
    memoryFutures[i] = comm.recvMemory(peers[i], 411);
  }
  std::array<mscclpp::Connection, 2> connections{connectionFutures[0].get(),
                                                 connectionFutures[1].get()};
  std::array<mscclpp::RegisteredMemory, 2> remoteMemory{memoryFutures[0].get(),
                                                        memoryFutures[1].get()};
  std::array<std::shared_future<mscclpp::Semaphore>, 2> semaphoreFutures{
      comm.buildSemaphore(connections[0], peers[0], 412),
      comm.buildSemaphore(connections[1], peers[1], 412)};

  int crossRank = rank ^ kLocalRanks;
  mscclpp::Connection directConnection;
  mscclpp::RegisteredMemory remoteGpuMemory;
  if (cfg.directGpuRecv) {
    auto connectionFuture = comm.connect(ibTransport(rail), crossRank, 420);
    comm.sendMemory(localMemory, crossRank, 421);
    auto memoryFuture = comm.recvMemory(crossRank, 421);
    directConnection = connectionFuture.get();
    remoteGpuMemory = memoryFuture.get();
  }
  uint64_t* inbandHostFlags = nullptr;
  mscclpp::RegisteredMemory inbandHostFlagMemory;
  mscclpp::RegisteredMemory remoteHostFlagMemory;
  if (cfg.inbandSignal) {
    if (posix_memalign(reinterpret_cast<void**>(&inbandHostFlags), 4096,
                       kInbandFlagBytes) != 0)
      MPI_Abort(MPI_COMM_WORLD, 2);
    std::memset(inbandHostFlags, 0, kInbandFlagBytes);
    inbandHostFlagMemory = comm.registerMemory(
        inbandHostFlags, kInbandFlagBytes, ibTransport(rail));
    comm.sendMemory(inbandHostFlagMemory, crossRank, 423);
    remoteHostFlagMemory = comm.recvMemory(crossRank, 423).get();
  }

  auto proxy = std::make_shared<mscclpp::ProxyService>();
  auto localMemoryId = proxy->addMemory(localMemory);
  auto sendMemoryId = proxy->addMemory(remoteMemory[0]);
  auto recvMemoryId = proxy->addMemory(remoteMemory[1]);
  auto sendSemaphore = proxy->addSemaphore(semaphoreFutures[0].get());
  auto recvSemaphore = proxy->addSemaphore(semaphoreFutures[1].get());
  auto sendObject =
      proxy->portChannel(sendSemaphore, sendMemoryId, localMemoryId);
  auto recvObject =
      proxy->portChannel(recvSemaphore, recvMemoryId, localMemoryId);
  PortChan localSend = sendObject.deviceHandle();
  PortChan localRecv = recvObject.deviceHandle();
  proxy->startProxy();

  size_t hostSlotBytes = std::min(slotBytes, cfg.pieceBytes + 4096);
  float *hostSend[2]{}, *hostRecv[2]{};
  for (int i = 0; i < 2; i++) {
    CUDACHECK(
        cudaHostAlloc(&hostSend[i], hostSlotBytes, cudaHostAllocPortable));
    CUDACHECK(
        cudaHostAlloc(&hostRecv[i], hostSlotBytes, cudaHostAllocPortable));
  }
  float* fusedHost = nullptr;
  float* fusedHostDev = nullptr;
  size_t fusedHostBytes = (slotBytes + 4095) & ~(size_t)4095;
  mscclpp::RegisteredMemory fusedHostMemory;
  if (cfg.fusedSmStage) {
    if (cfg.directGpuRecv) {
      if (cfg.cpuReduce) {
        if (rank == 0)
          std::fprintf(stderr, "direct receive requires GPU reduction\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
      }
      if (posix_memalign(reinterpret_cast<void**>(&fusedHost), 4096,
                         fusedHostBytes) != 0)
        MPI_Abort(MPI_COMM_WORLD, 2);
      std::memset(fusedHost, 0, fusedHostBytes);
      fusedHostMemory =
          comm.registerMemory(fusedHost, fusedHostBytes, ibTransport(rail));
      CUDACHECK(
          cudaHostRegister(fusedHost, fusedHostBytes,
                           cudaHostRegisterPortable | cudaHostRegisterMapped));
    } else {
      CUDACHECK(cudaHostAlloc(&fusedHost, fusedHostBytes,
                              cudaHostAllocPortable | cudaHostAllocMapped));
    }
    CUDACHECK(cudaHostGetDevicePointer(&fusedHostDev, fusedHost, 0));
  }
  if (cfg.directGpuRecv) {
    if (!cfg.fusedSmStage) {
      if (rank == 0)
        std::fprintf(stderr, "direct receive requires -m fused -r gpu\n");
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (cfg.inbandSignal && cfg.cpuReduce) {
      if (rank == 0)
        std::fprintf(stderr, "in-band signaling requires GPU reduction\n");
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaStream_t routeStream[2];
  cudaEvent_t rsReady, h2dReady[2];
  CUDACHECK(cudaEventCreateWithFlags(&rsReady, cudaEventDisableTiming));
  for (int i = 0; i < 2; i++) {
    CUDACHECK(
        cudaStreamCreateWithFlags(&routeStream[i], cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&h2dReady[i], cudaEventDisableTiming));
  }
  std::vector<float> hostInput, hostOutput;
  uint64_t sequence = 1;
  size_t flagBaseBytes = slotBytes - kInbandFlagBytes;

  auto fill = [&](size_t count) {
    hostInput.resize(count);
    for (size_t i = 0; i < count; i++)
      hostInput[i] = (float)(rank + 1) * (float)(1 + i % 97);
    CUDACHECK(cudaMemcpyAsync(buffer, hostInput.data(), count * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  };

  auto launch = [&](size_t count) {
    size_t chunkElems =
        ((count + kLocalRanks - 1) / kLocalRanks + 31) & ~(size_t)31;
    size_t ownBase =
        (size_t)((ringPosition(localRank) + 1) % kLocalRanks) * chunkElems;
    size_t ownLen =
        ownBase >= count ? 0 : std::min(chunkElems, count - ownBase);
    size_t pieceElems =
        std::max<size_t>(32, (cfg.pieceBytes / sizeof(float)) & ~(size_t)31);
    int pieces = (int)((ownLen + pieceElems - 1) / pieceElems);

    cpuRouterLocalRs<<<cfg.blocks, cfg.threads, 0, stream>>>(
        localSend, localRecv, reinterpret_cast<float*>(buffer), count, layout,
        ringPosition(localRank), fusedHostDev);
    CUDACHECK(cudaEventRecord(rsReady, stream));

    if (cfg.directGpuRecv && cfg.inbandSignal) {
      if (pieces > kMaxInbandPieces) {
        if (rank == 0)
          std::fprintf(stderr, "in-band piece count exceeds %d\n",
                       kMaxInbandPieces);
        MPI_Abort(MPI_COMM_WORLD, 2);
      }
      CUDACHECK(cudaEventSynchronize(rsReady));
      uint64_t currentSequence = sequence++;
      uint64_t* sourceFlags = reinterpret_cast<uint64_t*>(
          reinterpret_cast<char*>(fusedHost) + flagBaseBytes);
      for (int piece = 0; piece < pieces; piece++) {
        size_t off = (size_t)piece * pieceElems;
        size_t len = off >= ownLen ? 0 : std::min(pieceElems, ownLen - off);
        size_t bytes = len * sizeof(float);
        sourceFlags[piece] = currentSequence;
        directConnection.write(remoteGpuMemory,
                               layout.ibScratch + off * sizeof(float),
                               fusedHostMemory, off * sizeof(float), bytes);
        directConnection.write(
            remoteHostFlagMemory, piece * sizeof(uint64_t), fusedHostMemory,
            flagBaseBytes + piece * sizeof(uint64_t), sizeof(uint64_t));
      }
      directConnection.flush();
      for (int piece = 0; piece < pieces; piece++) {
        while (__atomic_load_n(inbandHostFlags + piece, __ATOMIC_ACQUIRE) !=
               currentSequence) {
        }
      }
      CUDACHECK(cudaDeviceFlushGPUDirectRDMAWrites(
          cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
          cudaFlushGPUDirectRDMAWritesToOwner));
      for (int piece = 0; piece < pieces; piece++) {
        size_t off = (size_t)piece * pieceElems;
        size_t len = off >= ownLen ? 0 : std::min(pieceElems, ownLen - off);
        cpuRouterFinishPiece<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localSend, localRecv, reinterpret_cast<float*>(buffer), count,
            layout, ringPosition(localRank), off, len, nullptr);
      }
      return;
    }

    auto routePiece = [&](int piece) {
      int slot = piece & 1;
      size_t off = (size_t)piece * pieceElems;
      size_t len = off >= ownLen ? 0 : std::min(pieceElems, ownLen - off);
      size_t bytes = len * sizeof(float);
      CUDACHECK(cudaStreamSynchronize(routeStream[slot]));
      if (cfg.fusedSmStage) {
        CUDACHECK(cudaEventSynchronize(rsReady));
      } else {
        CUDACHECK(cudaStreamWaitEvent(routeStream[slot], rsReady, 0));
        CUDACHECK(cudaMemcpyAsync(
            hostSend[slot], buffer + layout.ibSend + off * sizeof(float), bytes,
            cudaMemcpyDeviceToHost, routeStream[slot]));
      }
    };
    auto finishPiece = [&](int piece) {
      int slot = piece & 1;
      size_t off = (size_t)piece * pieceElems;
      size_t len = off >= ownLen ? 0 : std::min(pieceElems, ownLen - off);
      size_t bytes = len * sizeof(float);
      CUDACHECK(cudaStreamSynchronize(routeStream[slot]));
      float* sendPtr = cfg.fusedSmStage ? fusedHost + off : hostSend[slot];
      if (cfg.directGpuRecv) {
        directConnection.write(remoteGpuMemory,
                               layout.ibScratch + off * sizeof(float),
                               fusedHostMemory, off * sizeof(float), bytes);
        directConnection.flush();
        MPICHECK(MPI_Sendrecv(nullptr, 0, MPI_BYTE, crossRank, 501, nullptr, 0,
                              MPI_BYTE, crossRank, 501, MPI_COMM_WORLD,
                              MPI_STATUS_IGNORE));
      } else {
        MPICHECK(MPI_Sendrecv(sendPtr, (int)bytes, MPI_BYTE, crossRank, 500,
                              hostRecv[slot], (int)bytes, MPI_BYTE, crossRank,
                              500, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      }
      if (cfg.cpuReduce) {
        if (cfg.directGpuRecv)
          throw std::runtime_error("CPU reduction requires host receive");
        for (size_t i = 0; i < len; i++) sendPtr[i] += hostRecv[slot][i];
        size_t ownBaseBytes = ownBase * sizeof(float);
        CUDACHECK(cudaMemcpyAsync(
            buffer + layout.data + ownBaseBytes + off * sizeof(float), sendPtr,
            bytes, cudaMemcpyHostToDevice, routeStream[slot]));
      } else if (!cfg.directGpuRecv) {
        CUDACHECK(cudaMemcpyAsync(
            buffer + layout.ibScratch + off * sizeof(float), hostRecv[slot],
            bytes, cudaMemcpyHostToDevice, routeStream[slot]));
      }
      if (!cfg.directGpuRecv) {
        CUDACHECK(cudaEventRecord(h2dReady[slot], routeStream[slot]));
        CUDACHECK(cudaStreamWaitEvent(stream, h2dReady[slot], 0));
      }
      if (cfg.cpuReduce) {
        cpuRouterGatherPiece<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localSend, localRecv, reinterpret_cast<float*>(buffer), count,
            layout, ringPosition(localRank), off, len);
      } else {
        cpuRouterFinishPiece<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localSend, localRecv, reinterpret_cast<float*>(buffer), count,
            layout, ringPosition(localRank), off, len, nullptr);
      }
    };

    if (pieces > 0) routePiece(0);
    for (int piece = 1; piece < pieces; piece++) {
      routePiece(piece);  // D2H(piece) overlaps routing/AG(piece-1)
      finishPiece(piece - 1);
    }
    if (pieces > 0) finishPiece(pieces - 1);
  };

  auto verify = [&](size_t count) {
    CUDACHECK(cudaStreamSynchronize(stream));
    hostOutput.resize(count);
    CUDACHECK(cudaMemcpy(hostOutput.data(), buffer, count * sizeof(float),
                         cudaMemcpyDeviceToHost));
    constexpr double rankSum = (double)kWorld * (kWorld + 1) / 2.0;
    double worst = 0.0;
    for (size_t i = 0; i < count; i++) {
      double want = rankSum * (double)(1 + i % 97);
      worst = std::max(worst, std::fabs((double)hostOutput[i] - want) / want);
    }
    return worst;
  };

  if (rank == 0) {
    std::printf("# UCCL Lite CPU-router two-node AllReduce, float32\n");
    std::printf("# stage=%s recv=%s signal=%s reduction=%s piece=%zu B\n",
                cfg.fusedSmStage ? "fused-sm-host" : "memcpy",
                cfg.directGpuRecv ? "gpu-direct" : "host-h2d",
                cfg.inbandSignal ? "in-band" : "handshake",
                cfg.cpuReduce ? "cpu" : "gpu", cfg.pieceBytes);
    std::printf("# %12s %12s %10s %10s %10s %10s\n", "size(B)", "count",
                "time(us)", "algbw", "busbw", "maxrelerr");
  }

  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    bytes = (bytes + 511) & ~(size_t)511;
    size_t count = bytes / sizeof(float);
    double error = -1.0;
    if (cfg.check) {
      fill(count);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      launch(count);
      error = verify(count);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    fill(count);
    for (int i = 0; i < cfg.warmup; i++) launch(count);
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; i++) launch(count);
    CUDACHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - begin).count() /
                cfg.iters;
    double worstUs = 0.0, worstError = 0.0;
    MPICHECK(
        MPI_Reduce(&us, &worstUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&error, &worstError, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    if (rank == 0) {
      double algbw = (double)bytes / (worstUs * 1e-6) / 1e9;
      std::printf("  %12zu %12zu %10.2f %10.2f %10.2f %10.1e\n", bytes, count,
                  worstUs, algbw, algbw * 1.75, worstError);
      std::fflush(stdout);
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (bytes > cfg.maxBytes / 2) break;
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  proxy->stopProxy();
  CUDACHECK(cudaEventDestroy(rsReady));
  for (int i = 0; i < 2; i++) {
    CUDACHECK(cudaEventDestroy(h2dReady[i]));
    CUDACHECK(cudaStreamDestroy(routeStream[i]));
  }
  CUDACHECK(cudaStreamDestroy(stream));
  for (int i = 0; i < 2; i++) {
    CUDACHECK(cudaFreeHost(hostSend[i]));
    CUDACHECK(cudaFreeHost(hostRecv[i]));
  }
  if (fusedHost) {
    if (cfg.directGpuRecv) {
      fusedHostMemory = mscclpp::RegisteredMemory{};
      CUDACHECK(cudaHostUnregister(fusedHost));
      std::free(fusedHost);
    } else {
      CUDACHECK(cudaFreeHost(fusedHost));
    }
  }
  if (inbandHostFlags) {
    inbandHostFlagMemory = mscclpp::RegisteredMemory{};
    std::free(inbandHostFlags);
  }
  CUDACHECK(cudaFree(buffer));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}
