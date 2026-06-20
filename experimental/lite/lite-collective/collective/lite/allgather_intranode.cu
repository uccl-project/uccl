// Intra-node AllGather implementation: host-staged SHM + CudaIPC paths.
// Included by nccl.cu inside its anonymous namespace so the intra-node
// paths can reuse the NCCL shim's send/recv peer context.
//
// Uses HostStagingBuffer for all shared-memory management.

#include "host_staging_buffer.hpp"

namespace cg = cooperative_groups;

// ── Tuning constants ─────────────────────────────────────────────────────────
static constexpr int kHostAllGatherPollSpinsBeforeYield = 65536;
static constexpr size_t kIntraAllGatherRingMinTotalBytes = 8 * 1024 * 1024;

static constexpr size_t kHostAllGatherMaxTotalBytes   = 1024ULL * 1024 * 1024;
static constexpr size_t kHostAllGatherMinChunkBytes   = 256 * 1024;
static constexpr size_t kHostAllGatherMaxRankBytes    = kHostAllGatherMaxTotalBytes / 4;
static constexpr size_t kHostAllGatherMaxChunks       =
    kHostAllGatherMaxRankBytes / kHostAllGatherMinChunkBytes;
static constexpr size_t kHostAllGatherDefaultMinTotalBytes  = 0;
static constexpr size_t kHostAllGatherDefaultKernelMaxBytes = 4 * 1024;
static constexpr int    kHostAllGatherDefaultKernelThreads  = 256;
static constexpr size_t kHostAllGatherDefaultCoopMaxBytes   = 512 * 1024;
static constexpr size_t kHostAllGatherDefaultTwoKernelBlockBytes = 1024;
static constexpr int    kHostAllGatherDefaultCoopMaxBlocks   = 16;

static inline void hostAllGatherCpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  asm volatile("" ::: "memory");
#endif
}

template <typename Fn>
static ncclResult_t runHostAllGatherGuarded(char const* opName, Fn&& fn) {
  try { fn(); return ncclSuccess; }
  catch (std::exception const& ex) {
    WARN(MSCCLPP_NCCL, opName, " failed: ", ex.what());
    return mapMscclppException(ex);
  } catch (...) {
    WARN(MSCCLPP_NCCL, opName, " failed with an unknown exception");
    return ncclInternalError;
  }
}

// ── Env helpers ──────────────────────────────────────────────────────────────
static bool hostAllGatherMapSlabEnabled() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_MAP_SLAB");
  return e == nullptr || std::strcmp(e, "0") != 0;
}
static bool hostAllGatherSelfKernelEnabled() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_SELF_KERNEL");
  return e != nullptr && std::strcmp(e, "0") != 0;
}
static bool hostAllGatherNumaPlacementEnabled() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_NUMA_PLACE");
  return e != nullptr && std::strcmp(e, "0") != 0;
}
static size_t hostAllGatherMinTotalBytes() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_MIN_BYTES");
  if (!e || !e[0]) return kHostAllGatherDefaultMinTotalBytes;
  char* end = nullptr;
  unsigned long long v = std::strtoull(e, &end, 10);
  return (end == e || v == 0) ? kHostAllGatherDefaultMinTotalBytes : static_cast<size_t>(v);
}
static size_t hostAllGatherKernelMaxBytes() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_KERNEL_MAX_BYTES");
  if (!e || !e[0]) return kHostAllGatherDefaultKernelMaxBytes;
  char* end = nullptr;
  unsigned long long v = std::strtoull(e, &end, 10);
  return (end == e) ? kHostAllGatherDefaultKernelMaxBytes : static_cast<size_t>(v);
}
static int hostAllGatherKernelThreads() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_KERNEL_THREADS");
  if (!e || !e[0]) return kHostAllGatherDefaultKernelThreads;
  char* end = nullptr;
  long v = std::strtol(e, &end, 10);
  return (end == e || v <= 0) ? kHostAllGatherDefaultKernelThreads : static_cast<int>(v);
}
static size_t hostAllGatherCoopMaxBytes() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_COOP_MAX_BYTES");
  if (!e || !e[0]) return kHostAllGatherDefaultCoopMaxBytes;
  char* end = nullptr;
  unsigned long long v = std::strtoull(e, &end, 10);
  return (end == e) ? kHostAllGatherDefaultCoopMaxBytes : static_cast<size_t>(v);
}
static size_t hostAllGatherTwoKernelBlockBytes() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_TWO_KERNEL_BLOCK_BYTES");
  if (!e || !e[0]) return kHostAllGatherDefaultTwoKernelBlockBytes;
  char* end = nullptr;
  unsigned long long v = std::strtoull(e, &end, 10);
  return (end == e || v == 0) ? kHostAllGatherDefaultTwoKernelBlockBytes : static_cast<size_t>(v);
}
static int hostAllGatherCoopMaxBlocks() {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_COOP_MAX_BLOCKS");
  if (!e || !e[0]) return kHostAllGatherDefaultCoopMaxBlocks;
  char* end = nullptr;
  long v = std::strtol(e, &end, 10);
  return (end == e || v <= 0) ? kHostAllGatherDefaultCoopMaxBlocks : static_cast<int>(v);
}
static size_t hostAllGatherChunkBytes(size_t bytesPerRank) {
  char const* e = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_CHUNK_BYTES");
  if (!e || !e[0]) {
    if (bytesPerRank > 1024*1024 && bytesPerRank <= 32*1024*1024) return 1024*1024;
    if (bytesPerRank > 32*1024*1024) return 4*1024*1024;
    return bytesPerRank;
  }
  char* end = nullptr;
  unsigned long long v = std::strtoull(e, &end, 10);
  if (end == e || v < kHostAllGatherMinChunkBytes) return bytesPerRank;
  return std::min(static_cast<size_t>(v), bytesPerRank);
}

// ── GPU kernels (use HsbDeviceHandle for slab/ctrl access) ───────────────────

__device__ static unsigned long long volatile* hsbCtrlU64(char* ctrl, size_t off) {
  return reinterpret_cast<unsigned long long volatile*>(ctrl + off);
}

// Single-CTA kernel: one block writes own data to host-mapped slab via GPU SM,
// polls mapped ctrl flags for all peers, then reads all peers' data back to recv.
__global__ void hostAllGatherMappedKernel(
    char const* send, char* recv,
    char* slabDev, char* ctrlDev,
    size_t slotOffset, size_t bytesPerRank, int rank, int nRanks,
    unsigned long long epoch,
    size_t readyOffset, size_t doneOffset, size_t counterStride) {
  using Vec = unsigned long long;
  size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank;
  size_t fullBytes  = bytesPerRank * static_cast<size_t>(nRanks);
  size_t vecBytes   = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  size_t vecCount   = vecBytes / sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto*       selfSlot = reinterpret_cast<Vec*>(slabDev + slotOffset + selfOffset);
  for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x)  selfSlot[i] = sendVec[i];
  for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank; i += blockDim.x)
    slabDev[slotOffset + selfOffset + i] = send[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    *hsbCtrlU64(ctrlDev, readyOffset + static_cast<size_t>(rank) * counterStride) = epoch;
    for (int r = 0; r < nRanks; ++r)
      while (*hsbCtrlU64(ctrlDev, readyOffset + static_cast<size_t>(r) * counterStride) < epoch) {}
  }
  __syncthreads();
  size_t outVecBytes = (fullBytes / sizeof(Vec)) * sizeof(Vec);
  size_t outVecCount = outVecBytes / sizeof(Vec);
  auto const* slotVec = reinterpret_cast<Vec const*>(slabDev + slotOffset);
  auto* recvVec = reinterpret_cast<Vec*>(recv);
  for (size_t i = threadIdx.x; i < outVecCount; i += blockDim.x) recvVec[i] = slotVec[i];
  for (size_t i = outVecBytes + threadIdx.x; i < fullBytes; i += blockDim.x)
    recv[i] = slabDev[slotOffset + i];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    *hsbCtrlU64(ctrlDev, doneOffset + static_cast<size_t>(rank) * counterStride) = epoch;
  }
}

// Cooperative-launch kernel: multiple CTAs cooperate via grid.sync().
__global__ void hostAllGatherCoopKernel(
    char const* send, char* recv,
    char* slabDev, char* ctrlDev,
    size_t slotOffset, size_t bytesPerRank, int rank, int nRanks,
    unsigned long long epoch,
    size_t readyOffset, size_t doneOffset, size_t counterStride) {
  cg::grid_group grid = cg::this_grid();
  using Vec = unsigned long long;
  size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank;
  size_t fullBytes  = bytesPerRank * static_cast<size_t>(nRanks);
  size_t tid        = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride     = static_cast<size_t>(gridDim.x)  * blockDim.x;
  size_t vecBytes   = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  size_t vecCount   = vecBytes / sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto*       selfSlot = reinterpret_cast<Vec*>(slabDev + slotOffset + selfOffset);
  for (size_t i = tid; i < vecCount; i += stride)         selfSlot[i] = sendVec[i];
  for (size_t i = vecBytes + tid; i < bytesPerRank; i += stride)
    slabDev[slotOffset + selfOffset + i] = send[i];
  grid.sync();
  if (tid == 0) {
    __threadfence_system();
    *hsbCtrlU64(ctrlDev, readyOffset + static_cast<size_t>(rank) * counterStride) = epoch;
    for (int r = 0; r < nRanks; ++r)
      while (*hsbCtrlU64(ctrlDev, readyOffset + static_cast<size_t>(r) * counterStride) < epoch) {}
  }
  grid.sync();
  size_t outVecBytes = (fullBytes / sizeof(Vec)) * sizeof(Vec);
  size_t outVecCount = outVecBytes / sizeof(Vec);
  auto const* slotVec = reinterpret_cast<Vec const*>(slabDev + slotOffset);
  auto* recvVec = reinterpret_cast<Vec*>(recv);
  for (size_t i = tid; i < outVecCount; i += stride) recvVec[i] = slotVec[i];
  for (size_t i = outVecBytes + tid; i < fullBytes; i += stride) recv[i] = slabDev[slotOffset + i];
  grid.sync();
  if (tid == 0) {
    __threadfence_system();
    *hsbCtrlU64(ctrlDev, doneOffset + static_cast<size_t>(rank) * counterStride) = epoch;
  }
}

// Self-copy kernel: D2D copy on user stream (avoids cudaMemcpyAsync overhead
// for the self-rank slot when using GPU SM is beneficial).
__global__ void hostAllGatherSelfCopyKernel(char const* src, char* dst, size_t bytes) {
  using Vec = unsigned long long;
  size_t vecBytes = (bytes / sizeof(Vec)) * sizeof(Vec);
  size_t vecCount = vecBytes / sizeof(Vec);
  size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x)  * blockDim.x;
  auto const* srcVec = reinterpret_cast<Vec const*>(src);
  auto*       dstVec = reinterpret_cast<Vec*>(dst);
  for (size_t i = tid; i < vecCount; i += stride) dstVec[i] = srcVec[i];
  for (size_t i = vecBytes + tid; i < bytes; i += stride) dst[i] = src[i];
}

// ── Context ───────────────────────────────────────────────────────────────────
// Lightweight context: HostStagingBuffer holds all shared memory;
// this struct only owns the CUDA streams, events, and per-call epoch counter.

struct HostAllGatherContext {
  std::mutex initMutex;
  std::condition_variable initCv;
  bool initialized  = false;
  bool initializing = false;
  std::exception_ptr initException;

  bool mapSlab = true;
  bool cooperativeLaunch = false;
  uint64_t epoch = 0;

  // Shared staging buffer (initialized on first call).
  std::unique_ptr<HostStagingBuffer> buf;

  // Per-call CUDA streams and events.
  cudaStream_t d2hStream       = nullptr;
  cudaStream_t h2dStream       = nullptr;
  cudaStream_t h2dStream2      = nullptr;
  cudaEvent_t  inputReadyEvent = nullptr;
  cudaEvent_t  h2dDoneEvent    = nullptr;
  cudaEvent_t  h2dDoneEvent2   = nullptr;

  ~HostAllGatherContext() {
    if (h2dDoneEvent2)  cudaEventDestroy(h2dDoneEvent2);
    if (h2dDoneEvent)   cudaEventDestroy(h2dDoneEvent);
    if (inputReadyEvent) cudaEventDestroy(inputReadyEvent);
    if (h2dStream2)     cudaStreamDestroy(h2dStream2);
    if (h2dStream)      cudaStreamDestroy(h2dStream);
    if (d2hStream)      cudaStreamDestroy(d2hStream);
    // buf destructor handles HostStagingBuffer cleanup
  }
};

static std::mutex gHostAllGatherContextMutex;
static std::unordered_map<ncclComm_t, std::unique_ptr<HostAllGatherContext>>
    gHostAllGatherMappedContexts;
static std::unordered_map<ncclComm_t, std::unique_ptr<HostAllGatherContext>>
    gHostAllGatherPinnedContexts;

static HostAllGatherContext& getHostAllGatherContext(ncclComm_t comm, bool mapSlab) {
  std::lock_guard<std::mutex> lk(gHostAllGatherContextMutex);
  auto& ctxs = mapSlab ? gHostAllGatherMappedContexts : gHostAllGatherPinnedContexts;
  auto& ptr = ctxs[comm];
  if (!ptr) { ptr = std::make_unique<HostAllGatherContext>(); ptr->mapSlab = mapSlab; }
  return *ptr;
}

static void cleanupHostAllGatherContexts(ncclComm_t comm) {
  std::lock_guard<std::mutex> lk(gHostAllGatherContextMutex);
  gHostAllGatherMappedContexts.erase(comm);
  gHostAllGatherPinnedContexts.erase(comm);
}

// ── Context initialization ───────────────────────────────────────────────────

static void initializeHostAllGatherContext(
    HostAllGatherContext& ctx, ncclComm_t comm, int rank, int nRanks,
    int cudaDevice, std::shared_ptr<mscclpp::Communicator> bootstrapComm) {
  {
    std::unique_lock<std::mutex> lk(ctx.initMutex);
    if (ctx.initialized) return;
    if (ctx.initializing) {
      ctx.initCv.wait(lk, [&] { return !ctx.initializing; });
      if (ctx.initException) std::rethrow_exception(ctx.initException);
      if (ctx.initialized) return;
    }
    ctx.initializing = true;
  }

  // RAII guard: on exception, mark context as failed.
  auto fail = [&](std::exception_ptr ep) {
    std::lock_guard<std::mutex> lk(ctx.initMutex);
    ctx.initialized  = false;
    ctx.initializing = false;
    ctx.initException = ep;
    ctx.initCv.notify_all();
  };

  ncclResult_t result = ncclSuccess;
  std::string  errmsg;
  try {
    mscclpp::CudaDeviceGuard devGuard(cudaDevice);

    // Build unique name tag from comm pointer.
    auto nc = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(comm));
    std::string tag = (ctx.mapSlab ? "m" : "p");
    char buf[64]; std::snprintf(buf, 64, "%d_%llx", getpid(), nc); tag += buf;

    // Create HostStagingBuffer — handles slab/ctrl shm, NUMA placement, registration.
    ctx.buf = std::make_unique<HostStagingBuffer>(HostStagingBuffer::create(
        kHostAllGatherMaxTotalBytes / static_cast<size_t>(nRanks), // bytesPerRank
        kHsbMaxSlots,
        bootstrapComm,
        rank, nRanks, cudaDevice,
        ctx.mapSlab,
        hostAllGatherNumaPlacementEnabled(),
        tag));

    int leastPri = 0, greatestPri = 0;
    MSCCLPP_CUDATHROW(cudaDeviceGetStreamPriorityRange(&leastPri, &greatestPri));
    (void)leastPri;
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.d2hStream,  cudaStreamNonBlocking, greatestPri));
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.h2dStream,  cudaStreamNonBlocking, greatestPri));
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.h2dStream2, cudaStreamNonBlocking, greatestPri));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.inputReadyEvent, cudaEventDisableTiming));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent,    cudaEventDisableTiming));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent2,   cudaEventDisableTiming));
    int coop = 0;
    MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, cudaDevice));
    ctx.cooperativeLaunch = (coop != 0);
  } catch (std::exception const& ex) {
    result = mapMscclppException(ex); errmsg = ex.what();
  } catch (...) {
    result = ncclInternalError; errmsg = "unknown exception";
  }

  if (result != ncclSuccess) {
    fail(std::make_exception_ptr(
        mscclpp::Error("HostStagingBuffer init: " + errmsg,
                       mscclpp::ErrorCode::InternalError)));
    throw mscclpp::Error(errmsg, mscclpp::ErrorCode::InternalError);
  }

  bootstrapComm->bootstrap()->barrier();
  {
    std::lock_guard<std::mutex> lk(ctx.initMutex);
    ctx.initialized  = true;
    ctx.initializing = false;
    ctx.initException = nullptr;
  }
  ctx.initCv.notify_all();
}

// ── streamWriteFlag64 / streamWaitFlag64 (still used directly in nccl.cu)  ──

static void streamWriteFlag64(cudaStream_t stream, CUdeviceptr addr, uint64_t value) {
  CUresult res = cuStreamWriteValue64(reinterpret_cast<CUstream>(stream), addr,
                                      static_cast<cuuint64_t>(value),
                                      CU_STREAM_WRITE_VALUE_DEFAULT);
  if (res != CUDA_SUCCESS)
    throw mscclpp::Error("cuStreamWriteValue64 failed in SHM allgather",
                         mscclpp::ErrorCode::SystemError);
}
static void streamWaitFlag64(cudaStream_t stream, CUdeviceptr addr, uint64_t value) {
  CUresult res = cuStreamWaitValue64(reinterpret_cast<CUstream>(stream), addr,
                                     static_cast<cuuint64_t>(value),
                                     CU_STREAM_WAIT_VALUE_GEQ);
  if (res != CUDA_SUCCESS)
    throw mscclpp::Error("cuStreamWaitValue64 failed in SHM allgather",
                         mscclpp::ErrorCode::SystemError);
}

// ── runIntraNodeShmAllGather ──────────────────────────────────────────────────

ncclResult_t runIntraNodeShmAllGather(
    void const* sendbuff, void* recvbuff, size_t bytesPerRank,
    ncclComm_t comm, cudaStream_t stream, int rank, int nRanks,
    int nRanksPerNode, std::shared_ptr<mscclpp::Communicator> bootstrapComm,
    int cudaDevice) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr ||
      nRanks < 2 || nRanks > kHsbMaxRanks || nRanks != nRanksPerNode)
    return ncclInvalidUsage;
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);
  if (fullBytes < hostAllGatherMinTotalBytes()) return ncclInvalidUsage;
  if (bytesPerRank > kHostAllGatherMaxTotalBytes / static_cast<size_t>(nRanks))
    return ncclInvalidUsage;
  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &captureStatus) != cudaSuccess)
    return ncclUnhandledCudaError;
  if (captureStatus != cudaStreamCaptureStatusNone) return ncclInvalidUsage;

  return runHostAllGatherGuarded("intra-node SHM AllGather", [&]() {
    bool useMappedCtx = hostAllGatherMapSlabEnabled() &&
                        fullBytes <= hostAllGatherCoopMaxBytes();
    HostAllGatherContext& ctx = getHostAllGatherContext(comm, useMappedCtx);
    initializeHostAllGatherContext(ctx, comm, rank, nRanks, cudaDevice, bootstrapComm);

    mscclpp::CudaDeviceGuard devGuard(cudaDevice);
    uint64_t epoch = ++ctx.epoch;
    int slot = static_cast<int>((epoch - 1) % kHsbMaxSlots);

    // Slot reuse guard: ensure prior use of this slot is complete before reusing.
    if (epoch > static_cast<uint64_t>(kHsbMaxSlots)) {
      uint64_t reuseEpoch = epoch - static_cast<uint64_t>(kHsbMaxSlots);
      for (int r = 0; r < nRanks; ++r)
        ctx.buf->waitDone(slot, r, reuseEpoch);
    }

    HostStagingBuffer& buf  = *ctx.buf;
    HsbDeviceHandle   h     = buf.deviceHandle();

    size_t chunkBytes  = hostAllGatherChunkBytes(bytesPerRank);
    size_t chunkCount  = (bytesPerRank + chunkBytes - 1) / chunkBytes;
    if (chunkCount > kHostAllGatherMaxChunks)
      throw mscclpp::Error("chunk count exceeds kHostAllGatherMaxChunks",
                           mscclpp::ErrorCode::InvalidUsage);

    auto const* sendBytes = static_cast<char const*>(sendbuff);
    auto*       recvBytes = static_cast<char*>(recvbuff);
    void*       selfOutput = recvBytes + static_cast<size_t>(rank) * bytesPerRank;
    bool useVecKernels = (bytesPerRank % sizeof(unsigned long long) == 0) &&
                         ((reinterpret_cast<uintptr_t>(sendbuff)  & 7) == 0) &&
                         ((reinterpret_cast<uintptr_t>(recvbuff) & 7) == 0);

    // ── Path 1: Single-CTA kernel (tiny messages, ≤ kernelMaxBytes) ─────────
    // GPU SM writes own data to host-mapped slab, polls flags, reads all back.
    if (useVecKernels && chunkCount == 1 &&
        h.slabDev != nullptr && h.ctrlDev != nullptr &&
        fullBytes <= hostAllGatherKernelMaxBytes()) {
      hostAllGatherMappedKernel<<<1, hostAllGatherKernelThreads(), 0, stream>>>(
          sendBytes, recvBytes,
          h.slabDev, h.ctrlDev,
          h.slotOffset(slot), bytesPerRank, rank, nRanks, epoch,
          h.readyFlagOffset(slot), h.doneFlagOffset(slot), h.counterStride);
      MSCCLPP_CUDATHROW(cudaGetLastError());
      return;
    }

    // ── Path 2: Cooperative-launch kernel (small messages, ≤ coopMaxBytes) ──
    if (useVecKernels && chunkCount == 1 && ctx.cooperativeLaunch &&
        h.slabDev != nullptr && h.ctrlDev != nullptr &&
        fullBytes <= hostAllGatherCoopMaxBytes()) {
      size_t blockBytes = hostAllGatherTwoKernelBlockBytes();
      int blocks = std::min<int>(hostAllGatherCoopMaxBlocks(),
                                 std::max<int>(1, static_cast<int>(
                                     (fullBytes + blockBytes - 1) / blockBytes)));
      size_t slotOff    = h.slotOffset(slot);
      size_t readyOff   = h.readyFlagOffset(slot);
      size_t doneOff    = h.doneFlagOffset(slot);
      size_t cStride    = h.counterStride;
      void* args[] = { &sendBytes, &recvBytes,
                       &h.slabDev, &h.ctrlDev,
                       &slotOff, &bytesPerRank, &rank, &nRanks, &epoch,
                       &readyOff, &doneOff, &cStride };
      MSCCLPP_CUDATHROW(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(hostAllGatherCoopKernel),
          blocks, 256, args, 0, stream));
      return;
    }

    // ── Path 3: DMA-pipeline (large messages) ────────────────────────────────
    // Phase 1 (d2hStream): D2H own data chunk-by-chunk, write ready flag per chunk.
    //   Enqueue all D2H ops non-blocking; GPU copy engine executes autonomously.
    // Phase 2 (user stream): self D2D copy into recvbuff[rank] (concurrent).
    // Phase 3 (h2dStream + h2dStream2): per chunk, GPU-side wait for each peer's
    //   ready flag, then H2D [left half ‖ right half] in one batched DMA each.
    //   d2hStream and h2dStream run bidirectionally → bidirectional PCIe.

    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.d2hStream, ctx.inputReadyEvent, 0));

    // Phase 1: D2H
    for (size_t ci = 0; ci < chunkCount; ++ci) {
      size_t offset = ci * chunkBytes;
      size_t bytes  = std::min(chunkBytes, bytesPerRank - offset);
      buf.put(ctx.d2hStream, slot, static_cast<int>(ci),
              sendbuff, offset, bytes, epoch);
    }

    // Phase 2: self D2D copy (overlaps with D2H on separate stream)
    if (sendbuff != selfOutput) {
      if (hostAllGatherSelfKernelEnabled() && useVecKernels) {
        int blocks = std::min<int>(128, std::max<int>(1, static_cast<int>(
            (bytesPerRank + 4095) / 4096)));
        hostAllGatherSelfCopyKernel<<<blocks, 256, 0, stream>>>(
            sendBytes, static_cast<char*>(selfOutput), bytesPerRank);
        MSCCLPP_CUDATHROW(cudaGetLastError());
      } else {
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(selfOutput, sendbuff, bytesPerRank,
                                          cudaMemcpyDeviceToDevice, stream));
      }
    }

    // Phase 3: GPU-side wait + H2D per chunk (left ‖ right on two streams)
    bool usedH2dStream2 = false;
    for (size_t ci = 0; ci < chunkCount; ++ci) {
      size_t offset = ci * chunkBytes;
      size_t bytes  = std::min(chunkBytes, bytesPerRank - offset);

      cudaStream_t rightStream = (rank > 0) ? ctx.h2dStream2 : ctx.h2dStream;

      // Wait for left peers on h2dStream, right peers on rightStream.
      for (int r = 0; r < rank; ++r)
        buf.wait(ctx.h2dStream, slot, static_cast<int>(ci), r, epoch);
      for (int r = rank + 1; r < nRanks; ++r)
        buf.wait(rightStream, slot, static_cast<int>(ci), r, epoch);

      // H2D: one batched DMA for left [0..rank-1] and right [rank+1..N-1].
      if (rank > 0)
        buf.get(ctx.h2dStream, slot, 0, rank - 1, offset, bytes, recvbuff);
      if (rank + 1 < nRanks) {
        buf.get(rightStream, slot, rank + 1, nRanks - 1, offset, bytes, recvbuff);
        usedH2dStream2 = usedH2dStream2 || (rank > 0);
      }
    }

    // Sync h2dStream2 → h2dStream, then signal slot done, chain back to userStream.
    if (usedH2dStream2) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent2, ctx.h2dStream2));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.h2dStream, ctx.h2dDoneEvent2, 0));
    }
    buf.signalDone(ctx.h2dStream, slot, epoch);
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent, ctx.h2dStream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, ctx.h2dDoneEvent, 0));
  });
}

// ── CudaIPC AllGather (unchanged) ────────────────────────────────────────────
static ncclResult_t runIntraNodeCudaIpcAllGather(
    void const* sendbuff, void* recvbuff, size_t bytesPerRank,
    ncclComm_t comm, cudaStream_t stream, int rank, int nRanks) {
  if (comm == nullptr || nRanks <= 1 || nRanks != comm->nRanksPerNode ||
      sendbuff == nullptr || recvbuff == nullptr) {
    return ncclInvalidUsage;
  }
  if (bytesPerRank >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(nRanks)) {
    return ncclInvalidUsage;
  }
  if (!comm->hasIB || !cudaIpcEventSyncEnabled()) return ncclInvalidUsage;
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);
  if (fullBytes < kIntraAllGatherRingMinTotalBytes) return ncclInvalidUsage;
  if (fullBytes >
      std::numeric_limits<uintptr_t>::max() -
          reinterpret_cast<uintptr_t>(recvbuff)) {
    return ncclInvalidUsage;
  }
  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
  cudaError_t captureErr = cudaStreamIsCapturing(stream, &captureStatus);
  if (captureErr != cudaSuccess) return ncclUnhandledCudaError;
  if (captureStatus != cudaStreamCaptureStatusNone) return ncclInvalidUsage;

  return runNcclGuarded("intra-node CudaIpc AllGather", [&]() {
    struct LocalCudaEvents {
      std::vector<cudaEvent_t> events;
      cudaEvent_t create() {
        cudaEvent_t event = nullptr;
        MSCCLPP_CUDATHROW(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        events.push_back(event);
        return event;
      }
      ~LocalCudaEvents() {
        for (cudaEvent_t event : events) {
          if (event) cudaEventDestroy(event);
        }
      }
    } localEvents;

    mscclpp::CudaDeviceGuard deviceGuard(comm->cudaDevice);
    uintptr_t recvAddr = reinterpret_cast<uintptr_t>(recvbuff);
    CUdeviceptr recvBase = 0;
    size_t recvAllocSize = 0;
    CUresult rangeResult =
        cuMemGetAddressRange(&recvBase, &recvAllocSize,
                             static_cast<CUdeviceptr>(recvAddr));
    if (rangeResult != CUDA_SUCCESS) {
      throw mscclpp::Error("cuMemGetAddressRange failed",
                           mscclpp::ErrorCode::SystemError);
    }
    unsigned long long recvBufferId = 0;
    CUresult idResult = cuPointerGetAttribute(
        &recvBufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID,
        static_cast<CUdeviceptr>(recvAddr));
    if (idResult != CUDA_SUCCESS) {
      throw mscclpp::Error("cuPointerGetAttribute BUFFER_ID failed",
                           mscclpp::ErrorCode::SystemError);
    }
    uintptr_t recvBaseAddr = static_cast<uintptr_t>(recvBase);
    size_t recvOffset =
        recvAddr >= recvBaseAddr ? static_cast<size_t>(recvAddr - recvBaseAddr)
                                 : recvAllocSize + 1;
    if (recvOffset > recvAllocSize || fullBytes > recvAllocSize - recvOffset) {
      throw mscclpp::Error("AllGather recvbuff does not fit in allocation",
                           mscclpp::ErrorCode::InvalidUsage);
    }
    std::vector<int> peers;
    peers.reserve(static_cast<size_t>(nRanks - 1));
    for (int peer = 0; peer < nRanks; ++peer) {
      if (peer != rank) peers.push_back(peer);
    }

    int next = (rank + 1) % nRanks;
    int prev = (rank + nRanks - 1) % nRanks;

    for (int peer : peers) {
      auto& ctx = getSendRecvPeerContext(comm, peer, comm->cudaDevice);
      if (!ctx.isCudaIpc || ctx.ipcShmSemaphore == nullptr ||
          ctx.ipcReadyEvent == nullptr || ctx.remoteIpcReadyEvent == nullptr ||
          ctx.ipcAllGatherDoneEvent == nullptr ||
          ctx.remoteIpcAllGatherDoneEvent == nullptr ||
          ctx.ipcDoneEvent == nullptr || ctx.remoteIpcDoneEvent == nullptr) {
        throw mscclpp::Error("intra-node AllGather requires CudaIpc context",
                             mscclpp::ErrorCode::InvalidUsage);
      }

      bool needExchange = ctx.localAllocBase == 0 ||
                          ctx.localAllocBase != recvBaseAddr ||
                          ctx.localAllocSize != recvAllocSize ||
                          ctx.localAllocBufferId != recvBufferId;
      if (needExchange) {
        mscclpp::TransportFlags ipcFlags(mscclpp::Transport::CudaIpc);
        auto rm = comm->comm->registerMemory(
            reinterpret_cast<void*>(recvBaseAddr), recvAllocSize,
            ipcFlags);
        comm->comm->sendMemory(
            rm, peer, recvbuffExchangeTag(rank, peer, comm->worldSize));
        ctx.localRecvbuffMemory = rm;
        ctx.localAllocBase = recvBaseAddr;
        ctx.localAllocSize = recvAllocSize;
        ctx.localAllocBufferId = recvBufferId;
        ctx.ipcShmSemaphore->publishAllocBase(ctx.localAllocBase);
      }
      ctx.ipcShmSemaphore->publishRecvbuff(recvAddr);
    }

    std::vector<uintptr_t> peerRecvAddrs(static_cast<size_t>(nRanks), 0);
    for (int peer : peers) {
      auto& ctx = getSendRecvPeerContext(comm, peer, comm->cudaDevice);
      peerRecvAddrs[static_cast<size_t>(peer)] =
          ctx.ipcShmSemaphore->readPeerRecvbuff();
      uint64_t peerAllocGen = ctx.ipcShmSemaphore->readPeerAllocGeneration();
      if (peerAllocGen != ctx.peerAllocGeneration) {
        if (ctx.peerAllocGeneration != 0) {
          MSCCLPP_CUDATHROW(cudaStreamSynchronize(ctx.ipcStream));
        }
        auto rmFuture = comm->comm->recvMemory(
            peer, recvbuffExchangeTag(peer, rank, comm->worldSize));
        auto remoteRecvbuffRM = rmFuture.get();
        uintptr_t mappedAllocBase =
            reinterpret_cast<uintptr_t>(remoteRecvbuffRM.data());
        uintptr_t peerAllocBase = ctx.ipcShmSemaphore->readPeerAllocBase();
        ctx.remoteRecvbuffRM = remoteRecvbuffRM;
        ctx.mappedAllocBase = mappedAllocBase;
        ctx.peerAllocBase = peerAllocBase;
        ctx.peerAllocGeneration = peerAllocGen;
      }
    }

    auto* recvBytes = static_cast<char*>(recvbuff);
    void* selfOutput = recvBytes + static_cast<size_t>(rank) * bytesPerRank;
    auto& nextCtx = getSendRecvPeerContext(comm, next, comm->cudaDevice);
    auto& prevCtx = getSendRecvPeerContext(comm, prev, comm->cudaDevice);
    cudaEvent_t outboundDone = localEvents.create();

    MSCCLPP_CUDATHROW(cudaEventRecord(prevCtx.ipcReadyEvent, stream));
    prevCtx.ipcShmSemaphore->signalAllGatherReadyRecorded();
    nextCtx.ipcShmSemaphore->waitPeerAllGatherReadyRecorded();
    MSCCLPP_CUDATHROW(
        cudaStreamWaitEvent(nextCtx.ipcStream, nextCtx.remoteIpcReadyEvent, 0));
    nextCtx.ipcShmSemaphore->ackPeerAllGatherReadyWaitEnqueued();
    prevCtx.ipcShmSemaphore->waitPeerAllGatherReadyWaitEnqueued();

    cudaEvent_t inputReady = localEvents.create();
    MSCCLPP_CUDATHROW(cudaEventRecord(inputReady, stream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(nextCtx.ipcStream, inputReady, 0));

    if (sendbuff != selfOutput) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(selfOutput, sendbuff, bytesPerRank,
                                        cudaMemcpyDeviceToDevice, stream));
    }

    for (int step = 0; step < nRanks - 1; ++step) {
      int sendBlock = (rank - step + nRanks) % nRanks;
      if (step > 0) {
        cudaEvent_t blockReady = localEvents.create();
        MSCCLPP_CUDATHROW(cudaEventRecord(blockReady, stream));
        MSCCLPP_CUDATHROW(
            cudaStreamWaitEvent(nextCtx.ipcStream, blockReady, 0));
      }
      void const* src =
          step == 0 ? sendbuff
                    : static_cast<void const*>(
                          recvBytes +
                          static_cast<size_t>(sendBlock) * bytesPerRank);
      uintptr_t peerSlot =
          peerRecvAddrs[static_cast<size_t>(next)] +
          static_cast<uintptr_t>(sendBlock) * bytesPerRank;
      void* remoteDst = nextCtx.mapPeerPtr(peerSlot);
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(remoteDst, src, bytesPerRank,
                                        cudaMemcpyDeviceToDevice,
                                        nextCtx.ipcStream));
      MSCCLPP_CUDATHROW(
          cudaEventRecord(nextCtx.ipcAllGatherDoneEvent, nextCtx.ipcStream));
      nextCtx.ipcShmSemaphore->signalAllGatherEventRecorded();
      cudaStreamQuery(nextCtx.ipcStream);

      prevCtx.ipcShmSemaphore->waitPeerAllGatherEventRecorded();
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
          stream, prevCtx.remoteIpcAllGatherDoneEvent, 0));
      prevCtx.ipcShmSemaphore->ackPeerAllGatherEventWaitEnqueued();
      nextCtx.ipcShmSemaphore->waitPeerAllGatherEventWaitEnqueued();
    }
    MSCCLPP_CUDATHROW(cudaEventRecord(outboundDone, nextCtx.ipcStream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, outboundDone, 0));
  });
}
