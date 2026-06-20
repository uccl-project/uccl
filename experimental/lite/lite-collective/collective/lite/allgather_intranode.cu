// Intra-node AllGather implementation: host-staged SHM + CudaIPC paths.
// Included by nccl.cu inside its anonymous namespace so the intra-node
// paths can reuse the NCCL shim's send/recv peer context.
//
// Two staging channel variants:
//   CPU-driven: CpuStagingChannel (default) — CPU enqueues D2H to CUDA stream.
//   GPU-driven: GpuStagingChannel — GPU kernel posts D2H cmds to ring;
//               background service thread executes DMA.
//   Enable GPU-driven with MSCCLPP_NCCL_AG_GPU_STAGING=1.

#include "host_staging_buffer.hpp"
#include "gpu_staging_channel.hpp"

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

// ── GPU-driven AllGather (GpuStagingChannel) ─────────────────────────────────
// Enabled with MSCCLPP_NCCL_AG_GPU_STAGING=1.
//
// The only change vs the CPU-driven path: Phase 1 launches a tiny GPU kernel
// that posts a D2H command to the GpuStagingChannel ring instead of calling
// CpuStagingChannel::put() from the CPU.  A background service thread drains
// the ring and issues cudaMemcpyAsync + cuStreamWriteValue64 on its own stream.
// Phase 3 (wait + get) is identical — GpuStagingChannel shares CscCtrl with
// CpuStagingChannel so the same cuStreamWaitValue64 flags are reused.
//
// Expected latency vs CPU-driven:
//   extra overhead = kernel launch (~5µs) + ring-write-to-CPU-visible (~2µs)
//   dominates for small messages; DMA-dominated large messages are ~same.

static bool gpuStagingEnabled() {
  char const* e = std::getenv("MSCCLPP_NCCL_AG_GPU_STAGING");
  return e != nullptr && std::strcmp(e, "0") != 0;
}

// 1-thread kernel: posts one D2H command to the GpuStagingChannel ring.
__global__ static void gscPostD2HCmd(GscDeviceHandle h,
                                     int slot, int chunkId,
                                     char const* src,
                                     size_t offset, size_t size,
                                     uint64_t tag) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    h.put(slot, chunkId, src, offset, size, tag);
}

struct GpuAllGatherContext {
  std::mutex initMutex;
  std::condition_variable initCv;
  bool initialized  = false;
  bool initializing = false;
  std::exception_ptr initException;

  uint64_t epoch = 0;
  int cudaDevice  = -1;

  // GpuStagingChannel owns the slab + ring; CpuStagingChannel (inside it) owns
  // the ctrl.  Use unique_ptr since GpuStagingChannel has no default constructor.
  std::unique_ptr<GpuStagingChannel> gpuBuf;

  cudaStream_t d2hStream       = nullptr;  // GPU kernels post ring cmds here
  cudaStream_t h2dStream       = nullptr;
  cudaStream_t h2dStream2      = nullptr;
  cudaEvent_t  inputReadyEvent = nullptr;
  cudaEvent_t  h2dDoneEvent    = nullptr;
  cudaEvent_t  h2dDoneEvent2   = nullptr;

  // Background service thread: polls ring, issues DMA on its own private stream.
  std::thread serviceThread;

  ~GpuAllGatherContext() {
    if (gpuBuf) gpuBuf->stop();
    if (serviceThread.joinable()) serviceThread.join();
    if (h2dDoneEvent2)   cudaEventDestroy(h2dDoneEvent2);
    if (h2dDoneEvent)    cudaEventDestroy(h2dDoneEvent);
    if (inputReadyEvent) cudaEventDestroy(inputReadyEvent);
    if (h2dStream2)      cudaStreamDestroy(h2dStream2);
    if (h2dStream)       cudaStreamDestroy(h2dStream);
    if (d2hStream)       cudaStreamDestroy(d2hStream);
  }
};

static std::mutex gGpuAllGatherMutex;
static std::unordered_map<ncclComm_t, std::unique_ptr<GpuAllGatherContext>>
    gGpuAllGatherContexts;

static GpuAllGatherContext& getGpuAllGatherContext(ncclComm_t comm) {
  std::lock_guard<std::mutex> lk(gGpuAllGatherMutex);
  auto& ptr = gGpuAllGatherContexts[comm];
  if (!ptr) ptr = std::make_unique<GpuAllGatherContext>();
  return *ptr;
}

static void cleanupGpuAllGatherContexts(ncclComm_t comm) {
  std::lock_guard<std::mutex> lk(gGpuAllGatherMutex);
  gGpuAllGatherContexts.erase(comm);
}

// Service thread function: runs on a separate CPU thread, creates its own CUDA
// stream, drains the ring until stop() is called.
static void gscServiceThreadFunc(GpuStagingChannel* ch, int dev) {
  cudaSetDevice(dev);
  cudaStream_t s = nullptr;
  int lp = 0, gp = 0;
  cudaDeviceGetStreamPriorityRange(&lp, &gp); (void)lp;
  cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, gp);
  ch->serviceLoop(s);
  cudaStreamDestroy(s);
}

static void initializeGpuAllGatherContext(
    GpuAllGatherContext& ctx, ncclComm_t comm, int rank, int nRanks,
    int cudaDevice, std::shared_ptr<mscclpp::Communicator> bootstrapComm) {
  {
    std::unique_lock<std::mutex> lk(ctx.initMutex);
    if (ctx.initialized) return;
    if (ctx.initializing) {
      ctx.initCv.wait(lk, [&]{ return !ctx.initializing; });
      if (ctx.initException) std::rethrow_exception(ctx.initException);
      if (ctx.initialized) return;
    }
    ctx.initializing = true;
  }
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
    ctx.cudaDevice = cudaDevice;

    auto nc = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(comm));
    char nbuf[64]; std::snprintf(nbuf, 64, "%d_%llx", getpid(), nc);
    std::string tag = std::string("gsc") + nbuf;

    ctx.gpuBuf = std::make_unique<GpuStagingChannel>(
        GpuStagingChannel::create(
            kHostAllGatherMaxTotalBytes / static_cast<size_t>(nRanks),
            kCscMaxSlots,
            bootstrapComm,
            rank, nRanks, cudaDevice,
            /*mapSlab=*/true,
            hostAllGatherNumaPlacementEnabled(),
            tag));

    int lp = 0, gp = 0;
    MSCCLPP_CUDATHROW(cudaDeviceGetStreamPriorityRange(&lp, &gp)); (void)lp;
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.d2hStream,  cudaStreamNonBlocking, gp));
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.h2dStream,  cudaStreamNonBlocking, gp));
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.h2dStream2, cudaStreamNonBlocking, gp));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.inputReadyEvent, cudaEventDisableTiming));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent,    cudaEventDisableTiming));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent2,   cudaEventDisableTiming));

    // Start background service thread.
    ctx.serviceThread = std::thread(gscServiceThreadFunc, ctx.gpuBuf.get(), cudaDevice);
  } catch (std::exception const& ex) {
    result = mapMscclppException(ex); errmsg = ex.what();
  } catch (...) {
    result = ncclInternalError; errmsg = "unknown exception";
  }
  if (result != ncclSuccess) {
    fail(std::make_exception_ptr(
        mscclpp::Error("GpuStagingChannel init: " + errmsg,
                       mscclpp::ErrorCode::InternalError)));
    throw mscclpp::Error(errmsg, mscclpp::ErrorCode::InternalError);
  }
  {
    std::lock_guard<std::mutex> lk(ctx.initMutex);
    ctx.initialized  = true;
    ctx.initializing = false;
    ctx.initCv.notify_all();
  }
}

static ncclResult_t runIntraNodeGpuStagingAllGather(
    void const* sendbuff, void* recvbuff, size_t bytesPerRank,
    ncclComm_t comm, cudaStream_t stream, int rank, int nRanks,
    int cudaDevice,
    std::shared_ptr<mscclpp::Communicator> bootstrapComm) {
  if (comm == nullptr || nRanks <= 1 || bytesPerRank == 0) return ncclInvalidUsage;
  if (bytesPerRank > kHostAllGatherMaxTotalBytes / static_cast<size_t>(nRanks))
    return ncclInvalidUsage;

  return runHostAllGatherGuarded("GpuStagingAllGather", [&] {
    GpuAllGatherContext& ctx = getGpuAllGatherContext(comm);
    initializeGpuAllGatherContext(ctx, comm, rank, nRanks, cudaDevice, bootstrapComm);

    // Shortcut: empty / trivial cases.
    if (bytesPerRank == 0) return;
    void* selfOutput = static_cast<char*>(recvbuff)
                       + static_cast<size_t>(rank) * bytesPerRank;
    if (sendbuff == selfOutput && nRanks == 1) return;

    uint64_t epoch = ++ctx.epoch;
    int slot = static_cast<int>((epoch - 1) % kCscMaxSlots);

    // Slot reuse guard: wait for all peers to finish with this slot before reuse.
    if (epoch > static_cast<uint64_t>(kCscMaxSlots)) {
      // Use CpuStagingChannel (inner) waitDone since it shares CscCtrl.
      // Access inner csc via gpuBuf's CscDeviceHandle's ctrlDev... but simpler:
      // just call the inherited waitDone from the embedded CpuStagingChannel.
      // We expose it via service() by accessing the internal csc_ ... actually
      // the easiest path: add a small waitDone wrapper to GpuStagingChannel.
      // For now, do a stream sync as a conservative slot reuse guard.
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(ctx.h2dStream));
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(ctx.h2dStream2));
    }

    size_t chunkBytes = hostAllGatherChunkBytes(bytesPerRank);
    size_t chunkCount = (bytesPerRank + chunkBytes - 1) / chunkBytes;

    GscDeviceHandle handle = ctx.gpuBuf->deviceHandle();
    // CpuStagingChannel for wait/get (shares CscCtrl).
    // We need a CpuStagingChannel reference to call wait/get.
    // The GpuStagingChannel wraps a CpuStagingChannel; we use the same slab.
    // Build a lightweight wrapper using rankSlabHost/ctrlDev from handle.
    // Simplest: use gpuBuf->asCpu() — add an accessor to GpuStagingChannel.
    // For now: call wait/get via nRanks/bytesPerRank and raw stream writes.
    // Better: add CpuStagingChannel& csc() accessor to GpuStagingChannel.

    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.d2hStream, ctx.inputReadyEvent, 0));

    auto const* send = static_cast<char const*>(sendbuff);

    // Phase 1: GPU kernel posts D2H commands to ring (one kernel per chunk).
    // The service thread (running independently) will drain the ring and issue
    // cudaMemcpyAsync + cuStreamWriteValue64 on its private stream.
    for (size_t ci = 0; ci < chunkCount; ++ci) {
      size_t offset = ci * chunkBytes;
      size_t bytes  = std::min(chunkBytes, bytesPerRank - offset);
      gscPostD2HCmd<<<1, 1, 0, ctx.d2hStream>>>(
          handle, slot, static_cast<int>(ci),
          send, offset, bytes, epoch);
      MSCCLPP_CUDATHROW(cudaGetLastError());
    }

    // Phase 2: self D2D copy (overlaps with ring-servicing on d2hStream).
    if (sendbuff != selfOutput) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(selfOutput, sendbuff, bytesPerRank,
                                        cudaMemcpyDeviceToDevice, stream));
    }

    // Phase 3: GPU-side wait (cuStreamWaitValue64) + batched H2D.
    // Reuse CpuStagingChannel's wait/get via the inner slab channel.
    // We call them directly via the CpuStagingChannel& inside GpuStagingChannel.
    // Since GpuStagingChannel::create() builds a CpuStagingChannel internally
    // and its wait/get/signalDone are exposed via the inner CpuStagingChannel:
    // access it via: ctx.gpuBuf->csc() (accessor to be added below).
    CpuStagingChannel& csc = ctx.gpuBuf->csc();

    bool usedH2dStream2 = false;
    for (size_t ci = 0; ci < chunkCount; ++ci) {
      size_t offset = ci * chunkBytes;
      size_t bytes  = std::min(chunkBytes, bytesPerRank - offset);
      cudaStream_t rightStream = (rank > 0) ? ctx.h2dStream2 : ctx.h2dStream;
      for (int r = 0; r < rank; ++r)
        csc.wait(ctx.h2dStream, slot, static_cast<int>(ci), r, epoch);
      for (int r = rank + 1; r < nRanks; ++r)
        csc.wait(rightStream, slot, static_cast<int>(ci), r, epoch);
      if (rank > 0)
        csc.get(ctx.h2dStream, slot, 0, rank - 1, offset, bytes, recvbuff);
      if (rank + 1 < nRanks) {
        csc.get(rightStream, slot, rank + 1, nRanks - 1, offset, bytes, recvbuff);
        usedH2dStream2 = usedH2dStream2 || (rank > 0);
      }
    }

    if (usedH2dStream2) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent2, ctx.h2dStream2));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.h2dStream, ctx.h2dDoneEvent2, 0));
    }
    csc.signalDone(ctx.h2dStream, slot, epoch);
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent, ctx.h2dStream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, ctx.h2dDoneEvent, 0));
  });
}



// ── True GPU-kernel AllGather (CscDeviceHandle __device__ API) ────────────────
// Enabled with MSCCLPP_NCCL_AG_GPU_KERNEL=1.
//
// The ENTIRE AllGather runs inside a single GPU kernel per rank:
//   - put():  SM-copy own data to host-mapped slab + signal d2hReady flag
//   - wait(): GPU spin on each peer's flag
//   - get():  SM-copy peer data from slab to recvbuff
//
// CPU role: launch kernel on stream + chain events.  No CPU in hot path.
// This is the true "GPU-driven" path: the channel API (put/wait/get) is called
// from __device__ code, not from CPU stream-enqueue calls.

static bool gpuKernelEnabled() {
  char const* e = std::getenv("MSCCLPP_NCCL_AG_GPU_KERNEL");
  return e != nullptr && std::strcmp(e, "0") != 0;
}

// Per-rank GPU-kernel AllGather.  Each rank launches this kernel concurrently
// on its own GPU.  Ranks communicate via the host-mapped slab (h.slabDev) and
// ctrl flags (h.ctrlDev).
//
// Requires: h.slabDev != nullptr (mapSlab=true).
// Thread assignment: blockDim.x >= nRanks for parallel wait; more threads give
// faster SM-copy throughput (suggest 256).
__global__ static void gpuAllGatherKernel(
    CscDeviceHandle h,
    char const* send, char* recv,
    size_t bytesPerRank, int rank, int nRanks,
    int slot, uint64_t epoch) {
  // 1. Stage own data to slab + signal all peers.
  h.put(slot, /*chunkId=*/0, send, /*offset=*/0, bytesPerRank, epoch);

  // 2. Wait for all other ranks in parallel.
  //    Thread r waits for peer r (r != rank, r < nRanks).
  if (static_cast<int>(threadIdx.x) < nRanks
      && static_cast<int>(threadIdx.x) != rank)
    h.wait(slot, /*chunkId=*/0, static_cast<int>(threadIdx.x), epoch);
  __syncthreads();

  // 3. SM-copy own data into recvbuff (from send, not from slab).
  char* selfDst = recv + static_cast<size_t>(rank) * bytesPerRank;
  for (size_t i = threadIdx.x; i < bytesPerRank; i += blockDim.x)
    selfDst[i] = send[i];

  // 4. SM-copy peers' data from slab to recvbuff.
  if (rank > 0)
    h.get(slot, 0, rank - 1, /*offset=*/0, bytesPerRank, recv);
  if (rank + 1 < nRanks)
    h.get(slot, rank + 1, nRanks - 1, /*offset=*/0, bytesPerRank, recv);

  // 5. Signal slot done (for slot reuse guard).
  h.signalDone(slot, epoch);
}

static ncclResult_t runIntraNodeGpuKernelAllGather(
    void const* sendbuff, void* recvbuff, size_t bytesPerRank,
    ncclComm_t comm, cudaStream_t stream, int rank, int nRanks,
    int cudaDevice,
    std::shared_ptr<mscclpp::Communicator> bootstrapComm) {
  if (comm == nullptr || nRanks <= 1 || bytesPerRank == 0) return ncclInvalidUsage;
  if (bytesPerRank > kHostAllGatherMaxTotalBytes / static_cast<size_t>(nRanks))
    return ncclInvalidUsage;

  return runHostAllGatherGuarded("GpuKernelAllGather", [&] {
    // Reuse the CPU-driven context (same CpuStagingChannel slab).
    HostAllGatherContext& ctx = getHostAllGatherContext(comm, /*mapSlab=*/true);
    initializeHostAllGatherContext(ctx, comm, rank, nRanks, cudaDevice,
                                   bootstrapComm);

    if (bytesPerRank == 0) return;

    void* selfOutput = static_cast<char*>(recvbuff)
                       + static_cast<size_t>(rank) * bytesPerRank;
    if (sendbuff == selfOutput && nRanks == 1) return;

    CpuStagingChannel& buf = *ctx.buf;
    HsbDeviceHandle rawH  = buf.deviceHandle();

    // Require device-mapped slab.
    if (rawH.slabDev == nullptr)
      throw mscclpp::Error("GpuKernelAllGather: slabDev is null (need mapSlab)",
                           mscclpp::ErrorCode::InvalidUsage);

    uint64_t epoch = ++ctx.epoch;
    int slot = static_cast<int>((epoch - 1) % kCscMaxSlots);

    // Slot reuse guard: wait for previous use of this slot to finish.
    if (epoch > static_cast<uint64_t>(kCscMaxSlots)) {
      buf.waitDone(slot, (rank + 1) % nRanks, epoch - kCscMaxSlots);
    }

    CscDeviceHandle h;
    h.slabDev      = rawH.slabDev;
    h.ctrlDev      = rawH.ctrlDev;
    h.rank         = rawH.rank;
    h.nRanks       = rawH.nRanks;
    h.bytesPerRank = rawH.bytesPerRank;
    h.slotStride   = rawH.slotStride;
    h.counterStride = rawH.counterStride;

    // Chain userStream → kernel stream → userStream.
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.h2dStream, ctx.inputReadyEvent, 0));

    // One block per rank; threads = max(nRanks, 256) for parallel wait + fast copy.
    int threads = std::max(nRanks, 256);
    gpuAllGatherKernel<<<1, threads, 0, ctx.h2dStream>>>(
        h,
        static_cast<char const*>(sendbuff),
        static_cast<char*>(recvbuff),
        bytesPerRank, rank, nRanks, slot, epoch);
    MSCCLPP_CUDATHROW(cudaGetLastError());

    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent, ctx.h2dStream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, ctx.h2dDoneEvent, 0));
  });
}


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
