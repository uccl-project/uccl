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

// Thrown (silently) when a cached init failure is rethrown — no warning printed.
struct SilentRethrownError : std::exception {
  std::exception_ptr original;
  explicit SilentRethrownError(std::exception_ptr ep) : original(std::move(ep)) {}
  const char* what() const noexcept override { return "cached init failure"; }
};

template <typename Fn>
static ncclResult_t runHostAllGatherGuarded(char const* opName, Fn&& fn) {
  try { fn(); return ncclSuccess; }
  catch (SilentRethrownError const& e) {
    // Cached failure rethrown — already warned on first occurrence.
    try { std::rethrow_exception(e.original); }
    catch (std::exception const& ex) { return mapMscclppException(ex); }
    catch (...) { return ncclInternalError; }
  }
  catch (std::exception const& ex) {
    std::fprintf(stderr, "AllGather WARN %s: %s\n", opName, ex.what()); std::fflush(stderr);
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

// Probe kernel: verifies device-mapped pointers are accessible from GPU kernel.
// Used once during init to decide if GPU-kernel AllGather is viable.
__global__ static void gpuKernelProbe(char const* slabDev, char* ctrlDev) {
  // Attempt a volatile read from mapped ctrl; if pointers are invalid the
  // kernel launch itself will fail (caught by cudaGetLastError).
  (void)*reinterpret_cast<volatile unsigned long long const*>(ctrlDev);
  (void)*reinterpret_cast<char const volatile*>(slabDev);
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
  bool initFailed   = false;   // set on first failure; suppresses repeated warnings
  std::exception_ptr initException;

  bool mapSlab = true;
  bool cooperativeLaunch = false;
  bool gpuKernelWorks   = false;  // tested during init
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
    // If init failed previously, rethrow without printing warning again.
    if (ctx.initFailed) throw SilentRethrownError(ctx.initException);
    if (ctx.initializing) {
      ctx.initCv.wait(lk, [&] { return !ctx.initializing; });
      if (ctx.initFailed) throw SilentRethrownError(ctx.initException);
      if (ctx.initialized) return;
    }
    ctx.initializing = true;
  }

  // RAII guard: on exception, mark context as failed.
  auto fail = [&](std::exception_ptr ep) {
    std::lock_guard<std::mutex> lk(ctx.initMutex);
    ctx.initialized  = false;
    ctx.initializing = false;
    ctx.initFailed   = true;
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

    // Probe test: verify GPU-kernel with device-mapped pointers works in this
    // CUDA context.  Use cudaStreamSynchronize to catch both launch and execution errors.
    HsbDeviceHandle rawH = ctx.buf->deviceHandle();
    ctx.gpuKernelWorks = false;
    if (rawH.slabDev != nullptr && rawH.ctrlDev != nullptr) {
      gpuKernelProbe<<<1, 1, 0, ctx.h2dStream>>>(rawH.slabDev, rawH.ctrlDev);
      if (cudaGetLastError() == cudaSuccess) {
        cudaError_t execErr = cudaStreamSynchronize(ctx.h2dStream);
        ctx.gpuKernelWorks = (execErr == cudaSuccess);
      }
      cudaGetLastError();  // clear any sticky error from the probe
    }
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

// ── Lightweight GPU ring for Path 3 (GPU posts D2H commands, CPU services) ──
// Owns only the ring buffer (small mmap + cudaHostRegister) and service thread.
// Reuses the HostAllGatherContext's CpuStagingChannel for actual DMA (no second slab).
struct GpuRingContext {
  GscRingEntry* ring     = nullptr;  // host ptr (mmap)
  GscRingEntry* ringDev  = nullptr;  // device ptr
  GscRingCtrl*  ctrl     = nullptr;  // host ptr (mmap)
  GscRingCtrl*  ctrlDev  = nullptr;  // device ptr
  void* ringMapping = nullptr;
  void* ctrlMapping = nullptr;
  volatile bool stopFlag = false;
  std::thread   serviceThread;

  ~GpuRingContext() {
    stopFlag = true;
    if (serviceThread.joinable()) serviceThread.join();
    size_t ringBytes = kGscRingSize * sizeof(GscRingEntry);
    if (ringMapping) { cudaHostUnregister(ringMapping); ::munmap(ringMapping, ringBytes); }
    if (ctrlMapping) { cudaHostUnregister(ctrlMapping); ::munmap(ctrlMapping, sizeof(GscRingCtrl)); }
  }
};

struct GpuAllGatherContext {
  std::mutex initMutex;
  std::condition_variable initCv;
  bool initialized  = false;
  bool initializing = false;
  bool initFailed   = false;
  std::exception_ptr initException;

  uint64_t epoch = 0;
  int cudaDevice  = -1;

  // Ring for GPU→CPU D2H command posting (small, no second slab).
  std::unique_ptr<GpuRingContext> ring;

  cudaStream_t d2hStream       = nullptr;
  cudaStream_t h2dStream       = nullptr;
  cudaStream_t h2dStream2      = nullptr;
  cudaEvent_t  inputReadyEvent = nullptr;
  cudaEvent_t  h2dDoneEvent    = nullptr;
  cudaEvent_t  h2dDoneEvent2   = nullptr;

  ~GpuAllGatherContext() {
    ring.reset();  // stops service thread before destroying streams
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

// Service thread: drains GPU ring, calls csc.put() for each D2H command.
static void gpuRingServiceFunc(GpuRingContext* ring, CpuStagingChannel* csc, int dev) {
  cudaSetDevice(dev);
  cudaStream_t svcStream = nullptr;
  int lp = 0, gp = 0;
  cudaDeviceGetStreamPriorityRange(&lp, &gp); (void)lp;
  cudaStreamCreateWithPriority(&svcStream, cudaStreamNonBlocking, gp);
  GscRingCtrl* ctrl = ring->ctrl;
  constexpr int kYield = 65536;
  int spins = 0;
  while (!ring->stopFlag) {
    uint64_t head = ctrl->head;
    uint64_t tail = ctrl->tail;
    if (tail < head) {
      GscRingEntry const& e = ring->ring[tail & kGscRingMask];
      auto* src = reinterpret_cast<void const*>(e.srcDevPtr);
      // D2H + flag: reuse CpuStagingChannel.put() (cudaMemcpyAsync + streamWriteValue)
      try {
        csc->put(svcStream, e.slot, e.chunkId, src, e.offset, e.size, e.tag);
      } catch (...) {}
      ctrl->tail = tail + 1;
      spins = 0;
    } else {
      if (spins++ < kYield) {
#if defined(__x86_64__) || defined(__i386__)
        asm volatile("pause" ::: "memory");
#else
        asm volatile("" ::: "memory");
#endif
      } else {
        std::this_thread::yield(); spins = 0;
      }
    }
  }
  cudaStreamDestroy(svcStream);
}

static void initializeGpuAllGatherContext(
    GpuAllGatherContext& ctx, ncclComm_t comm, int rank, int nRanks,
    int cudaDevice, std::shared_ptr<mscclpp::Communicator> bootstrapComm,
    CpuStagingChannel* csc) {
  {
    std::unique_lock<std::mutex> lk(ctx.initMutex);
    if (ctx.initialized) return;
    if (ctx.initFailed) throw SilentRethrownError(ctx.initException);
    if (ctx.initializing) {
      ctx.initCv.wait(lk, [&]{ return !ctx.initializing; });
      if (ctx.initFailed) throw SilentRethrownError(ctx.initException);
      if (ctx.initialized) return;
    }
    ctx.initializing = true;
  }
  auto fail = [&](std::exception_ptr ep) {
    std::lock_guard<std::mutex> lk(ctx.initMutex);
    ctx.initialized  = false;
    ctx.initializing = false;
    ctx.initFailed   = true;
    ctx.initException = ep;
    ctx.initCv.notify_all();
  };

  ncclResult_t result = ncclSuccess;
  std::string  errmsg;
  try {
    mscclpp::CudaDeviceGuard devGuard(cudaDevice);
    ctx.cudaDevice = cudaDevice;

    // Allocate small ring buffer using mmap (same memory type as POSIX shm,
    // kernel-accessible after cudaHostRegister in this context).
    auto mmap_alloc = [](size_t sz) -> void* {
      void* p = ::mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED)
        throw mscclpp::Error("mmap failed", mscclpp::ErrorCode::SystemError);
      std::memset(p, 0, sz);
      return p;
    };
    size_t ringBytes = kGscRingSize * sizeof(GscRingEntry);
    auto gr = std::make_unique<GpuRingContext>();
    gr->ringMapping = mmap_alloc(ringBytes);
    gr->ring        = static_cast<GscRingEntry*>(gr->ringMapping);
    MSCCLPP_CUDATHROW(cudaHostRegister(gr->ringMapping, ringBytes,
                                       cudaHostRegisterPortable |
                                       cudaHostRegisterMapped));
    { void* dp = nullptr;
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, gr->ringMapping, 0));
      gr->ringDev = static_cast<GscRingEntry*>(dp); }

    gr->ctrlMapping = mmap_alloc(sizeof(GscRingCtrl));
    gr->ctrl        = static_cast<GscRingCtrl*>(gr->ctrlMapping);
    MSCCLPP_CUDATHROW(cudaHostRegister(gr->ctrlMapping, sizeof(GscRingCtrl),
                                       cudaHostRegisterPortable |
                                       cudaHostRegisterMapped));
    { void* dp = nullptr;
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, gr->ctrlMapping, 0));
      gr->ctrlDev = static_cast<GscRingCtrl*>(dp); }

    int lp = 0, gp = 0;
    MSCCLPP_CUDATHROW(cudaDeviceGetStreamPriorityRange(&lp, &gp)); (void)lp;
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.d2hStream,  cudaStreamNonBlocking, gp));
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.h2dStream,  cudaStreamNonBlocking, gp));
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(&ctx.h2dStream2, cudaStreamNonBlocking, gp));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.inputReadyEvent, cudaEventDisableTiming));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent,    cudaEventDisableTiming));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent2,   cudaEventDisableTiming));

    gr->serviceThread = std::thread(gpuRingServiceFunc, gr.get(), csc, cudaDevice);
    ctx.ring = std::move(gr);
  } catch (std::exception const& ex) {
    result = mapMscclppException(ex); errmsg = ex.what();
  } catch (...) {
    result = ncclInternalError; errmsg = "unknown exception";
  }
  if (result != ncclSuccess) {
    fail(std::make_exception_ptr(
        mscclpp::Error("GpuRing init: " + errmsg, mscclpp::ErrorCode::InternalError)));
    throw mscclpp::Error(errmsg, mscclpp::ErrorCode::InternalError);
  }
  {
    std::lock_guard<std::mutex> lk(ctx.initMutex);
    ctx.initialized  = true;
    ctx.initializing = false;
    ctx.initCv.notify_all();
  }
}



// ── True GPU-kernel AllGather (CscDeviceHandle + GscDeviceHandle __device__ API)
// Enabled with MSCCLPP_NCCL_AG_GPU_KERNEL=1.
//
// Three paths matching the CPU-driven version:
//   Path 1 (tiny,  ≤ kernelMaxBytes):    single-block SM-write via CscDeviceHandle
//   Path 2 (small, ≤ coopMaxBytes):      same kernel launched with more blocks
//   Path 3 (large, > coopMaxBytes):      chunked DMA pipeline:
//       GPU posts ring cmd (GscDeviceHandle::put) → service thread calls
//       cudaMemcpyAsync → writes ready flag → GPU spin-waits (CscDeviceHandle::wait)
//       → GPU SM-reads slab to recv (CscDeviceHandle::get).
//
// CPU role: launch kernel + maintain background service thread.

static bool gpuKernelEnabled() {
  char const* e = std::getenv("MSCCLPP_NCCL_AG_GPU_KERNEL");
  return e != nullptr && std::strcmp(e, "0") != 0;
}

// ── Path 1+2: single kernel, SM write/wait/read ───────────────────────────────
// Uses COMPACT slab layout: rank r's data at slab[slot][r * bytesPerRank]
// (actual message bytesPerRank, not the slab's max capacity).
// This matches hostAllGatherMappedKernel and avoids demand-page issues in MPI
// mode where large slab offsets (rank * SLAB_bytesPerRank) may access
// physically unallocated pages.
__global__ static void gpuAllGatherSmKernel(
    CscDeviceHandle h,
    char const* send, char* recv,
    size_t bytesPerRank, int rank, int nRanks,
    int slot, uint64_t epoch) {
  using Vec = unsigned long long;
  // COMPACT slab: slot data begins at slotBase; rank r at slotBase + r*bytesPerRank.
  size_t slotBase = h.slotOffset(slot);  // slot * h.slotStride (based on max capacity)

  // 1. SM-write own data at compact offset (rank * ACTUAL bytesPerRank).
  size_t selfBase  = slotBase + static_cast<size_t>(rank) * bytesPerRank;
  size_t vecBytes  = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  auto const* sv   = reinterpret_cast<Vec const*>(send);
  auto*       ds   = reinterpret_cast<Vec*>(h.slabDev + selfBase);
  for (size_t i = threadIdx.x; i < vecBytes / sizeof(Vec); i += blockDim.x)
    ds[i] = sv[i];
  for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank; i += blockDim.x)
    h.slabDev[selfBase + i] = send[i];
  __syncthreads();

  // 2. Thread 0: fence, signal flag, then spin-wait for all peers.
  if (threadIdx.x == 0) {
    __threadfence_system();
    size_t flagOff = h.readyFlagOffset(slot)
                     + static_cast<size_t>(rank) * h.counterStride;
    *reinterpret_cast<volatile unsigned long long*>(h.ctrlDev + flagOff) = epoch;
    __threadfence_system();
    for (int r = 0; r < nRanks; ++r) {
      size_t peerFlagOff = h.readyFlagOffset(slot)
                           + static_cast<size_t>(r) * h.counterStride;
      while (*reinterpret_cast<volatile unsigned long long*>(
                 h.ctrlDev + peerFlagOff) < epoch) {}
    }
  }
  __syncthreads();

  // 3. SM-read all ranks' data from compact slab to recv.
  size_t fullBytes = static_cast<size_t>(nRanks) * bytesPerRank;
  size_t outVecBytes = (fullBytes / sizeof(Vec)) * sizeof(Vec);
  auto const* slotVec = reinterpret_cast<Vec const*>(h.slabDev + slotBase);
  auto*       recvVec = reinterpret_cast<Vec*>(recv);
  for (size_t i = threadIdx.x; i < outVecBytes / sizeof(Vec); i += blockDim.x)
    recvVec[i] = slotVec[i];
  for (size_t i = outVecBytes + threadIdx.x; i < fullBytes; i += blockDim.x)
    recv[i] = h.slabDev[slotBase + i];

  // 4. Signal slot done.
  h.signalDone(slot, epoch);
}

// ── Path 3: GPU posts D2H ring commands; CPU drives wait + H2D ───────────────
// GPU kernel: only posts D2H ring commands (one per chunk).
// CPU: cuStreamWaitValue64 for each peer's flag + cudaMemcpyAsync H2D.
// This avoids GPU SM reads from POSIX shm (unreliable cross-process in MPI).
__global__ static void gpuAllGatherPostCmdsKernel(
    GscDeviceHandle g,
    char const* send, size_t bytesPerRank, int rank,
    int slot, size_t chunkBytes, int nChunks, uint64_t epochBase) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int ci = 0; ci < nChunks; ++ci) {
    size_t offset = static_cast<size_t>(ci) * chunkBytes;
    size_t bytes  = (offset + chunkBytes <= bytesPerRank)
                        ? chunkBytes : bytesPerRank - offset;
    uint64_t epoch = epochBase + static_cast<uint64_t>(ci);
    g.put(slot, ci, send + offset, /*offset=*/0, bytes, epoch);
  }
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
    // Set the correct CUDA device for all kernel launches and CUDA ops.
    mscclpp::CudaDeviceGuard devGuard(cudaDevice);
    size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);

    // Path 1+2 use CPU-driven context (CpuStagingChannel with SM slab).
    // Path 3 uses GpuAllGatherContext (GpuStagingChannel with ring + service thread).
    bool useDma = fullBytes > hostAllGatherCoopMaxBytes();

    if (!useDma) {
      // ── Path 1+2: SM write/wait/read kernel ──────────────────────────────
      HostAllGatherContext& ctx = getHostAllGatherContext(comm, /*mapSlab=*/true);
      initializeHostAllGatherContext(ctx, comm, rank, nRanks, cudaDevice,
                                     bootstrapComm);

      // gpuKernelWorks is probed once during init: fall back if unavailable.
      if (!ctx.gpuKernelWorks)
        return runIntraNodeShmAllGather(sendbuff, recvbuff, bytesPerRank,
                                        comm, stream, rank, nRanks,
                                        nRanks, bootstrapComm, cudaDevice);

      CpuStagingChannel& buf = *ctx.buf;
      HsbDeviceHandle rawH  = buf.deviceHandle();

      uint64_t epoch = ++ctx.epoch;
      int slot = static_cast<int>((epoch - 1) % kCscMaxSlots);
      if (epoch > static_cast<uint64_t>(kCscMaxSlots))
        buf.waitDone(slot, (rank + 1) % nRanks, epoch - kCscMaxSlots);

      CscDeviceHandle h;
      h.slabDev = rawH.slabDev; h.ctrlDev = rawH.ctrlDev;
      h.rank = rawH.rank; h.nRanks = rawH.nRanks;
      h.bytesPerRank = rawH.bytesPerRank; h.slotStride = rawH.slotStride;
      h.counterStride = rawH.counterStride;

      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.h2dStream, ctx.inputReadyEvent, 0));

      int threads = std::max(nRanks, 256);
      gpuAllGatherSmKernel<<<1, threads, 0, ctx.h2dStream>>>(
          h, static_cast<char const*>(sendbuff), static_cast<char*>(recvbuff),
          bytesPerRank, rank, nRanks, slot, epoch);
      MSCCLPP_CUDATHROW(cudaGetLastError());

      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent, ctx.h2dStream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, ctx.h2dDoneEvent, 0));

    } else {
      // ── Path 3: chunked DMA pipeline ─────────────────────────────────────
      // HostAllGatherContext (already init for Path 1+2 probe) provides slab/ctrl.
      // GpuAllGatherContext provides only a small ring buffer (no second slab).
      // The ring service thread reuses hostCtx.buf (CpuStagingChannel) for D2H.
      HostAllGatherContext& hostCtx = getHostAllGatherContext(comm, /*mapSlab=*/true);
      initializeHostAllGatherContext(hostCtx, comm, rank, nRanks, cudaDevice,
                                     bootstrapComm);
      CpuStagingChannel& csc = *hostCtx.buf;
      HsbDeviceHandle rawH = csc.deviceHandle();

      GpuAllGatherContext& gpuCtx = getGpuAllGatherContext(comm);
      initializeGpuAllGatherContext(gpuCtx, comm, rank, nRanks, cudaDevice,
                                    bootstrapComm, &csc);

      uint64_t epochBase = ++hostCtx.epoch;
      int slot = static_cast<int>((epochBase - 1) % kCscMaxSlots);
      if (epochBase > static_cast<uint64_t>(kCscMaxSlots))
        csc.waitDone(slot, (rank + 1) % nRanks, epochBase - kCscMaxSlots);

      size_t chunkBytes = hostAllGatherChunkBytes(bytesPerRank);
      int    nChunks    = static_cast<int>(
          (bytesPerRank + chunkBytes - 1) / chunkBytes);

      MSCCLPP_CUDATHROW(cudaEventRecord(gpuCtx.inputReadyEvent, stream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(gpuCtx.h2dStream, gpuCtx.inputReadyEvent, 0));

      void* selfOutput = static_cast<char*>(recvbuff)
                         + static_cast<size_t>(rank) * bytesPerRank;
      if (sendbuff != selfOutput) {
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(selfOutput, sendbuff, bytesPerRank,
                                          cudaMemcpyDeviceToDevice, stream));
      }

      // Phase 1: GPU posts D2H ring commands (or CPU fallback if ring unavailable).
      // Build a GscDeviceHandle using the ring from GpuRingContext.
      GscDeviceHandle g{};
      g.ring      = gpuCtx.ring->ringDev;
      g.ringCtrl  = gpuCtx.ring->ctrlDev;
      g.ringMask  = kGscRingMask;
      g.rank      = rank;
      // slabDev/ctrlDev not needed for put() — ring-only handle.

      bool ringOk = (g.ring != static_cast<GscRingEntry*>(
                                    static_cast<void*>(gpuCtx.ring->ring)));
      if (ringOk) {
        gpuAllGatherPostCmdsKernel<<<1, 1, 0, gpuCtx.h2dStream>>>(
            g, static_cast<char const*>(sendbuff),
            bytesPerRank, rank, slot, chunkBytes, nChunks, epochBase);
        cudaError_t kErr = cudaGetLastError();
        if (kErr != cudaSuccess) { ringOk = false; cudaGetLastError(); }
      }
      if (!ringOk) {
        // CPU fallback: post D2H commands directly.
        for (int ci = 0; ci < nChunks; ++ci) {
          size_t off2 = static_cast<size_t>(ci) * chunkBytes;
          size_t byt2 = std::min(chunkBytes, bytesPerRank - off2);
          csc.put(gpuCtx.d2hStream, slot, ci, sendbuff, off2, byt2,
                  epochBase + static_cast<uint64_t>(ci));
        }
      }

      // Phase 3: CPU-driven wait + H2D (cuStreamWaitValue64 + cudaMemcpyAsync).
      bool usedH2dStream2 = false;
      for (int ci = 0; ci < nChunks; ++ci) {
        size_t offset = static_cast<size_t>(ci) * chunkBytes;
        size_t bytes  = std::min(chunkBytes, bytesPerRank - offset);
        uint64_t epoch = epochBase + static_cast<uint64_t>(ci);
        cudaStream_t rightStream = (rank > 0) ? gpuCtx.h2dStream2 : gpuCtx.h2dStream;
        for (int r = 0; r < rank; ++r)
          csc.wait(gpuCtx.h2dStream, slot, ci, r, epoch);
        for (int r = rank + 1; r < nRanks; ++r)
          csc.wait(rightStream, slot, ci, r, epoch);
        if (rank > 0)
          csc.get(gpuCtx.h2dStream, slot, 0, rank - 1, offset, bytes, recvbuff);
        if (rank + 1 < nRanks) {
          csc.get(rightStream, slot, rank + 1, nRanks - 1, offset, bytes, recvbuff);
          usedH2dStream2 = usedH2dStream2 || (rank > 0);
        }
      }
      if (usedH2dStream2) {
        MSCCLPP_CUDATHROW(cudaEventRecord(gpuCtx.h2dDoneEvent2, gpuCtx.h2dStream2));
        MSCCLPP_CUDATHROW(cudaStreamWaitEvent(gpuCtx.h2dStream, gpuCtx.h2dDoneEvent2, 0));
      }
      csc.signalDone(gpuCtx.h2dStream, slot, epochBase);
      MSCCLPP_CUDATHROW(cudaEventRecord(gpuCtx.h2dDoneEvent, gpuCtx.h2dStream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, gpuCtx.h2dDoneEvent, 0));
    }
    return ncclSuccess;
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
