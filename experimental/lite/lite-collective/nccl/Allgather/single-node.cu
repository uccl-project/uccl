// Included by nccl.cu inside its anonymous namespace so the intra-node
// AllGather paths can reuse the NCCL shim send/recv peer context.

namespace cg = cooperative_groups;

static constexpr int kHostAllGatherPollSpinsBeforeYield = 65536;
static constexpr size_t kIntraAllGatherRingMinTotalBytes = 8 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Single-node no-CudaIpc SHM AllGather (1nx4g opt-in).
// ---------------------------------------------------------------------------


static inline void hostAllGatherCpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  asm volatile("" ::: "memory");
#endif
}

static int hostAllGatherGpuNumaNode(int cudaDeviceId) {
  try {
    return mscclpp::getDeviceNumaNode(cudaDeviceId);
  } catch (...) {
    return -1;
  }
}

template <typename Fn>
ncclResult_t runHostAllGatherGuarded(char const* opName, Fn&& fn) {
  try {
    fn();
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN(MSCCLPP_NCCL, opName, " failed: ", ex.what());
    return mapMscclppException(ex);
  } catch (...) {
    WARN(MSCCLPP_NCCL, opName, " failed with an unknown exception");
    return ncclInternalError;
  }
}

static constexpr int kHostAllGatherMaxRanks = 8;
static constexpr int kHostAllGatherSlots = 2;
static constexpr size_t kHostAllGatherMaxTotalBytes = 1024ULL * 1024 * 1024;
static constexpr size_t kHostAllGatherMinChunkBytes = 256 * 1024;
static constexpr size_t kHostAllGatherMaxRankBytes =
    kHostAllGatherMaxTotalBytes / 4;
static constexpr size_t kHostAllGatherMaxChunks =
    kHostAllGatherMaxRankBytes / kHostAllGatherMinChunkBytes;
static constexpr size_t kHostAllGatherDefaultMinTotalBytes = 0;
static constexpr size_t kHostAllGatherDefaultKernelMaxBytes = 4 * 1024;
static constexpr int kHostAllGatherDefaultKernelThreads = 256;
static constexpr size_t kHostAllGatherDefaultCoopMaxBytes = 512 * 1024;
static constexpr size_t kHostAllGatherDefaultTwoKernelBlockBytes = 1024;
static constexpr int kHostAllGatherDefaultCoopMaxBlocks = 16;

struct HostAllGatherCounter {
  alignas(64) std::atomic<uint64_t> value{0};
};

struct HostAllGatherControl {
  HostAllGatherCounter d2hReady[kHostAllGatherSlots]
                               [kHostAllGatherMaxChunks]
                               [kHostAllGatherMaxRanks];
  HostAllGatherCounter mappedReadyCount[kHostAllGatherSlots];
  HostAllGatherCounter slotDone[kHostAllGatherSlots]
                               [kHostAllGatherMaxRanks];
};

__device__ unsigned long long volatile* hostAllGatherCtrlU64(
    char* ctrl, size_t offset) {
  return reinterpret_cast<unsigned long long volatile*>(ctrl + offset);
}

__global__ void hostAllGatherMappedKernel(
    char const* send, char* recv, char* slot, char* ctrl, size_t slotOffset,
    size_t bytesPerRank, int rank, int nRanks, unsigned long long epoch,
    size_t readyOffset, size_t doneOffset, size_t counterStride) {
  using Vec = unsigned long long;
  size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank;
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);
  size_t vecBytes = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  size_t vecCount = vecBytes / sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto* selfSlot =
      reinterpret_cast<Vec*>(slot + slotOffset + selfOffset);
  for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
    selfSlot[i] = sendVec[i];
  }
  for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank;
       i += blockDim.x) {
    slot[slotOffset + selfOffset + i] = send[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    *hostAllGatherCtrlU64(
        ctrl, readyOffset + static_cast<size_t>(rank) * counterStride) = epoch;
    for (int r = 0; r < nRanks; ++r) {
      while (*hostAllGatherCtrlU64(
                 ctrl, readyOffset + static_cast<size_t>(r) * counterStride) <
             epoch) {
      }
    }
  }
  __syncthreads();

  size_t outVecBytes = (fullBytes / sizeof(Vec)) * sizeof(Vec);
  size_t outVecCount = outVecBytes / sizeof(Vec);
  auto const* slotVec = reinterpret_cast<Vec const*>(slot + slotOffset);
  auto* recvVec = reinterpret_cast<Vec*>(recv);
  for (size_t i = threadIdx.x; i < outVecCount; i += blockDim.x) {
    recvVec[i] = slotVec[i];
  }
  for (size_t i = outVecBytes + threadIdx.x; i < fullBytes;
       i += blockDim.x) {
    recv[i] = slot[slotOffset + i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    *hostAllGatherCtrlU64(
        ctrl, doneOffset + static_cast<size_t>(rank) * counterStride) = epoch;
  }
}

__global__ void hostAllGatherCoopKernel(
    char const* send, char* recv, char* slot, char* ctrl, size_t slotOffset,
    size_t bytesPerRank, int rank, int nRanks, unsigned long long epoch,
    size_t readyOffset, size_t doneOffset, size_t counterStride) {
  cg::grid_group grid = cg::this_grid();
  using Vec = unsigned long long;
  size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank;
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  size_t vecBytes = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  size_t vecCount = vecBytes / sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto* selfSlot =
      reinterpret_cast<Vec*>(slot + slotOffset + selfOffset);
  for (size_t i = tid; i < vecCount; i += stride) {
    selfSlot[i] = sendVec[i];
  }
  for (size_t i = vecBytes + tid; i < bytesPerRank; i += stride) {
    slot[slotOffset + selfOffset + i] = send[i];
  }
  grid.sync();

  if (tid == 0) {
    __threadfence_system();
    *hostAllGatherCtrlU64(
        ctrl, readyOffset + static_cast<size_t>(rank) * counterStride) = epoch;
    for (int r = 0; r < nRanks; ++r) {
      while (*hostAllGatherCtrlU64(
                 ctrl, readyOffset + static_cast<size_t>(r) * counterStride) <
             epoch) {
      }
    }
  }
  grid.sync();

  size_t outVecBytes = (fullBytes / sizeof(Vec)) * sizeof(Vec);
  size_t outVecCount = outVecBytes / sizeof(Vec);
  auto const* slotVec = reinterpret_cast<Vec const*>(slot + slotOffset);
  auto* recvVec = reinterpret_cast<Vec*>(recv);
  for (size_t i = tid; i < outVecCount; i += stride) {
    recvVec[i] = slotVec[i];
  }
  for (size_t i = outVecBytes + tid; i < fullBytes; i += stride) {
    recv[i] = slot[slotOffset + i];
  }
  grid.sync();
  if (tid == 0) {
    __threadfence_system();
    *hostAllGatherCtrlU64(
        ctrl, doneOffset + static_cast<size_t>(rank) * counterStride) = epoch;
  }
}

__global__ void hostAllGatherSelfCopyKernel(char const* src, char* dst,
                                            size_t bytes) {
  using Vec = unsigned long long;
  size_t vecBytes = (bytes / sizeof(Vec)) * sizeof(Vec);
  size_t vecCount = vecBytes / sizeof(Vec);
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  auto const* srcVec = reinterpret_cast<Vec const*>(src);
  auto* dstVec = reinterpret_cast<Vec*>(dst);
  for (size_t i = tid; i < vecCount; i += stride) {
    dstVec[i] = srcVec[i];
  }
  for (size_t i = vecBytes + tid; i < bytes; i += stride) {
    dst[i] = src[i];
  }
}

struct HostAllGatherNames {
  char slabName[128] = {};
  char ctrlName[128] = {};
};

struct HostAllGatherContext {
  std::mutex initMutex;
  std::condition_variable initCv;
  bool initialized = false;
  bool initializing = false;
  std::exception_ptr initException;

  int rank = -1;
  int nRanks = 0;
  int cudaDevice = -1;
  bool isLeader = false;
  bool mapSlab = true;
  bool cooperativeLaunch = false;
  size_t slotStrideBytes = kHostAllGatherMaxTotalBytes;
  size_t slabBytes = kHostAllGatherSlots * kHostAllGatherMaxTotalBytes;
  uint64_t epoch = 0;

  std::string slabName;
  std::string ctrlName;
  void* slabMapping = nullptr;
  void* ctrlMapping = nullptr;
  char* slab = nullptr;
  char* slabDevice = nullptr;
  HostAllGatherControl* ctrl = nullptr;
  char* ctrlDevice = nullptr;
  bool slabRegistered = false;
  bool ctrlRegistered = false;
  cudaStream_t d2hStream = nullptr;
  cudaStream_t h2dStream = nullptr;
  cudaStream_t h2dStream2 = nullptr;
  cudaEvent_t inputReadyEvent = nullptr;
  cudaEvent_t h2dDoneEvent = nullptr;
  cudaEvent_t h2dDoneEvent2 = nullptr;

  ~HostAllGatherContext() {
    if (h2dDoneEvent) cudaEventDestroy(h2dDoneEvent);
    if (h2dDoneEvent2) cudaEventDestroy(h2dDoneEvent2);
    if (inputReadyEvent) cudaEventDestroy(inputReadyEvent);
    if (h2dStream) cudaStreamDestroy(h2dStream);
    if (h2dStream2) cudaStreamDestroy(h2dStream2);
    if (d2hStream) cudaStreamDestroy(d2hStream);
    if (ctrlRegistered) cudaHostUnregister(ctrlMapping);
    if (slabRegistered) cudaHostUnregister(slabMapping);
    if (ctrlMapping) munmap(ctrlMapping, sizeof(HostAllGatherControl));
    if (slabMapping) munmap(slabMapping, slabBytes);
    if (isLeader) {
      if (!ctrlName.empty()) shm_unlink(ctrlName.c_str());
      if (!slabName.empty()) shm_unlink(slabName.c_str());
    }
  }
};

// Compute device address of a ctrl flag given its host address.
// Requires ctx.ctrlDevice != nullptr (ctrl is always device-mapped).
static CUdeviceptr ctrlFlagDevPtr(HostAllGatherContext const& ctx,
                                   std::atomic<uint64_t> const* flagHost) {
  ptrdiff_t off = reinterpret_cast<char const*>(flagHost) -
                  reinterpret_cast<char const*>(ctx.ctrl);
  return reinterpret_cast<CUdeviceptr>(ctx.ctrlDevice + off);
}

// Queue a GPU DMA write of a 64-bit epoch to a ctrl flag — no CPU callback.
static void streamWriteFlag64(cudaStream_t stream, CUdeviceptr addr,
                               uint64_t value) {
  CUresult res = cuStreamWriteValue64(reinterpret_cast<CUstream>(stream), addr,
                                      static_cast<cuuint64_t>(value),
                                      CU_STREAM_WRITE_VALUE_DEFAULT);
  if (res != CUDA_SUCCESS)
    throw mscclpp::Error("cuStreamWriteValue64 failed in SHM allgather",
                         mscclpp::ErrorCode::SystemError);
}

// Queue a GPU-side wait on a ctrl flag — no CPU spin, enables D2H/H2D overlap.
static void streamWaitFlag64(cudaStream_t stream, CUdeviceptr addr,
                              uint64_t value) {
  CUresult res = cuStreamWaitValue64(reinterpret_cast<CUstream>(stream), addr,
                                     static_cast<cuuint64_t>(value),
                                     CU_STREAM_WAIT_VALUE_GEQ);
  if (res != CUDA_SUCCESS)
    throw mscclpp::Error("cuStreamWaitValue64 failed in SHM allgather",
                         mscclpp::ErrorCode::SystemError);
}

static std::mutex gHostAllGatherContextMutex;
static std::unordered_map<ncclComm_t, std::unique_ptr<HostAllGatherContext>>
    gHostAllGatherMappedContexts;
static std::unordered_map<ncclComm_t, std::unique_ptr<HostAllGatherContext>>
    gHostAllGatherPinnedContexts;

static bool hostAllGatherMapSlabEnabled() {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_MAP_SLAB");
  return env == nullptr || std::strcmp(env, "0") != 0;
}

static bool hostAllGatherSelfKernelEnabled() {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_SELF_KERNEL");
  return env != nullptr && std::strcmp(env, "0") != 0;
}

static bool hostAllGatherNumaPlacementEnabled() {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_NUMA_PLACE");
  return env != nullptr && std::strcmp(env, "0") != 0;
}

static size_t hostAllGatherMinTotalBytes() {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_MIN_BYTES");
  if (env == nullptr || env[0] == '\0') {
    return kHostAllGatherDefaultMinTotalBytes;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env || parsed == 0) return kHostAllGatherDefaultMinTotalBytes;
  return static_cast<size_t>(parsed);
}

static size_t hostAllGatherKernelMaxBytes() {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_KERNEL_MAX_BYTES");
  if (env == nullptr || env[0] == '\0') {
    return kHostAllGatherDefaultKernelMaxBytes;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env) return kHostAllGatherDefaultKernelMaxBytes;
  return static_cast<size_t>(parsed);
}

static int hostAllGatherKernelThreads() {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_KERNEL_THREADS");
  if (env == nullptr || env[0] == '\0') return kHostAllGatherDefaultKernelThreads;
  char* end = nullptr;
  long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed <= 0) return kHostAllGatherDefaultKernelThreads;
  return static_cast<int>(parsed);
}

static size_t hostAllGatherCoopMaxBytes() {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_COOP_MAX_BYTES");
  if (env == nullptr || env[0] == '\0') {
    return kHostAllGatherDefaultCoopMaxBytes;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env) return kHostAllGatherDefaultCoopMaxBytes;
  return static_cast<size_t>(parsed);
}

static size_t hostAllGatherTwoKernelBlockBytes() {
  char const* env =
      std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_TWO_KERNEL_BLOCK_BYTES");
  if (env == nullptr || env[0] == '\0') {
    return kHostAllGatherDefaultTwoKernelBlockBytes;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env || parsed == 0) {
    return kHostAllGatherDefaultTwoKernelBlockBytes;
  }
  return static_cast<size_t>(parsed);
}

static int hostAllGatherCoopMaxBlocks() {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_COOP_MAX_BLOCKS");
  if (env == nullptr || env[0] == '\0') {
    return kHostAllGatherDefaultCoopMaxBlocks;
  }
  char* end = nullptr;
  long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed <= 0) return kHostAllGatherDefaultCoopMaxBlocks;
  return static_cast<int>(parsed);
}

static size_t hostAllGatherChunkBytes(size_t bytesPerRank) {
  char const* env = std::getenv("MSCCLPP_NCCL_HOST_ALLGATHER_CHUNK_BYTES");
  if (env == nullptr || env[0] == '\0') {
    // Use 1 MiB chunks for everything up to 32 MiB/rank.
    // This gives N = bytesPerRank/1MiB overlap stages (4–32 stages for
    // 4–32 MiB/rank), enabling much better D2H/H2D pipeline overlap than
    // the old bytesPerRank/2 rule which always gave only 2 stages.
    if (bytesPerRank > 1024 * 1024 && bytesPerRank <= 32 * 1024 * 1024) {
      return 1024 * 1024;
    }
    // For larger messages, use 4 MiB chunks: enough stages for good overlap
    // while keeping per-chunk overhead low (cuStreamWaitValue64 per chunk).
    if (bytesPerRank > 32 * 1024 * 1024) {
      return 4 * 1024 * 1024;
    }
    return bytesPerRank;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(env, &end, 10);
  if (end == env || parsed < kHostAllGatherMinChunkBytes) {
    return bytesPerRank;
  }
  size_t bytes = static_cast<size_t>(parsed);
  return std::min(bytes, bytesPerRank);
}

static HostAllGatherContext& getHostAllGatherContext(ncclComm_t comm,
                                                     bool mapSlab) {
  std::lock_guard<std::mutex> lock(gHostAllGatherContextMutex);
  auto& contexts =
      mapSlab ? gHostAllGatherMappedContexts : gHostAllGatherPinnedContexts;
  auto& ptr = contexts[comm];
  if (!ptr) {
    ptr = std::make_unique<HostAllGatherContext>();
    ptr->mapSlab = mapSlab;
  }
  return *ptr;
}

static void cleanupHostAllGatherContexts(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(gHostAllGatherContextMutex);
  gHostAllGatherMappedContexts.erase(comm);
  gHostAllGatherPinnedContexts.erase(comm);
}

struct HostAllGatherInitStatus {
  int result = static_cast<int>(ncclSuccess);
  char message[160] = {};
};

class HostAllGatherInitGuard {
 public:
  explicit HostAllGatherInitGuard(HostAllGatherContext* ctx) : ctx_(ctx) {}
  ~HostAllGatherInitGuard() {
    if (committed_) return;
    auto failure = std::current_exception();
    if (!failure) {
      failure = std::make_exception_ptr(mscclpp::Error(
          "host allgather context initialization did not complete",
          mscclpp::ErrorCode::InternalError));
    }
    {
      std::lock_guard<std::mutex> lock(ctx_->initMutex);
      ctx_->initialized = false;
      ctx_->initializing = false;
      ctx_->initException = failure;
    }
    ctx_->initCv.notify_all();
  }
  void commit() {
    {
      std::lock_guard<std::mutex> lock(ctx_->initMutex);
      ctx_->initialized = true;
      ctx_->initializing = false;
      ctx_->initException = nullptr;
    }
    committed_ = true;
    ctx_->initCv.notify_all();
  }

 private:
  HostAllGatherContext* ctx_;
  bool committed_ = false;
};

static mscclpp::ErrorCode hostAllGatherInitErrorCode(ncclResult_t result) {
  switch (result) {
    case ncclInvalidArgument:
    case ncclInvalidUsage:
      return mscclpp::ErrorCode::InvalidUsage;
    case ncclSystemError:
      return mscclpp::ErrorCode::SystemError;
    default:
      return mscclpp::ErrorCode::InternalError;
  }
}

static void publishHostAllGatherInitStatus(
    std::shared_ptr<mscclpp::Communicator> bootstrapComm, int rank, int nRanks,
    ncclResult_t result, std::string const& message, char const* stage) {
  std::vector<HostAllGatherInitStatus> statuses(static_cast<size_t>(nRanks));
  auto& local = statuses[static_cast<size_t>(rank)];
  local.result = static_cast<int>(result);
  if (!message.empty()) {
    std::snprintf(local.message, sizeof(local.message), "%s",
                  message.c_str());
  }
  bootstrapComm->bootstrap()->allGather(statuses.data(),
                                        sizeof(HostAllGatherInitStatus));
  for (int peer = 0; peer < nRanks; ++peer) {
    auto peerResult =
        static_cast<ncclResult_t>(statuses[static_cast<size_t>(peer)].result);
    if (peerResult != ncclSuccess) {
      std::string detail(statuses[static_cast<size_t>(peer)].message);
      if (detail.empty()) detail = "unknown initialization error";
      throw mscclpp::Error(std::string(stage) + " failed on rank " +
                               std::to_string(peer) + ": " + detail,
                           hostAllGatherInitErrorCode(peerResult));
    }
  }
}

static void createHostAllGatherShm(std::string const& name, size_t size) {
  shm_unlink(name.c_str());
  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) {
    throw mscclpp::Error("shm_open failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  if (ftruncate(fd, size) < 0) {
    close(fd);
    shm_unlink(name.c_str());
    throw mscclpp::Error("ftruncate failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  close(fd);
}

static void* mapHostAllGatherShm(std::string const& name, size_t size) {
  int fd = shm_open(name.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    throw mscclpp::Error("shm_open failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  void* mapping =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (mapping == MAP_FAILED) {
    throw mscclpp::Error("mmap failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
#ifdef MADV_HUGEPAGE
  (void)madvise(mapping, size, MADV_HUGEPAGE);
#endif
  return mapping;
}

static void waitHostAllGatherEpoch(std::atomic<uint64_t> const& value,
                                   uint64_t epoch) {
  int spins = 0;
  while (value.load(std::memory_order_acquire) < epoch) {
    if (spins++ < kHostAllGatherPollSpinsBeforeYield) {
      hostAllGatherCpuRelax();
    } else {
      std::this_thread::yield();
    }
  }
}

static void placeHostAllGatherOnNuma(void* mapping, size_t size, int numaNode,
                                     char const* name) {
  if (mapping == nullptr || size == 0 || numaNode < 0) return;
  if (numa_available() < 0) {
    WARN(MSCCLPP_NCCL, "NUMA placement unavailable for ", name);
    return;
  }
  std::memset(mapping, 0, size);
  numa_tonode_memory(mapping, size, numaNode);
}

static void initializeHostAllGatherContext(
    HostAllGatherContext& ctx, ncclComm_t comm, int rank, int nRanks,
    int cudaDevice, std::shared_ptr<mscclpp::Communicator> bootstrapComm) {
  {
    std::unique_lock<std::mutex> lock(ctx.initMutex);
    if (ctx.initialized) return;
    if (ctx.initializing) {
      ctx.initCv.wait(lock, [&] { return !ctx.initializing; });
      if (ctx.initException) std::rethrow_exception(ctx.initException);
      if (ctx.initialized) return;
    }
    ctx.initializing = true;
  }

  HostAllGatherInitGuard guard(&ctx);
  auto bootstrap = bootstrapComm->bootstrap();
  ctx.rank = rank;
  ctx.nRanks = nRanks;
  ctx.cudaDevice = cudaDevice;
  ctx.isLeader = rank == 0;
  ctx.slabBytes = kHostAllGatherSlots * ctx.slotStrideBytes;

  HostAllGatherNames localNames;
  ncclResult_t createResult = ncclSuccess;
  std::string createMessage;
  try {
    if (ctx.isLeader) {
      unsigned long long commNonce =
          static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(comm));
      std::snprintf(localNames.slabName, sizeof(localNames.slabName),
                    "/mint_ag_host_%s_slab_%d_%llx_%d",
                    ctx.mapSlab ? "mapped" : "pinned", getpid(), commNonce,
                    rank);
      std::snprintf(localNames.ctrlName, sizeof(localNames.ctrlName),
                    "/mint_ag_host_%s_ctrl_%d_%llx_%d",
                    ctx.mapSlab ? "mapped" : "pinned", getpid(), commNonce,
                    rank);
      createHostAllGatherShm(localNames.slabName, ctx.slabBytes);
      createHostAllGatherShm(localNames.ctrlName,
                             sizeof(HostAllGatherControl));
    }
  } catch (std::exception const& ex) {
    createResult = mapMscclppException(ex);
    createMessage = ex.what();
  } catch (...) {
    createResult = ncclInternalError;
    createMessage = "unknown host allgather shm create exception";
  }
  publishHostAllGatherInitStatus(
      bootstrapComm, rank, nRanks, createResult, createMessage,
      "host allgather shared-memory create");

  std::vector<HostAllGatherNames> allNames(static_cast<size_t>(nRanks));
  allNames[static_cast<size_t>(rank)] = localNames;
  bootstrap->allGather(allNames.data(), sizeof(HostAllGatherNames));
  HostAllGatherNames const& leaderNames = allNames[0];
  ctx.slabName = leaderNames.slabName;
  ctx.ctrlName = leaderNames.ctrlName;

  ncclResult_t setupResult = ncclSuccess;
  std::string setupMessage;
  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    ctx.slabMapping = mapHostAllGatherShm(ctx.slabName, ctx.slabBytes);
    ctx.ctrlMapping =
        mapHostAllGatherShm(ctx.ctrlName, sizeof(HostAllGatherControl));
    ctx.slab = static_cast<char*>(ctx.slabMapping);
    ctx.ctrl = static_cast<HostAllGatherControl*>(ctx.ctrlMapping);
    if (ctx.isLeader) {
      std::memset(ctx.ctrlMapping, 0, sizeof(HostAllGatherControl));
      new (ctx.ctrl) HostAllGatherControl{};
    }
    bootstrap->barrier();

    size_t rankStride = kHostAllGatherMaxTotalBytes /
                        static_cast<size_t>(nRanks);
    size_t rankOffset = static_cast<size_t>(rank) * rankStride;
    if (hostAllGatherNumaPlacementEnabled()) {
      for (int slot = 0; slot < kHostAllGatherSlots; ++slot) {
        placeHostAllGatherOnNuma(
            ctx.slab + static_cast<size_t>(slot) * ctx.slotStrideBytes +
                rankOffset,
            rankStride, hostAllGatherGpuNumaNode(cudaDevice),
            "single-node SHM allgather default rank slab");
      }
    }

    MSCCLPP_CUDATHROW(cudaHostRegister(
        ctx.slabMapping, ctx.slabBytes,
        ctx.mapSlab ? (cudaHostRegisterPortable | cudaHostRegisterMapped)
                    : cudaHostRegisterPortable));
    ctx.slabRegistered = true;
    if (ctx.mapSlab) {
      void* slabDevice = nullptr;
      MSCCLPP_CUDATHROW(
          cudaHostGetDevicePointer(&slabDevice, ctx.slabMapping, 0));
      ctx.slabDevice = static_cast<char*>(slabDevice);
    }
    // ctrl is always device-mapped so GPU streams can read/write flags directly
    // via cuStreamWriteValue64/cuStreamWaitValue64. Only the slab follows mapSlab.
    MSCCLPP_CUDATHROW(cudaHostRegister(
        ctx.ctrlMapping, sizeof(HostAllGatherControl),
        cudaHostRegisterPortable | cudaHostRegisterMapped));
    ctx.ctrlRegistered = true;
    {
      void* ctrlDevice = nullptr;
      MSCCLPP_CUDATHROW(
          cudaHostGetDevicePointer(&ctrlDevice, ctx.ctrlMapping, 0));
      ctx.ctrlDevice = static_cast<char*>(ctrlDevice);
    }
    int leastPriority = 0;
    int greatestPriority = 0;
    MSCCLPP_CUDATHROW(
        cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    (void)leastPriority;
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(
        &ctx.d2hStream, cudaStreamNonBlocking, greatestPriority));
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(
        &ctx.h2dStream, cudaStreamNonBlocking, greatestPriority));
    MSCCLPP_CUDATHROW(cudaStreamCreateWithPriority(
        &ctx.h2dStream2, cudaStreamNonBlocking, greatestPriority));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.inputReadyEvent,
                                               cudaEventDisableTiming));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent,
                                               cudaEventDisableTiming));
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent2,
                                               cudaEventDisableTiming));
    int coop = 0;
    MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(
        &coop, cudaDevAttrCooperativeLaunch, cudaDevice));
    ctx.cooperativeLaunch = coop != 0;
  } catch (std::exception const& ex) {
    setupResult = mapMscclppException(ex);
    setupMessage = ex.what();
  } catch (...) {
    setupResult = ncclInternalError;
    setupMessage = "unknown SHM allgather setup exception";
  }
  publishHostAllGatherInitStatus(bootstrapComm, rank, nRanks, setupResult,
                                 setupMessage, "SHM allgather setup");
  bootstrap->barrier();
  guard.commit();
}

ncclResult_t runIntraNodeShmAllGather(
    void const* sendbuff, void* recvbuff, size_t bytesPerRank,
    ncclComm_t comm, cudaStream_t stream, int rank, int nRanks,
    int nRanksPerNode, std::shared_ptr<mscclpp::Communicator> bootstrapComm,
    int cudaDevice) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr ||
      nRanks < 2 || nRanks > kHostAllGatherMaxRanks ||
      nRanks != nRanksPerNode) {
    return ncclInvalidUsage;
  }
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);
  if (fullBytes < hostAllGatherMinTotalBytes()) return ncclInvalidUsage;
  if (bytesPerRank >
      kHostAllGatherMaxTotalBytes / static_cast<size_t>(nRanks)) {
    return ncclInvalidUsage;
  }
  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
  cudaError_t captureErr = cudaStreamIsCapturing(stream, &captureStatus);
  if (captureErr != cudaSuccess) return ncclUnhandledCudaError;
  if (captureStatus != cudaStreamCaptureStatusNone) return ncclInvalidUsage;

  return runHostAllGatherGuarded("intra-node SHM AllGather", [&]() {
    bool useMappedContext =
        hostAllGatherMapSlabEnabled() &&
        fullBytes <= hostAllGatherCoopMaxBytes();
    HostAllGatherContext& ctx = getHostAllGatherContext(comm, useMappedContext);
    initializeHostAllGatherContext(ctx, comm, rank, nRanks, cudaDevice,
                                   bootstrapComm);

    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    uint64_t epoch = ++ctx.epoch;
    int slot = static_cast<int>((epoch - 1) % kHostAllGatherSlots);
    if (epoch > kHostAllGatherSlots) {
      uint64_t reuseEpoch = epoch - kHostAllGatherSlots;
      for (int r = 0; r < nRanks; ++r) {
        waitHostAllGatherEpoch(ctx.ctrl->slotDone[slot][r].value,
                               reuseEpoch);
      }
    }

    size_t chunkBytes = hostAllGatherChunkBytes(bytesPerRank);
    size_t chunkCount = (bytesPerRank + chunkBytes - 1) / chunkBytes;
    if (chunkCount > kHostAllGatherMaxChunks) {
      throw mscclpp::Error("host allgather chunk count exceeds control slab",
                           mscclpp::ErrorCode::InvalidUsage);
    }
    char* slotBase =
        ctx.slab + static_cast<size_t>(slot) * ctx.slotStrideBytes;
    char* selfHost = slotBase + static_cast<size_t>(rank) * bytesPerRank;
    auto const* sendBytes = static_cast<char const*>(sendbuff);
    auto* recvBytes = static_cast<char*>(recvbuff);
    void* selfOutput = recvBytes + static_cast<size_t>(rank) * bytesPerRank;
    bool useVectorHostKernels =
        (bytesPerRank % sizeof(unsigned long long) == 0) &&
        ((reinterpret_cast<uintptr_t>(sendbuff) &
          (sizeof(unsigned long long) - 1)) == 0) &&
        ((reinterpret_cast<uintptr_t>(recvbuff) &
          (sizeof(unsigned long long) - 1)) == 0);

    // ── Single-CTA kernel (tiny messages, ≤ kernelMaxBytes) ──────────────────
    // One block reads its own data to mapped SHM, waits for all peers via
    // __syncthreads() on mapped ctrl flags, then reads peers' data back to GPU.
    // No DMA engine, no streams — pure GPU kernel on mapped host memory.
    if (useVectorHostKernels && chunkCount == 1 &&
        ctx.slabDevice != nullptr &&
        ctx.ctrlDevice != nullptr &&
        fullBytes <= hostAllGatherKernelMaxBytes()) {
      size_t slotOffset = static_cast<size_t>(slot) * ctx.slotStrideBytes;
      size_t readyOffset =
          reinterpret_cast<char*>(&ctx.ctrl->d2hReady[slot][0][0].value) -
          reinterpret_cast<char*>(ctx.ctrl);
      size_t doneOffset =
          reinterpret_cast<char*>(&ctx.ctrl->slotDone[slot][0].value) -
          reinterpret_cast<char*>(ctx.ctrl);
      hostAllGatherMappedKernel<<<1, hostAllGatherKernelThreads(), 0, stream>>>(
          sendBytes, recvBytes, ctx.slabDevice, ctx.ctrlDevice, slotOffset,
          bytesPerRank, rank, nRanks, epoch, readyOffset, doneOffset,
          sizeof(HostAllGatherCounter));
      MSCCLPP_CUDATHROW(cudaGetLastError());
      return;
    }

    // ── Cooperative-launch kernel (small messages, ≤ coopMaxBytes) ───────────
    // Up to coopMaxBlocks CTAs cooperate via grid.sync() on mapped host memory:
    // all blocks write self-data to SHM, sync across CTAs, then each block
    // reads the assigned range of peer data back to recvbuff.
    // Requires cooperative launch support (sm_60+, always available on L4/5090).
    if (useVectorHostKernels && chunkCount == 1 &&
        ctx.cooperativeLaunch &&
        ctx.slabDevice != nullptr &&
        ctx.ctrlDevice != nullptr &&
        fullBytes <= hostAllGatherCoopMaxBytes()) {
      size_t slotOffset = static_cast<size_t>(slot) * ctx.slotStrideBytes;
      size_t readyOffset =
          reinterpret_cast<char*>(&ctx.ctrl->d2hReady[slot][0][0].value) -
          reinterpret_cast<char*>(ctx.ctrl);
      size_t doneOffset =
          reinterpret_cast<char*>(&ctx.ctrl->slotDone[slot][0].value) -
          reinterpret_cast<char*>(ctx.ctrl);
      size_t blockBytes = hostAllGatherTwoKernelBlockBytes();
      int blocks = std::min<int>(
          hostAllGatherCoopMaxBlocks(),
          std::max<int>(1, static_cast<int>((fullBytes + blockBytes - 1) /
                                            blockBytes)));
      char const* sendArg = sendBytes;
      char* recvArg = recvBytes;
      char* slabArg = ctx.slabDevice;
      char* ctrlArg = ctx.ctrlDevice;
      size_t counterStride = sizeof(HostAllGatherCounter);
      void* args[] = {&sendArg,
                      &recvArg,
                      &slabArg,
                      &ctrlArg,
                      &slotOffset,
                      &bytesPerRank,
                      &rank,
                      &nRanks,
                      &epoch,
                      &readyOffset,
                      &doneOffset,
                      &counterStride};
      MSCCLPP_CUDATHROW(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(hostAllGatherCoopKernel), blocks, 256, args,
          0, stream));
      return;
    }

    // ── DMA-pipeline: GPU-signaled H2D overlapping D2H (large messages) ──────
    // Phase 1 (d2hStream): D2H of own data to SHM; GPU writes d2hReady flag
    //   via cuStreamWriteValue64 after each chunk — no CPU callback latency.
    // Phase 2 (user stream): D2D self-copy into recvbuff[rank] concurrently.
    // Phase 3 (h2dStream/h2dStream2): for each chunk, queue cuStreamWaitValue64
    //   per peer then immediately issue H2D — CPU never blocks; PCIe DMA engine
    //   runs D2H and H2D concurrently in both directions for high throughput.

    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
    MSCCLPP_CUDATHROW(
        cudaStreamWaitEvent(ctx.d2hStream, ctx.inputReadyEvent, 0));
    for (size_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
      size_t chunkOffset = chunkIdx * chunkBytes;
      size_t bytes = std::min(chunkBytes, bytesPerRank - chunkOffset);
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(selfHost + chunkOffset,
                                        sendBytes + chunkOffset, bytes,
                                        cudaMemcpyDeviceToHost,
                                        ctx.d2hStream));
      streamWriteFlag64(ctx.d2hStream,
                        ctrlFlagDevPtr(ctx, &ctx.ctrl->d2hReady[slot][chunkIdx][rank].value),
                        epoch);
    }

    if (sendbuff != selfOutput) {
      if (hostAllGatherSelfKernelEnabled() && useVectorHostKernels) {
        int blocks = std::min<int>(
            128, std::max<int>(
                     1, static_cast<int>((bytesPerRank + 4095) / 4096)));
        hostAllGatherSelfCopyKernel<<<blocks, 256, 0, stream>>>(
            sendBytes, static_cast<char*>(selfOutput), bytesPerRank);
        MSCCLPP_CUDATHROW(cudaGetLastError());
      } else {
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(selfOutput, sendbuff, bytesPerRank,
                                          cudaMemcpyDeviceToDevice, stream));
      }
    }

    // Phase 3: queue GPU-side waits + H2D per chunk.
    // h2dStream:  left half (ranks < self), h2dStream2: right half (ranks > self).
    bool usedSecondH2dStream = false;
    for (size_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
      size_t chunkOffset = chunkIdx * chunkBytes;
      size_t bytes = std::min(chunkBytes, bytesPerRank - chunkOffset);

      cudaStream_t rightStream = rank > 0 ? ctx.h2dStream2 : ctx.h2dStream;
      for (int r = 0; r < rank; ++r) {
        streamWaitFlag64(ctx.h2dStream,
                         ctrlFlagDevPtr(ctx, &ctx.ctrl->d2hReady[slot][chunkIdx][r].value),
                         epoch);
      }
      for (int r = rank + 1; r < nRanks; ++r) {
        streamWaitFlag64(rightStream,
                         ctrlFlagDevPtr(ctx, &ctx.ctrl->d2hReady[slot][chunkIdx][r].value),
                         epoch);
      }

      if (bytes == bytesPerRank) {
        if (rank > 0) {
          size_t copyBytes = static_cast<size_t>(rank) * bytesPerRank;
          MSCCLPP_CUDATHROW(cudaMemcpyAsync(
              recvBytes, slotBase, copyBytes, cudaMemcpyHostToDevice,
              ctx.h2dStream));
        }
        if (rank + 1 < nRanks) {
          size_t first = static_cast<size_t>(rank + 1);
          size_t copyBytes =
              static_cast<size_t>(nRanks - rank - 1) * bytesPerRank;
          MSCCLPP_CUDATHROW(cudaMemcpyAsync(
              recvBytes + first * bytesPerRank,
              slotBase + first * bytesPerRank, copyBytes,
              cudaMemcpyHostToDevice, rightStream));
          usedSecondH2dStream = usedSecondH2dStream || rank > 0;
        }
      } else {
        if (rank > 0) {
          MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
              recvBytes + chunkOffset, bytesPerRank,
              slotBase + chunkOffset, bytesPerRank, bytes,
              static_cast<size_t>(rank), cudaMemcpyHostToDevice,
              ctx.h2dStream));
        }
        if (rank + 1 < nRanks) {
          size_t first = static_cast<size_t>(rank + 1);
          MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
              recvBytes + first * bytesPerRank + chunkOffset, bytesPerRank,
              slotBase + first * bytesPerRank + chunkOffset, bytesPerRank,
              bytes, static_cast<size_t>(nRanks - rank - 1),
              cudaMemcpyHostToDevice, rightStream));
          usedSecondH2dStream = usedSecondH2dStream || rank > 0;
        }
      }
    }
    if (usedSecondH2dStream) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent2, ctx.h2dStream2));
      MSCCLPP_CUDATHROW(
          cudaStreamWaitEvent(ctx.h2dStream, ctx.h2dDoneEvent2, 0));
    }
    streamWriteFlag64(ctx.h2dStream,
                      ctrlFlagDevPtr(ctx, &ctx.ctrl->slotDone[slot][rank].value),
                      epoch);
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
