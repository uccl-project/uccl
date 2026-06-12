#include "native_collectives.hpp"
#include "lite_common.h"
#include "debug.h"
#include "gpu_utils.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

namespace mscclpp {
namespace nccl {
namespace {

static constexpr int kTagBase = 0x572000;
static constexpr int kTagStride = 32;
static constexpr int kMaxRanksPerNode = 8;
static constexpr int kScratchRowsPerChunk = 16;
static constexpr int kPipelineSlots = 4;
static constexpr int kTwoRankPipelineSlots = 5;
static constexpr int kLocalFourPipelineSlots = 1024;
static constexpr size_t kLocalLeadChunks = 3;
static constexpr size_t kLongLocalLeadChunks = 3;
static constexpr size_t kDefaultChunkBytes = 2 * 1024 * 1024;
static constexpr size_t kDefaultTwoRankChunkBytes = 2 * 1024 * 1024;
static constexpr size_t kMaxNativeBytesPerRank = 1024ULL * 1024 * 1024;
static constexpr size_t kRdmaChunkBytes = 4 * 1024 * 1024;
static constexpr size_t kSmallHostFullBytes = 512 * 1024;
static constexpr size_t kDefaultTwoRankSmallHostBytes = 512 * 1024;
static constexpr int kSmallSignalEveryN = 256;
static constexpr int kPairSignalEveryN = 1024;
static constexpr int kIpcDeviceFlagPhases = 2;

using lite::cudaResult;
using lite::createOwnedShm;
using lite::getAvailableIBTransports;
using lite::InitGuard;
using lite::mapException;
using lite::mapShm;
using lite::placeOnNuma;
using lite::publishInitStatus;
using lite::selectIBTransportForGpu;
using lite::waitForEpoch;

struct RsControl {
  alignas(64) std::atomic<uint64_t> localCopyReady[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> localCrossReady[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> localScratchDone[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> smallLocalDone[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> pairRdmaReady[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> pairRdmaSignal[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> pairAckReady[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> pairAckSignal[kMaxRanksPerNode];
};

struct RsContext {
  bool initialized = false;
  bool initializing = false;
  int rank = -1;
  int worldSize = -1;
  int nRanksPerNode = -1;
  int localRank = -1;
  int nodeId = -1;
  int localLeader = -1;
  int cudaDevice = -1;
  int numaNode = -1;
  bool owner = false;
  uint64_t epoch = 0;
  uint64_t smallOpCount = 0;
  int pipelineSlots = kPipelineSlots;
  size_t chunkCapacity = 0;
  size_t partialBlockCapacity = 0;
  size_t smallSlotBytes = 0;
  std::vector<uint64_t> smallSlotEpochs;

  std::string smallSendName;
  std::string smallRecvName;
  std::string ctrlName;
  void* sendMapping = nullptr;
  void* recvMapping = nullptr;
  void* smallSendMapping = nullptr;
  void* smallRecvMapping = nullptr;
  void* ctrlMapping = nullptr;
  char* sendSlab = nullptr;
  char* recvSlab = nullptr;
  char* smallSendSlab = nullptr;
  char* smallRecvSlab = nullptr;
  char* sendDeviceSlab = nullptr;
  char* recvDeviceSlab = nullptr;
  char* smallRecvDeviceSlab = nullptr;
  char* ctrlDeviceSlab = nullptr;
  RsControl* ctrl = nullptr;
  bool sendHostRegistered = false;
  bool recvHostRegistered = false;
  bool smallSendHostRegistered = false;
  bool smallRecvHostRegistered = false;
  bool ctrlHostRegistered = false;

  mscclpp::Transport transport = mscclpp::Transport::Unknown;
  mscclpp::RegisteredMemory sendMemory;
  mscclpp::RegisteredMemory recvMemory;
  mscclpp::RegisteredMemory smallRecvMemory;
  mscclpp::RegisteredMemory ctrlMemory;
  mscclpp::RegisteredMemory remoteRecvMemory;
  mscclpp::RegisteredMemory remoteSmallRecvMemory;
  mscclpp::RegisteredMemory remoteCtrlMemory;
  mscclpp::Connection pairConnection;
  std::shared_ptr<mscclpp::IbQp> smallQp;
  mscclpp::IbMr const* sendMr = nullptr;
  mscclpp::IbMr const* smallRecvMr = nullptr;
  mscclpp::IbMr const* smallCtrlMr = nullptr;
  mscclpp::IbMrInfo remoteRecvMrInfo{};
  mscclpp::IbMrInfo smallRemoteRecvMrInfo{};
  mscclpp::IbMrInfo smallRemoteCtrlMrInfo{};
  int smallWrCount = 0;
  uint64_t smallPendingAckEpoch = 0;
  uint64_t smallAckPostedEpoch = 0;

  bool localScratchIpcReady = false;
  void* localScratchBuffer = nullptr;
  size_t localScratchBufferSize = 0;
  mscclpp::RegisteredMemory localScratchMemory;
  std::vector<mscclpp::RegisteredMemory> remoteScratchMemories;
  std::vector<char*> remoteScratchPtrs;
  char* localDeviceFlags = nullptr;
  mscclpp::RegisteredMemory localDeviceFlagMemory;
  std::vector<mscclpp::RegisteredMemory> remoteDeviceFlagMemories;
  std::vector<char*> remoteDeviceFlagPtrs;
  size_t localFourRowStrideBytes = 0;
  bool localScratchEventIpcReady = false;
  std::vector<cudaEvent_t> localCopyEvents;
  std::vector<cudaEvent_t> localCrossEvents;
  std::vector<cudaEvent_t> localCopyStartEvents;
  std::vector<cudaEvent_t> localCopyDoneEvents;
  std::vector<std::vector<cudaEvent_t>> remoteCopyEvents;
  std::vector<std::vector<cudaEvent_t>> remoteCrossEvents;
  std::vector<cudaStream_t> localCopyStreams;
  cudaStream_t d2hStream = nullptr;
  cudaStream_t h2dStream = nullptr;
  std::vector<cudaEvent_t> reduceDoneEvents;
  std::vector<cudaEvent_t> d2hDoneEvents;
  std::vector<cudaEvent_t> h2dDoneEvents;
  std::vector<cudaEvent_t> slotDoneEvents;

  std::mutex initMutex;
  std::condition_variable initCv;
  std::exception_ptr initException = nullptr;

  ~RsContext() {
    pairConnection = mscclpp::Connection{};
    remoteScratchPtrs.clear();
    remoteDeviceFlagPtrs.clear();
    remoteScratchMemories.clear();
    remoteDeviceFlagMemories.clear();
    localDeviceFlagMemory = mscclpp::RegisteredMemory{};
    localScratchMemory = mscclpp::RegisteredMemory{};
    remoteCtrlMemory = mscclpp::RegisteredMemory{};
    remoteSmallRecvMemory = mscclpp::RegisteredMemory{};
    remoteRecvMemory = mscclpp::RegisteredMemory{};
    smallQp.reset();
    sendMr = nullptr;
    smallRecvMr = nullptr;
    smallCtrlMr = nullptr;
    remoteRecvMrInfo = {};
    smallRemoteRecvMrInfo = {};
    smallRemoteCtrlMrInfo = {};
    ctrlMemory = mscclpp::RegisteredMemory{};
    smallRecvMemory = mscclpp::RegisteredMemory{};
    recvMemory = mscclpp::RegisteredMemory{};
    sendMemory = mscclpp::RegisteredMemory{};
    if (sendHostRegistered) cudaHostUnregister(sendMapping);
    if (recvHostRegistered) cudaHostUnregister(recvMapping);
    if (smallSendHostRegistered) cudaHostUnregister(smallSendMapping);
    if (smallRecvHostRegistered) cudaHostUnregister(smallRecvMapping);
    if (ctrlHostRegistered) cudaHostUnregister(ctrlMapping);
    if (sendMapping) cudaFreeHost(sendMapping);
    if (recvMapping) cudaFreeHost(recvMapping);
    if (localDeviceFlags) cudaFree(localDeviceFlags);
    if (smallSendMapping) munmap(smallSendMapping, partialBlockCapacity);
    if (smallRecvMapping) munmap(smallRecvMapping, partialBlockCapacity);
    if (ctrlMapping) munmap(ctrlMapping, sizeof(RsControl));
    if (owner && !smallSendName.empty()) shm_unlink(smallSendName.c_str());
    if (owner && !smallRecvName.empty()) shm_unlink(smallRecvName.c_str());
    if (owner && !ctrlName.empty()) shm_unlink(ctrlName.c_str());
    for (auto event : reduceDoneEvents) cudaEventDestroy(event);
    for (auto event : d2hDoneEvents) cudaEventDestroy(event);
    for (auto event : h2dDoneEvents) cudaEventDestroy(event);
    for (auto event : slotDoneEvents) cudaEventDestroy(event);
    for (auto event : localCopyEvents) cudaEventDestroy(event);
    for (auto event : localCrossEvents) cudaEventDestroy(event);
    for (auto event : localCopyStartEvents) cudaEventDestroy(event);
    for (auto event : localCopyDoneEvents) cudaEventDestroy(event);
    for (auto const& events : remoteCopyEvents) {
      for (auto event : events) cudaEventDestroy(event);
    }
    for (auto const& events : remoteCrossEvents) {
      for (auto event : events) cudaEventDestroy(event);
    }
    for (auto copyStream : localCopyStreams) cudaStreamDestroy(copyStream);
    if (d2hStream) cudaStreamDestroy(d2hStream);
    if (h2dStream) cudaStreamDestroy(h2dStream);
  }
};

struct RsControlName {
  char smallSendName[96] = {};
  char smallRecvName[96] = {};
  char ctrlName[96] = {};
};

std::mutex gReduceScatterMutex;
enum RsTransportPolicy {
  kRsTransportLocality = 0,
  kRsTransportRoundRobinHca = 1,
};

struct RsContextKey {
  ncclComm_t comm = nullptr;
  int transportPolicy = kRsTransportLocality;

  bool operator==(RsContextKey const& other) const {
    return comm == other.comm && transportPolicy == other.transportPolicy;
  }
};

struct RsContextKeyHash {
  size_t operator()(RsContextKey const& key) const {
    auto commValue = reinterpret_cast<uintptr_t>(key.comm);
    return std::hash<uintptr_t>{}(commValue) ^
           (std::hash<int>{}(key.transportPolicy) + 0x9e3779b9 +
            (commValue << 6) + (commValue >> 2));
  }
};

struct RsPolicyCacheKey {
  ncclComm_t comm = nullptr;
  int policyClass = 0;

  bool operator==(RsPolicyCacheKey const& other) const {
    return comm == other.comm && policyClass == other.policyClass;
  }
};

struct RsPolicyCacheKeyHash {
  size_t operator()(RsPolicyCacheKey const& key) const {
    auto commValue = reinterpret_cast<uintptr_t>(key.comm);
    return std::hash<uintptr_t>{}(commValue) ^
           (std::hash<int>{}(key.policyClass) + 0x9e3779b9 +
            (commValue << 6) + (commValue >> 2));
  }
};

std::unordered_map<RsContextKey, std::unique_ptr<RsContext>, RsContextKeyHash>
    gReduceScatterContexts;
std::unordered_map<RsPolicyCacheKey, int, RsPolicyCacheKeyHash>
    gReduceScatterTransportPolicies;

int rsTag(int rank, int worldSize, int peer, int slot) {
  int lo = std::min(rank, peer);
  int hi = std::max(rank, peer);
  return kTagBase + (lo * worldSize + hi) * kTagStride + slot;
}

int localPeerIndex(int localRank, int peerLocalRank);

bool isTargetLayout(int nRanks, int nRanksPerNode) {
  return (nRanks == 4 && nRanksPerNode == 4) ||
         (nRanks == 4 && nRanksPerNode == 2) ||
         (nRanks == 8 && nRanksPerNode == 4) ||
         (nRanks == 2 && nRanksPerNode == 1);
}

bool isTwoRankLayout(int nRanks, int nRanksPerNode) {
  return nRanks == 2 && nRanksPerNode == 1;
}

bool isIntraFourRankLayout(int nRanks, int nRanksPerNode) {
  return nRanks == 4 && nRanksPerNode == 4;
}

bool isTwoNodeTwoGpuLayout(int nRanks, int nRanksPerNode) {
  return nRanks == 4 && nRanksPerNode == 2;
}

void waitForCudaStream(cudaStream_t stream) {
  int spins = 0;
  while (true) {
    cudaError_t result = cudaStreamQuery(stream);
    if (result == cudaSuccess) return;
    if (result != cudaErrorNotReady) MSCCLPP_CUDATHROW(result);
    if (spins++ < 65536) {
      asm volatile("pause" ::: "memory");
    } else {
      std::this_thread::yield();
    }
  }
}

void waitForCudaEvent(cudaEvent_t event) {
  int spins = 0;
  while (true) {
    cudaError_t result = cudaEventQuery(event);
    if (result == cudaSuccess) return;
    if (result != cudaErrorNotReady) MSCCLPP_CUDATHROW(result);
    if (spins++ < 65536) {
      asm volatile("pause" ::: "memory");
    } else {
      std::this_thread::yield();
    }
  }
}

size_t configuredChunkBytes() {
  static size_t value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_CHUNK_BYTES");
    if (env == nullptr || env[0] == '\0') return kDefaultChunkBytes;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 0);
    if (end == env || parsed == 0) return kDefaultChunkBytes;
    return static_cast<size_t>(parsed);
  }();
  return value;
}

size_t configuredTwoRankChunkBytes() {
  static size_t value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_CHUNK_BYTES");
    if (env == nullptr || env[0] == '\0') return kDefaultTwoRankChunkBytes;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 0);
    if (end == env || parsed == 0) return kDefaultTwoRankChunkBytes;
    return static_cast<size_t>(parsed);
  }();
  return value;
}

size_t configuredLayoutChunkBytesOverride() {
  static size_t value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_LAYOUT_CHUNK_BYTES");
    if (env == nullptr || env[0] == '\0') return static_cast<size_t>(0);
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 0);
    if (end == env) return static_cast<size_t>(0);
    size_t bytes = static_cast<size_t>(parsed);
    bytes = bytes / sizeof(float) * sizeof(float);
    return bytes;
  }();
  return value;
}

size_t configuredTwoRankSmallHostBytes() {
  static size_t value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_TWO_RANK_SMALL_HOST_BYTES");
    if (env == nullptr || env[0] == '\0') return kDefaultTwoRankSmallHostBytes;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 0);
    if (end == env) return kDefaultTwoRankSmallHostBytes;
    return static_cast<size_t>(parsed);
  }();
  return value;
}

int configuredIpcEventSyncMode() {
  static int value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_IPC_EVENT_SYNC");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return value;
}

bool useIpcEventSyncFor(size_t bytesPerRank) {
  int mode = configuredIpcEventSyncMode();
  if (mode >= 0) return mode != 0;
  (void)bytesPerRank;
  return true;
}

bool useTwoNodeTwoGpuIpcEventSyncForPath(bool pipelined, size_t fullBytes) {
  int mode = configuredIpcEventSyncMode();
  if (mode >= 0) return mode != 0;
  return pipelined && fullBytes >= 8 * 1024 * 1024;
}

int configuredAsyncFinalAddMode() {
  static int mode = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_ASYNC_FINAL_ADD");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return mode;
}

bool useAsyncFinalAddFor(size_t bytesPerRank) {
  int mode = configuredAsyncFinalAddMode();
  if (mode >= 0) return mode != 0;
  return bytesPerRank >= 32 * 1024 * 1024;
}

bool usePipelinedAsyncFinalAddFor(RsContext const& ctx,
                                  size_t messageBytes) {
  int mode = configuredAsyncFinalAddMode();
  if (mode >= 0) return mode != 0;
  (void)messageBytes;
  if (ctx.worldSize == 4 && ctx.nRanksPerNode == 2) return false;
  return true;
}

size_t configuredLocalLeadChunks() {
  static size_t value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_LOCAL_LEAD_CHUNKS");
    if (env == nullptr || env[0] == '\0') return kLocalLeadChunks;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 0);
    if (end == env) return kLocalLeadChunks;
    return static_cast<size_t>(parsed);
  }();
  return value;
}

size_t configuredShortLocalLeadChunks() {
  static size_t value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_SHORT_LOCAL_LEAD_CHUNKS");
    if (env == nullptr || env[0] == '\0') return static_cast<size_t>(1);
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 0);
    if (end == env) return static_cast<size_t>(1);
    return static_cast<size_t>(parsed);
  }();
  return value;
}

size_t configuredLongLocalLeadChunks() {
  static size_t value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_LONG_LOCAL_LEAD_CHUNKS");
    if (env == nullptr || env[0] == '\0') return kLongLocalLeadChunks;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 0);
    if (end == env) return kLongLocalLeadChunks;
    return static_cast<size_t>(parsed);
  }();
  return value;
}

bool useLocalFourGpuFlags() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_LOCAL_GPU_FLAGS");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useLocalFourParallelCopies() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_LOCAL_PARALLEL_COPY");
    if (env == nullptr || env[0] == '\0') return true;
    return env[0] != '0';
  }();
  return value;
}

int configuredLocalFourRingMode() {
  static int value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_LOCAL_RING");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return value;
}

bool useLocalFourRingEvents() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_LOCAL_RING_EVENTS");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useLocalFourRingFor(size_t bytesPerRank) {
  if (bytesPerRank < 1024 * 1024) return false;
  int mode = configuredLocalFourRingMode();
  if (mode >= 0) return mode != 0;
  return bytesPerRank >= 2 * 1024 * 1024 &&
         bytesPerRank <= 32ULL * 1024 * 1024;
}

bool useP2pRing() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_P2P_RING");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useP2pRingForLocalFour(size_t bytesPerRank) {
  return useP2pRing() ||
         (bytesPerRank >= 256 * 1024 && bytesPerRank <= 2 * 1024 * 1024);
}

bool useTwoNodeTwoGpuHier() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_2N2G_HIER");
    if (env == nullptr || env[0] == '\0') return true;
    return env[0] != '0';
  }();
  return value;
}

size_t configuredTwoNodeTwoGpuLeadChunks(size_t totalChunks) {
  char const* env = totalChunks <= 2
                        ? std::getenv("MSCCLPP_NCCL_RS_SHORT_LOCAL_LEAD_CHUNKS")
                        : (totalChunks >= 8
                               ? std::getenv("MSCCLPP_NCCL_RS_LONG_LOCAL_LEAD_CHUNKS")
                               : std::getenv("MSCCLPP_NCCL_RS_LOCAL_LEAD_CHUNKS"));
  if (env == nullptr || env[0] == '\0') return 1;
  return totalChunks <= 2
             ? configuredShortLocalLeadChunks()
             : (totalChunks >= 8 ? configuredLongLocalLeadChunks()
                                 : configuredLocalLeadChunks());
}

int configuredDirectPartnerCopyMode() {
  static int value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_DIRECT_PARTNER_COPY");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return value;
}

bool useDirectPartnerCopyFor(size_t messageBytes) {
  int mode = configuredDirectPartnerCopyMode();
  if (mode >= 0) return mode != 0;
  return messageBytes == 4 * 1024 * 1024 ||
         messageBytes >= 32 * 1024 * 1024;
}

int configuredDirectPartnerCopy2DMode() {
  static int value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_DIRECT_PARTNER_COPY_2D");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return value;
}

bool useDirectPartnerCopy2DFor(size_t messageBytes) {
  int mode = configuredDirectPartnerCopy2DMode();
  if (mode >= 0) return mode != 0;
  return useDirectPartnerCopyFor(messageBytes);
}

bool useDirectCrossWrite() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_DIRECT_CROSS_WRITE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

int configuredHostReadFinalAddMode() {
  static int value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_HOST_READ_FINAL_ADD");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return value;
}

bool useDirectTargetReduce() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_DIRECT_TARGET_REDUCE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useDirectTargetScatterKernel() {
  static bool value = [] {
    char const* env =
        std::getenv("MSCCLPP_NCCL_RS_DIRECT_TARGET_SCATTER_KERNEL");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useSingleChunkAsyncCopy() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_SINGLE_ASYNC_COPY");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useSingleChunkSlotRing() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_SINGLE_SLOT_RING");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useDeviceFlagSync() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_DEVICE_FLAG_SYNC");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useIpcDeviceFlagSync() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_IPC_DEVICE_FLAGS");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool usePairConnectionWrite() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_PAIR_CONNECTION_WRITE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useTwoRankMappedHost() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_TWO_RANK_MAPPED_HOST");
    if (env == nullptr || env[0] == '\0') return true;
    return env[0] != '0';
  }();
  return value;
}

bool useCpuFinalAdd() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_CPU_FINAL_ADD");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

int configuredMappedSendFinalReduceMode() {
  static int value = [] {
    char const* env =
        std::getenv("MSCCLPP_NCCL_RS_MAPPED_SEND_FINAL_REDUCE");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return value;
}

bool useMappedSendFinalReduceFor(size_t bytesPerRank) {
  int mode = configuredMappedSendFinalReduceMode();
  if (mode >= 0) return mode != 0;
  return bytesPerRank <= 1024 * 1024;
}

bool useMappedHostSingleChunk() {
  static bool value = [] {
    char const* env =
        std::getenv("MSCCLPP_NCCL_RS_MAPPED_HOST_SINGLE_CHUNK");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

int configuredSplitFinalReduceMode() {
  static int value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_SPLIT_FINAL_REDUCE");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return value;
}

bool useSplitFinalReduceFor(size_t bytesPerRank, size_t messageBytes,
                            bool recordAsyncD2h) {
  int mode = configuredSplitFinalReduceMode();
  if (mode >= 0) return mode != 0;
  if (messageBytes >= 16 * 1024 * 1024) return true;
  return recordAsyncD2h && bytesPerRank <= 1024 * 1024;
}

bool useParallelSplitFinalReduce() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_PARALLEL_SPLIT_FINAL");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useEagerRdmaPost() {
  static bool value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_EAGER_RDMA_POST");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

bool useParallelSplitFinalReduceFor(RsContext const& ctx,
                                    size_t messageBytes) {
  return useParallelSplitFinalReduce() ||
         (ctx.worldSize == 8 && ctx.nRanksPerNode == 4 &&
          messageBytes >= 8 * 1024 * 1024);
}

bool useEagerRdmaPostFor(RsContext const& ctx, size_t messageBytes) {
  return useEagerRdmaPost() ||
         (ctx.worldSize == 4 && ctx.nRanksPerNode == 2 &&
          messageBytes >= 4 * 1024 * 1024) ||
         (ctx.worldSize == 8 && ctx.nRanksPerNode == 4 &&
          messageBytes >= 8 * 1024 * 1024);
}

int configuredRoundRobinHcaMode() {
  static int value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_ROUND_ROBIN_HCA");
    if (env == nullptr || env[0] == '\0') return -1;
    return env[0] == '0' ? 0 : 1;
  }();
  return value;
}

mscclpp::Transport selectReduceScatterTransport(int cudaDeviceId,
                                                int localRank,
                                                int transportPolicy) {
  if (transportPolicy == kRsTransportRoundRobinHca) {
    auto available = getAvailableIBTransports();
    if (!available.empty()) {
      return available[static_cast<size_t>(localRank) % available.size()];
    }
  }
  return selectIBTransportForGpu(cudaDeviceId);
}

int transportPolicyForLayout(ncclComm_t comm,
                             std::shared_ptr<Communicator> bootstrapComm,
                             int rank, int nRanks, int nRanksPerNode,
                             size_t bytesPerRank) {
  int forced = configuredRoundRobinHcaMode();
  int policyClass = forced >= 0 ? 2 : 0;
  int proposed = forced > 0 ? kRsTransportRoundRobinHca : kRsTransportLocality;
  size_t messageBytes = bytesPerRank * static_cast<size_t>(nRanks);
  if (forced < 0 && isTwoNodeTwoGpuLayout(nRanks, nRanksPerNode) &&
      messageBytes <= 2 * 1024 * 1024) {
    policyClass = 1;
    proposed = getAvailableIBTransports().size() > 1
                   ? kRsTransportRoundRobinHca
                   : kRsTransportLocality;
  }
  if (policyClass == 0) return kRsTransportLocality;

  RsPolicyCacheKey key{comm, policyClass};
  {
    std::lock_guard<std::mutex> lock(gReduceScatterMutex);
    auto it = gReduceScatterTransportPolicies.find(key);
    if (it != gReduceScatterTransportPolicies.end()) return it->second;
  }

  std::vector<int> policies(static_cast<size_t>(nRanks),
                            kRsTransportLocality);
  policies[static_cast<size_t>(rank)] = proposed;
  bootstrapComm->bootstrap()->allGather(policies.data(), sizeof(int));

  int agreed = policies[0];
  for (int policy : policies) {
    if (policy != agreed) {
      agreed = kRsTransportLocality;
      break;
    }
  }
  if (policyClass == 1 && agreed != kRsTransportRoundRobinHca) {
    agreed = kRsTransportLocality;
  }
  {
    std::lock_guard<std::mutex> lock(gReduceScatterMutex);
    auto [it, inserted] = gReduceScatterTransportPolicies.emplace(key, agreed);
    (void)inserted;
    return it->second;
  }
}

size_t configuredSmallHostFullBytes() {
  static size_t value = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_SMALL_HOST_FULL_BYTES");
    if (env == nullptr || env[0] == '\0') return kSmallHostFullBytes;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(env, &end, 0);
    if (end == env || parsed == 0) return kSmallHostFullBytes;
    return static_cast<size_t>(parsed);
  }();
  return value;
}

int pipelineSlotsForLayout(bool twoRankLayout) {
  return twoRankLayout ? kTwoRankPipelineSlots : kPipelineSlots;
}

int pipelineSlotsForLayout(int nRanks, int nRanksPerNode) {
  if (isIntraFourRankLayout(nRanks, nRanksPerNode)) {
    return kLocalFourPipelineSlots;
  }
  return pipelineSlotsForLayout(isTwoRankLayout(nRanks, nRanksPerNode));
}

size_t chunkCapacityBytes(size_t scratchBufferSize, bool twoRankLayout,
                          int slots) {
  size_t denominator =
      twoRankLayout ? static_cast<size_t>(slots)
                    : static_cast<size_t>(kScratchRowsPerChunk) *
                          static_cast<size_t>(slots);
  size_t rowBytes = scratchBufferSize / denominator;
  rowBytes = std::min(rowBytes, twoRankLayout ? configuredTwoRankChunkBytes()
                                              : configuredChunkBytes());
  return rowBytes / sizeof(float) * sizeof(float);
}

size_t effectiveChunkBytes(size_t bytesPerRank, size_t chunkCapacity) {
  if (useMappedHostSingleChunk() && useMappedSendFinalReduceFor(bytesPerRank) &&
      configuredHostReadFinalAddMode() != 0 &&
      bytesPerRank <= 2 * 1024 * 1024) {
    return std::min(chunkCapacity, bytesPerRank);
  }
  if (bytesPerRank <= 512 * 1024) {
    return std::min(chunkCapacity, static_cast<size_t>(512 * 1024));
  }
  if (bytesPerRank <= 1024 * 1024) {
    return std::min(chunkCapacity, static_cast<size_t>(512 * 1024));
  }
  if (bytesPerRank <= 2 * 1024 * 1024) {
    return std::min(chunkCapacity, static_cast<size_t>(512 * 1024));
  }
  return std::min(chunkCapacity, static_cast<size_t>(1024 * 1024));
}

size_t effectiveChunkBytesForLayout(int nRanks, int nRanksPerNode,
                                    size_t bytesPerRank,
                                    size_t chunkCapacity) {
  size_t overrideBytes = configuredLayoutChunkBytesOverride();
  if (overrideBytes != 0) {
    return std::min(chunkCapacity, overrideBytes);
  }
  if (nRanks == 8 && nRanksPerNode == 4 &&
      bytesPerRank >= 2 * 1024 * 1024) {
    return std::min(chunkCapacity, static_cast<size_t>(1024 * 1024));
  }
  return effectiveChunkBytes(bytesPerRank, chunkCapacity);
}

__device__ __forceinline__ int4 addFloat4Vec(int4 a, int4 b) {
  int4 out;
  out.x = __float_as_int(__int_as_float(a.x) + __int_as_float(b.x));
  out.y = __float_as_int(__int_as_float(a.y) + __int_as_float(b.y));
  out.z = __float_as_int(__int_as_float(a.z) + __int_as_float(b.z));
  out.w = __float_as_int(__int_as_float(a.w) + __int_as_float(b.w));
  return out;
}

__device__ __forceinline__ unsigned long long volatile* rsFlagPtr(
    char* ctrl, size_t baseOffset, int index);
__device__ __forceinline__ void waitForRsFlag(char* ctrl, size_t baseOffset,
                                              int index,
                                              unsigned long long epoch);

__global__ void packPartnerRowsFloat4Kernel(
    float const* send, char* partnerSendRows, size_t chunkVecs,
    size_t fullVecs, size_t elemOffsetVec, size_t rowBytes, int localRank,
    int localBase, int remoteBase) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= chunkVecs) return;

  int partnerLocal = localRank ^ 1;
  int crossPairBase = localRank < 2 ? 2 : 0;
  int partnerCrossAssigned = crossPairBase + (partnerLocal & 1);
  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto* partner4 = reinterpret_cast<int4*>(partnerSendRows);
  size_t idx = elemOffsetVec + vecIdx;
  partner4[vecIdx] =
      send4[(static_cast<size_t>(localBase + partnerLocal) * fullVecs) + idx];
  partner4[rowVecs + vecIdx] =
      send4[(static_cast<size_t>(remoteBase + partnerLocal) * fullVecs) + idx];
  partner4[2 * rowVecs + vecIdx] =
      send4[(static_cast<size_t>(localBase + partnerCrossAssigned) * fullVecs) + idx];
  partner4[3 * rowVecs + vecIdx] =
      send4[(static_cast<size_t>(remoteBase + partnerCrossAssigned) * fullVecs) + idx];
}

__global__ void packPartnerRowsFloatKernel(
    float const* send, char* partnerSendRows, size_t chunkElems,
    size_t fullElems, size_t elemOffset, size_t rowBytes, int localRank,
    int localBase, int remoteBase) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= chunkElems) return;

  int partnerLocal = localRank ^ 1;
  int crossPairBase = localRank < 2 ? 2 : 0;
  int partnerCrossAssigned = crossPairBase + (partnerLocal & 1);
  size_t rowStride = rowBytes / sizeof(float);
  auto* partner = reinterpret_cast<float*>(partnerSendRows);
  size_t elem = elemOffset + idx;
  partner[idx] =
      send[(static_cast<size_t>(localBase + partnerLocal) * fullElems) + elem];
  partner[rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + partnerLocal) * fullElems) + elem];
  partner[2 * rowStride + idx] =
      send[(static_cast<size_t>(localBase + partnerCrossAssigned) * fullElems) + elem];
  partner[3 * rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + partnerCrossAssigned) * fullElems) + elem];
}

__global__ void reduceNumaPairFromInputFloat4Kernel(
    float const* send, char const* partnerRecvRows, char* ownPartialRows,
    char* crossSendRows, size_t nVec, size_t fullVecs, size_t elemOffsetVec,
    size_t rowBytes, int localRank, int localBase, int remoteBase) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  int crossAssigned = localRank < 2 ? 2 + (localRank & 1)
                                   : (localRank & 1);
  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto const* partner4 = reinterpret_cast<int4 const*>(partnerRecvRows);
  auto* own4 = reinterpret_cast<int4*>(ownPartialRows);
  auto* cross4 = reinterpret_cast<int4*>(crossSendRows);
  size_t idx = elemOffsetVec + vecIdx;
  int4 self0 =
      send4[(static_cast<size_t>(localBase + localRank) * fullVecs) + idx];
  int4 self1 =
      send4[(static_cast<size_t>(remoteBase + localRank) * fullVecs) + idx];
  int4 self2 =
      send4[(static_cast<size_t>(localBase + crossAssigned) * fullVecs) + idx];
  int4 self3 =
      send4[(static_cast<size_t>(remoteBase + crossAssigned) * fullVecs) + idx];
  own4[vecIdx] = addFloat4Vec(self0, partner4[vecIdx]);
  own4[rowVecs + vecIdx] =
      addFloat4Vec(self1, partner4[rowVecs + vecIdx]);
  cross4[vecIdx] =
      addFloat4Vec(self2, partner4[2 * rowVecs + vecIdx]);
  cross4[rowVecs + vecIdx] =
      addFloat4Vec(self3, partner4[3 * rowVecs + vecIdx]);
}

__global__ void reduceNumaPairFromInputFloatKernel(
    float const* send, char const* partnerRecvRows, char* ownPartialRows,
    char* crossSendRows, size_t count, size_t fullElems, size_t elemOffset,
    size_t rowBytes, int localRank, int localBase, int remoteBase) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  int crossAssigned = localRank < 2 ? 2 + (localRank & 1)
                                   : (localRank & 1);
  size_t rowStride = rowBytes / sizeof(float);
  auto const* partner = reinterpret_cast<float const*>(partnerRecvRows);
  auto* own = reinterpret_cast<float*>(ownPartialRows);
  auto* cross = reinterpret_cast<float*>(crossSendRows);
  size_t elem = elemOffset + idx;
  own[idx] =
      send[(static_cast<size_t>(localBase + localRank) * fullElems) + elem] +
      partner[idx];
  own[rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + localRank) * fullElems) + elem] +
      partner[rowStride + idx];
  cross[idx] =
      send[(static_cast<size_t>(localBase + crossAssigned) * fullElems) + elem] +
      partner[2 * rowStride + idx];
  cross[rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + crossAssigned) * fullElems) + elem] +
      partner[3 * rowStride + idx];
}

__global__ void reduceNumaPairFromInputWaitFloat4Kernel(
    char* ctrl, size_t readyOffset, int readyLocalRank,
    unsigned long long epoch, float const* send, char const* partnerRecvRows,
    char* ownPartialRows, char* crossSendRows, size_t nVec, size_t fullVecs,
    size_t elemOffsetVec, size_t rowBytes, int localRank, int localBase,
    int remoteBase) {
  waitForRsFlag(ctrl, readyOffset, readyLocalRank, epoch);
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  int crossAssigned = localRank < 2 ? 2 + (localRank & 1)
                                   : (localRank & 1);
  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto const* partner4 = reinterpret_cast<int4 const*>(partnerRecvRows);
  auto* own4 = reinterpret_cast<int4*>(ownPartialRows);
  auto* cross4 = reinterpret_cast<int4*>(crossSendRows);
  size_t idx = elemOffsetVec + vecIdx;
  int4 self0 =
      send4[(static_cast<size_t>(localBase + localRank) * fullVecs) + idx];
  int4 self1 =
      send4[(static_cast<size_t>(remoteBase + localRank) * fullVecs) + idx];
  int4 self2 =
      send4[(static_cast<size_t>(localBase + crossAssigned) * fullVecs) + idx];
  int4 self3 =
      send4[(static_cast<size_t>(remoteBase + crossAssigned) * fullVecs) + idx];
  own4[vecIdx] = addFloat4Vec(self0, partner4[vecIdx]);
  own4[rowVecs + vecIdx] =
      addFloat4Vec(self1, partner4[rowVecs + vecIdx]);
  cross4[vecIdx] =
      addFloat4Vec(self2, partner4[2 * rowVecs + vecIdx]);
  cross4[rowVecs + vecIdx] =
      addFloat4Vec(self3, partner4[3 * rowVecs + vecIdx]);
}

__global__ void reduceNumaPairFromInputWaitFloatKernel(
    char* ctrl, size_t readyOffset, int readyLocalRank,
    unsigned long long epoch, float const* send, char const* partnerRecvRows,
    char* ownPartialRows, char* crossSendRows, size_t count, size_t fullElems,
    size_t elemOffset, size_t rowBytes, int localRank, int localBase,
    int remoteBase) {
  waitForRsFlag(ctrl, readyOffset, readyLocalRank, epoch);
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  int crossAssigned = localRank < 2 ? 2 + (localRank & 1)
                                   : (localRank & 1);
  size_t rowStride = rowBytes / sizeof(float);
  auto const* partner = reinterpret_cast<float const*>(partnerRecvRows);
  auto* own = reinterpret_cast<float*>(ownPartialRows);
  auto* cross = reinterpret_cast<float*>(crossSendRows);
  size_t elem = elemOffset + idx;
  own[idx] =
      send[(static_cast<size_t>(localBase + localRank) * fullElems) + elem] +
      partner[idx];
  own[rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + localRank) * fullElems) + elem] +
      partner[rowStride + idx];
  cross[idx] =
      send[(static_cast<size_t>(localBase + crossAssigned) * fullElems) + elem] +
      partner[2 * rowStride + idx];
  cross[rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + crossAssigned) * fullElems) + elem] +
      partner[3 * rowStride + idx];
}

__global__ void finalNumaPairReduceFloat4Kernel(
    char const* ownPartialRows, char const* crossRecvRows, float* localOutput,
    float* remoteOutput, size_t nVec, size_t rowBytes) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* own4 = reinterpret_cast<int4 const*>(ownPartialRows);
  auto const* cross4 = reinterpret_cast<int4 const*>(crossRecvRows);
  reinterpret_cast<int4*>(localOutput)[vecIdx] =
      addFloat4Vec(cross4[vecIdx], own4[vecIdx]);
  reinterpret_cast<int4*>(remoteOutput)[vecIdx] =
      addFloat4Vec(cross4[rowVecs + vecIdx], own4[rowVecs + vecIdx]);
}

__global__ void finalNumaPairReduceFloatKernel(
    char const* ownPartialRows, char const* crossRecvRows, float* localOutput,
    float* remoteOutput, size_t count, size_t rowBytes) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  size_t rowStride = rowBytes / sizeof(float);
  auto const* own = reinterpret_cast<float const*>(ownPartialRows);
  auto const* cross = reinterpret_cast<float const*>(crossRecvRows);
  localOutput[idx] = own[idx] + cross[idx];
  remoteOutput[idx] = own[rowStride + idx] + cross[rowStride + idx];
}

__global__ void finalNumaPairReduceLocalFloat4Kernel(
    char const* ownPartialRows, char const* crossRecvRows, float* localOutput,
    size_t nVec, size_t rowBytes) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  auto const* own4 = reinterpret_cast<int4 const*>(ownPartialRows);
  auto const* cross4 = reinterpret_cast<int4 const*>(crossRecvRows);
  (void)rowBytes;
  reinterpret_cast<int4*>(localOutput)[vecIdx] =
      addFloat4Vec(cross4[vecIdx], own4[vecIdx]);
}

__global__ void finalNumaPairReduceLocalFloatKernel(
    char const* ownPartialRows, char const* crossRecvRows, float* localOutput,
    size_t count, size_t rowBytes) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  auto const* own = reinterpret_cast<float const*>(ownPartialRows);
  auto const* cross = reinterpret_cast<float const*>(crossRecvRows);
  (void)rowBytes;
  localOutput[idx] = own[idx] + cross[idx];
}

__global__ void finalNumaPairReduceRemoteFloat4Kernel(
    char const* ownPartialRows, char const* crossRecvRows, float* remoteOutput,
    size_t nVec, size_t rowBytes) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* own4 = reinterpret_cast<int4 const*>(ownPartialRows);
  auto const* cross4 = reinterpret_cast<int4 const*>(crossRecvRows);
  reinterpret_cast<int4*>(remoteOutput)[vecIdx] =
      addFloat4Vec(cross4[rowVecs + vecIdx], own4[rowVecs + vecIdx]);
}

__global__ void finalNumaPairReduceRemoteFloatKernel(
    char const* ownPartialRows, char const* crossRecvRows, float* remoteOutput,
    size_t count, size_t rowBytes) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  size_t rowStride = rowBytes / sizeof(float);
  auto const* own = reinterpret_cast<float const*>(ownPartialRows);
  auto const* cross = reinterpret_cast<float const*>(crossRecvRows);
  remoteOutput[idx] = own[rowStride + idx] + cross[rowStride + idx];
}

__global__ void finalNumaPairReduceWaitFloat4Kernel(
    char* ctrl, size_t readyOffset, int readyLocalRank,
    unsigned long long epoch, char const* ownPartialRows,
    char const* crossRecvRows, float* localOutput, float* remoteOutput,
    size_t nVec, size_t rowBytes) {
  waitForRsFlag(ctrl, readyOffset, readyLocalRank, epoch);
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* own4 = reinterpret_cast<int4 const*>(ownPartialRows);
  auto const* cross4 = reinterpret_cast<int4 const*>(crossRecvRows);
  reinterpret_cast<int4*>(localOutput)[vecIdx] =
      addFloat4Vec(cross4[vecIdx], own4[vecIdx]);
  reinterpret_cast<int4*>(remoteOutput)[vecIdx] =
      addFloat4Vec(cross4[rowVecs + vecIdx], own4[rowVecs + vecIdx]);
}

__global__ void finalNumaPairReduceWaitFloatKernel(
    char* ctrl, size_t readyOffset, int readyLocalRank,
    unsigned long long epoch, char const* ownPartialRows,
    char const* crossRecvRows, float* localOutput, float* remoteOutput,
    size_t count, size_t rowBytes) {
  waitForRsFlag(ctrl, readyOffset, readyLocalRank, epoch);
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  size_t rowStride = rowBytes / sizeof(float);
  auto const* own = reinterpret_cast<float const*>(ownPartialRows);
  auto const* cross = reinterpret_cast<float const*>(crossRecvRows);
  localOutput[idx] = own[idx] + cross[idx];
  remoteOutput[idx] = own[rowStride + idx] + cross[rowStride + idx];
}

__global__ void directTargetReduceFloat4Kernel(char const* rows,
                                               float* localOutput,
                                               float* remoteOutput,
                                               size_t nVec,
                                               size_t rowBytes) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* rows4 = reinterpret_cast<int4 const*>(rows);
  int4 local01 = addFloat4Vec(rows4[vecIdx], rows4[rowVecs + vecIdx]);
  int4 local23 =
      addFloat4Vec(rows4[2 * rowVecs + vecIdx],
                   rows4[3 * rowVecs + vecIdx]);
  int4 remote01 =
      addFloat4Vec(rows4[4 * rowVecs + vecIdx],
                   rows4[5 * rowVecs + vecIdx]);
  int4 remote23 =
      addFloat4Vec(rows4[6 * rowVecs + vecIdx],
                   rows4[7 * rowVecs + vecIdx]);
  reinterpret_cast<int4*>(localOutput)[vecIdx] = addFloat4Vec(local01, local23);
  reinterpret_cast<int4*>(remoteOutput)[vecIdx] =
      addFloat4Vec(remote01, remote23);
}

__global__ void directTargetReduceFloatKernel(char const* rows,
                                              float* localOutput,
                                              float* remoteOutput,
                                              size_t count,
                                              size_t rowBytes) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  size_t rowStride = rowBytes / sizeof(float);
  auto const* f = reinterpret_cast<float const*>(rows);
  localOutput[idx] = f[idx] + f[rowStride + idx] +
                     f[2 * rowStride + idx] + f[3 * rowStride + idx];
  remoteOutput[idx] = f[4 * rowStride + idx] +
                      f[5 * rowStride + idx] +
                      f[6 * rowStride + idx] +
                      f[7 * rowStride + idx];
}

__global__ void directTargetScatterFloat4Kernel(
    float const* send, char* dst0, char* dst1, char* dst2, char* dst3,
    size_t nVec, size_t fullVecs, size_t elemOffsetVec, size_t rowBytes,
    int localRank, int localBase, int remoteBase) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  char* dsts[4] = {dst0, dst1, dst2, dst3};
  size_t idx = elemOffsetVec + vecIdx;
  for (int target = 0; target < 4; ++target) {
    auto* out = reinterpret_cast<int4*>(dsts[target]);
    out[static_cast<size_t>(localRank) * rowVecs + vecIdx] =
        send4[(static_cast<size_t>(localBase + target) * fullVecs) + idx];
    out[static_cast<size_t>(4 + localRank) * rowVecs + vecIdx] =
        send4[(static_cast<size_t>(remoteBase + target) * fullVecs) + idx];
  }
}

__global__ void directTargetScatterFloatKernel(
    float const* send, char* dst0, char* dst1, char* dst2, char* dst3,
    size_t count, size_t fullElems, size_t elemOffset, size_t rowBytes,
    int localRank, int localBase, int remoteBase) {
  size_t elemIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (elemIdx >= count) return;

  size_t rowStride = rowBytes / sizeof(float);
  char* dsts[4] = {dst0, dst1, dst2, dst3};
  size_t idx = elemOffset + elemIdx;
  for (int target = 0; target < 4; ++target) {
    auto* out = reinterpret_cast<float*>(dsts[target]);
    out[static_cast<size_t>(localRank) * rowStride + elemIdx] =
        send[(static_cast<size_t>(localBase + target) * fullElems) + idx];
    out[static_cast<size_t>(4 + localRank) * rowStride + elemIdx] =
        send[(static_cast<size_t>(remoteBase + target) * fullElems) + idx];
  }
}

__global__ void addFloat4Kernel(float const* lhs, float const* rhs, float* out,
                                size_t nVec) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;
  auto const* lhs4 = reinterpret_cast<int4 const*>(lhs);
  auto const* rhs4 = reinterpret_cast<int4 const*>(rhs);
  reinterpret_cast<int4*>(out)[vecIdx] = addFloat4Vec(lhs4[vecIdx], rhs4[vecIdx]);
}

__global__ void addFloatKernel(float const* lhs, float const* rhs, float* out,
                               size_t count) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  out[idx] = lhs[idx] + rhs[idx];
}

__device__ __forceinline__ void waitForIpcDeviceFlag(
    char* flags, int phase, int readyLocalRank, unsigned long long epoch);

__global__ void twoNodeTwoGpuLocalReduceFloat4Kernel(
    float const* send, char const* partnerRecvRows, char* localPartial,
    char* remotePartial, size_t nVec, size_t fullVecs, size_t elemOffsetVec,
    size_t rowBytes, bool partnerRowsReversed, int localRank, int localBase,
    int remoteBase) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;
  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto const* partner4 = reinterpret_cast<int4 const*>(partnerRecvRows);
  auto* local4 = reinterpret_cast<int4*>(localPartial);
  auto* remote4 = reinterpret_cast<int4*>(remotePartial);
  size_t idx = elemOffsetVec + vecIdx;
  size_t partnerLocalOffset =
      (partnerRowsReversed ? rowVecs : 0) + vecIdx;
  size_t partnerRemoteOffset =
      (partnerRowsReversed ? 0 : rowVecs) + vecIdx;
  local4[vecIdx] = addFloat4Vec(
      send4[(static_cast<size_t>(localBase + localRank) * fullVecs) + idx],
      partner4[partnerLocalOffset]);
  remote4[vecIdx] = addFloat4Vec(
      send4[(static_cast<size_t>(remoteBase + localRank) * fullVecs) + idx],
      partner4[partnerRemoteOffset]);
}

__global__ void twoNodeTwoGpuLocalReduceFloatKernel(
    float const* send, char const* partnerRecvRows, char* localPartial,
    char* remotePartial, size_t count, size_t fullElems, size_t elemOffset,
    size_t rowBytes, bool partnerRowsReversed, int localRank, int localBase,
    int remoteBase) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  size_t rowStride = rowBytes / sizeof(float);
  auto const* partner = reinterpret_cast<float const*>(partnerRecvRows);
  auto* local = reinterpret_cast<float*>(localPartial);
  auto* remote = reinterpret_cast<float*>(remotePartial);
  size_t elem = elemOffset + idx;
  size_t partnerLocalOffset =
      (partnerRowsReversed ? rowStride : 0) + idx;
  size_t partnerRemoteOffset =
      (partnerRowsReversed ? 0 : rowStride) + idx;
  local[idx] =
      send[(static_cast<size_t>(localBase + localRank) * fullElems) + elem] +
      partner[partnerLocalOffset];
  remote[idx] =
      send[(static_cast<size_t>(remoteBase + localRank) * fullElems) + elem] +
      partner[partnerRemoteOffset];
}

__global__ void twoNodeTwoGpuLocalReduceWaitFloat4Kernel(
    char* flags, int readyLocalRank, unsigned long long epoch, float const* send,
    char const* partnerRecvRows, char* localPartial, char* remotePartial,
    size_t nVec, size_t fullVecs, size_t elemOffsetVec, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int localBase, int remoteBase) {
  waitForIpcDeviceFlag(flags, /*phase=*/0, readyLocalRank, epoch);
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;
  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto const* partner4 = reinterpret_cast<int4 const*>(partnerRecvRows);
  auto* local4 = reinterpret_cast<int4*>(localPartial);
  auto* remote4 = reinterpret_cast<int4*>(remotePartial);
  size_t idx = elemOffsetVec + vecIdx;
  size_t partnerLocalOffset =
      (partnerRowsReversed ? rowVecs : 0) + vecIdx;
  size_t partnerRemoteOffset =
      (partnerRowsReversed ? 0 : rowVecs) + vecIdx;
  local4[vecIdx] = addFloat4Vec(
      send4[(static_cast<size_t>(localBase + localRank) * fullVecs) + idx],
      partner4[partnerLocalOffset]);
  remote4[vecIdx] = addFloat4Vec(
      send4[(static_cast<size_t>(remoteBase + localRank) * fullVecs) + idx],
      partner4[partnerRemoteOffset]);
}

__global__ void twoNodeTwoGpuLocalReduceWaitFloatKernel(
    char* flags, int readyLocalRank, unsigned long long epoch, float const* send,
    char const* partnerRecvRows, char* localPartial, char* remotePartial,
    size_t count, size_t fullElems, size_t elemOffset, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int localBase, int remoteBase) {
  waitForIpcDeviceFlag(flags, /*phase=*/0, readyLocalRank, epoch);
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  size_t rowStride = rowBytes / sizeof(float);
  auto const* partner = reinterpret_cast<float const*>(partnerRecvRows);
  auto* local = reinterpret_cast<float*>(localPartial);
  auto* remote = reinterpret_cast<float*>(remotePartial);
  size_t elem = elemOffset + idx;
  size_t partnerLocalOffset =
      (partnerRowsReversed ? rowStride : 0) + idx;
  size_t partnerRemoteOffset =
      (partnerRowsReversed ? 0 : rowStride) + idx;
  local[idx] =
      send[(static_cast<size_t>(localBase + localRank) * fullElems) + elem] +
      partner[partnerLocalOffset];
  remote[idx] =
      send[(static_cast<size_t>(remoteBase + localRank) * fullElems) + elem] +
      partner[partnerRemoteOffset];
}

__global__ void twoNodeTwoGpuRemoteReduceFloat4Kernel(
    float const* send, char const* partnerRecvRows, char* remotePartial,
    size_t nVec, size_t fullVecs, size_t elemOffsetVec, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int remoteBase) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;
  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto const* partner4 = reinterpret_cast<int4 const*>(partnerRecvRows);
  auto* remote4 = reinterpret_cast<int4*>(remotePartial);
  size_t idx = elemOffsetVec + vecIdx;
  size_t partnerRemoteOffset =
      (partnerRowsReversed ? 0 : rowVecs) + vecIdx;
  remote4[vecIdx] = addFloat4Vec(
      send4[(static_cast<size_t>(remoteBase + localRank) * fullVecs) + idx],
      partner4[partnerRemoteOffset]);
}

__global__ void twoNodeTwoGpuRemoteReduceFloatKernel(
    float const* send, char const* partnerRecvRows, char* remotePartial,
    size_t count, size_t fullElems, size_t elemOffset, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int remoteBase) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  size_t rowStride = rowBytes / sizeof(float);
  auto const* partner = reinterpret_cast<float const*>(partnerRecvRows);
  auto* remote = reinterpret_cast<float*>(remotePartial);
  size_t elem = elemOffset + idx;
  size_t partnerRemoteOffset =
      (partnerRowsReversed ? 0 : rowStride) + idx;
  remote[idx] =
      send[(static_cast<size_t>(remoteBase + localRank) * fullElems) + elem] +
      partner[partnerRemoteOffset];
}

__global__ void twoNodeTwoGpuLocalOnlyReduceFloat4Kernel(
    float const* send, char const* partnerRecvRows, char* localPartial,
    size_t nVec, size_t fullVecs, size_t elemOffsetVec, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int localBase) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;
  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto const* partner4 = reinterpret_cast<int4 const*>(partnerRecvRows);
  auto* local4 = reinterpret_cast<int4*>(localPartial);
  size_t idx = elemOffsetVec + vecIdx;
  size_t partnerLocalOffset =
      (partnerRowsReversed ? rowVecs : 0) + vecIdx;
  local4[vecIdx] = addFloat4Vec(
      send4[(static_cast<size_t>(localBase + localRank) * fullVecs) + idx],
      partner4[partnerLocalOffset]);
}

__global__ void twoNodeTwoGpuLocalOnlyReduceFloatKernel(
    float const* send, char const* partnerRecvRows, char* localPartial,
    size_t count, size_t fullElems, size_t elemOffset, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int localBase) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  size_t rowStride = rowBytes / sizeof(float);
  auto const* partner = reinterpret_cast<float const*>(partnerRecvRows);
  auto* local = reinterpret_cast<float*>(localPartial);
  size_t elem = elemOffset + idx;
  size_t partnerLocalOffset =
      (partnerRowsReversed ? rowStride : 0) + idx;
  local[idx] =
      send[(static_cast<size_t>(localBase + localRank) * fullElems) + elem] +
      partner[partnerLocalOffset];
}

__global__ void twoRankStoreRemoteShardKernel(float const* send,
                                              char* mappedSlot,
                                              size_t count,
                                              size_t recvcount,
                                              int remoteRank,
                                              size_t remoteOffsetBytes) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  auto* out = reinterpret_cast<float*>(mappedSlot + remoteOffsetBytes);
  out[idx] = send[static_cast<size_t>(remoteRank) * recvcount + idx];
}

__global__ void twoRankFinalizeMappedKernel(float const* send,
                                            char const* mappedIncoming,
                                            float* recv, size_t count,
                                            size_t recvcount, int rank) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  auto const* incoming = reinterpret_cast<float const*>(mappedIncoming);
  recv[idx] = send[static_cast<size_t>(rank) * recvcount + idx] + incoming[idx];
}

__global__ void localFourRankScratchReduceFloat4Kernel(
    char const* scratch, float* recv, size_t nVec, size_t rowStrideVecs) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;
  auto const* rows = reinterpret_cast<int4 const*>(scratch);
  reinterpret_cast<int4*>(recv)[vecIdx] =
      addFloat4Vec(addFloat4Vec(rows[vecIdx], rows[rowStrideVecs + vecIdx]),
                   addFloat4Vec(rows[2 * rowStrideVecs + vecIdx],
                                rows[3 * rowStrideVecs + vecIdx]));
}

__global__ void localFourRankScratchReduceFloatKernel(
    char const* scratch, float* recv, size_t count, size_t rowStrideElems) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  auto const* rows = reinterpret_cast<float const*>(scratch);
  recv[idx] = rows[idx] + rows[rowStrideElems + idx] +
              rows[2 * rowStrideElems + idx] +
              rows[3 * rowStrideElems + idx];
}

__global__ void localFourRankScratchReduceSelfFloat4Kernel(
    char const* scratch, float const* selfShard, float* recv, size_t nVec,
    size_t rowStrideVecs, int selfLocalRank) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;
  auto const* rows = reinterpret_cast<int4 const*>(scratch);
  auto const* self = reinterpret_cast<int4 const*>(selfShard);
  int4 sum = self[vecIdx];
  for (int row = 0; row < 4; ++row) {
    if (row == selfLocalRank) continue;
    sum = addFloat4Vec(sum, rows[static_cast<size_t>(row) * rowStrideVecs +
                                vecIdx]);
  }
  reinterpret_cast<int4*>(recv)[vecIdx] = sum;
}

__global__ void localFourRankScratchReduceSelfFloatKernel(
    char const* scratch, float const* selfShard, float* recv, size_t count,
    size_t rowStrideElems, int selfLocalRank) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  auto const* rows = reinterpret_cast<float const*>(scratch);
  float sum = selfShard[idx];
  for (int row = 0; row < 4; ++row) {
    if (row == selfLocalRank) continue;
    sum += rows[static_cast<size_t>(row) * rowStrideElems + idx];
  }
  recv[idx] = sum;
}

__device__ __forceinline__ unsigned long long volatile* rsFlagPtr(
    char* ctrl, size_t baseOffset, int index) {
  return reinterpret_cast<unsigned long long volatile*>(
      ctrl + baseOffset + static_cast<size_t>(index) * sizeof(uint64_t));
}

__device__ __forceinline__ void waitForRsFlag(char* ctrl, size_t baseOffset,
                                              int index,
                                              unsigned long long epoch) {
  auto* flag = rsFlagPtr(ctrl, baseOffset, index);
  while (*flag != epoch) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(16);
#endif
  }
}

__global__ void localFourRankScratchWaitReduceFloat4Kernel(
    char* ctrl, size_t readyOffset, size_t doneOffset, char const* scratch,
    float* recv, size_t nVec, size_t rowStrideVecs, int localRank,
    unsigned long long epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    *rsFlagPtr(ctrl, readyOffset, localRank) = epoch;
  }
  for (int i = 0; i < 4; ++i) waitForRsFlag(ctrl, readyOffset, i, epoch);

  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx < nVec) {
    auto const* rows = reinterpret_cast<int4 const*>(scratch);
    reinterpret_cast<int4*>(recv)[vecIdx] =
        addFloat4Vec(addFloat4Vec(rows[vecIdx], rows[rowStrideVecs + vecIdx]),
                     addFloat4Vec(rows[2 * rowStrideVecs + vecIdx],
                                  rows[3 * rowStrideVecs + vecIdx]));
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    *rsFlagPtr(ctrl, doneOffset, localRank) = epoch;
  }
}

__global__ void localFourRankScratchWaitReduceFloatKernel(
    char* ctrl, size_t readyOffset, size_t doneOffset, char const* scratch,
    float* recv, size_t count, size_t rowStrideElems, int localRank,
    unsigned long long epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    *rsFlagPtr(ctrl, readyOffset, localRank) = epoch;
  }
  for (int i = 0; i < 4; ++i) waitForRsFlag(ctrl, readyOffset, i, epoch);

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    auto const* rows = reinterpret_cast<float const*>(scratch);
    recv[idx] = rows[idx] + rows[rowStrideElems + idx] +
                rows[2 * rowStrideElems + idx] +
                rows[3 * rowStrideElems + idx];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    *rsFlagPtr(ctrl, doneOffset, localRank) = epoch;
  }
}

__global__ void localFourRankTinyScatterKernel(
    float const* send, float* dst0, float* dst1, float* dst2, float* dst3,
    unsigned long long* flag0, unsigned long long* flag1,
    unsigned long long* flag2, unsigned long long* flag3, size_t count,
    size_t recvcount, unsigned long long epoch) {
  for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
    dst0[idx] = send[idx];
    dst1[idx] = send[recvcount + idx];
    dst2[idx] = send[2 * recvcount + idx];
    dst3[idx] = send[3 * recvcount + idx];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    *flag0 = epoch;
    *flag1 = epoch;
    *flag2 = epoch;
    *flag3 = epoch;
  }
}

__global__ void localFourRankTinyScatterFloat4Kernel(
    float const* send, char* dst0, char* dst1, char* dst2, char* dst3,
    unsigned long long* flag0, unsigned long long* flag1,
    unsigned long long* flag2, unsigned long long* flag3, size_t nVec,
    size_t recvVecs, unsigned long long epoch) {
  for (size_t vecIdx = threadIdx.x; vecIdx < nVec; vecIdx += blockDim.x) {
    auto const* send4 = reinterpret_cast<int4 const*>(send);
    reinterpret_cast<int4*>(dst0)[vecIdx] = send4[vecIdx];
    reinterpret_cast<int4*>(dst1)[vecIdx] = send4[recvVecs + vecIdx];
    reinterpret_cast<int4*>(dst2)[vecIdx] = send4[2 * recvVecs + vecIdx];
    reinterpret_cast<int4*>(dst3)[vecIdx] = send4[3 * recvVecs + vecIdx];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    *flag0 = epoch;
    *flag1 = epoch;
    *flag2 = epoch;
    *flag3 = epoch;
  }
}

__global__ void storeRsControlFlagKernel(char* ctrl, size_t readyOffset,
                                         int localRank,
                                         unsigned long long epoch) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence_system();
    *rsFlagPtr(ctrl, readyOffset, localRank) = epoch;
  }
}

__device__ __forceinline__ int ipcDeviceFlagIndex(int phase, int localRank) {
  return phase * kMaxRanksPerNode + localRank;
}

__global__ void storeIpcDeviceFlagKernel(char* flags, int phase, int localRank,
                                         unsigned long long epoch) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence_system();
    auto* flagWords = reinterpret_cast<unsigned long long*>(flags);
    flagWords[ipcDeviceFlagIndex(phase, localRank)] = epoch;
  }
}

__device__ __forceinline__ void waitForDeviceFlag(
    unsigned long long volatile* flags, int index, unsigned long long epoch) {
  while (flags[index] != epoch) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(16);
#endif
  }
}

__device__ __forceinline__ void waitForIpcDeviceFlag(
    char* flags, int phase, int readyLocalRank, unsigned long long epoch) {
  auto* flagWords = reinterpret_cast<unsigned long long volatile*>(flags);
  waitForDeviceFlag(flagWords, ipcDeviceFlagIndex(phase, readyLocalRank),
                    epoch);
}

__global__ void waitIpcDeviceFlagKernel(char* flags, int phase,
                                        int readyLocalRank,
                                        unsigned long long epoch) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    waitForIpcDeviceFlag(flags, phase, readyLocalRank, epoch);
  }
}

__global__ void localFourRankDeviceFlagReduceFloat4Kernel(
    unsigned long long volatile* flags, char const* scratch, float* recv,
    size_t nVec, size_t rowStrideVecs, unsigned long long epoch) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; ++i) waitForDeviceFlag(flags, i, epoch);
  }
  __syncthreads();
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;
  auto const* rows = reinterpret_cast<int4 const*>(scratch);
  reinterpret_cast<int4*>(recv)[vecIdx] =
      addFloat4Vec(addFloat4Vec(rows[vecIdx], rows[rowStrideVecs + vecIdx]),
                   addFloat4Vec(rows[2 * rowStrideVecs + vecIdx],
                                rows[3 * rowStrideVecs + vecIdx]));
}

__global__ void localFourRankDeviceFlagReduceFloatKernel(
    unsigned long long volatile* flags, char const* scratch, float* recv,
    size_t count, size_t rowStrideElems, unsigned long long epoch) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; ++i) waitForDeviceFlag(flags, i, epoch);
  }
  __syncthreads();
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  auto const* rows = reinterpret_cast<float const*>(scratch);
  recv[idx] = rows[idx] + rows[rowStrideElems + idx] +
              rows[2 * rowStrideElems + idx] +
              rows[3 * rowStrideElems + idx];
}

ncclResult_t launchPackPartnerRows(
    void const* sendbuff, void* partnerSendRows, size_t chunkElems,
    size_t fullElems, size_t elemOffset, size_t rowBytes, int localRank,
    int localBase, int remoteBase, cudaStream_t stream) {
  constexpr int threads = 256;
  if (chunkElems % 4 == 0 && fullElems % 4 == 0 && elemOffset % 4 == 0 &&
      rowBytes % sizeof(int4) == 0) {
    size_t nVec = chunkElems / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    packPartnerRowsFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff), static_cast<char*>(partnerSendRows),
        nVec, fullElems / 4, elemOffset / 4, rowBytes, localRank, localBase,
        remoteBase);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter partner-pack vector kernel");
  }
  int blocks = static_cast<int>((chunkElems + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  packPartnerRowsFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff), static_cast<char*>(partnerSendRows),
      chunkElems, fullElems, elemOffset, rowBytes, localRank, localBase,
      remoteBase);
  return cudaResult(cudaGetLastError(), "ReduceScatter partner-pack kernel");
}

ncclResult_t copyPartnerRowsDirect(void const* sendbuff, char* partnerRecvRows,
                                   size_t chunkElems, size_t fullElems,
                                   size_t elemOffset, size_t rowBytes,
                                   int localRank, int localBase,
                                   int remoteBase, bool use2D,
                                   cudaStream_t stream) {
  auto const* sendBytes = static_cast<char const*>(sendbuff);
  int partnerLocal = localRank ^ 1;
  int crossPairBase = localRank < 2 ? 2 : 0;
  int partnerCrossAssigned = crossPairBase + (partnerLocal & 1);
  int rows[4] = {localBase + partnerLocal, remoteBase + partnerLocal,
                 localBase + partnerCrossAssigned,
                 remoteBase + partnerCrossAssigned};
  size_t rowOffset = elemOffset * sizeof(float);
  size_t fullRowBytes = fullElems * sizeof(float);
  size_t copyBytes = chunkElems * sizeof(float);
  if (use2D && localBase < remoteBase) {
    size_t srcPitch =
        static_cast<size_t>(kMaxRanksPerNode / 2) * fullRowBytes;
    MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
        partnerRecvRows, rowBytes,
        sendBytes + static_cast<size_t>(rows[0]) * fullRowBytes + rowOffset,
        srcPitch, copyBytes, 2, cudaMemcpyDeviceToDevice, stream));
    MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
        partnerRecvRows + static_cast<size_t>(2) * rowBytes, rowBytes,
        sendBytes + static_cast<size_t>(rows[2]) * fullRowBytes + rowOffset,
        srcPitch, copyBytes, 2, cudaMemcpyDeviceToDevice, stream));
    return ncclSuccess;
  }
  for (int i = 0; i < 4; ++i) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        partnerRecvRows + static_cast<size_t>(i) * rowBytes,
        sendBytes + static_cast<size_t>(rows[i]) * fullRowBytes + rowOffset,
        copyBytes, cudaMemcpyDeviceToDevice, stream));
  }
  return ncclSuccess;
}

ncclResult_t launchPairReduceFromInput(
    void const* sendbuff, void* partnerRecvRows, void* ownPartialRows,
    void* crossSendRows, size_t count, size_t fullElems, size_t elemOffset,
    size_t rowBytes, int localRank, int localBase, int remoteBase,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && fullElems % 4 == 0 && elemOffset % 4 == 0 &&
      rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    reduceNumaPairFromInputFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff),
        static_cast<char const*>(partnerRecvRows),
        static_cast<char*>(ownPartialRows), static_cast<char*>(crossSendRows),
        nVec, fullElems / 4, elemOffset / 4, rowBytes, localRank, localBase,
        remoteBase);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter input pair-reduce vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  reduceNumaPairFromInputFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff),
      static_cast<char const*>(partnerRecvRows),
      static_cast<char*>(ownPartialRows), static_cast<char*>(crossSendRows),
      count, fullElems, elemOffset, rowBytes, localRank, localBase, remoteBase);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter input pair-reduce kernel");
}

ncclResult_t launchStoreRsControlFlag(RsContext& ctx, size_t readyOffset,
                                      uint64_t epoch, cudaStream_t stream) {
  if (ctx.ctrlDeviceSlab == nullptr) return ncclInvalidUsage;
  storeRsControlFlagKernel<<<1, 1, 0, stream>>>(
      ctx.ctrlDeviceSlab, readyOffset, ctx.localRank,
      static_cast<unsigned long long>(epoch));
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter device control flag store");
}

ncclResult_t launchStoreIpcDeviceFlag(RsContext& ctx, int phase,
                                      uint64_t epoch, cudaStream_t stream) {
  if (ctx.localDeviceFlags == nullptr) return ncclInvalidUsage;
  storeIpcDeviceFlagKernel<<<1, 1, 0, stream>>>(
      ctx.localDeviceFlags, phase, ctx.localRank,
      static_cast<unsigned long long>(epoch));
  return cudaResult(cudaGetLastError(), "ReduceScatter IPC device flag store");
}

ncclResult_t launchWaitIpcDeviceFlag(RsContext& ctx, int phase,
                                     int readyLocalRank, uint64_t epoch,
                                     cudaStream_t stream) {
  int peerIdx = localPeerIndex(ctx.localRank, readyLocalRank);
  if (peerIdx < 0 ||
      peerIdx >= static_cast<int>(ctx.remoteDeviceFlagPtrs.size()) ||
      ctx.remoteDeviceFlagPtrs[peerIdx] == nullptr) {
    return ncclInvalidUsage;
  }
  waitIpcDeviceFlagKernel<<<1, 1, 0, stream>>>(
      ctx.remoteDeviceFlagPtrs[peerIdx], phase, readyLocalRank,
      static_cast<unsigned long long>(epoch));
  return cudaResult(cudaGetLastError(), "ReduceScatter IPC device flag wait");
}

ncclResult_t launchPairReduceFromInputWait(
    RsContext& ctx, size_t readyOffset, int readyLocalRank, uint64_t epoch,
    void const* sendbuff, void* partnerRecvRows, void* ownPartialRows,
    void* crossSendRows, size_t count, size_t fullElems, size_t elemOffset,
    size_t rowBytes, int localRank, int localBase, int remoteBase,
    cudaStream_t stream) {
  if (ctx.ctrlDeviceSlab == nullptr) return ncclInvalidUsage;
  constexpr int threads = 256;
  if (count % 4 == 0 && fullElems % 4 == 0 && elemOffset % 4 == 0 &&
      rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    reduceNumaPairFromInputWaitFloat4Kernel<<<blocks, threads, 0, stream>>>(
        ctx.ctrlDeviceSlab, readyOffset, readyLocalRank,
        static_cast<unsigned long long>(epoch),
        static_cast<float const*>(sendbuff),
        static_cast<char const*>(partnerRecvRows),
        static_cast<char*>(ownPartialRows), static_cast<char*>(crossSendRows),
        nVec, fullElems / 4, elemOffset / 4, rowBytes, localRank, localBase,
        remoteBase);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter device-wait pair-reduce vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  reduceNumaPairFromInputWaitFloatKernel<<<blocks, threads, 0, stream>>>(
      ctx.ctrlDeviceSlab, readyOffset, readyLocalRank,
      static_cast<unsigned long long>(epoch),
      static_cast<float const*>(sendbuff),
      static_cast<char const*>(partnerRecvRows),
      static_cast<char*>(ownPartialRows), static_cast<char*>(crossSendRows),
      count, fullElems, elemOffset, rowBytes, localRank, localBase, remoteBase);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter device-wait pair-reduce kernel");
}

ncclResult_t launchFinalLocalReduce(void* ownPartialRows, void* crossRecvRows,
                                    void* localOutput, void* remoteOutput,
                                    size_t count, size_t rowBytes,
                                    cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    finalNumaPairReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(ownPartialRows),
        static_cast<char const*>(crossRecvRows), static_cast<float*>(localOutput),
        static_cast<float*>(remoteOutput), nVec, rowBytes);
    return cudaResult(cudaGetLastError(), "ReduceScatter final local vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  finalNumaPairReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(ownPartialRows),
      static_cast<char const*>(crossRecvRows), static_cast<float*>(localOutput),
      static_cast<float*>(remoteOutput), count, rowBytes);
  return cudaResult(cudaGetLastError(), "ReduceScatter final local kernel");
}

ncclResult_t launchFinalLocalOnlyReduce(void* ownPartialRows,
                                        void* crossRecvRows,
                                        void* localOutput, size_t count,
                                        size_t rowBytes, cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    finalNumaPairReduceLocalFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(ownPartialRows),
        static_cast<char const*>(crossRecvRows), static_cast<float*>(localOutput),
        nVec, rowBytes);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter final local-only vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  finalNumaPairReduceLocalFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(ownPartialRows),
      static_cast<char const*>(crossRecvRows), static_cast<float*>(localOutput),
      count, rowBytes);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter final local-only kernel");
}

ncclResult_t launchFinalRemoteOnlyReduce(void* ownPartialRows,
                                         void* crossRecvRows,
                                         void* remoteOutput, size_t count,
                                         size_t rowBytes, cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    finalNumaPairReduceRemoteFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(ownPartialRows),
        static_cast<char const*>(crossRecvRows),
        static_cast<float*>(remoteOutput), nVec, rowBytes);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter final remote-only vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  finalNumaPairReduceRemoteFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(ownPartialRows),
      static_cast<char const*>(crossRecvRows), static_cast<float*>(remoteOutput),
      count, rowBytes);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter final remote-only kernel");
}

ncclResult_t launchFinalLocalReduceWait(
    RsContext& ctx, size_t readyOffset, int readyLocalRank, uint64_t epoch,
    void* ownPartialRows, void* crossRecvRows, void* localOutput,
    void* remoteOutput, size_t count, size_t rowBytes, cudaStream_t stream) {
  if (ctx.ctrlDeviceSlab == nullptr) return ncclInvalidUsage;
  constexpr int threads = 256;
  if (count % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    finalNumaPairReduceWaitFloat4Kernel<<<blocks, threads, 0, stream>>>(
        ctx.ctrlDeviceSlab, readyOffset, readyLocalRank,
        static_cast<unsigned long long>(epoch),
        static_cast<char const*>(ownPartialRows),
        static_cast<char const*>(crossRecvRows),
        static_cast<float*>(localOutput), static_cast<float*>(remoteOutput),
        nVec, rowBytes);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter device-wait final vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  finalNumaPairReduceWaitFloatKernel<<<blocks, threads, 0, stream>>>(
      ctx.ctrlDeviceSlab, readyOffset, readyLocalRank,
      static_cast<unsigned long long>(epoch),
      static_cast<char const*>(ownPartialRows),
      static_cast<char const*>(crossRecvRows), static_cast<float*>(localOutput),
      static_cast<float*>(remoteOutput), count, rowBytes);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter device-wait final kernel");
}

ncclResult_t launchDirectTargetReduce(void* rows, void* localOutput,
                                      void* remoteOutput, size_t count,
                                      size_t rowBytes, cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    directTargetReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(rows), static_cast<float*>(localOutput),
        static_cast<float*>(remoteOutput), nVec, rowBytes);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter direct-target vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  directTargetReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(rows), static_cast<float*>(localOutput),
      static_cast<float*>(remoteOutput), count, rowBytes);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter direct-target kernel");
}

ncclResult_t launchDirectTargetScatter(
    void const* sendbuff, char* dst0, char* dst1, char* dst2, char* dst3,
    size_t count, size_t fullElems, size_t elemOffset, size_t rowBytes,
    int localRank, int localBase, int remoteBase, cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && fullElems % 4 == 0 && elemOffset % 4 == 0 &&
      rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    directTargetScatterFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff), dst0, dst1, dst2, dst3, nVec,
        fullElems / 4, elemOffset / 4, rowBytes, localRank, localBase,
        remoteBase);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter direct-target scatter vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  directTargetScatterFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff), dst0, dst1, dst2, dst3, count,
      fullElems, elemOffset, rowBytes, localRank, localBase, remoteBase);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter direct-target scatter kernel");
}

ncclResult_t launchAdd(void const* lhs, void const* rhs, void* output,
                       size_t count, cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    addFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(lhs), static_cast<float const*>(rhs),
        static_cast<float*>(output), nVec);
    return cudaResult(cudaGetLastError(), "ReduceScatter final add vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  addFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(lhs), static_cast<float const*>(rhs),
      static_cast<float*>(output), count);
  return cudaResult(cudaGetLastError(), "ReduceScatter final add kernel");
}

ncclResult_t launchTwoNodeTwoGpuLocalReduce(
    void const* sendbuff, void* partnerRecvRows, void* localPartialGpu,
    void* remotePartialGpu, size_t count, size_t fullElems, size_t elemOffset,
    size_t rowBytes, bool partnerRowsReversed, int localRank, int localBase,
    int remoteBase, cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && fullElems % 4 == 0 && elemOffset % 4 == 0 &&
      rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    twoNodeTwoGpuLocalReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff),
        static_cast<char const*>(partnerRecvRows),
        static_cast<char*>(localPartialGpu), static_cast<char*>(remotePartialGpu),
        nVec, fullElems / 4, elemOffset / 4, rowBytes, partnerRowsReversed,
        localRank, localBase, remoteBase);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter 2nx2g local reduce vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  twoNodeTwoGpuLocalReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff),
      static_cast<char const*>(partnerRecvRows),
      static_cast<char*>(localPartialGpu), static_cast<char*>(remotePartialGpu),
      count, fullElems, elemOffset, rowBytes, partnerRowsReversed, localRank,
      localBase, remoteBase);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter 2nx2g local reduce kernel");
}

ncclResult_t launchTwoNodeTwoGpuLocalReduceWait(
    RsContext& ctx, int readyLocalRank, uint64_t epoch, void const* sendbuff,
    void* partnerRecvRows, void* localPartialGpu, void* remotePartialGpu,
    size_t count, size_t fullElems, size_t elemOffset, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int localBase, int remoteBase,
    cudaStream_t stream) {
  int peerIdx = localPeerIndex(ctx.localRank, readyLocalRank);
  if (peerIdx < 0 ||
      peerIdx >= static_cast<int>(ctx.remoteDeviceFlagPtrs.size()) ||
      ctx.remoteDeviceFlagPtrs[peerIdx] == nullptr) {
    return ncclInvalidUsage;
  }
  constexpr int threads = 256;
  if (count % 4 == 0 && fullElems % 4 == 0 && elemOffset % 4 == 0 &&
      rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    twoNodeTwoGpuLocalReduceWaitFloat4Kernel<<<blocks, threads, 0, stream>>>(
        ctx.remoteDeviceFlagPtrs[peerIdx], readyLocalRank,
        static_cast<unsigned long long>(epoch),
        static_cast<float const*>(sendbuff),
        static_cast<char const*>(partnerRecvRows),
        static_cast<char*>(localPartialGpu), static_cast<char*>(remotePartialGpu),
        nVec, fullElems / 4, elemOffset / 4, rowBytes, partnerRowsReversed,
        localRank, localBase, remoteBase);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter 2nx2g device-wait local reduce vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  twoNodeTwoGpuLocalReduceWaitFloatKernel<<<blocks, threads, 0, stream>>>(
      ctx.remoteDeviceFlagPtrs[peerIdx], readyLocalRank,
      static_cast<unsigned long long>(epoch),
      static_cast<float const*>(sendbuff),
      static_cast<char const*>(partnerRecvRows),
      static_cast<char*>(localPartialGpu), static_cast<char*>(remotePartialGpu),
      count, fullElems, elemOffset, rowBytes, partnerRowsReversed, localRank,
      localBase, remoteBase);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter 2nx2g device-wait local reduce kernel");
}

ncclResult_t launchTwoNodeTwoGpuRemoteOnlyReduce(
    void const* sendbuff, void* partnerRecvRows, void* remotePartialGpu,
    size_t count, size_t fullElems, size_t elemOffset, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int remoteBase,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && fullElems % 4 == 0 && elemOffset % 4 == 0 &&
      rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    twoNodeTwoGpuRemoteReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff),
        static_cast<char const*>(partnerRecvRows),
        static_cast<char*>(remotePartialGpu), nVec, fullElems / 4,
        elemOffset / 4, rowBytes, partnerRowsReversed, localRank, remoteBase);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter 2nx2g remote-only reduce vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  twoNodeTwoGpuRemoteReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff),
      static_cast<char const*>(partnerRecvRows),
      static_cast<char*>(remotePartialGpu), count, fullElems, elemOffset,
      rowBytes, partnerRowsReversed, localRank, remoteBase);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter 2nx2g remote-only reduce kernel");
}

ncclResult_t launchTwoNodeTwoGpuLocalOnlyReduce(
    void const* sendbuff, void* partnerRecvRows, void* localPartialGpu,
    size_t count, size_t fullElems, size_t elemOffset, size_t rowBytes,
    bool partnerRowsReversed, int localRank, int localBase,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (count % 4 == 0 && fullElems % 4 == 0 && elemOffset % 4 == 0 &&
      rowBytes % sizeof(int4) == 0) {
    size_t nVec = count / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    twoNodeTwoGpuLocalOnlyReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff),
        static_cast<char const*>(partnerRecvRows),
        static_cast<char*>(localPartialGpu), nVec, fullElems / 4,
        elemOffset / 4, rowBytes, partnerRowsReversed, localRank, localBase);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter 2nx2g local-only reduce vector kernel");
  }
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  twoNodeTwoGpuLocalOnlyReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff),
      static_cast<char const*>(partnerRecvRows),
      static_cast<char*>(localPartialGpu), count, fullElems, elemOffset,
      rowBytes, partnerRowsReversed, localRank, localBase);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter 2nx2g local-only reduce kernel");
}

ncclResult_t launchTwoRankStoreRemoteShard(void const* sendbuff,
                                           void* mappedSlot,
                                           size_t count, size_t recvcount,
                                           int remoteRank,
                                           size_t remoteOffsetBytes,
                                           cudaStream_t stream) {
  constexpr int threads = 128;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  twoRankStoreRemoteShardKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff), static_cast<char*>(mappedSlot),
      count, recvcount, remoteRank, remoteOffsetBytes);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter 2-rank mapped store kernel");
}

ncclResult_t launchTwoRankFinalizeMapped(void const* sendbuff,
                                         void const* mappedIncoming,
                                         void* recvbuff, size_t count,
                                         size_t recvcount, int rank,
                                         cudaStream_t stream) {
  constexpr int threads = 128;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  twoRankFinalizeMappedKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff),
      static_cast<char const*>(mappedIncoming), static_cast<float*>(recvbuff),
      count, recvcount, rank);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter 2-rank mapped finalize kernel");
}

ncclResult_t launchLocalFourRankScratchReduce(void const* scratch,
                                              void* recvbuff,
                                              size_t chunkElems,
                                              size_t rowStrideBytes,
                                              cudaStream_t stream) {
  constexpr int threads = 256;
  if (chunkElems % 4 == 0 && rowStrideBytes % sizeof(int4) == 0) {
    size_t nVec = chunkElems / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    localFourRankScratchReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(scratch), static_cast<float*>(recvbuff), nVec,
        rowStrideBytes / sizeof(int4));
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter local scratch vector kernel");
  }
  int blocks = static_cast<int>((chunkElems + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  localFourRankScratchReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(scratch), static_cast<float*>(recvbuff),
      chunkElems, rowStrideBytes / sizeof(float));
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter local scratch kernel");
}

ncclResult_t launchLocalFourRankScratchReduceSelf(
    void const* scratch, void const* selfShard, void* recvbuff,
    size_t chunkElems, size_t rowStrideBytes, int localRank,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (chunkElems % 4 == 0 && rowStrideBytes % sizeof(int4) == 0) {
    size_t nVec = chunkElems / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    localFourRankScratchReduceSelfFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(scratch),
        static_cast<float const*>(selfShard), static_cast<float*>(recvbuff),
        nVec, rowStrideBytes / sizeof(int4), localRank);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter local scratch self vector kernel");
  }
  int blocks = static_cast<int>((chunkElems + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  localFourRankScratchReduceSelfFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(scratch), static_cast<float const*>(selfShard),
      static_cast<float*>(recvbuff), chunkElems, rowStrideBytes / sizeof(float),
      localRank);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter local scratch self kernel");
}

ncclResult_t launchLocalFourRankScratchWaitReduce(
    char* ctrlDeviceSlab, size_t readyOffset, size_t doneOffset,
    void const* scratch, void* recvbuff, size_t chunkElems,
    size_t rowStrideBytes, int localRank, uint64_t epoch,
    cudaStream_t stream) {
  if (ctrlDeviceSlab == nullptr) return ncclInvalidUsage;
  constexpr int threads = 256;
  if (chunkElems % 4 == 0 && rowStrideBytes % sizeof(int4) == 0) {
    size_t nVec = chunkElems / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    localFourRankScratchWaitReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        ctrlDeviceSlab, readyOffset, doneOffset,
        static_cast<char const*>(scratch), static_cast<float*>(recvbuff), nVec,
        rowStrideBytes / sizeof(int4), localRank, epoch);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter local scratch wait vector kernel");
  }
  int blocks = static_cast<int>((chunkElems + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  localFourRankScratchWaitReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      ctrlDeviceSlab, readyOffset, doneOffset, static_cast<char const*>(scratch),
      static_cast<float*>(recvbuff), chunkElems, rowStrideBytes / sizeof(float),
      localRank, epoch);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter local scratch wait kernel");
}

ncclResult_t launchLocalFourRankTinyScatter(
    void const* sendbuff, void* dst0, void* dst1, void* dst2, void* dst3,
    unsigned long long* flag0, unsigned long long* flag1,
    unsigned long long* flag2, unsigned long long* flag3, size_t recvcount,
    uint64_t epoch, cudaStream_t stream) {
  constexpr int threads = 256;
  if (recvcount % 4 == 0) {
    size_t nVec = recvcount / 4;
    if (nVec == 0) return ncclSuccess;
    localFourRankTinyScatterFloat4Kernel<<<1, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff), static_cast<char*>(dst0),
        static_cast<char*>(dst1), static_cast<char*>(dst2),
        static_cast<char*>(dst3), flag0, flag1, flag2, flag3, nVec,
        recvcount / 4, epoch);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter local tiny scatter vector kernel");
  }
  if (recvcount == 0) return ncclSuccess;
  localFourRankTinyScatterKernel<<<1, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff), static_cast<float*>(dst0),
      static_cast<float*>(dst1), static_cast<float*>(dst2),
      static_cast<float*>(dst3), flag0, flag1, flag2, flag3, recvcount,
      recvcount, epoch);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter local tiny scatter kernel");
}

ncclResult_t launchLocalFourRankDeviceFlagReduce(
    unsigned long long volatile* flags, void const* scratch, void* recvbuff,
    size_t chunkElems, size_t rowStrideBytes, uint64_t epoch,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (chunkElems % 4 == 0 && rowStrideBytes % sizeof(int4) == 0) {
    size_t nVec = chunkElems / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    localFourRankDeviceFlagReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        flags, static_cast<char const*>(scratch), static_cast<float*>(recvbuff),
        nVec, rowStrideBytes / sizeof(int4), epoch);
    return cudaResult(cudaGetLastError(),
                      "ReduceScatter local device-flag vector kernel");
  }
  int blocks = static_cast<int>((chunkElems + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  localFourRankDeviceFlagReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      flags, static_cast<char const*>(scratch), static_cast<float*>(recvbuff),
      chunkElems, rowStrideBytes / sizeof(float), epoch);
  return cudaResult(cudaGetLastError(),
                    "ReduceScatter local device-flag kernel");
}

RsContext& getContext(ncclComm_t commHandle,
                      std::shared_ptr<Communicator> bootstrapComm, int rank,
                      int nRanks, int nRanksPerNode, int cudaDevice,
                      size_t chunkCapacity, int pipelineSlots,
                      int transportPolicy) {
  RsContext* ctx = nullptr;
  bool shouldInitialize = false;
  {
    std::lock_guard<std::mutex> lock(gReduceScatterMutex);
    RsContextKey key{commHandle, transportPolicy};
    auto& existing = gReduceScatterContexts[key];
    if (!existing) {
      existing = std::make_unique<RsContext>();
      existing->initializing = true;
      shouldInitialize = true;
    }
    ctx = existing.get();
  }
  {
    std::unique_lock<std::mutex> initLock(ctx->initMutex);
    if (ctx->initialized) return *ctx;
    if (!shouldInitialize) {
      ctx->initCv.wait(initLock,
                       [&] { return ctx->initialized || !ctx->initializing; });
      if (ctx->initialized) return *ctx;
      if (ctx->initException) std::rethrow_exception(ctx->initException);
      throw mscclpp::Error("ReduceScatter context initialization failed",
                           mscclpp::ErrorCode::InternalError);
    }
  }

  InitGuard<RsContext> initGuard(ctx);
  ctx->rank = rank;
  ctx->worldSize = nRanks;
  ctx->nRanksPerNode = nRanksPerNode;
  ctx->localRank = rank % nRanksPerNode;
  ctx->nodeId = rank / nRanksPerNode;
  ctx->localLeader = ctx->nodeId * nRanksPerNode;
  ctx->cudaDevice = cudaDevice;
  try {
    ctx->numaNode = mscclpp::getDeviceNumaNode(cudaDevice);
  } catch (...) {
    ctx->numaNode = -1;
  }
  ctx->owner = rank == ctx->localLeader;
  ctx->pipelineSlots = pipelineSlots;
  ctx->chunkCapacity = chunkCapacity;
  ctx->partialBlockCapacity = static_cast<size_t>(nRanksPerNode) *
                              chunkCapacity *
                              static_cast<size_t>(ctx->pipelineSlots);

  RsControlName localName;
  ncclResult_t createResult = ncclSuccess;
  std::string createMessage;
  try {
    if (ctx->owner) {
      auto commNonce = static_cast<unsigned long long>(
          reinterpret_cast<uintptr_t>(commHandle));
      std::snprintf(localName.smallSendName, sizeof(localName.smallSendName),
                    "/mint_rs_multi_%llx_%d_%d_%d_p%d_s", commNonce, getpid(),
                    rank, nRanks, transportPolicy);
      std::snprintf(localName.smallRecvName, sizeof(localName.smallRecvName),
                    "/mint_rs_multi_%llx_%d_%d_%d_p%d_r", commNonce, getpid(),
                    rank, nRanks, transportPolicy);
      std::snprintf(localName.ctrlName, sizeof(localName.ctrlName),
                    "/mint_rs_multi_%llx_%d_%d_%d_p%d_c", commNonce, getpid(),
                    rank, nRanks, transportPolicy);
      createOwnedShm(localName.smallSendName, ctx->partialBlockCapacity);
      createOwnedShm(localName.smallRecvName, ctx->partialBlockCapacity);
      createOwnedShm(localName.ctrlName, sizeof(RsControl));
    }
  } catch (std::exception const& ex) {
    createResult = mapException(ex);
    createMessage = ex.what();
  } catch (...) {
    createResult = ncclInternalError;
    createMessage = "unknown shared-memory create exception";
  }
  publishInitStatus(bootstrapComm, rank, nRanks, createResult, createMessage,
                    "ReduceScatter shared-memory create");

  std::vector<RsControlName> allNames(nRanks);
  allNames[rank] = localName;
  bootstrapComm->bootstrap()->allGather(allNames.data(), sizeof(RsControlName));
  auto const& leaderNames = allNames[ctx->localLeader];
  ctx->smallSendName = leaderNames.smallSendName;
  ctx->smallRecvName = leaderNames.smallRecvName;
  ctx->ctrlName = leaderNames.ctrlName;

  ncclResult_t mapResult = ncclSuccess;
  std::string mapMessage;
  try {
    ctx->smallSendMapping =
        mapShm(ctx->smallSendName, ctx->partialBlockCapacity);
    ctx->smallRecvMapping =
        mapShm(ctx->smallRecvName, ctx->partialBlockCapacity);
    ctx->ctrlMapping = mapShm(ctx->ctrlName, sizeof(RsControl));
    ctx->smallSendSlab = static_cast<char*>(ctx->smallSendMapping);
    ctx->smallRecvSlab = static_cast<char*>(ctx->smallRecvMapping);
    ctx->ctrl = static_cast<RsControl*>(ctx->ctrlMapping);
    if (ctx->owner) {
      std::memset(ctx->ctrlMapping, 0, sizeof(RsControl));
      new (ctx->ctrl) RsControl{};
    }
  } catch (std::exception const& ex) {
    mapResult = mapException(ex);
    mapMessage = ex.what();
  } catch (...) {
    mapResult = ncclInternalError;
    mapMessage = "unknown shared-memory map exception";
  }
  publishInitStatus(bootstrapComm, rank, nRanks, mapResult, mapMessage,
                    "ReduceScatter shared-memory map");
  bootstrapComm->bootstrap()->barrier();

  size_t smallRankStride =
      ctx->chunkCapacity * static_cast<size_t>(ctx->pipelineSlots);
  size_t localSliceOffset = static_cast<size_t>(ctx->localRank) *
                            smallRankStride;
  placeOnNuma(ctx->smallSendSlab + localSliceOffset, smallRankStride,
              ctx->numaNode, "ReduceScatter small send slice");
  placeOnNuma(ctx->smallRecvSlab + localSliceOffset, smallRankStride,
              ctx->numaNode, "ReduceScatter small recv slice");
  bootstrapComm->bootstrap()->barrier();

  ncclResult_t setupResult = ncclSuccess;
  std::string setupMessage;
  bool const singleNode = nRanks == nRanksPerNode;
  try {
    MSCCLPP_CUDATHROW(cudaMallocHost(&ctx->sendMapping,
                                     ctx->partialBlockCapacity));
    MSCCLPP_CUDATHROW(cudaMallocHost(&ctx->recvMapping,
                                     ctx->partialBlockCapacity));
    ctx->sendSlab = static_cast<char*>(ctx->sendMapping);
    ctx->recvSlab = static_cast<char*>(ctx->recvMapping);
    void* sendDevicePtr = nullptr;
    cudaError_t sendDeviceResult =
        cudaHostGetDevicePointer(&sendDevicePtr, ctx->sendMapping, 0);
    if (sendDeviceResult == cudaSuccess) {
      ctx->sendDeviceSlab = static_cast<char*>(sendDevicePtr);
    } else {
      cudaGetLastError();
      ctx->sendDeviceSlab = nullptr;
    }
    void* recvDevicePtr = nullptr;
    cudaError_t recvDeviceResult =
        cudaHostGetDevicePointer(&recvDevicePtr, ctx->recvMapping, 0);
    if (recvDeviceResult == cudaSuccess) {
      ctx->recvDeviceSlab = static_cast<char*>(recvDevicePtr);
    } else {
      cudaGetLastError();
      ctx->recvDeviceSlab = nullptr;
    }
    placeOnNuma(ctx->sendMapping, ctx->partialBlockCapacity, ctx->numaNode,
                "ReduceScatter send slab");
    placeOnNuma(ctx->recvMapping, ctx->partialBlockCapacity, ctx->numaNode,
                "ReduceScatter recv slab");
    MSCCLPP_CUDATHROW(cudaHostRegister(ctx->smallSendMapping,
                                       ctx->partialBlockCapacity,
                                       cudaHostRegisterPortable));
    ctx->smallSendHostRegistered = true;
    cudaError_t smallRecvRegister = cudaHostRegister(
        ctx->smallRecvMapping, ctx->partialBlockCapacity,
        cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (smallRecvRegister == cudaSuccess) {
      void* devicePtr = nullptr;
      MSCCLPP_CUDATHROW(
          cudaHostGetDevicePointer(&devicePtr, ctx->smallRecvMapping, 0));
      ctx->smallRecvDeviceSlab = static_cast<char*>(devicePtr);
    } else {
      cudaGetLastError();
      MSCCLPP_CUDATHROW(cudaHostRegister(ctx->smallRecvMapping,
                                         ctx->partialBlockCapacity,
                                         cudaHostRegisterPortable));
      ctx->smallRecvDeviceSlab = nullptr;
    }
    ctx->smallRecvHostRegistered = true;
    cudaError_t ctrlRegister = cudaHostRegister(
        ctx->ctrlMapping, sizeof(RsControl),
        cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (ctrlRegister == cudaSuccess) {
      void* devicePtr = nullptr;
      MSCCLPP_CUDATHROW(
          cudaHostGetDevicePointer(&devicePtr, ctx->ctrlMapping, 0));
      ctx->ctrlDeviceSlab = static_cast<char*>(devicePtr);
    } else {
      cudaGetLastError();
      MSCCLPP_CUDATHROW(cudaHostRegister(ctx->ctrlMapping, sizeof(RsControl),
                                         cudaHostRegisterPortable));
      ctx->ctrlDeviceSlab = nullptr;
    }
    ctx->ctrlHostRegistered = true;

    if (!singleNode) {
      ctx->transport =
          selectReduceScatterTransport(cudaDevice, ctx->localRank,
                                       transportPolicy);
      if (ctx->transport == mscclpp::Transport::Unknown) {
        throw mscclpp::Error("ReduceScatter requires IB transport",
                             mscclpp::ErrorCode::InvalidUsage);
      }
      mscclpp::TransportFlags transportFlags(ctx->transport);
      ctx->sendMemory = bootstrapComm->registerMemory(
          ctx->sendSlab, ctx->partialBlockCapacity, transportFlags);
      ctx->recvMemory = bootstrapComm->registerMemory(
          ctx->recvSlab, ctx->partialBlockCapacity, transportFlags);
      ctx->smallRecvMemory = bootstrapComm->registerMemory(
          ctx->smallRecvSlab, ctx->partialBlockCapacity, transportFlags);
      ctx->ctrlMemory =
          bootstrapComm->registerMemory(ctx->ctrlMapping, sizeof(RsControl),
                                        transportFlags);
    }
  } catch (std::exception const& ex) {
    setupResult = mapException(ex);
    setupMessage = ex.what();
  } catch (...) {
    setupResult = ncclInternalError;
    setupMessage = "unknown setup exception";
  }
  publishInitStatus(bootstrapComm, rank, nRanks, setupResult, setupMessage,
                    "ReduceScatter setup");

  ncclResult_t connectResult = ncclSuccess;
  std::string connectMessage;
  try {
    if (singleNode) {
      connectResult = ncclSuccess;
      connectMessage.clear();
    } else {
      int remotePeer = (1 - ctx->nodeId) * nRanksPerNode + ctx->localRank;
      mscclpp::EndpointConfig::Ib ibCfg;
      ibCfg.maxCqPollNum = 128;
      ibCfg.mode = mscclpp::EndpointConfig::Ib::Mode::Host;
      mscclpp::EndpointConfig endpointConfig(
          ctx->transport, mscclpp::Device(mscclpp::DeviceType::CPU),
          /*maxWriteQueueSize=*/-1, ibCfg);
      int tagBase = transportPolicy * 4;
      int tag0 = rsTag(rank, nRanks, remotePeer, tagBase + 0);
      int tag1 = rsTag(rank, nRanks, remotePeer, tagBase + 1);
      int tag2 = rsTag(rank, nRanks, remotePeer, tagBase + 2);
      int tag3 = rsTag(rank, nRanks, remotePeer, tagBase + 3);
      auto connectionFuture =
          bootstrapComm->connect(endpointConfig, remotePeer, tag0);
      bootstrapComm->sendMemory(ctx->recvMemory, remotePeer, tag1);
      auto remoteRecvFuture = bootstrapComm->recvMemory(remotePeer, tag1);
      bootstrapComm->sendMemory(ctx->ctrlMemory, remotePeer, tag2);
      auto remoteCtrlFuture = bootstrapComm->recvMemory(remotePeer, tag2);
      bootstrapComm->sendMemory(ctx->smallRecvMemory, remotePeer, tag3);
      auto remoteSmallRecvFuture = bootstrapComm->recvMemory(remotePeer, tag3);
      ctx->pairConnection = connectionFuture.get();
      ctx->remoteRecvMemory = remoteRecvFuture.get();
      ctx->remoteCtrlMemory = remoteCtrlFuture.get();
      ctx->remoteSmallRecvMemory = remoteSmallRecvFuture.get();
      ctx->smallQp = ctx->pairConnection.getIbQp();
      if (ctx->smallQp) {
        ctx->sendMemory.getIbMrInfo(ctx->transport, &ctx->sendMr, nullptr);
        ctx->remoteRecvMemory.getIbMrInfo(ctx->transport, nullptr,
                                          &ctx->remoteRecvMrInfo);
        ctx->smallRecvMemory.getIbMrInfo(ctx->transport, &ctx->smallRecvMr,
                                         nullptr);
        ctx->ctrlMemory.getIbMrInfo(ctx->transport, &ctx->smallCtrlMr, nullptr);
        ctx->remoteSmallRecvMemory.getIbMrInfo(ctx->transport, nullptr,
                                               &ctx->smallRemoteRecvMrInfo);
        ctx->remoteCtrlMemory.getIbMrInfo(ctx->transport, nullptr,
                                          &ctx->smallRemoteCtrlMrInfo);
      }
    }
  } catch (std::exception const& ex) {
    connectResult = mapException(ex);
    connectMessage = ex.what();
  } catch (...) {
    connectResult = ncclInternalError;
    connectMessage = "unknown connection exception";
  }
  publishInitStatus(bootstrapComm, rank, nRanks, connectResult, connectMessage,
                    "ReduceScatter connect");

  initGuard.commit();
  return *ctx;
}

void ensureLocalScratchIpc(RsContext& ctx,
                           std::shared_ptr<Communicator> bootstrapComm,
                           int rank, int nRanks, int nRanksPerNode,
                           void* scratchBuffer, size_t scratchBufferSize) {
  bool wantDeviceFlags = useIpcDeviceFlagSync() && nRanksPerNode <= kMaxRanksPerNode;
  size_t expectedPeerCount = static_cast<size_t>(std::max(0, nRanksPerNode - 1));
  if (ctx.localScratchIpcReady && ctx.localScratchBuffer == scratchBuffer &&
      ctx.localScratchBufferSize == scratchBufferSize &&
      (!wantDeviceFlags ||
       (ctx.localDeviceFlags != nullptr &&
        ctx.remoteDeviceFlagPtrs.size() == expectedPeerCount))) {
    return;
  }

  mscclpp::TransportFlags ipcFlags(mscclpp::Transport::CudaIpc);
  ctx.localScratchMemory =
      bootstrapComm->registerMemory(scratchBuffer, scratchBufferSize, ipcFlags);
  ctx.remoteScratchMemories.clear();
  ctx.remoteScratchPtrs.clear();
  ctx.remoteDeviceFlagMemories.clear();
  ctx.remoteDeviceFlagPtrs.clear();
  if (wantDeviceFlags) {
    size_t flagBytes = static_cast<size_t>(kIpcDeviceFlagPhases) *
                       static_cast<size_t>(kMaxRanksPerNode) * sizeof(uint64_t);
    if (ctx.localDeviceFlags == nullptr) {
      MSCCLPP_CUDATHROW(cudaMalloc(&ctx.localDeviceFlags, flagBytes));
      MSCCLPP_CUDATHROW(cudaMemset(ctx.localDeviceFlags, 0, flagBytes));
    }
    ctx.localDeviceFlagMemory =
        bootstrapComm->registerMemory(ctx.localDeviceFlags, flagBytes, ipcFlags);
  }

  int localBase = (rank / nRanksPerNode) * nRanksPerNode;
  std::vector<decltype(bootstrapComm->recvMemory(0, 0))> memoryFutures;
  std::vector<decltype(bootstrapComm->recvMemory(0, 0))> flagFutures;
  for (int i = 0; i < nRanksPerNode; ++i) {
    int peer = localBase + i;
    if (peer == rank) continue;
    int scratchTag = rsTag(rank, nRanks, peer, 4);
    bootstrapComm->sendMemory(ctx.localScratchMemory, peer, scratchTag);
    memoryFutures.push_back(bootstrapComm->recvMemory(peer, scratchTag));
    if (wantDeviceFlags) {
      int flagTag = rsTag(rank, nRanks, peer, 6);
      bootstrapComm->sendMemory(ctx.localDeviceFlagMemory, peer, flagTag);
      flagFutures.push_back(bootstrapComm->recvMemory(peer, flagTag));
    }
  }
  for (auto& future : memoryFutures) {
    ctx.remoteScratchMemories.push_back(future.get());
  }
  for (auto const& memory : ctx.remoteScratchMemories) {
    ctx.remoteScratchPtrs.push_back(static_cast<char*>(memory.data()));
  }
  if (wantDeviceFlags) {
    for (auto& future : flagFutures) {
      ctx.remoteDeviceFlagMemories.push_back(future.get());
    }
    for (auto const& memory : ctx.remoteDeviceFlagMemories) {
      ctx.remoteDeviceFlagPtrs.push_back(static_cast<char*>(memory.data()));
    }
  }

  ctx.localScratchBuffer = scratchBuffer;
  ctx.localScratchBufferSize = scratchBufferSize;
  ctx.localScratchIpcReady = true;
}

int localPeerIndex(int localRank, int peerLocalRank) {
  return peerLocalRank < localRank ? peerLocalRank : peerLocalRank - 1;
}

void ensureLocalScratchEvents(RsContext& ctx,
                              std::shared_ptr<Communicator> bootstrapComm,
                              int rank, int nRanks, int nRanksPerNode) {
  if (ctx.localScratchEventIpcReady) return;

  int slots = ctx.pipelineSlots;
  while (ctx.localCopyEvents.size() < static_cast<size_t>(slots)) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(
        &event, cudaEventDisableTiming | cudaEventInterprocess));
    ctx.localCopyEvents.push_back(event);
  }
  while (ctx.localCrossEvents.size() < static_cast<size_t>(slots)) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(
        &event, cudaEventDisableTiming | cudaEventInterprocess));
    ctx.localCrossEvents.push_back(event);
  }

  int localBase = (rank / nRanksPerNode) * nRanksPerNode;
  int localRank = rank % nRanksPerNode;
  int peerCount = nRanksPerNode - 1;
  ctx.remoteCopyEvents.assign(peerCount, {});
  ctx.remoteCrossEvents.assign(peerCount, {});
  std::vector<char> localPayload(
      static_cast<size_t>(2 * slots) * sizeof(cudaIpcEventHandle_t));
  auto* localHandles =
      reinterpret_cast<cudaIpcEventHandle_t*>(localPayload.data());
  for (int slot = 0; slot < slots; ++slot) {
    MSCCLPP_CUDATHROW(
        cudaIpcGetEventHandle(&localHandles[slot], ctx.localCopyEvents[slot]));
    MSCCLPP_CUDATHROW(cudaIpcGetEventHandle(
        &localHandles[slots + slot], ctx.localCrossEvents[slot]));
  }

  for (int peerLocal = 0; peerLocal < nRanksPerNode; ++peerLocal) {
    int peer = localBase + peerLocal;
    if (peer == rank) continue;
    int peerIdx = localPeerIndex(localRank, peerLocal);
    int tag = rsTag(rank, nRanks, peer, 5);
    bootstrapComm->bootstrap()->send(localPayload, peer, tag);
    std::vector<char> peerPayload;
    bootstrapComm->bootstrap()->recv(peerPayload, peer, tag);
    auto const* peerHandles =
        reinterpret_cast<cudaIpcEventHandle_t const*>(peerPayload.data());
    ctx.remoteCopyEvents[peerIdx].resize(slots);
    ctx.remoteCrossEvents[peerIdx].resize(slots);
    for (int slot = 0; slot < slots; ++slot) {
      MSCCLPP_CUDATHROW(cudaIpcOpenEventHandle(
          &ctx.remoteCopyEvents[peerIdx][slot], peerHandles[slot]));
      MSCCLPP_CUDATHROW(cudaIpcOpenEventHandle(
          &ctx.remoteCrossEvents[peerIdx][slot], peerHandles[slots + slot]));
    }
  }
  ctx.localScratchEventIpcReady = true;
}

void ensureLocalParallelCopyResources(RsContext& ctx) {
  while (ctx.localCopyStartEvents.size() <
         static_cast<size_t>(ctx.pipelineSlots)) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ctx.localCopyStartEvents.push_back(event);
  }
  while (ctx.localCopyDoneEvents.size() <
         static_cast<size_t>(ctx.nRanksPerNode)) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ctx.localCopyDoneEvents.push_back(event);
  }
  while (ctx.localCopyStreams.size() <
         static_cast<size_t>(ctx.nRanksPerNode)) {
    cudaStream_t copyStream = nullptr;
    MSCCLPP_CUDATHROW(
        cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
    ctx.localCopyStreams.push_back(copyStream);
  }
}

void ensurePipelineResources(RsContext& ctx) {
  if (ctx.d2hStream == nullptr) {
    MSCCLPP_CUDATHROW(
        cudaStreamCreateWithFlags(&ctx.d2hStream, cudaStreamNonBlocking));
  }

  if (ctx.h2dStream == nullptr) {
    MSCCLPP_CUDATHROW(
        cudaStreamCreateWithFlags(&ctx.h2dStream, cudaStreamNonBlocking));
  }
  while (ctx.reduceDoneEvents.size() <
         static_cast<size_t>(ctx.pipelineSlots)) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ctx.reduceDoneEvents.push_back(event);
  }
  while (ctx.d2hDoneEvents.size() <
         static_cast<size_t>(ctx.pipelineSlots)) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ctx.d2hDoneEvents.push_back(event);
  }
  while (ctx.h2dDoneEvents.size() <
         static_cast<size_t>(ctx.pipelineSlots)) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ctx.h2dDoneEvents.push_back(event);
  }
  while (ctx.slotDoneEvents.size() <
         static_cast<size_t>(ctx.pipelineSlots)) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ctx.slotDoneEvents.push_back(event);
  }
}

ncclResult_t runLocalFourRankRingReduceScatter(
    RsContext& ctx, void const* sendbuff, void* recvbuff, size_t recvcount,
    size_t bytesPerRank, cudaStream_t stream,
    std::shared_ptr<Communicator> bootstrapComm, void* scratchBuffer,
    size_t scratchBufferSize) {
  ensureLocalScratchIpc(ctx, bootstrapComm, ctx.rank, ctx.worldSize,
                        ctx.nRanksPerNode, scratchBuffer, scratchBufferSize);
  bool ringEvents = useLocalFourRingEvents();
  if (ringEvents) {
    ensureLocalScratchEvents(ctx, bootstrapComm, ctx.rank, ctx.worldSize,
                             ctx.nRanksPerNode);
  }
  size_t rowStrideBytes = (bytesPerRank + sizeof(float) - 1) / sizeof(float) *
                          sizeof(float);
  size_t slotBytes = 2 * rowStrideBytes;
  if (slotBytes == 0 || slotBytes > scratchBufferSize) {
    return ncclInvalidUsage;
  }
  size_t slotCount = std::min<size_t>(scratchBufferSize / slotBytes, 1024);
  if (ringEvents) {
    slotCount = std::min<size_t>(
        slotCount, static_cast<size_t>(ctx.pipelineSlots) /
                       static_cast<size_t>(ctx.nRanksPerNode - 1));
  }
  if (slotCount == 0) return ncclInvalidUsage;

  uint64_t epoch = ++ctx.epoch;
  size_t slot = static_cast<size_t>((epoch - 1) % slotCount);
  if (epoch > slotCount) {
    uint64_t reuseEpoch = epoch - slotCount;
    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->localCrossReady[i], reuseEpoch);
    }
    if (ringEvents) {
      for (int peerLocal = 0; peerLocal < ctx.nRanksPerNode; ++peerLocal) {
        if (peerLocal == ctx.localRank) continue;
        int peerIdx = localPeerIndex(ctx.localRank, peerLocal);
        MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
            stream, ctx.remoteCrossEvents[peerIdx][slot], 0));
      }
    }
  }

  int const n = ctx.nRanksPerNode;
  int const nextLocal = (ctx.localRank + 1) % n;
  int const prevLocal = (ctx.localRank + n - 1) % n;
  int const nextIdx = localPeerIndex(ctx.localRank, nextLocal);
  int const prevIdx = localPeerIndex(ctx.localRank, prevLocal);
  char const* send = static_cast<char const*>(sendbuff);
  char* scratch = static_cast<char*>(scratchBuffer);
  size_t slotBase = slot * slotBytes;
  uint64_t signalBase = epoch * 8;

  for (int step = 0; step < n - 1; ++step) {
    int sendChunk = step == 0 ? (ctx.localRank + n - 1) % n
                              : (ctx.localRank - step - 1 + n) % n;
    char const* src =
        step == 0 ? send + static_cast<size_t>(sendChunk) * bytesPerRank
                  : scratch + slotBase + static_cast<size_t>((step - 1) & 1) *
                                        rowStrideBytes;
    char* dst = ctx.remoteScratchPtrs[nextIdx] + slotBase +
                static_cast<size_t>(step & 1) * rowStrideBytes;
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, bytesPerRank,
                                      cudaMemcpyDeviceToDevice, stream));
    size_t eventIndex = 0;
    if (ringEvents) {
      eventIndex = slot * static_cast<size_t>(n - 1) +
                   static_cast<size_t>(step);
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCopyEvents[eventIndex],
                                        stream));
    } else {
      waitForCudaStream(stream);
    }

    uint64_t signal = signalBase + static_cast<uint64_t>(step + 1);
    ctx.ctrl->localCopyReady[ctx.localRank].store(
        signal, std::memory_order_release);
    waitForEpoch(ctx.ctrl->localCopyReady[prevLocal], signal);
    if (ringEvents) {
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
          stream, ctx.remoteCopyEvents[prevIdx][eventIndex], 0));
    }

    int recvChunk = (ctx.localRank - step - 2 + n) % n;
    char* incoming = scratch + slotBase +
                     static_cast<size_t>(step & 1) * rowStrideBytes;
    char const* local =
        send + static_cast<size_t>(recvChunk) * bytesPerRank;
    void* out = step == n - 2 ? recvbuff : incoming;
    ncclResult_t result = launchAdd(local, incoming, out, recvcount, stream);
    if (result != ncclSuccess) return result;
  }

  if (ringEvents) {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCrossEvents[slot], stream));
  } else {
    waitForCudaStream(stream);
  }
  ctx.ctrl->localCrossReady[ctx.localRank].store(epoch,
                                                 std::memory_order_release);
  return ncclSuccess;
}

ncclResult_t runP2pRingReduceScatter(void const* sendbuff, void* recvbuff,
                                     size_t recvcount, size_t bytesPerRank,
                                     ncclDataType_t datatype, ncclComm_t comm,
                                     cudaStream_t stream, int rank, int nRanks,
                                     void* scratchBuffer,
                                     size_t scratchBufferSize) {
  if (datatype != ncclFloat32) return ncclInvalidUsage;
  size_t slotBytes = bytesPerRank;
  if (slotBytes == 0 || scratchBufferSize < 2 * slotBytes) {
    return ncclInvalidUsage;
  }
  int next = (rank + 1) % nRanks;
  int prev = (rank + nRanks - 1) % nRanks;
  auto const* send = static_cast<char const*>(sendbuff);
  auto* scratch = static_cast<char*>(scratchBuffer);

  for (int step = 0; step < nRanks - 1; ++step) {
    int sendChunk = step == 0 ? (rank + nRanks - 1) % nRanks
                              : (rank - step - 1 + nRanks) % nRanks;
    char const* src =
        step == 0 ? send + static_cast<size_t>(sendChunk) * bytesPerRank
                  : scratch + static_cast<size_t>((step - 1) & 1) * slotBytes;
    char* dst = scratch + static_cast<size_t>(step & 1) * slotBytes;

    ncclResult_t result = ncclGroupStart();
    if (result != ncclSuccess) return result;
    ncclResult_t enqueueResult =
        ncclSend(src, recvcount, datatype, next, comm, stream);
    if (enqueueResult == ncclSuccess) {
      enqueueResult = ncclRecv(dst, recvcount, datatype, prev, comm, stream);
    }
    result = ncclGroupEnd();
    if (enqueueResult != ncclSuccess) return enqueueResult;
    if (result != ncclSuccess) return result;

    int recvChunk = (rank - step - 2 + nRanks) % nRanks;
    char const* local =
        send + static_cast<size_t>(recvChunk) * bytesPerRank;
    void* out = step == nRanks - 2 ? recvbuff : dst;
    result = launchAdd(local, dst, out, recvcount, stream);
    if (result != ncclSuccess) return result;
  }
  return ncclSuccess;
}

ncclResult_t runLocalFourRankReduceScatter(
    RsContext& ctx, void const* sendbuff, void* recvbuff, size_t recvcount,
    size_t bytesPerRank, cudaStream_t stream,
    std::shared_ptr<Communicator> bootstrapComm, void* scratchBuffer,
    size_t scratchBufferSize) {
  if (useLocalFourRingFor(bytesPerRank)) {
    return runLocalFourRankRingReduceScatter(
        ctx, sendbuff, recvbuff, recvcount, bytesPerRank, stream,
        bootstrapComm, scratchBuffer, scratchBufferSize);
  }
  if (scratchBuffer == nullptr || scratchBufferSize == 0) {
    return ncclInvalidArgument;
  }
  ensureLocalScratchIpc(ctx, bootstrapComm, ctx.rank, ctx.worldSize,
                        ctx.nRanksPerNode, scratchBuffer, scratchBufferSize);
  ensureLocalScratchEvents(ctx, bootstrapComm, ctx.rank, ctx.worldSize,
                           ctx.nRanksPerNode);

  size_t rowStrideBytes = (bytesPerRank + sizeof(float) - 1) / sizeof(float) *
                          sizeof(float);
  size_t flagBytes = 256;
  size_t slotBytes =
      flagBytes + static_cast<size_t>(ctx.nRanksPerNode) * rowStrideBytes;
  if (slotBytes == 0 || slotBytes > scratchBufferSize) {
    return ncclInvalidUsage;
  }
  size_t slotCount = scratchBufferSize / slotBytes;
  slotCount = std::min<size_t>(slotCount, 1024);
  if (slotCount == 0) return ncclInvalidUsage;
  bool rowStrideChanged = ctx.localFourRowStrideBytes != 0 &&
                          ctx.localFourRowStrideBytes != rowStrideBytes;
  if (rowStrideChanged) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    bootstrapComm->bootstrap()->barrier();
  }
  ctx.localFourRowStrideBytes = rowStrideBytes;

  uint64_t epoch = ++ctx.epoch;
  size_t slot = static_cast<size_t>((epoch - 1) % slotCount);
  int eventSlot = static_cast<int>(slot);
  bool useGpuFlags = ctx.ctrlDeviceSlab != nullptr && useLocalFourGpuFlags();
  bool useDeviceFlagScatter =
      !useGpuFlags && slotCount >= 16 && bytesPerRank <= 32 * 1024;
  bool useDeviceFlags = useDeviceFlagScatter;
  bool skipSelfCopy = !useDeviceFlags && !useGpuFlags &&
                      bytesPerRank <= 32 * 1024;
  bool useParallelCopies = !useDeviceFlags && !useGpuFlags && !skipSelfCopy &&
                           bytesPerRank >= 64 * 1024 &&
                           useLocalFourParallelCopies();
  if (useParallelCopies) ensureLocalParallelCopyResources(ctx);
  if (epoch > slotCount && !rowStrideChanged) {
    uint64_t reuseEpoch = epoch - slotCount;
    if (useDeviceFlags) {
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      bootstrapComm->bootstrap()->barrier();
    } else {
      for (int i = 0; i < ctx.nRanksPerNode; ++i) {
        waitForEpoch(ctx.ctrl->localCrossReady[i], reuseEpoch);
      }
      if (!useGpuFlags) {
        for (int peerLocal = 0; peerLocal < ctx.nRanksPerNode; ++peerLocal) {
          if (peerLocal == ctx.localRank) continue;
          int peerIdx = localPeerIndex(ctx.localRank, peerLocal);
          MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
              stream, ctx.remoteCrossEvents[peerIdx][eventSlot], 0));
        }
      }
    }
  }

  auto const* send = static_cast<char const*>(sendbuff);
  auto* scratch = static_cast<char*>(scratchBuffer);
  size_t slotBase = slot * slotBytes;
  size_t dataBase = slotBase + flagBytes;
  unsigned long long* flagPtrs[4] = {};
  char* dstPtrs[4] = {};
  if (useParallelCopies) {
    MSCCLPP_CUDATHROW(
        cudaEventRecord(ctx.localCopyStartEvents[eventSlot], stream));
  }
  for (int targetLocal = 0; targetLocal < ctx.nRanksPerNode; ++targetLocal) {
    size_t sourceOffset = static_cast<size_t>(targetLocal) * bytesPerRank;
    char* targetScratch = nullptr;
    if (targetLocal == ctx.localRank) {
      targetScratch = scratch;
    } else {
      int peerIdx = localPeerIndex(ctx.localRank, targetLocal);
      targetScratch = ctx.remoteScratchPtrs[peerIdx];
    }
    char* dst = targetScratch + dataBase +
                static_cast<size_t>(ctx.localRank) * rowStrideBytes;
    dstPtrs[targetLocal] = dst;
    flagPtrs[targetLocal] = reinterpret_cast<unsigned long long*>(
        targetScratch + slotBase +
        static_cast<size_t>(ctx.localRank) * sizeof(unsigned long long));
    if (!useDeviceFlagScatter &&
        !(skipSelfCopy && targetLocal == ctx.localRank)) {
      cudaStream_t copyStream = stream;
      if (useParallelCopies) {
        copyStream = ctx.localCopyStreams[static_cast<size_t>(targetLocal)];
        MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
            copyStream, ctx.localCopyStartEvents[eventSlot], 0));
      }
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, send + sourceOffset, bytesPerRank,
                                        cudaMemcpyDeviceToDevice, copyStream));
      if (useParallelCopies) {
        MSCCLPP_CUDATHROW(cudaEventRecord(
            ctx.localCopyDoneEvents[static_cast<size_t>(targetLocal)],
            copyStream));
      }
    }
  }
  if (useParallelCopies) {
    for (int targetLocal = 0; targetLocal < ctx.nRanksPerNode; ++targetLocal) {
      if (skipSelfCopy && targetLocal == ctx.localRank) continue;
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
          stream, ctx.localCopyDoneEvents[static_cast<size_t>(targetLocal)],
          0));
    }
  }

  if (useDeviceFlags) {
    ncclResult_t result = launchLocalFourRankTinyScatter(
        sendbuff, dstPtrs[0], dstPtrs[1], dstPtrs[2], dstPtrs[3], flagPtrs[0],
        flagPtrs[1], flagPtrs[2], flagPtrs[3], recvcount, epoch, stream);
    if (result != ncclSuccess) return result;
    result = launchLocalFourRankDeviceFlagReduce(
        reinterpret_cast<unsigned long long volatile*>(scratch + slotBase),
        scratch + dataBase, recvbuff, recvcount, rowStrideBytes, epoch, stream);
    if (result != ncclSuccess) return result;
    ctx.ctrl->localCrossReady[ctx.localRank].store(epoch,
                                                   std::memory_order_release);
    return ncclSuccess;
  }

  if (useGpuFlags) {
    ncclResult_t result = launchLocalFourRankScratchWaitReduce(
        ctx.ctrlDeviceSlab, offsetof(RsControl, localCopyReady),
        offsetof(RsControl, localCrossReady), scratch + dataBase, recvbuff,
        recvcount, rowStrideBytes, ctx.localRank, epoch, stream);
    if (result != ncclSuccess) return result;
    return ncclSuccess;
  } else {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCopyEvents[eventSlot], stream));
    ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->localCopyReady[i], epoch);
    }
    for (int peerLocal = 0; peerLocal < ctx.nRanksPerNode; ++peerLocal) {
      if (peerLocal == ctx.localRank) continue;
      int peerIdx = localPeerIndex(ctx.localRank, peerLocal);
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
          stream, ctx.remoteCopyEvents[peerIdx][eventSlot], 0));
    }

    ncclResult_t result =
        skipSelfCopy
            ? launchLocalFourRankScratchReduceSelf(
                  scratch + dataBase,
                  send + static_cast<size_t>(ctx.localRank) * bytesPerRank,
                  recvbuff, recvcount, rowStrideBytes, ctx.localRank, stream)
            : launchLocalFourRankScratchReduce(
                  scratch + dataBase, recvbuff, recvcount, rowStrideBytes,
                  stream);
    if (result != ncclSuccess) return result;

    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCrossEvents[eventSlot], stream));
    ctx.ctrl->localCrossReady[ctx.localRank].store(epoch,
                                                   std::memory_order_release);
  }
  return ncclSuccess;
}

size_t ctrlArrayOffset(size_t baseOffset, int localRank) {
  return baseOffset + static_cast<size_t>(localRank) *
                          sizeof(std::atomic<uint64_t>);
}

void pollSmallQp(RsContext& ctx) {
  if (!ctx.smallQp) return;
  while (ctx.smallQp->getNumSendCqItems() > 0) {
    int wcNum = ctx.smallQp->pollSendCq();
    if (wcNum < 0) {
      throw mscclpp::Error("ReduceScatter small-QP pollSendCq failed",
                           mscclpp::ErrorCode::SystemError);
    }
    for (int i = 0; i < wcNum; ++i) {
      if (ctx.smallQp->getSendWcStatus(i) != 0) {
        throw mscclpp::Error("ReduceScatter small-QP RDMA write failed: " +
                                 ctx.smallQp->getSendWcStatusString(i),
                             mscclpp::ErrorCode::SystemError);
      }
    }
  }
}

void postSmallDataAndSignal(RsContext& ctx, size_t remoteDataOffset,
                            size_t localDataOffset, size_t bytes,
                            size_t remoteSignalOffset,
                            size_t localSignalOffset) {
  if (usePairConnectionWrite() || !ctx.smallQp || ctx.smallRecvMr == nullptr ||
      ctx.smallCtrlMr == nullptr) {
    ctx.pairConnection.write(ctx.remoteSmallRecvMemory, remoteDataOffset,
                             ctx.smallRecvMemory, localDataOffset, bytes);
    ctx.pairConnection.write(ctx.remoteCtrlMemory, remoteSignalOffset,
                             ctx.ctrlMemory, localSignalOffset,
                             sizeof(uint64_t));
    ctx.pairConnection.flush();
    return;
  }
  bool signaled = (++ctx.smallWrCount % kSmallSignalEveryN) == 0;
  ctx.smallQp->stageSendWrite(ctx.smallRecvMr, ctx.smallRemoteRecvMrInfo,
                              static_cast<uint32_t>(bytes), /*wrId=*/0,
                              localDataOffset, remoteDataOffset, false);
  ctx.smallQp->stageSendWrite(ctx.smallCtrlMr, ctx.smallRemoteCtrlMrInfo,
                              sizeof(uint64_t), /*wrId=*/0, localSignalOffset,
                              remoteSignalOffset, signaled);
  ctx.smallQp->postSend();
  if (signaled) pollSmallQp(ctx);
}

void postPairDataAndSignal(RsContext& ctx, size_t remoteDataOffset,
                           size_t localDataOffset, size_t bytes,
                           size_t remoteSignalOffset,
                           size_t localSignalOffset) {
  if (usePairConnectionWrite() || !ctx.smallQp || ctx.sendMr == nullptr ||
      ctx.smallCtrlMr == nullptr) {
    ctx.pairConnection.write(ctx.remoteRecvMemory, remoteDataOffset,
                             ctx.sendMemory, localDataOffset, bytes);
    ctx.pairConnection.write(ctx.remoteCtrlMemory, remoteSignalOffset,
                             ctx.ctrlMemory, localSignalOffset,
                             sizeof(uint64_t));
    ctx.pairConnection.flush();
    return;
  }
  bool signaled = (++ctx.smallWrCount % kPairSignalEveryN) == 0;
  ctx.smallQp->stageSendWrite(ctx.sendMr, ctx.remoteRecvMrInfo,
                              static_cast<uint32_t>(bytes), /*wrId=*/0,
                              localDataOffset, remoteDataOffset, false);
  ctx.smallQp->stageSendWrite(ctx.smallCtrlMr, ctx.smallRemoteCtrlMrInfo,
                              sizeof(uint64_t), /*wrId=*/0, localSignalOffset,
                              remoteSignalOffset, signaled);
  ctx.smallQp->postSend();
  if (signaled) pollSmallQp(ctx);
}

void postSmallSignal(RsContext& ctx, size_t remoteSignalOffset,
                     size_t localSignalOffset,
                     int signalEvery = kSmallSignalEveryN) {
  if (!ctx.smallQp || ctx.smallCtrlMr == nullptr) {
    ctx.pairConnection.write(ctx.remoteCtrlMemory, remoteSignalOffset,
                             ctx.ctrlMemory, localSignalOffset,
                             sizeof(uint64_t));
    ctx.pairConnection.flush();
    return;
  }
  bool signaled = (++ctx.smallWrCount % signalEvery) == 0;
  ctx.smallQp->stageSendWrite(ctx.smallCtrlMr, ctx.smallRemoteCtrlMrInfo,
                              sizeof(uint64_t), /*wrId=*/0, localSignalOffset,
                              remoteSignalOffset, signaled);
  ctx.smallQp->postSend();
  if (signaled) pollSmallQp(ctx);
}

void postSmallAck(RsContext& ctx, uint64_t epoch) {
  if (epoch == 0 || epoch <= ctx.smallAckPostedEpoch) return;
  size_t ackReadyOffset =
      ctrlArrayOffset(offsetof(RsControl, pairAckReady), ctx.localRank);
  size_t ackSignalOffset =
      ctrlArrayOffset(offsetof(RsControl, pairAckSignal), ctx.localRank);
  ctx.ctrl->pairAckSignal[ctx.localRank].store(epoch,
                                               std::memory_order_release);
  postSmallSignal(ctx, ackReadyOffset, ackSignalOffset, kPairSignalEveryN);
  ctx.smallAckPostedEpoch = epoch;
}

void flushSmallPendingAck(RsContext& ctx) {
  postSmallAck(ctx, ctx.smallPendingAckEpoch);
}

void reduceSmallPartialsScalar(float const* rank0, float const* rank1,
                               float const* rank2, float const* rank3,
                               size_t localElemOffset,
                               size_t remoteElemOffset, float* localPartial,
                               float* remotePartial, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    localPartial[i] = rank0[localElemOffset + i] +
                      rank1[localElemOffset + i] +
                      rank2[localElemOffset + i] +
                      rank3[localElemOffset + i];
    remotePartial[i] = rank0[remoteElemOffset + i] +
                       rank1[remoteElemOffset + i] +
                       rank2[remoteElemOffset + i] +
                       rank3[remoteElemOffset + i];
  }
}

void addSmallPartialsScalar(float* output, float const* remoteIncoming,
                            size_t count) {
  for (size_t i = 0; i < count; ++i) {
    output[i] += remoteIncoming[i];
  }
}

void reduceSmallPartialsTwoLocalScalar(float const* rank0, float const* rank1,
                                       size_t localElemOffset,
                                       size_t remoteElemOffset,
                                       float* localPartial,
                                       float* remotePartial, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    localPartial[i] =
        rank0[localElemOffset + i] + rank1[localElemOffset + i];
    remotePartial[i] =
        rank0[remoteElemOffset + i] + rank1[remoteElemOffset + i];
  }
}

#if defined(__x86_64__) && defined(__GNUC__)
using Avx512Float __attribute__((vector_size(64), aligned(1))) = float;

__attribute__((target("avx512f")))
void reduceSmallPartialsAvx512(float const* rank0, float const* rank1,
                               float const* rank2, float const* rank3,
                               size_t localElemOffset,
                               size_t remoteElemOffset, float* localPartial,
                               float* remotePartial, size_t count) {
  size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    auto l0 = *reinterpret_cast<Avx512Float const*>(rank0 + localElemOffset + i);
    auto l1 = *reinterpret_cast<Avx512Float const*>(rank1 + localElemOffset + i);
    auto l2 = *reinterpret_cast<Avx512Float const*>(rank2 + localElemOffset + i);
    auto l3 = *reinterpret_cast<Avx512Float const*>(rank3 + localElemOffset + i);
    auto r0 = *reinterpret_cast<Avx512Float const*>(rank0 + remoteElemOffset + i);
    auto r1 = *reinterpret_cast<Avx512Float const*>(rank1 + remoteElemOffset + i);
    auto r2 = *reinterpret_cast<Avx512Float const*>(rank2 + remoteElemOffset + i);
    auto r3 = *reinterpret_cast<Avx512Float const*>(rank3 + remoteElemOffset + i);
    *reinterpret_cast<Avx512Float*>(localPartial + i) =
        (l0 + l1) + (l2 + l3);
    *reinterpret_cast<Avx512Float*>(remotePartial + i) =
        (r0 + r1) + (r2 + r3);
  }
  reduceSmallPartialsScalar(rank0, rank1, rank2, rank3, localElemOffset + i,
                            remoteElemOffset + i, localPartial + i,
                            remotePartial + i, count - i);
}

__attribute__((target("avx512f")))
void addSmallPartialsAvx512(float* output, float const* remoteIncoming,
                            size_t count) {
  size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    auto lhs = *reinterpret_cast<Avx512Float const*>(output + i);
    auto rhs = *reinterpret_cast<Avx512Float const*>(remoteIncoming + i);
    *reinterpret_cast<Avx512Float*>(output + i) = lhs + rhs;
  }
  addSmallPartialsScalar(output + i, remoteIncoming + i, count - i);
}

__attribute__((target("avx512f")))
void reduceSmallPartialsTwoLocalAvx512(float const* rank0, float const* rank1,
                                       size_t localElemOffset,
                                       size_t remoteElemOffset,
                                       float* localPartial,
                                       float* remotePartial, size_t count) {
  size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    auto l0 = *reinterpret_cast<Avx512Float const*>(rank0 + localElemOffset + i);
    auto l1 = *reinterpret_cast<Avx512Float const*>(rank1 + localElemOffset + i);
    auto r0 = *reinterpret_cast<Avx512Float const*>(rank0 + remoteElemOffset + i);
    auto r1 = *reinterpret_cast<Avx512Float const*>(rank1 + remoteElemOffset + i);
    *reinterpret_cast<Avx512Float*>(localPartial + i) = l0 + l1;
    *reinterpret_cast<Avx512Float*>(remotePartial + i) = r0 + r1;
  }
  reduceSmallPartialsTwoLocalScalar(
      rank0, rank1, localElemOffset + i, remoteElemOffset + i,
      localPartial + i, remotePartial + i, count - i);
}

bool hasAvx512F() {
  static bool supported = [] {
    char const* env = std::getenv("MSCCLPP_NCCL_RS_DISABLE_AVX512");
    if (env != nullptr && env[0] != '\0' && env[0] != '0') return false;
    return static_cast<bool>(__builtin_cpu_supports("avx512f"));
  }();
  return supported;
}
#endif

void reduceSmallPartials(float const* rank0, float const* rank1,
                         float const* rank2, float const* rank3,
                         size_t localElemOffset, size_t remoteElemOffset,
                         float* localPartial, float* remotePartial,
                         size_t count) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (hasAvx512F()) {
    reduceSmallPartialsAvx512(rank0, rank1, rank2, rank3, localElemOffset,
                              remoteElemOffset, localPartial, remotePartial,
                              count);
    return;
  }
#endif
  reduceSmallPartialsScalar(rank0, rank1, rank2, rank3, localElemOffset,
                            remoteElemOffset, localPartial, remotePartial,
                            count);
}

void addSmallPartials(float* output, float const* remoteIncoming,
                      size_t count) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (count >= 2048 && hasAvx512F()) {
    addSmallPartialsAvx512(output, remoteIncoming, count);
    return;
  }
#endif
  addSmallPartialsScalar(output, remoteIncoming, count);
}

void reduceSmallPartialsTwoLocal(float const* rank0, float const* rank1,
                                 size_t localElemOffset,
                                 size_t remoteElemOffset, float* localPartial,
                                 float* remotePartial, size_t count) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (hasAvx512F()) {
    reduceSmallPartialsTwoLocalAvx512(
        rank0, rank1, localElemOffset, remoteElemOffset, localPartial,
        remotePartial, count);
    return;
  }
#endif
  reduceSmallPartialsTwoLocalScalar(rank0, rank1, localElemOffset,
                                    remoteElemOffset, localPartial,
                                    remotePartial, count);
}

ncclResult_t runSmallHostReduceScatter(RsContext& ctx, void const* sendbuff,
                                       void* recvbuff, size_t recvcount,
                                       size_t bytesPerRank,
                                       cudaStream_t stream,
                                       std::shared_ptr<Communicator>
                                           bootstrapComm) {
  size_t fullBytes = bytesPerRank * static_cast<size_t>(ctx.worldSize);
  size_t smallRankStride =
      ctx.chunkCapacity * static_cast<size_t>(ctx.pipelineSlots);
  if (fullBytes >= configuredSmallHostFullBytes() ||
      fullBytes > smallRankStride ||
      3 * bytesPerRank > smallRankStride) {
    return ncclInvalidUsage;
  }
  size_t slotBytes = std::max(fullBytes, 3 * bytesPerRank);
  size_t slotCount = smallRankStride / slotBytes;
  if (slotCount < 1) return ncclInvalidUsage;
  slotCount = std::min<size_t>(slotCount, 1024);
  if (ctx.smallSlotBytes != 0 && ctx.smallSlotBytes != slotBytes) {
    waitForCudaStream(stream);
    flushSmallPendingAck(ctx);
    pollSmallQp(ctx);
    bootstrapComm->bootstrap()->barrier();
    ctx.smallOpCount = 0;
    ctx.smallSlotEpochs.assign(slotCount, 0);
  }
  ctx.smallSlotBytes = slotBytes;
  if (ctx.smallSlotEpochs.size() != slotCount) {
    ctx.smallOpCount = 0;
    ctx.smallSlotEpochs.assign(slotCount, 0);
  }

  uint64_t epoch = ++ctx.epoch;
  uint64_t smallSeq = ++ctx.smallOpCount;
  size_t slot = static_cast<size_t>((smallSeq - 1) % slotCount);
  uint64_t reuseEpoch = ctx.smallSlotEpochs[slot];
  if (reuseEpoch != 0) {
    if (slotCount == 1 && ctx.smallPendingAckEpoch > ctx.smallAckPostedEpoch) {
      waitForCudaStream(stream);
      flushSmallPendingAck(ctx);
      pollSmallQp(ctx);
    }
    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->smallLocalDone[i], reuseEpoch);
    }
    waitForEpoch(ctx.ctrl->pairAckReady[ctx.localRank], reuseEpoch);
    pollSmallQp(ctx);
  }
  ctx.smallSlotEpochs[slot] = epoch;

  size_t slotOffset = slot * slotBytes;
  size_t inputOffset =
      static_cast<size_t>(ctx.localRank) * smallRankStride + slotOffset;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(ctx.smallSendSlab + inputOffset, sendbuff,
                                    fullBytes, cudaMemcpyDeviceToHost,
                                    stream));
  waitForCudaStream(stream);
  flushSmallPendingAck(ctx);
  ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                std::memory_order_release);

  size_t readyOffset =
      ctrlArrayOffset(offsetof(RsControl, pairRdmaReady), ctx.localRank);
  size_t signalOffset =
      ctrlArrayOffset(offsetof(RsControl, pairRdmaSignal), ctx.localRank);
  int localBase = ctx.nodeId * ctx.nRanksPerNode;
  int remoteBase = (1 - ctx.nodeId) * ctx.nRanksPerNode;

  for (int i = 0; i < ctx.nRanksPerNode; ++i) {
    waitForEpoch(ctx.ctrl->localCopyReady[i], epoch);
  }

  size_t recvSlotOffset =
      static_cast<size_t>(ctx.localRank) * smallRankStride + slotOffset;
  auto* localPartial =
      reinterpret_cast<float*>(ctx.smallRecvSlab + recvSlotOffset);
  auto* remotePartial =
      reinterpret_cast<float*>(ctx.smallRecvSlab + recvSlotOffset +
                               bytesPerRank);
  auto const* rank0 =
      reinterpret_cast<float const*>(ctx.smallSendSlab + slotOffset);
  auto const* rank1 =
      reinterpret_cast<float const*>(ctx.smallSendSlab + smallRankStride +
                                     slotOffset);
  size_t elemsPerRank = recvcount;
  size_t localElemOffset =
      static_cast<size_t>(localBase + ctx.localRank) * elemsPerRank;
  size_t remoteElemOffset =
      static_cast<size_t>(remoteBase + ctx.localRank) * elemsPerRank;
  size_t remoteIncomingOffset = recvSlotOffset + 2 * bytesPerRank;
  if (ctx.nRanksPerNode == 2) {
    reduceSmallPartialsTwoLocal(rank0, rank1, localElemOffset,
                                remoteElemOffset, localPartial, remotePartial,
                                elemsPerRank);
  } else {
    auto const* rank2 =
        reinterpret_cast<float const*>(ctx.smallSendSlab + 2 * smallRankStride +
                                       slotOffset);
    auto const* rank3 =
        reinterpret_cast<float const*>(ctx.smallSendSlab + 3 * smallRankStride +
                                       slotOffset);
    reduceSmallPartials(rank0, rank1, rank2, rank3, localElemOffset,
                        remoteElemOffset, localPartial, remotePartial,
                        elemsPerRank);
  }
  ctx.ctrl->smallLocalDone[ctx.localRank].store(epoch,
                                                std::memory_order_release);

  std::atomic_thread_fence(std::memory_order_release);
  ctx.ctrl->pairRdmaSignal[ctx.localRank].store(epoch,
                                                std::memory_order_release);
  postSmallDataAndSignal(ctx, remoteIncomingOffset,
                         recvSlotOffset + bytesPerRank, bytesPerRank,
                         readyOffset, signalOffset);

  waitForEpoch(ctx.ctrl->pairRdmaReady[ctx.localRank], epoch);

  auto const* remoteIncoming =
      reinterpret_cast<float const*>(ctx.smallRecvSlab + remoteIncomingOffset);
  auto* output = localPartial;
  addSmallPartials(output, remoteIncoming, elemsPerRank);

  MSCCLPP_CUDATHROW(cudaMemcpyAsync(
      recvbuff, output, bytesPerRank,
      cudaMemcpyHostToDevice, stream));
  ctx.ctrl->localScratchDone[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
  ctx.smallPendingAckEpoch = epoch;

  return ncclSuccess;
}

ncclResult_t runTwoRankSmallHostReduceScatter(RsContext& ctx,
                                              void const* sendbuff,
                                              void* recvbuff,
                                              size_t recvcount,
                                              size_t bytesPerRank,
                                              cudaStream_t stream,
                                              std::shared_ptr<Communicator>
                                                  bootstrapComm) {
  size_t fullBytes = 2 * bytesPerRank;
  size_t smallRankStride =
      ctx.chunkCapacity * static_cast<size_t>(ctx.pipelineSlots);
  size_t slotBytes = std::max(fullBytes, 3 * bytesPerRank);
  if (bytesPerRank == 0 ||
      bytesPerRank > configuredTwoRankSmallHostBytes() ||
      slotBytes > smallRankStride) {
    return ncclInvalidUsage;
  }
  size_t slotCount = smallRankStride / slotBytes;
  if (slotCount < 1) return ncclInvalidUsage;
  slotCount = std::min<size_t>(slotCount, 1024);
  if (ctx.smallSlotBytes != 0 && ctx.smallSlotBytes != slotBytes) {
    waitForCudaStream(stream);
    flushSmallPendingAck(ctx);
    pollSmallQp(ctx);
    bootstrapComm->bootstrap()->barrier();
    ctx.smallOpCount = 0;
    ctx.smallSlotEpochs.assign(slotCount, 0);
  }
  ctx.smallSlotBytes = slotBytes;
  if (ctx.smallSlotEpochs.size() != slotCount) {
    ctx.smallOpCount = 0;
    ctx.smallSlotEpochs.assign(slotCount, 0);
  }

  uint64_t epoch = ++ctx.epoch;
  uint64_t smallSeq = ++ctx.smallOpCount;
  size_t slot = static_cast<size_t>((smallSeq - 1) % slotCount);
  uint64_t reuseEpoch = ctx.smallSlotEpochs[slot];
  if (reuseEpoch != 0) {
    if (slotCount == 1 && ctx.smallPendingAckEpoch > ctx.smallAckPostedEpoch) {
      waitForCudaStream(stream);
      flushSmallPendingAck(ctx);
      pollSmallQp(ctx);
    }
    waitForEpoch(ctx.ctrl->smallLocalDone[ctx.localRank], reuseEpoch);
    waitForEpoch(ctx.ctrl->pairAckReady[ctx.localRank], reuseEpoch);
    pollSmallQp(ctx);
  }
  ctx.smallSlotEpochs[slot] = epoch;

  size_t slotOffset = slot * slotBytes;
  size_t localOffset = static_cast<size_t>(ctx.rank) * bytesPerRank;
  size_t remoteOffset = static_cast<size_t>(1 - ctx.rank) * bytesPerRank;
  size_t readyOffset =
      ctrlArrayOffset(offsetof(RsControl, pairRdmaReady), ctx.localRank);
  size_t signalOffset =
      ctrlArrayOffset(offsetof(RsControl, pairRdmaSignal), ctx.localRank);
  size_t remoteIncomingOffset = slotOffset + 2 * bytesPerRank;
  if (useTwoRankMappedHost() && ctx.smallRecvDeviceSlab != nullptr) {
    char* recvSlotDevice = ctx.smallRecvDeviceSlab + slotOffset;
    ncclResult_t result = launchTwoRankStoreRemoteShard(
        sendbuff, recvSlotDevice, recvcount, recvcount, 1 - ctx.rank,
        remoteOffset, stream);
    if (result != ncclSuccess) return result;
    waitForCudaStream(stream);
    flushSmallPendingAck(ctx);
    ctx.ctrl->smallLocalDone[ctx.localRank].store(
        epoch, std::memory_order_release);

    std::atomic_thread_fence(std::memory_order_release);
    ctx.ctrl->pairRdmaSignal[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
    postSmallDataAndSignal(ctx, remoteIncomingOffset, slotOffset + remoteOffset,
                           bytesPerRank, readyOffset, signalOffset);

    waitForEpoch(ctx.ctrl->pairRdmaReady[ctx.localRank], epoch);

    result = launchTwoRankFinalizeMapped(
        sendbuff, ctx.smallRecvDeviceSlab + remoteIncomingOffset, recvbuff,
        recvcount, recvcount, ctx.rank, stream);
    if (result != ncclSuccess) return result;
    ctx.ctrl->localScratchDone[ctx.localRank].store(
        epoch, std::memory_order_release);
    ctx.smallPendingAckEpoch = epoch;
    return ncclSuccess;
  }

  auto* recvSlot = ctx.smallRecvSlab + slotOffset;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(recvSlot, sendbuff,
                                    fullBytes, cudaMemcpyDeviceToHost,
                                    stream));
  waitForCudaStream(stream);
  flushSmallPendingAck(ctx);

  ctx.ctrl->smallLocalDone[ctx.localRank].store(epoch,
                                                std::memory_order_release);

  std::atomic_thread_fence(std::memory_order_release);
  ctx.ctrl->pairRdmaSignal[ctx.localRank].store(epoch,
                                                std::memory_order_release);
  postSmallDataAndSignal(ctx, remoteIncomingOffset, slotOffset + remoteOffset,
                         bytesPerRank, readyOffset, signalOffset);

  waitForEpoch(ctx.ctrl->pairRdmaReady[ctx.localRank], epoch);

  auto* output = reinterpret_cast<float*>(recvSlot + localOffset);
  auto const* remoteIncoming =
      reinterpret_cast<float const*>(recvSlot + 2 * bytesPerRank);
  addSmallPartials(output, remoteIncoming, recvcount);

  MSCCLPP_CUDATHROW(cudaMemcpyAsync(recvbuff, output, bytesPerRank,
                                    cudaMemcpyHostToDevice, stream));
  ctx.ctrl->localScratchDone[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
  ctx.smallPendingAckEpoch = epoch;
  return ncclSuccess;
}

size_t hostRankStride(RsContext const& ctx) {
  return ctx.chunkCapacity * static_cast<size_t>(ctx.pipelineSlots);
}

void waitForScratchReuse(RsContext& ctx, uint64_t epoch) {
  if (useDirectTargetReduce()) {
    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->localScratchDone[i], epoch);
    }
    return;
  }
  int peers[3] = {ctx.localRank, ctx.localRank ^ 1,
                  ctx.localRank < 2 ? ctx.localRank + 2
                                    : ctx.localRank - 2};
  for (int i = 0; i < 3; ++i) {
    bool seen = false;
    for (int j = 0; j < i; ++j) seen = seen || peers[j] == peers[i];
    if (!seen) waitForEpoch(ctx.ctrl->localScratchDone[peers[i]], epoch);
  }
}

struct RsChunkWork {
  uint64_t epoch = 0;
  size_t slot = 0;
  size_t elemOffset = 0;
  size_t chunkElems = 0;
  size_t chunkBytes = 0;
  size_t pairSlotOffset = 0;
  char* localPartialGpu = nullptr;
  char* remotePartialGpu = nullptr;
  bool ackAfterSlotDone = false;
  bool localScratchDoneAfterSlotDone = false;
  bool ackSent = false;
  bool rdmaPosted = false;
  bool remoteCompleted = false;
};

ncclResult_t finishScheduledChunkLocal(RsContext& ctx, uint64_t epoch,
                                       size_t slot, size_t elemOffset,
                                       size_t chunkElems, size_t chunkBytes,
                                       char* localPartialGpu,
                                       char* remotePartialGpu,
                                       cudaStream_t stream,
                                       bool recordAsyncD2h,
                                       bool remotePartialInSendSlab,
                                       RsChunkWork* work) {
  size_t pairSlotOffset = static_cast<size_t>(ctx.localRank) *
                              hostRankStride(ctx) +
                          slot * ctx.chunkCapacity;
  if (recordAsyncD2h) {
    if (remotePartialInSendSlab) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.d2hDoneEvents[slot], stream));
    } else {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.reduceDoneEvents[slot], stream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.d2hStream,
                                           ctx.reduceDoneEvents[slot], 0));
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(ctx.sendSlab + pairSlotOffset,
                                       remotePartialGpu, chunkBytes,
                                       cudaMemcpyDeviceToHost, ctx.d2hStream));
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.d2hDoneEvents[slot],
                                       ctx.d2hStream));
    }
  } else {
    if (remotePartialInSendSlab) {
      waitForCudaStream(stream);
      ctx.ctrl->localScratchDone[ctx.localRank].store(epoch,
                                                     std::memory_order_release);
    } else {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(ctx.sendSlab + pairSlotOffset,
                                       remotePartialGpu, chunkBytes,
                                       cudaMemcpyDeviceToHost, stream));
      if (useCpuFinalAdd()) {
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(ctx.smallRecvSlab + pairSlotOffset,
                                         localPartialGpu, chunkBytes,
                                         cudaMemcpyDeviceToHost, stream));
      }
      waitForCudaStream(stream);
      ctx.ctrl->localScratchDone[ctx.localRank].store(
          epoch, std::memory_order_release);
    }
  }

  if (work != nullptr) {
    work->epoch = epoch;
    work->slot = slot;
    work->elemOffset = elemOffset;
    work->chunkElems = chunkElems;
    work->chunkBytes = chunkBytes;
    work->pairSlotOffset = pairSlotOffset;
    work->localPartialGpu = localPartialGpu;
    work->remotePartialGpu = remotePartialGpu;
  }
  return ncclSuccess;
}

ncclResult_t scheduleChunkLocal(RsContext& ctx, void const* sendbuff,
                                size_t fullElems, size_t elemOffset,
                                size_t chunkElems, size_t rowBytes,
                                size_t slot, cudaStream_t stream,
                                std::shared_ptr<Communicator> bootstrapComm,
                                ncclComm_t comm, int nRanks,
                                void* scratchBuffer,
                                size_t scratchBufferSize,
                                bool recordAsyncD2h, RsChunkWork* work) {
  (void)comm;
  uint64_t epoch = ++ctx.epoch;

  int localBase = ctx.nodeId * ctx.nRanksPerNode;
  int remoteBase = (1 - ctx.nodeId) * ctx.nRanksPerNode;
  size_t slotBase =
      slot * static_cast<size_t>(kScratchRowsPerChunk) * ctx.chunkCapacity;
  auto* scratch = static_cast<char*>(scratchBuffer) + slotBase;
  char* selfRows = scratch;
  char* partnerSendRows = selfRows + 4 * rowBytes;
  char* partnerRecvRows = partnerSendRows + 4 * rowBytes;
  char* ownPartialRows = partnerSendRows;
  char* crossSendRows = ownPartialRows + 2 * rowBytes;
  char* crossRecvRows = partnerRecvRows + 4 * rowBytes;
  char* localPartialGpu = crossRecvRows + 2 * rowBytes;
  char* remotePartialGpu = localPartialGpu + rowBytes;

  ensureLocalScratchIpc(ctx, bootstrapComm, ctx.rank, nRanks,
                        ctx.nRanksPerNode, scratchBuffer, scratchBufferSize);
  bool ipcDeviceFlagSync =
      useIpcDeviceFlagSync() && ctx.nRanksPerNode == 4 &&
      !useDirectTargetReduce() && ctx.localDeviceFlags != nullptr &&
      ctx.remoteDeviceFlagPtrs.size() ==
          static_cast<size_t>(ctx.nRanksPerNode - 1);
  bool useIpcEvents =
      !ipcDeviceFlagSync && useIpcEventSyncFor(fullElems * sizeof(float));
  if (useIpcEvents) {
    ensureLocalScratchEvents(ctx, bootstrapComm, ctx.rank, nRanks,
                             ctx.nRanksPerNode);
  }
  bool deviceFlagSync = useDeviceFlagSync() && ctx.ctrlDeviceSlab != nullptr;
  ncclResult_t result = ncclSuccess;

  if (useDirectTargetReduce()) {
    char* targetRows = scratch;
    localPartialGpu = targetRows + static_cast<size_t>(8) * rowBytes;
    remotePartialGpu = localPartialGpu + rowBytes;
    if (useDirectTargetScatterKernel()) {
      char* dsts[4] = {};
      for (int targetLocal = 0; targetLocal < ctx.nRanksPerNode; ++targetLocal) {
        if (targetLocal == ctx.localRank) {
          dsts[targetLocal] = targetRows;
        } else {
          int peerIdx = localPeerIndex(ctx.localRank, targetLocal);
          dsts[targetLocal] = ctx.remoteScratchPtrs[peerIdx] + slotBase;
        }
      }
      result = launchDirectTargetScatter(
          sendbuff, dsts[0], dsts[1], dsts[2], dsts[3], chunkElems,
          fullElems, elemOffset, rowBytes, ctx.localRank, localBase,
          remoteBase, stream);
      if (result != ncclSuccess) return result;
    } else {
      auto const* sendBytes = static_cast<char const*>(sendbuff);
      size_t fullRowBytes = fullElems * sizeof(float);
      size_t rowOffset = elemOffset * sizeof(float);
      size_t copyBytes = chunkElems * sizeof(float);
      for (int targetLocal = 0; targetLocal < ctx.nRanksPerNode; ++targetLocal) {
        char* dstBase = nullptr;
        if (targetLocal == ctx.localRank) {
          dstBase = targetRows;
        } else {
          int peerIdx = localPeerIndex(ctx.localRank, targetLocal);
          dstBase = ctx.remoteScratchPtrs[peerIdx] + slotBase;
        }
        int rows[2] = {localBase + targetLocal, remoteBase + targetLocal};
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(
            dstBase + static_cast<size_t>(ctx.localRank) * rowBytes,
            sendBytes + static_cast<size_t>(rows[0]) * fullRowBytes + rowOffset,
            copyBytes, cudaMemcpyDeviceToDevice, stream));
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(
            dstBase + (static_cast<size_t>(4 + ctx.localRank) * rowBytes),
            sendBytes + static_cast<size_t>(rows[1]) * fullRowBytes + rowOffset,
            copyBytes, cudaMemcpyDeviceToDevice, stream));
      }
    }
    if (deviceFlagSync) {
      result = launchStoreRsControlFlag(
          ctx, offsetof(RsControl, localCopyReady), epoch, stream);
      if (result != ncclSuccess) return result;
    } else if (useIpcEvents) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCopyEvents[slot], stream));
      ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                    std::memory_order_release);
      for (int peerLocal = 0; peerLocal < ctx.nRanksPerNode; ++peerLocal) {
        if (peerLocal == ctx.localRank) continue;
        waitForEpoch(ctx.ctrl->localCopyReady[peerLocal], epoch);
        int peerIdx = localPeerIndex(ctx.localRank, peerLocal);
        MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
            stream, ctx.remoteCopyEvents[peerIdx][slot], 0));
      }
    } else {
      waitForCudaStream(stream);
      ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                    std::memory_order_release);
      for (int peerLocal = 0; peerLocal < ctx.nRanksPerNode; ++peerLocal) {
        if (peerLocal == ctx.localRank) continue;
        waitForEpoch(ctx.ctrl->localCopyReady[peerLocal], epoch);
      }
    }
    ncclResult_t directResult =
        launchDirectTargetReduce(targetRows, localPartialGpu, remotePartialGpu,
                                 chunkElems, rowBytes, stream);
    if (directResult != ncclSuccess) return directResult;
    return finishScheduledChunkLocal(ctx, epoch, slot, elemOffset, chunkElems,
                                     chunkElems * sizeof(float),
                                     localPartialGpu, remotePartialGpu, stream,
                                     recordAsyncD2h,
                                     /*remotePartialInSendSlab=*/false, work);
  }

  int partnerLocal = ctx.localRank ^ 1;
  int partnerPeerIdx = localPeerIndex(ctx.localRank, partnerLocal);
  size_t partnerRecvOffset = slotBase + static_cast<size_t>(8) * rowBytes;
  size_t bytesPerRankForPolicy = fullElems * sizeof(float);
  size_t messageBytesForPolicy =
      bytesPerRankForPolicy * static_cast<size_t>(nRanks);
  bool directPartnerCopy = useDirectPartnerCopyFor(messageBytesForPolicy);
  if (directPartnerCopy) {
    result = copyPartnerRowsDirect(
        sendbuff, ctx.remoteScratchPtrs[partnerPeerIdx] + partnerRecvOffset,
        chunkElems, fullElems, elemOffset, rowBytes, ctx.localRank, localBase,
        remoteBase, useDirectPartnerCopy2DFor(messageBytesForPolicy), stream);
    if (result != ncclSuccess) return result;
  } else {
    result = launchPackPartnerRows(
        sendbuff, partnerSendRows, chunkElems, fullElems, elemOffset, rowBytes,
        ctx.localRank, localBase, remoteBase, stream);
    if (result != ncclSuccess) return result;
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        ctx.remoteScratchPtrs[partnerPeerIdx] + partnerRecvOffset,
        partnerSendRows, 4 * rowBytes, cudaMemcpyDeviceToDevice, stream));
  }
  if (ipcDeviceFlagSync) {
    result = launchStoreIpcDeviceFlag(ctx, /*phase=*/0, epoch, stream);
    if (result != ncclSuccess) return result;
  } else if (useIpcEvents) {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCopyEvents[slot], stream));
    ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
    waitForEpoch(ctx.ctrl->localCopyReady[partnerLocal], epoch);
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
        stream, ctx.remoteCopyEvents[partnerPeerIdx][slot], 0));
  } else {
    waitForCudaStream(stream);
    ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
    waitForEpoch(ctx.ctrl->localCopyReady[partnerLocal], epoch);
  }

  int crossTargetLocal = ctx.localRank < 2 ? ctx.localRank + 2
                                           : ctx.localRank - 2;
  int crossPeerIdx = localPeerIndex(ctx.localRank, crossTargetLocal);
  size_t crossRecvOffset = slotBase + static_cast<size_t>(12) * rowBytes;
  bool directCrossWrite =
      useDirectCrossWrite() && !deviceFlagSync && !ipcDeviceFlagSync;
  char* crossOutputRows =
      directCrossWrite ? ctx.remoteScratchPtrs[crossPeerIdx] + crossRecvOffset
                       : crossSendRows;

  if (ipcDeviceFlagSync) {
    result = launchWaitIpcDeviceFlag(ctx, /*phase=*/0, partnerLocal, epoch,
                                     stream);
    if (result != ncclSuccess) return result;
    result = launchPairReduceFromInput(
        sendbuff, partnerRecvRows, ownPartialRows, crossOutputRows, chunkElems,
        fullElems, elemOffset, rowBytes, ctx.localRank, localBase, remoteBase,
        stream);
  } else if (deviceFlagSync) {
    result = launchPairReduceFromInputWait(
        ctx, offsetof(RsControl, localCopyReady), partnerLocal, epoch,
        sendbuff, partnerRecvRows, ownPartialRows, crossOutputRows, chunkElems,
        fullElems, elemOffset, rowBytes, ctx.localRank, localBase, remoteBase,
        stream);
  } else {
    result = launchPairReduceFromInput(
        sendbuff, partnerRecvRows, ownPartialRows, crossOutputRows, chunkElems,
        fullElems, elemOffset, rowBytes, ctx.localRank, localBase, remoteBase,
        stream);
  }
  if (result != ncclSuccess) return result;

  if (!directCrossWrite) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        ctx.remoteScratchPtrs[crossPeerIdx] + crossRecvOffset, crossSendRows,
        2 * rowBytes, cudaMemcpyDeviceToDevice, stream));
  }
  if (ipcDeviceFlagSync) {
    result = launchStoreIpcDeviceFlag(ctx, /*phase=*/1, epoch, stream);
    if (result != ncclSuccess) return result;
  } else if (deviceFlagSync) {
    result = launchStoreRsControlFlag(
        ctx, offsetof(RsControl, localCrossReady), epoch, stream);
    if (result != ncclSuccess) return result;
  } else if (useIpcEvents) {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCrossEvents[slot], stream));
    ctx.ctrl->localCrossReady[ctx.localRank].store(epoch,
                                                   std::memory_order_release);
    waitForEpoch(ctx.ctrl->localCrossReady[crossTargetLocal], epoch);
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
        stream, ctx.remoteCrossEvents[crossPeerIdx][slot], 0));
  } else {
    waitForCudaStream(stream);
    ctx.ctrl->localCrossReady[ctx.localRank].store(epoch,
                                                   std::memory_order_release);
    waitForEpoch(ctx.ctrl->localCrossReady[crossTargetLocal], epoch);
  }

  size_t pairSlotOffset = static_cast<size_t>(ctx.localRank) *
                              hostRankStride(ctx) +
                          slot * ctx.chunkCapacity;
  size_t bytesPerRank = fullElems * sizeof(float);
  bool remotePartialInSendSlab =
      useMappedSendFinalReduceFor(bytesPerRank) && ctx.sendDeviceSlab != nullptr;
  char* remotePartialOutput =
      remotePartialInSendSlab ? ctx.sendDeviceSlab + pairSlotOffset
                              : remotePartialGpu;

  if (ipcDeviceFlagSync) {
    result = launchWaitIpcDeviceFlag(ctx, /*phase=*/1, crossTargetLocal, epoch,
                                     stream);
    if (result != ncclSuccess) return result;
  }

  bool splitFinalReduce =
      !deviceFlagSync && !ipcDeviceFlagSync &&
      !(ctx.worldSize == 8 && ctx.nRanksPerNode == 4 &&
        bytesPerRank <= 512 * 1024) &&
      useSplitFinalReduceFor(bytesPerRank, messageBytesForPolicy,
                             recordAsyncD2h);
  if (splitFinalReduce) {
    ensurePipelineResources(ctx);
    bool parallelSplit =
        useParallelSplitFinalReduceFor(ctx, messageBytesForPolicy);
    cudaStream_t remoteStream = stream;
    if (parallelSplit) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.reduceDoneEvents[slot], stream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.d2hStream,
                                          ctx.reduceDoneEvents[slot], 0));
      remoteStream = ctx.d2hStream;
    }
    result = launchFinalRemoteOnlyReduce(ownPartialRows, crossRecvRows,
                                        remotePartialOutput, chunkElems,
                                        rowBytes, remoteStream);
    if (result != ncclSuccess) return result;
    if (remotePartialInSendSlab) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.d2hDoneEvents[slot],
                                        remoteStream));
    } else {
      if (!parallelSplit) {
        MSCCLPP_CUDATHROW(cudaEventRecord(ctx.reduceDoneEvents[slot], stream));
        MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.d2hStream,
                                            ctx.reduceDoneEvents[slot], 0));
      }
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(ctx.sendSlab + pairSlotOffset,
                                        remotePartialOutput,
                                        chunkElems * sizeof(float),
                                        cudaMemcpyDeviceToHost, ctx.d2hStream));
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.d2hDoneEvents[slot],
                                        ctx.d2hStream));
    }
    result = launchFinalLocalOnlyReduce(ownPartialRows, crossRecvRows,
                                        localPartialGpu, chunkElems, rowBytes,
                                        stream);
    if (result != ncclSuccess) return result;
    if (work != nullptr) {
      work->epoch = epoch;
      work->slot = slot;
      work->elemOffset = elemOffset;
      work->chunkElems = chunkElems;
      work->chunkBytes = chunkElems * sizeof(float);
      work->pairSlotOffset = pairSlotOffset;
      work->localPartialGpu = localPartialGpu;
      work->remotePartialGpu = remotePartialOutput;
      work->localScratchDoneAfterSlotDone = true;
    }
    return ncclSuccess;
  }

  if (ipcDeviceFlagSync) {
    result = launchFinalLocalReduce(ownPartialRows, crossRecvRows,
                                    localPartialGpu, remotePartialOutput,
                                    chunkElems, rowBytes, stream);
  } else if (deviceFlagSync) {
    result = launchFinalLocalReduceWait(
        ctx, offsetof(RsControl, localCrossReady), crossTargetLocal, epoch,
        ownPartialRows, crossRecvRows, localPartialGpu, remotePartialOutput,
        chunkElems, rowBytes, stream);
  } else {
    result = launchFinalLocalReduce(ownPartialRows, crossRecvRows,
                                    localPartialGpu, remotePartialOutput,
                                    chunkElems, rowBytes, stream);
  }
  if (result != ncclSuccess) return result;

  return finishScheduledChunkLocal(ctx, epoch, slot, elemOffset, chunkElems,
                                   chunkElems * sizeof(float),
                                   localPartialGpu, remotePartialOutput, stream,
                                   recordAsyncD2h, remotePartialInSendSlab,
                                   work);
}

void sendChunkAck(RsContext& ctx, RsChunkWork& work) {
  if (work.ackSent) return;
  waitForCudaEvent(work.ackAfterSlotDone || work.localScratchDoneAfterSlotDone
                       ? ctx.slotDoneEvents[work.slot]
                       : ctx.h2dDoneEvents[work.slot]);
  if (work.localScratchDoneAfterSlotDone) {
    ctx.ctrl->localScratchDone[ctx.localRank].store(
        work.epoch, std::memory_order_release);
  }
  size_t ackReadyOffset =
      ctrlArrayOffset(offsetof(RsControl, pairAckReady), ctx.localRank);
  size_t ackSignalOffset =
      ctrlArrayOffset(offsetof(RsControl, pairAckSignal), ctx.localRank);
  ctx.ctrl->pairAckSignal[ctx.localRank].store(work.epoch,
                                               std::memory_order_release);
  postSmallSignal(ctx, ackReadyOffset, ackSignalOffset, kPairSignalEveryN);
  work.ackSent = true;
}

ncclResult_t postChunkRdma(RsContext& ctx, RsChunkWork& work, bool asyncCopy,
                           bool deferAck, size_t asyncAckDistance) {
  if (work.rdmaPosted) return ncclSuccess;
  if (asyncCopy || work.localScratchDoneAfterSlotDone) {
    waitForCudaEvent(ctx.d2hDoneEvents[work.slot]);
  }
  if (!work.localScratchDoneAfterSlotDone) {
    ctx.ctrl->localScratchDone[ctx.localRank].store(
        work.epoch, std::memory_order_release);
  }
  uint64_t ackDistance = (asyncCopy || deferAck) ? asyncAckDistance : 1;
  if (work.epoch > ackDistance) {
    waitForEpoch(ctx.ctrl->pairAckReady[ctx.localRank],
                 work.epoch - ackDistance);
  }
  size_t readyOffset =
      ctrlArrayOffset(offsetof(RsControl, pairRdmaReady), ctx.localRank);
  size_t signalOffset =
      ctrlArrayOffset(offsetof(RsControl, pairRdmaSignal), ctx.localRank);

  ctx.ctrl->pairRdmaSignal[ctx.localRank].store(work.epoch,
                                                std::memory_order_release);
  size_t off = 0;
  while (off < work.chunkBytes) {
    size_t bytes = std::min(kRdmaChunkBytes, work.chunkBytes - off);
    bool last = off + bytes == work.chunkBytes;
    if (last) {
      postPairDataAndSignal(ctx, work.pairSlotOffset + off,
                            work.pairSlotOffset + off, bytes, readyOffset,
                            signalOffset);
    } else {
      ctx.pairConnection.write(ctx.remoteRecvMemory, work.pairSlotOffset + off,
                               ctx.sendMemory, work.pairSlotOffset + off,
                               bytes);
    }
    off += bytes;
  }
  work.rdmaPosted = true;
  return ncclSuccess;
}

ncclResult_t completeChunkRemote(RsContext& ctx, void* recvbuff,
                                 RsChunkWork& work, cudaStream_t stream,
                                 bool asyncCopy, bool deferAck,
                                 size_t asyncAckDistance,
                                 bool asyncFinalAdd = false) {
  ncclResult_t postResult =
      postChunkRdma(ctx, work, asyncCopy, deferAck, asyncAckDistance);
  if (postResult != ncclSuccess) return postResult;
  if (work.remoteCompleted) return ncclSuccess;

  size_t ackReadyOffset =
      ctrlArrayOffset(offsetof(RsControl, pairAckReady), ctx.localRank);
  size_t ackSignalOffset =
      ctrlArrayOffset(offsetof(RsControl, pairAckSignal), ctx.localRank);

  waitForEpoch(ctx.ctrl->pairRdmaReady[ctx.localRank], work.epoch);

  cudaStream_t h2dStream = asyncCopy ? ctx.h2dStream : stream;
  auto* recvBytes = static_cast<char*>(recvbuff);
  auto* outputGpu = recvBytes + work.elemOffset * sizeof(float);
  if (!asyncCopy && useCpuFinalAdd()) {
    auto* localHost =
        reinterpret_cast<float*>(ctx.smallRecvSlab + work.pairSlotOffset);
    auto const* remoteHost =
        reinterpret_cast<float const*>(ctx.recvSlab + work.pairSlotOffset);
    addSmallPartials(localHost, remoteHost, work.chunkElems);
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(outputGpu, localHost, work.chunkBytes,
                                      cudaMemcpyHostToDevice, stream));
    waitForCudaStream(stream);
    if (!deferAck) {
      ctx.ctrl->pairAckSignal[ctx.localRank].store(
          work.epoch, std::memory_order_release);
      postSmallSignal(ctx, ackReadyOffset, ackSignalOffset);
      work.ackSent = true;
    }
    work.remoteCompleted = true;
    return ncclSuccess;
  }
  int hostReadMode = configuredHostReadFinalAddMode();
  bool hostReadRequested =
      hostReadMode >= 0
          ? hostReadMode != 0
          : ((!asyncCopy && work.chunkBytes <= 512 * 1024) ||
             work.localScratchDoneAfterSlotDone);
  bool hostReadFinalAdd =
      hostReadRequested && ctx.recvDeviceSlab != nullptr &&
      (!asyncCopy || work.localScratchDoneAfterSlotDone);
  if (hostReadFinalAdd) {
    auto* incomingGpu = ctx.recvDeviceSlab + work.pairSlotOffset;
    ncclResult_t result =
        launchAdd(work.localPartialGpu, incomingGpu, outputGpu,
                  work.chunkElems, stream);
    if (result != ncclSuccess) return result;
    if (asyncCopy) {
      work.ackAfterSlotDone = true;
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.slotDoneEvents[work.slot], stream));
    } else {
      waitForCudaStream(stream);
    }
    if (!deferAck) {
      if (asyncCopy) waitForCudaEvent(ctx.slotDoneEvents[work.slot]);
      if (work.localScratchDoneAfterSlotDone) {
        ctx.ctrl->localScratchDone[ctx.localRank].store(
            work.epoch, std::memory_order_release);
      }
      ctx.ctrl->pairAckSignal[ctx.localRank].store(
          work.epoch, std::memory_order_release);
      postSmallSignal(ctx, ackReadyOffset, ackSignalOffset);
      work.ackSent = true;
    }
    work.remoteCompleted = true;
    return ncclSuccess;
  }
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(
      outputGpu, ctx.recvSlab + work.pairSlotOffset, work.chunkBytes,
      cudaMemcpyHostToDevice, h2dStream));
  if (asyncCopy) {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvents[work.slot],
                                      ctx.h2dStream));
    if (!asyncFinalAdd) {
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream,
                                            ctx.h2dDoneEvents[work.slot], 0));
    }
  } else {
    waitForCudaStream(stream);
  }

  if (!deferAck) {
    if (asyncCopy) {
      sendChunkAck(ctx, work);
    } else {
      ctx.ctrl->pairAckSignal[ctx.localRank].store(work.epoch,
                                                   std::memory_order_release);
      postSmallSignal(ctx, ackReadyOffset, ackSignalOffset);
      work.ackSent = true;
    }
  } else if (!asyncCopy) {
    ctx.ctrl->pairAckSignal[ctx.localRank].store(work.epoch,
                                                 std::memory_order_release);
    postSmallSignal(ctx, ackReadyOffset, ackSignalOffset);
    work.ackSent = true;
  }

  cudaStream_t addStream = (asyncCopy && asyncFinalAdd) ? ctx.h2dStream : stream;
  ncclResult_t result =
      launchAdd(work.localPartialGpu, outputGpu, outputGpu, work.chunkElems,
                addStream);
  if (result != ncclSuccess) return result;
  if (asyncCopy) {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.slotDoneEvents[work.slot], addStream));
  }
  work.remoteCompleted = true;
  return ncclSuccess;
}

ncclResult_t runChunk(RsContext& ctx, void const* sendbuff, void* recvbuff,
                      size_t fullElems, size_t elemOffset, size_t chunkElems,
                      size_t rowBytes, cudaStream_t stream,
                      std::shared_ptr<Communicator> bootstrapComm,
                      ncclComm_t comm, int nRanks, void* scratchBuffer,
                      size_t scratchBufferSize) {
  bool slotRing = useSingleChunkSlotRing();
  size_t slot = 0;
  size_t ackDistance = 1;
  if (slotRing) {
    ensurePipelineResources(ctx);
    size_t pipelineSlots = static_cast<size_t>(ctx.pipelineSlots);
    uint64_t nextEpoch = ctx.epoch + 1;
    slot = static_cast<size_t>((nextEpoch - 1) % pipelineSlots);
    ackDistance = pipelineSlots;
    if (nextEpoch > pipelineSlots) {
      uint64_t reuseEpoch = nextEpoch - pipelineSlots;
      waitForScratchReuse(ctx, reuseEpoch);
      waitForEpoch(ctx.ctrl->pairAckReady[ctx.localRank], reuseEpoch);
    }
  } else {
    uint64_t prevEpoch = ctx.epoch;
    if (prevEpoch > 0) {
      waitForScratchReuse(ctx, prevEpoch);
    }
  }

  RsChunkWork work;
  if (useSingleChunkAsyncCopy()) {
    ensurePipelineResources(ctx);
    ncclResult_t result = scheduleChunkLocal(
        ctx, sendbuff, fullElems, elemOffset, chunkElems, rowBytes, 0, stream,
        bootstrapComm, comm, nRanks, scratchBuffer, scratchBufferSize,
        /*recordAsyncD2h=*/true, &work);
    if (result != ncclSuccess) return result;
    bool asyncFinalAdd = useAsyncFinalAddFor(fullElems * sizeof(float));
    result = completeChunkRemote(ctx, recvbuff, work, stream,
                                 /*asyncCopy=*/true, /*deferAck=*/true,
                                 ctx.pipelineSlots, asyncFinalAdd);
    if (result != ncclSuccess) return result;
    sendChunkAck(ctx, work);
    if (asyncFinalAdd) {
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream,
                                            ctx.slotDoneEvents[work.slot], 0));
    }
    return ncclSuccess;
  }
  ncclResult_t result = scheduleChunkLocal(
      ctx, sendbuff, fullElems, elemOffset, chunkElems, rowBytes, slot, stream,
      bootstrapComm, comm, nRanks, scratchBuffer, scratchBufferSize,
      /*recordAsyncD2h=*/false, &work);
  if (result != ncclSuccess) return result;

  return completeChunkRemote(ctx, recvbuff, work, stream, /*asyncCopy=*/false,
                             /*deferAck=*/slotRing,
                             /*asyncAckDistance=*/ackDistance);
}

ncclResult_t runPipelinedChunks(RsContext& ctx, void const* sendbuff,
                                void* recvbuff, size_t recvcount,
                                size_t runChunkBytes,
                                cudaStream_t stream,
                                std::shared_ptr<Communicator> bootstrapComm,
                                ncclComm_t comm, int nRanks,
                                void* scratchBuffer,
                                size_t scratchBufferSize) {
  ensurePipelineResources(ctx);
  size_t pipelineSlots = static_cast<size_t>(ctx.pipelineSlots);
  size_t totalChunks = (recvcount * sizeof(float) + runChunkBytes - 1) /
                       runChunkBytes;
  size_t localLead = configuredTwoNodeTwoGpuLeadChunks(totalChunks);
  size_t messageBytes = recvcount * sizeof(float) *
                        static_cast<size_t>(ctx.worldSize);
  if (useParallelSplitFinalReduceFor(ctx, messageBytes) &&
      localLead >= pipelineSlots) {
    localLead = pipelineSlots - 1;
  }
  bool asyncFinalAdd = usePipelinedAsyncFinalAddFor(ctx, messageBytes);
  bool eagerRdmaPost = useEagerRdmaPostFor(ctx, messageBytes);
  std::vector<RsChunkWork> works;
  works.reserve(totalChunks);

  size_t nextPost = 0;
  size_t nextRemote = 0;
  size_t chunkIndex = 0;
  for (size_t elemOffset = 0; elemOffset < recvcount;) {
    size_t slot = chunkIndex % pipelineSlots;
    if (chunkIndex >= pipelineSlots) {
      auto& reuseWork = works[chunkIndex - pipelineSlots];
      if (eagerRdmaPost && !reuseWork.remoteCompleted) {
        ncclResult_t result =
            completeChunkRemote(ctx, recvbuff, reuseWork, stream,
                                /*asyncCopy=*/true, /*deferAck=*/true,
                                ctx.pipelineSlots, asyncFinalAdd);
        if (result != ncclSuccess) return result;
        nextRemote = std::max(nextRemote, chunkIndex - pipelineSlots + 1);
      }
      sendChunkAck(ctx, reuseWork);
      uint64_t reuseEpoch = reuseWork.epoch;
      waitForScratchReuse(ctx, reuseEpoch);
      waitForCudaEvent(ctx.slotDoneEvents[slot]);
    }
    size_t chunkBytes = std::min(runChunkBytes,
                                 recvcount * sizeof(float) -
                                     elemOffset * sizeof(float));
    size_t chunkElems = chunkBytes / sizeof(float);
    RsChunkWork work;
    ncclResult_t result = scheduleChunkLocal(
        ctx, sendbuff, recvcount, elemOffset, chunkElems, chunkBytes, slot,
        stream, bootstrapComm, comm, nRanks, scratchBuffer, scratchBufferSize,
        /*recordAsyncD2h=*/true, &work);
    if (result != ncclSuccess) return result;
    works.push_back(work);

    if (eagerRdmaPost) {
      while (works.size() > nextPost + localLead) {
        result = postChunkRdma(ctx, works[nextPost], /*asyncCopy=*/true,
                               /*deferAck=*/true, ctx.pipelineSlots);
        if (result != ncclSuccess) return result;
        ++nextPost;
      }
    } else {
      while (works.size() > nextRemote + localLead) {
        result = completeChunkRemote(ctx, recvbuff, works[nextRemote], stream,
                                     /*asyncCopy=*/true, /*deferAck=*/true,
                                     ctx.pipelineSlots, asyncFinalAdd);
        if (result != ncclSuccess) return result;
        ++nextRemote;
      }
    }

    elemOffset += chunkElems;
    ++chunkIndex;
  }

  if (eagerRdmaPost) {
    while (nextPost < works.size()) {
      ncclResult_t result =
          postChunkRdma(ctx, works[nextPost], /*asyncCopy=*/true,
                        /*deferAck=*/true, ctx.pipelineSlots);
      if (result != ncclSuccess) return result;
      ++nextPost;
    }
    while (nextRemote < works.size()) {
      ncclResult_t result =
          completeChunkRemote(ctx, recvbuff, works[nextRemote], stream,
                              /*asyncCopy=*/true, /*deferAck=*/true,
                              ctx.pipelineSlots, asyncFinalAdd);
      if (result != ncclSuccess) return result;
      ++nextRemote;
    }
  } else {
    while (nextRemote < works.size()) {
      ncclResult_t result =
          completeChunkRemote(ctx, recvbuff, works[nextRemote], stream,
                              /*asyncCopy=*/true, /*deferAck=*/true,
                              ctx.pipelineSlots, asyncFinalAdd);
      if (result != ncclSuccess) return result;
      ++nextRemote;
    }
  }
  for (auto& work : works) {
    sendChunkAck(ctx, work);
  }
  if (asyncFinalAdd) {
    size_t waitStart =
        works.size() > pipelineSlots ? works.size() - pipelineSlots : 0;
    for (size_t i = waitStart; i < works.size(); ++i) {
      MSCCLPP_CUDATHROW(
          cudaStreamWaitEvent(stream, ctx.slotDoneEvents[works[i].slot], 0));
    }
  }
  return ncclSuccess;
}

void waitForLocalPairScratchReuse(RsContext& ctx, uint64_t epoch) {
  for (int local = 0; local < ctx.nRanksPerNode; ++local) {
    waitForEpoch(ctx.ctrl->localScratchDone[local], epoch);
  }
}

ncclResult_t scheduleTwoNodeTwoGpuChunk(
    RsContext& ctx, void const* sendbuff, size_t fullElems, size_t elemOffset,
    size_t chunkElems, size_t rowBytes, size_t slot, cudaStream_t stream,
    std::shared_ptr<Communicator> bootstrapComm, int nRanks, void* scratchBuffer,
    size_t scratchBufferSize, bool recordAsyncD2h, RsChunkWork* work) {
  if (ctx.nRanksPerNode != 2) return ncclInvalidUsage;
  uint64_t epoch = ++ctx.epoch;
  int localBase = ctx.nodeId * ctx.nRanksPerNode;
  int remoteBase = (1 - ctx.nodeId) * ctx.nRanksPerNode;
  int partnerLocal = ctx.localRank ^ 1;
  int partnerPeerIdx = localPeerIndex(ctx.localRank, partnerLocal);
  if (partnerPeerIdx < 0) return ncclInvalidUsage;

  ensureLocalScratchIpc(ctx, bootstrapComm, ctx.rank, nRanks,
                        ctx.nRanksPerNode, scratchBuffer, scratchBufferSize);
  bool ipcDeviceFlagSync =
      useIpcDeviceFlagSync() && ctx.localDeviceFlags != nullptr &&
      ctx.remoteDeviceFlagPtrs.size() ==
          static_cast<size_t>(ctx.nRanksPerNode - 1);

  size_t slotBase =
      slot * static_cast<size_t>(kScratchRowsPerChunk) * ctx.chunkCapacity;
  auto* scratch = static_cast<char*>(scratchBuffer) + slotBase;
  char* partnerRecvRows = scratch;
  char* localPartialGpu = partnerRecvRows + 2 * rowBytes;
  char* remotePartialGpu = localPartialGpu + rowBytes;
  auto const* sendBytes = static_cast<char const*>(sendbuff);
  size_t fullRowBytes = fullElems * sizeof(float);
  size_t rowOffset = elemOffset * sizeof(float);
  size_t copyBytes = chunkElems * sizeof(float);
  size_t pairSlotOffset = static_cast<size_t>(ctx.localRank) *
                              hostRankStride(ctx) +
                          slot * ctx.chunkCapacity;
  bool remotePartialInSendSlab =
      useMappedSendFinalReduceFor(fullRowBytes) && ctx.sendDeviceSlab != nullptr;
  char* remotePartialOutput =
      remotePartialInSendSlab ? ctx.sendDeviceSlab + pairSlotOffset
                              : remotePartialGpu;
  char* remotePartnerRows =
      ctx.remoteScratchPtrs[partnerPeerIdx] + slotBase;
  bool useIpcEvents =
      !ipcDeviceFlagSync &&
      useTwoNodeTwoGpuIpcEventSyncForPath(
          recordAsyncD2h, fullElems * sizeof(float) * static_cast<size_t>(nRanks));
  if (useIpcEvents) {
    ensureLocalScratchEvents(ctx, bootstrapComm, ctx.rank, nRanks,
                             ctx.nRanksPerNode);
  }

  bool partnerRowsReversed = localBase > remoteBase;
  if (localBase < remoteBase) {
    MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
        remotePartnerRows, rowBytes,
        sendBytes + static_cast<size_t>(localBase + partnerLocal) *
                        fullRowBytes +
            rowOffset,
        static_cast<size_t>(ctx.nRanksPerNode) * fullRowBytes, copyBytes, 2,
        cudaMemcpyDeviceToDevice, stream));
  } else {
    MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
        remotePartnerRows, rowBytes,
        sendBytes + static_cast<size_t>(remoteBase + partnerLocal) *
                        fullRowBytes +
            rowOffset,
        static_cast<size_t>(ctx.nRanksPerNode) * fullRowBytes, copyBytes, 2,
        cudaMemcpyDeviceToDevice, stream));
  }
  if (ipcDeviceFlagSync) {
    ncclResult_t result = launchStoreIpcDeviceFlag(ctx, /*phase=*/0, epoch,
                                                   stream);
    if (result != ncclSuccess) return result;
  } else if (useIpcEvents) {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.localCopyEvents[slot], stream));
    ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
    waitForEpoch(ctx.ctrl->localCopyReady[partnerLocal], epoch);
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
        stream, ctx.remoteCopyEvents[partnerPeerIdx][slot], 0));
  } else {
    waitForCudaStream(stream);
    ctx.ctrl->localCopyReady[ctx.localRank].store(epoch,
                                                  std::memory_order_release);
    waitForEpoch(ctx.ctrl->localCopyReady[partnerLocal], epoch);
  }

  ncclResult_t result = ncclSuccess;
  bool splitLocalReduce =
      recordAsyncD2h && remotePartialInSendSlab && !ipcDeviceFlagSync &&
      useSplitFinalReduceFor(fullRowBytes,
                             fullRowBytes * static_cast<size_t>(nRanks),
                             recordAsyncD2h);
  if (splitLocalReduce) {
    ensurePipelineResources(ctx);
    bool parallelSplit = useParallelSplitFinalReduceFor(
        ctx, fullRowBytes * static_cast<size_t>(nRanks));
    cudaStream_t remoteStream = stream;
    if (parallelSplit) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.reduceDoneEvents[slot], stream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.d2hStream,
                                            ctx.reduceDoneEvents[slot], 0));
      remoteStream = ctx.d2hStream;
    }
    result = launchTwoNodeTwoGpuRemoteOnlyReduce(
        sendbuff, partnerRecvRows, remotePartialOutput, chunkElems, fullElems,
        elemOffset, rowBytes, partnerRowsReversed, ctx.localRank, remoteBase,
        remoteStream);
    if (result != ncclSuccess) return result;
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.d2hDoneEvents[slot], remoteStream));
    result = launchTwoNodeTwoGpuLocalOnlyReduce(
        sendbuff, partnerRecvRows, localPartialGpu, chunkElems, fullElems,
        elemOffset, rowBytes, partnerRowsReversed, ctx.localRank, localBase,
        stream);
    if (result != ncclSuccess) return result;
    if (work != nullptr) {
      work->epoch = epoch;
      work->slot = slot;
      work->elemOffset = elemOffset;
      work->chunkElems = chunkElems;
      work->chunkBytes = copyBytes;
      work->pairSlotOffset = pairSlotOffset;
      work->localPartialGpu = localPartialGpu;
      work->remotePartialGpu = remotePartialOutput;
      work->localScratchDoneAfterSlotDone = true;
    }
    return ncclSuccess;
  }

  result =
      ipcDeviceFlagSync
          ? launchTwoNodeTwoGpuLocalReduceWait(
                ctx, partnerLocal, epoch, sendbuff, partnerRecvRows,
                localPartialGpu, remotePartialOutput, chunkElems, fullElems,
                elemOffset, rowBytes, partnerRowsReversed, ctx.localRank,
                localBase, remoteBase, stream)
          : launchTwoNodeTwoGpuLocalReduce(
                sendbuff, partnerRecvRows, localPartialGpu, remotePartialOutput,
                chunkElems, fullElems, elemOffset, rowBytes,
                partnerRowsReversed, ctx.localRank, localBase, remoteBase,
                stream);
  if (result != ncclSuccess) return result;

  return finishScheduledChunkLocal(ctx, epoch, slot, elemOffset, chunkElems,
                                  copyBytes, localPartialGpu, remotePartialOutput,
                                   stream, recordAsyncD2h,
                                  remotePartialInSendSlab, work);
}

ncclResult_t runTwoNodeTwoGpuPipelinedChunks(
    RsContext& ctx, void const* sendbuff, void* recvbuff, size_t recvcount,
    size_t runChunkBytes, cudaStream_t stream,
    std::shared_ptr<Communicator> bootstrapComm, int nRanks, void* scratchBuffer,
    size_t scratchBufferSize) {
  ensurePipelineResources(ctx);
  size_t pipelineSlots = static_cast<size_t>(ctx.pipelineSlots);
  size_t totalChunks =
      (recvcount * sizeof(float) + runChunkBytes - 1) / runChunkBytes;
  size_t localLead = totalChunks <= 2
                         ? configuredShortLocalLeadChunks()
                         : (totalChunks >= 8 ? configuredLongLocalLeadChunks()
                                             : configuredLocalLeadChunks());
  size_t messageBytes = recvcount * sizeof(float) *
                        static_cast<size_t>(ctx.worldSize);
  if (useParallelSplitFinalReduceFor(ctx, messageBytes) &&
      localLead >= pipelineSlots) {
    localLead = pipelineSlots - 1;
  }
  bool asyncFinalAdd = usePipelinedAsyncFinalAddFor(ctx, messageBytes);
  bool eagerRdmaPost = useEagerRdmaPostFor(ctx, messageBytes);
  std::vector<RsChunkWork> works;
  works.reserve(totalChunks);

  size_t nextPost = 0;
  size_t nextRemote = 0;
  size_t chunkIndex = 0;
  for (size_t elemOffset = 0; elemOffset < recvcount;) {
    size_t slot = chunkIndex % pipelineSlots;
    if (chunkIndex >= pipelineSlots) {
      auto& reuseWork = works[chunkIndex - pipelineSlots];
      if (eagerRdmaPost && !reuseWork.remoteCompleted) {
        ncclResult_t result =
            completeChunkRemote(ctx, recvbuff, reuseWork, stream,
                                /*asyncCopy=*/true, /*deferAck=*/true,
                                ctx.pipelineSlots, asyncFinalAdd);
        if (result != ncclSuccess) return result;
        nextRemote = std::max(nextRemote, chunkIndex - pipelineSlots + 1);
      }
      sendChunkAck(ctx, reuseWork);
      waitForLocalPairScratchReuse(ctx, reuseWork.epoch);
      waitForCudaEvent(ctx.slotDoneEvents[slot]);
    }
    size_t chunkBytes = std::min(runChunkBytes,
                                 recvcount * sizeof(float) -
                                     elemOffset * sizeof(float));
    size_t chunkElems = chunkBytes / sizeof(float);
    RsChunkWork work;
    ncclResult_t result = scheduleTwoNodeTwoGpuChunk(
        ctx, sendbuff, recvcount, elemOffset, chunkElems, chunkBytes, slot,
        stream, bootstrapComm, nRanks, scratchBuffer, scratchBufferSize,
        /*recordAsyncD2h=*/true, &work);
    if (result != ncclSuccess) return result;
    works.push_back(work);

    if (eagerRdmaPost) {
      while (works.size() > nextPost + localLead) {
        result = postChunkRdma(ctx, works[nextPost], /*asyncCopy=*/true,
                               /*deferAck=*/true, ctx.pipelineSlots);
        if (result != ncclSuccess) return result;
        ++nextPost;
      }
    } else {
      while (works.size() > nextRemote + localLead) {
        result = completeChunkRemote(ctx, recvbuff, works[nextRemote], stream,
                                     /*asyncCopy=*/true, /*deferAck=*/true,
                                     ctx.pipelineSlots, asyncFinalAdd);
        if (result != ncclSuccess) return result;
        ++nextRemote;
      }
    }

    elemOffset += chunkElems;
    ++chunkIndex;
  }

  if (eagerRdmaPost) {
    while (nextPost < works.size()) {
      ncclResult_t result =
          postChunkRdma(ctx, works[nextPost], /*asyncCopy=*/true,
                        /*deferAck=*/true, ctx.pipelineSlots);
      if (result != ncclSuccess) return result;
      ++nextPost;
    }
    while (nextRemote < works.size()) {
      ncclResult_t result =
          completeChunkRemote(ctx, recvbuff, works[nextRemote], stream,
                              /*asyncCopy=*/true, /*deferAck=*/true,
                              ctx.pipelineSlots, asyncFinalAdd);
      if (result != ncclSuccess) return result;
      ++nextRemote;
    }
  } else {
    while (nextRemote < works.size()) {
      ncclResult_t result =
          completeChunkRemote(ctx, recvbuff, works[nextRemote], stream,
                              /*asyncCopy=*/true, /*deferAck=*/true,
                              ctx.pipelineSlots, asyncFinalAdd);
      if (result != ncclSuccess) return result;
      ++nextRemote;
    }
  }
  for (auto& work : works) {
    sendChunkAck(ctx, work);
  }
  if (asyncFinalAdd) {
    size_t waitStart =
        works.size() > pipelineSlots ? works.size() - pipelineSlots : 0;
    for (size_t i = waitStart; i < works.size(); ++i) {
      MSCCLPP_CUDATHROW(
          cudaStreamWaitEvent(stream, ctx.slotDoneEvents[works[i].slot], 0));
    }
  }
  return ncclSuccess;
}

ncclResult_t runTwoNodeTwoGpuHierReduceScatter(
    RsContext& ctx, void const* sendbuff, void* recvbuff, size_t recvcount,
    size_t bytesPerRank, size_t runChunkBytes, cudaStream_t stream,
    std::shared_ptr<Communicator> bootstrapComm, int nRanks, void* scratchBuffer,
    size_t scratchBufferSize) {
  if (ctx.smallPendingAckEpoch > ctx.smallAckPostedEpoch) {
    waitForCudaStream(stream);
    flushSmallPendingAck(ctx);
  }
  if (bytesPerRank <= runChunkBytes) {
    ensurePipelineResources(ctx);
    uint64_t prevEpoch = ctx.epoch;
    if (prevEpoch > 0) waitForLocalPairScratchReuse(ctx, prevEpoch);
    RsChunkWork work;
    bool asyncSingleChunk =
        useSingleChunkAsyncCopy() || bytesPerRank <= 512 * 1024;
    ncclResult_t result = scheduleTwoNodeTwoGpuChunk(
        ctx, sendbuff, recvcount, /*elemOffset=*/0, recvcount, bytesPerRank,
        /*slot=*/0, stream, bootstrapComm, nRanks, scratchBuffer,
        scratchBufferSize, asyncSingleChunk, &work);
    if (result != ncclSuccess) return result;
    if (!asyncSingleChunk) {
      return completeChunkRemote(ctx, recvbuff, work, stream,
                                 /*asyncCopy=*/false, /*deferAck=*/false,
                                 /*asyncAckDistance=*/1);
    }
    bool asyncFinalAdd = useAsyncFinalAddFor(bytesPerRank);
    result = completeChunkRemote(ctx, recvbuff, work, stream,
                                 /*asyncCopy=*/true, /*deferAck=*/true,
                                 ctx.pipelineSlots, asyncFinalAdd);
    if (result != ncclSuccess) return result;
    sendChunkAck(ctx, work);
    if (asyncFinalAdd) {
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream,
                                            ctx.slotDoneEvents[work.slot], 0));
    }
    return ncclSuccess;
  }
  return runTwoNodeTwoGpuPipelinedChunks(ctx, sendbuff, recvbuff, recvcount,
                                         runChunkBytes, stream, bootstrapComm,
                                         nRanks, scratchBuffer,
                                         scratchBufferSize);
}

ncclResult_t scheduleTwoRankChunk(RsContext& ctx, void const* sendbuff,
                                  void* /*recvbuff*/, size_t recvcount,
                                  size_t elemOffset, size_t chunkElems, size_t slot,
                                  void* scratchBuffer, cudaStream_t stream,
                                  bool recordAsyncD2h, RsChunkWork* work) {
  uint64_t epoch = ++ctx.epoch;
  size_t chunkBytes = chunkElems * sizeof(float);
  size_t pairSlotOffset = slot * ctx.chunkCapacity;
  auto* incomingScratch = static_cast<char*>(scratchBuffer) + pairSlotOffset;
  auto const* sendBytes = static_cast<char const*>(sendbuff);
  size_t localOffset =
      (static_cast<size_t>(ctx.rank) * recvcount + elemOffset) * sizeof(float);
  int remoteRank = 1 - ctx.rank;
  size_t remoteOffset =
      (static_cast<size_t>(remoteRank) * recvcount + elemOffset) *
      sizeof(float);

  if (recordAsyncD2h) {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.reduceDoneEvents[slot], stream));
  }

  if (recordAsyncD2h) {
    MSCCLPP_CUDATHROW(
        cudaStreamWaitEvent(ctx.d2hStream, ctx.reduceDoneEvents[slot], 0));
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        ctx.sendSlab + pairSlotOffset, sendBytes + remoteOffset, chunkBytes,
        cudaMemcpyDeviceToHost, ctx.d2hStream));
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.d2hDoneEvents[slot],
                                      ctx.d2hStream));
  } else {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(ctx.sendSlab + pairSlotOffset,
                                      sendBytes + remoteOffset, chunkBytes,
                                      cudaMemcpyDeviceToHost, stream));
    waitForCudaStream(stream);
    ctx.ctrl->localScratchDone[ctx.localRank].store(
        epoch, std::memory_order_release);
  }

  if (work != nullptr) {
    work->epoch = epoch;
    work->slot = slot;
    work->elemOffset = elemOffset;
    work->chunkElems = chunkElems;
    work->chunkBytes = chunkBytes;
    work->pairSlotOffset = pairSlotOffset;
    work->localPartialGpu = incomingScratch;
    work->remotePartialGpu = const_cast<char*>(sendBytes + localOffset);
  }
  return ncclSuccess;
}

ncclResult_t completeTwoRankRemote(RsContext& ctx, void* recvbuff,
                                   RsChunkWork const& work,
                                   cudaStream_t stream, bool asyncCopy,
                                   bool deferAck, int asyncAckDistance) {
  if (asyncCopy) {
    waitForCudaEvent(ctx.d2hDoneEvents[work.slot]);
    ctx.ctrl->localScratchDone[ctx.localRank].store(
        work.epoch, std::memory_order_release);
  }

  uint64_t ackWaitEpoch = 0;
  if (work.epoch > static_cast<uint64_t>(asyncAckDistance)) {
    ackWaitEpoch = work.epoch - static_cast<uint64_t>(asyncAckDistance);
  }
  if (ackWaitEpoch != 0) {
    waitForEpoch(ctx.ctrl->pairAckReady[ctx.localRank], ackWaitEpoch);
  }

  size_t readyOffset =
      ctrlArrayOffset(offsetof(RsControl, pairRdmaReady), ctx.localRank);
  size_t signalOffset =
      ctrlArrayOffset(offsetof(RsControl, pairRdmaSignal), ctx.localRank);
  size_t ackReadyOffset =
      ctrlArrayOffset(offsetof(RsControl, pairAckReady), ctx.localRank);
  size_t ackSignalOffset =
      ctrlArrayOffset(offsetof(RsControl, pairAckSignal), ctx.localRank);

  ctx.ctrl->pairRdmaSignal[ctx.localRank].store(work.epoch,
                                                std::memory_order_release);
  size_t off = 0;
  while (off < work.chunkBytes) {
    size_t bytes = std::min(kRdmaChunkBytes, work.chunkBytes - off);
    bool last = off + bytes == work.chunkBytes;
    if (last) {
      postPairDataAndSignal(ctx, work.pairSlotOffset + off,
                            work.pairSlotOffset + off, bytes, readyOffset,
                            signalOffset);
    } else {
      ctx.pairConnection.write(ctx.remoteRecvMemory, work.pairSlotOffset + off,
                               ctx.sendMemory, work.pairSlotOffset + off,
                               bytes);
    }
    off += bytes;
  }

  waitForEpoch(ctx.ctrl->pairRdmaReady[ctx.localRank], work.epoch);

  cudaStream_t h2dStream = asyncCopy ? ctx.h2dStream : stream;
  auto* recvBytes = static_cast<char*>(recvbuff);
  auto* outputGpu = recvBytes + work.elemOffset * sizeof(float);
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(
      work.localPartialGpu, ctx.recvSlab + work.pairSlotOffset,
      work.chunkBytes, cudaMemcpyHostToDevice, h2dStream));
  MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvents[work.slot], h2dStream));
  if (asyncCopy) {
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream,
                                          ctx.h2dDoneEvents[work.slot], 0));
  }

  ncclResult_t result =
      launchAdd(work.remotePartialGpu, work.localPartialGpu, outputGpu,
                work.chunkElems, stream);
  if (result != ncclSuccess) return result;
  if (asyncCopy) {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.slotDoneEvents[work.slot], stream));
  }

  if (!deferAck) {
    waitForCudaEvent(ctx.h2dDoneEvents[work.slot]);
    ctx.ctrl->pairAckSignal[ctx.localRank].store(work.epoch,
                                                 std::memory_order_release);
    postSmallSignal(ctx, ackReadyOffset, ackSignalOffset);
  }
  return ncclSuccess;
}

ncclResult_t runTwoRankPipelinedChunks(RsContext& ctx, void const* sendbuff,
                                       void* recvbuff, size_t recvcount,
                                       size_t runChunkBytes,
                                       cudaStream_t stream,
                                       void* scratchBuffer) {
  ensurePipelineResources(ctx);
  size_t pipelineSlots = static_cast<size_t>(ctx.pipelineSlots);
  size_t totalChunks = (recvcount * sizeof(float) + runChunkBytes - 1) /
                       runChunkBytes;
  size_t localLead = totalChunks <= 2
                         ? configuredShortLocalLeadChunks()
                         : (totalChunks >= 8 ? configuredLongLocalLeadChunks()
                                            : configuredLocalLeadChunks());
  std::vector<RsChunkWork> works;
  works.reserve(totalChunks);

  size_t nextRemote = 0;
  size_t chunkIndex = 0;
  for (size_t elemOffset = 0; elemOffset < recvcount;) {
    size_t slot = chunkIndex % pipelineSlots;
    if (chunkIndex >= pipelineSlots) {
      auto& reuseWork = works[chunkIndex - pipelineSlots];
      waitForEpoch(ctx.ctrl->localScratchDone[ctx.localRank],
                   reuseWork.epoch);
      waitForCudaEvent(ctx.slotDoneEvents[slot]);
      sendChunkAck(ctx, reuseWork);
    }

    size_t chunkBytes = std::min(runChunkBytes,
                                 recvcount * sizeof(float) -
                                     elemOffset * sizeof(float));
    size_t chunkElems = chunkBytes / sizeof(float);
    RsChunkWork work;
    ncclResult_t result = scheduleTwoRankChunk(
        ctx, sendbuff, recvbuff, recvcount, elemOffset, chunkElems, slot,
        scratchBuffer, stream, /*recordAsyncD2h=*/true, &work);
    if (result != ncclSuccess) return result;
    works.push_back(work);

    while (works.size() > nextRemote + localLead) {
      result = completeTwoRankRemote(ctx, recvbuff, works[nextRemote], stream,
                                     /*asyncCopy=*/true, /*deferAck=*/true,
                                     ctx.pipelineSlots);
      if (result != ncclSuccess) return result;
      ++nextRemote;
    }

    elemOffset += chunkElems;
    ++chunkIndex;
  }

  while (nextRemote < works.size()) {
    ncclResult_t result =
        completeTwoRankRemote(ctx, recvbuff, works[nextRemote], stream,
                              /*asyncCopy=*/true, /*deferAck=*/true,
                              ctx.pipelineSlots);
    if (result != ncclSuccess) return result;
    ++nextRemote;
  }
  size_t ackStart =
      works.size() > pipelineSlots ? works.size() - pipelineSlots : 0;
  for (size_t i = ackStart; i < works.size(); ++i) {
    waitForCudaEvent(ctx.slotDoneEvents[works[i].slot]);
    sendChunkAck(ctx, works[i]);
  }
  return ncclSuccess;
}

ncclResult_t runTwoRankReduceScatter(RsContext& ctx, void const* sendbuff,
                                     void* recvbuff, size_t recvcount,
                                     size_t bytesPerRank,
                                     size_t runChunkBytes,
                                     cudaStream_t stream,
                                     void* scratchBuffer,
                                     std::shared_ptr<Communicator>
                                         bootstrapComm) {
  if (bytesPerRank <= configuredTwoRankSmallHostBytes()) {
    size_t smallRankStride =
        ctx.chunkCapacity * static_cast<size_t>(ctx.pipelineSlots);
    size_t slotBytes = std::max(2 * bytesPerRank, 3 * bytesPerRank);
    if (slotBytes <= smallRankStride) {
      return runTwoRankSmallHostReduceScatter(ctx, sendbuff, recvbuff, recvcount,
                                              bytesPerRank, stream,
                                              bootstrapComm);
    }
  }
  if (ctx.smallPendingAckEpoch > ctx.smallAckPostedEpoch) {
    waitForCudaStream(stream);
    flushSmallPendingAck(ctx);
  }
  if (bytesPerRank <= runChunkBytes) {
    ensurePipelineResources(ctx);
    for (size_t elemOffset = 0; elemOffset < recvcount;) {
      size_t chunkBytes =
          std::min(runChunkBytes, bytesPerRank - elemOffset * sizeof(float));
      size_t chunkElems = chunkBytes / sizeof(float);
      uint64_t prevEpoch = ctx.epoch;
      if (prevEpoch > 0) {
        waitForEpoch(ctx.ctrl->localScratchDone[ctx.localRank], prevEpoch);
      }
      RsChunkWork work;
      ncclResult_t result = scheduleTwoRankChunk(
          ctx, sendbuff, recvbuff, recvcount, elemOffset, chunkElems,
          /*slot=*/0, scratchBuffer, stream, /*recordAsyncD2h=*/false, &work);
      if (result != ncclSuccess) return result;
      result = completeTwoRankRemote(ctx, recvbuff, work, stream,
                                     /*asyncCopy=*/false,
                                     /*deferAck=*/false,
                                     /*asyncAckDistance=*/1);
      if (result != ncclSuccess) return result;
      elemOffset += chunkElems;
    }
    return ncclSuccess;
  }
  return runTwoRankPipelinedChunks(ctx, sendbuff, recvbuff, recvcount,
                                   runChunkBytes, stream, scratchBuffer);
}

}  // namespace

ncclResult_t runLiteInterReduceScatter(void const* sendbuff, void* recvbuff,
                                      size_t recvcount, size_t bytesPerRank,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream,
                                      int rank, int nRanks, void* scratchBuffer,
                                      size_t scratchBufferSize,
                                      int nRanksPerNode,
                                      std::shared_ptr<Communicator> bootstrapComm,
                                      int cudaDevice) {
  if (!isTargetLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  if (datatype != ncclFloat32 || op != ncclSum) return ncclInvalidUsage;
  if (sendbuff == nullptr || recvbuff == nullptr || scratchBuffer == nullptr) {
    return ncclInvalidArgument;
  }
  if (recvcount > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return ncclInvalidArgument;
  }
  if (recvcount * sizeof(float) != bytesPerRank) return ncclInvalidUsage;
  if (bytesPerRank >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(nRanks)) {
    return ncclInvalidArgument;
  }
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);
  if (bytesPerRank > kMaxNativeBytesPerRank) return ncclInvalidUsage;

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    bool twoRankLayout = isTwoRankLayout(nRanks, nRanksPerNode);
    int pipelineSlots = pipelineSlotsForLayout(nRanks, nRanksPerNode);
    size_t chunkCapacity =
        chunkCapacityBytes(scratchBufferSize, twoRankLayout, pipelineSlots);
    if (chunkCapacity == 0) return ncclInvalidUsage;
    int transportPolicy = transportPolicyForLayout(
        comm, bootstrapComm, rank, nRanks, nRanksPerNode, bytesPerRank);
    auto& ctx = getContext(comm, bootstrapComm, rank, nRanks, nRanksPerNode,
                           cudaDevice, chunkCapacity, pipelineSlots,
                           transportPolicy);
    if (ctx.chunkCapacity != chunkCapacity ||
        ctx.pipelineSlots != pipelineSlots) {
      return ncclInvalidUsage;
    }
    size_t runChunkBytes = effectiveChunkBytesForLayout(
        nRanks, nRanksPerNode, bytesPerRank, chunkCapacity);

    if (isIntraFourRankLayout(nRanks, nRanksPerNode) &&
        useP2pRingForLocalFour(bytesPerRank)) {
      return runP2pRingReduceScatter(
          sendbuff, recvbuff, recvcount, bytesPerRank, datatype, comm, stream,
          rank, nRanks, scratchBuffer, scratchBufferSize);
    }

    if (isIntraFourRankLayout(nRanks, nRanksPerNode)) {
      return runLocalFourRankReduceScatter(ctx, sendbuff, recvbuff, recvcount,
                                           bytesPerRank, stream,
                                           bootstrapComm, scratchBuffer,
                                           scratchBufferSize);
    }

    if (twoRankLayout) {
      return runTwoRankReduceScatter(ctx, sendbuff, recvbuff, recvcount,
                                     bytesPerRank, runChunkBytes, stream,
                                     scratchBuffer, bootstrapComm);
    }

    if (isTwoNodeTwoGpuLayout(nRanks, nRanksPerNode)) {
      if (useTwoNodeTwoGpuHier()) {
        size_t twoNodeTwoGpuChunkBytes =
            configuredLayoutChunkBytesOverride() != 0
                ? runChunkBytes
                : (fullBytes < 8 * 1024 * 1024
                       ? runChunkBytes
                       : std::min(chunkCapacity,
                                  static_cast<size_t>(1024 * 1024)));
        return runTwoNodeTwoGpuHierReduceScatter(
            ctx, sendbuff, recvbuff, recvcount, bytesPerRank,
            twoNodeTwoGpuChunkBytes, stream, bootstrapComm, nRanks,
            scratchBuffer, scratchBufferSize);
      }
      return runSendRecvReduceScatter(
          sendbuff, recvbuff, recvcount, bytesPerRank, datatype, op, comm,
          stream, rank, nRanks, scratchBuffer, scratchBufferSize,
          nRanksPerNode, bootstrapComm, cudaDevice);
    }
    if (fullBytes < configuredSmallHostFullBytes()) {
      size_t smallRankStride =
          ctx.chunkCapacity * static_cast<size_t>(ctx.pipelineSlots);
      size_t slotBytes = std::max(fullBytes, 3 * bytesPerRank);
      if (fullBytes <= smallRankStride &&
          3 * bytesPerRank <= smallRankStride &&
          slotBytes <= smallRankStride) {
        return runSmallHostReduceScatter(ctx, sendbuff, recvbuff, recvcount,
                                         bytesPerRank, stream, bootstrapComm);
      }
    }
    if (ctx.smallPendingAckEpoch > ctx.smallAckPostedEpoch) {
      waitForCudaStream(stream);
      flushSmallPendingAck(ctx);
    }
    if (bytesPerRank > runChunkBytes) {
      return runPipelinedChunks(ctx, sendbuff, recvbuff, recvcount,
                                runChunkBytes, stream, bootstrapComm, comm,
                                nRanks, scratchBuffer, scratchBufferSize);
    }

    for (size_t elemOffset = 0; elemOffset < recvcount;) {
      size_t chunkBytes = std::min(runChunkBytes, bytesPerRank -
                                                    elemOffset * sizeof(float));
      size_t chunkElems = chunkBytes / sizeof(float);
      ncclResult_t result = runChunk(
          ctx, sendbuff, recvbuff, recvcount, elemOffset, chunkElems,
          chunkBytes, stream, bootstrapComm, comm, nRanks, scratchBuffer,
          scratchBufferSize);
      if (result != ncclSuccess) return result;
      elemOffset += chunkElems;
    }
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN("inter-node ReduceScatter failed: %s", ex.what());
    return mapException(ex);
  } catch (...) {
    WARN("inter-node ReduceScatter failed with an unknown exception");
    return ncclInternalError;
  }
}

void cleanupLiteReduceScatterContexts(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(gReduceScatterMutex);
  for (auto it = gReduceScatterContexts.begin();
       it != gReduceScatterContexts.end();) {
    if (it->first.comm == comm) {
      it = gReduceScatterContexts.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = gReduceScatterTransportPolicies.begin();
       it != gReduceScatterTransportPolicies.end();) {
    if (it->first.comm == comm) {
      it = gReduceScatterTransportPolicies.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace nccl
}  // namespace mscclpp
