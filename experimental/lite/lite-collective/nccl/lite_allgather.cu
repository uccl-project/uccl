#include "native_collectives.hpp"
#include "lite_common.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <numa.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mscclpp {
namespace nccl {
namespace {

// Multi-node AllGather is deliberately host-memory based.  Do not route it
// through the generic grouped P2P send/recv path: on this no-GDR testbed NCCL
// also stages through host memory, so the win comes from explicit host slots,
// contiguous slabs, and topology-specific overlap.
//
// Dispatch sketch:
// - tiny two-node rows: ordered host slots (`runSmallOrdered`);
// - 2nx1g large rows: one-rank 512KiB chunk pipeline (`runSingleSlab` delegates);
// - 2nx2g: single host slab with 512KiB chunks;
// - 2nx4g: NUMA/NIC split slabs when the local GPU layout is symmetric.


static constexpr int kHostTagBase = 0x560000;
static constexpr int kHostTagStride = 6;
static constexpr int kNumaTagBase = 0x565000;
static constexpr int kGpuDirectTagBase = 0x568000;
static constexpr int kGpuDirectTagStride = 1;
static constexpr int kMaxRanksPerNode = 8;
static constexpr int kMaxNodes = 16;
static constexpr int kMaxNicGroups = kMaxRanksPerNode;
static constexpr size_t kSmallCutoffBytes = 128 * 1024;
static constexpr size_t kTwoGpuSmallCutoffBytes = 128 * 1024;
static constexpr size_t kOneRankSmallCutoffBytes = 2 * 1024 * 1024;
static constexpr size_t kOneRankDirectCopyCutoffBytes = 128 * 1024;
static constexpr size_t kOneRankMappedOutputCutoffBytes = 64 * 1024;
static constexpr size_t kSmallMaxSlots = 1024;
static constexpr int kSmallSignalEvery = 256;
static constexpr size_t kMaxBytesPerRank = 16 * 1024 * 1024;
// The 2nx1g pipeline stages a full message per slot, so size its slabs to
// cover the largest optimized benchmark point instead of falling back at 16MiB.
static constexpr size_t kOneRankMaxBytesPerRank = 1024ULL * 1024 * 1024;
static constexpr size_t kDefaultPipelineChunkBytes = 2 * 1024 * 1024;
static constexpr size_t kTwoGpuPipelineChunkBytes = 512 * 1024;
static constexpr size_t kOneRankPipelineChunkBytes = 512 * 1024;
static constexpr size_t kOneRankPipelineMaxBytes = 1024ULL * 1024 * 1024;
static constexpr size_t kOneRankPipelineSendWindow = 1;
static constexpr size_t kRdmaChunkBytes = 2 * 1024 * 1024;
static constexpr size_t kDualRailMinBytes = 2 * 1024 * 1024;
static constexpr size_t kDirectSelfCopyMinBytes = 512 * 1024;
static constexpr int kSignalEveryN = 256;
static constexpr int kPollSpinsBeforeYield = 65536;
static constexpr uint64_t kPipeValueStride =
    (kOneRankPipelineMaxBytes + kOneRankPipelineChunkBytes - 1) /
    kOneRankPipelineChunkBytes;
static constexpr bool kEnableOneRankGpuDirect = false;

using lite::createOwnedShm;
using lite::cudaResult;
using lite::getAvailableIBTransports;
using lite::InitGuard;
using lite::mapException;
using lite::mapShm;
using lite::placeOnNuma;
using lite::publishInitStatus;
using lite::selectIBTransportForGpu;
using lite::waitForEpoch;

struct HostControl {
  alignas(64) std::atomic<uint64_t>
      d2hReady[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> rdmaReady[kMaxNodes];
  alignas(64) std::atomic<uint64_t> rdmaSignal[kMaxNodes];
  alignas(64) std::atomic<uint64_t>
      h2dDone[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> ackReady[kMaxNodes];
  alignas(64) std::atomic<uint64_t> ackSignal[kMaxNodes];
  alignas(64) std::atomic<uint64_t> pipeReady[kMaxNodes];
};

struct HostNames {
  char sendName[96] = {};
  char recvName[96] = {};
  char ctrlName[96] = {};
};

struct AgContext {
  bool initialized = false;
  bool initializing = false;
  bool owner = false;
  bool isLeader = false;
  int rank = -1;
  int worldSize = -1;
  int nRanksPerNode = -1;
  int nodeCount = -1;
  int localRank = -1;
  int nodeId = -1;
  int localLeader = -1;
  int remoteLeader = -1;
  int cudaDevice = -1;
  int transportDevice = -1;
  int numaNode = -1;
  int groupId = 0;
  int groupBase = 0;
  int groupSize = 0;
  bool numaSplit = false;
  size_t chunkCapacity = kMaxBytesPerRank;
  size_t slabBytes = 0;
  uint64_t epoch = 0;
  size_t smallPerSlotBytes = 0;

  std::string sendName;
  std::string recvName;
  std::string ctrlName;
  void* sendMapping = nullptr;
  void* recvMapping = nullptr;
  void* ctrlMapping = nullptr;
  char* sendSlab = nullptr;
  char* recvSlab = nullptr;
  HostControl* ctrl = nullptr;
  char const* sendDeviceSlab = nullptr;
  char* ctrlDeviceSlab = nullptr;
  bool sendHostRegistered = false;
  bool recvHostRegistered = false;
  bool ctrlHostRegistered = false;

  mscclpp::Transport transport = mscclpp::Transport::Unknown;
  mscclpp::RegisteredMemory sendMemory;
  mscclpp::RegisteredMemory recvMemory;
  mscclpp::RegisteredMemory ctrlMemory;
  mscclpp::RegisteredMemory remoteSendMemory;
  mscclpp::RegisteredMemory remoteRecvMemory;
  mscclpp::RegisteredMemory remoteCtrlMemory;
  mscclpp::Connection connection;
  mscclpp::Transport rail2Transport = mscclpp::Transport::Unknown;
  mscclpp::RegisteredMemory rail2SendMemory;
  mscclpp::RegisteredMemory rail2RecvMemory;
  mscclpp::RegisteredMemory rail2RemoteRecvMemory;
  mscclpp::Connection rail2Connection;
  std::vector<int> peerNodeIds;
  std::vector<int> peerLeaders;
  std::vector<mscclpp::Connection> peerConnections;
  std::vector<mscclpp::RegisteredMemory> peerRemoteRecvMemory;
  std::vector<mscclpp::RegisteredMemory> peerRemoteCtrlMemory;
  std::shared_ptr<mscclpp::IbQp> smallQp;
  mscclpp::IbMr const* smallSendMr = nullptr;
  mscclpp::IbMr const* smallCtrlMr = nullptr;
  mscclpp::IbMrInfo smallRemoteSendMrInfo{};
  mscclpp::IbMrInfo smallRemoteRecvMrInfo{};
  mscclpp::IbMrInfo smallRemoteCtrlMrInfo{};
  int smallWrCount = 0;
  std::array<uint64_t, kMaxNodes> rdmaReadyAtomicEpoch{};
  void const* gdrSendPtr = nullptr;
  size_t gdrSendBytes = 0;
  mscclpp::RegisteredMemory gdrSendMemory;
  mscclpp::IbMr const* gdrSendMr = nullptr;
  void* gdrRecvPtr = nullptr;
  size_t gdrRecvBytes = 0;
  mscclpp::RegisteredMemory gdrRecvMemory;
  mscclpp::RegisteredMemory gdrRemoteRecvMemory;
  mscclpp::IbMrInfo gdrRemoteRecvMrInfo{};
  bool gdrRemoteRecvValid = false;
  bool gdrDisabled = false;
  cudaStream_t d2hStream = nullptr;
  cudaStream_t h2dStream = nullptr;
  cudaEvent_t inputReadyEvent = nullptr;
  cudaEvent_t h2dDoneEvent = nullptr;
  std::vector<cudaEvent_t> h2dSlotEvents;
  std::vector<cudaEvent_t> d2hChunkEvents;
  std::mutex initMutex;
  std::condition_variable initCv;
  std::exception_ptr initException = nullptr;

  ~AgContext() {
    smallQp.reset();
    smallSendMr = nullptr;
    smallCtrlMr = nullptr;
    if (inputReadyEvent) cudaEventDestroy(inputReadyEvent);
    if (h2dDoneEvent) cudaEventDestroy(h2dDoneEvent);
    for (auto event : h2dSlotEvents) {
      if (event) cudaEventDestroy(event);
    }
    for (auto event : d2hChunkEvents) {
      if (event) cudaEventDestroy(event);
    }
    if (d2hStream) cudaStreamDestroy(d2hStream);
    if (h2dStream) cudaStreamDestroy(h2dStream);
    connection = mscclpp::Connection{};
    rail2Connection = mscclpp::Connection{};
    rail2RemoteRecvMemory = mscclpp::RegisteredMemory{};
    rail2RecvMemory = mscclpp::RegisteredMemory{};
    rail2SendMemory = mscclpp::RegisteredMemory{};
    remoteCtrlMemory = mscclpp::RegisteredMemory{};
    remoteRecvMemory = mscclpp::RegisteredMemory{};
    remoteSendMemory = mscclpp::RegisteredMemory{};
    ctrlMemory = mscclpp::RegisteredMemory{};
    recvMemory = mscclpp::RegisteredMemory{};
    sendMemory = mscclpp::RegisteredMemory{};
    if (sendHostRegistered) cudaHostUnregister(sendMapping);
    if (recvHostRegistered) cudaHostUnregister(recvMapping);
    if (ctrlHostRegistered) cudaHostUnregister(ctrlMapping);
    if (sendMapping) munmap(sendMapping, slabBytes);
    if (recvMapping) munmap(recvMapping, slabBytes);
    if (ctrlMapping) munmap(ctrlMapping, sizeof(HostControl));
    if (owner) {
      if (!sendName.empty()) shm_unlink(sendName.c_str());
      if (!recvName.empty()) shm_unlink(recvName.c_str());
      if (!ctrlName.empty()) shm_unlink(ctrlName.c_str());
    }
  }
};

struct NicGroupLayout {
  int count = 0;
  std::array<int, kMaxNicGroups> base = {};
  std::array<int, kMaxNicGroups> size = {};
  std::array<int, kMaxNicGroups> transportDevice = {};
};

std::mutex gAllGatherContextMutex;
std::unordered_map<ncclComm_t, std::unique_ptr<AgContext>>
    gSingleContexts;
std::unordered_map<
    ncclComm_t,
    std::array<std::unique_ptr<AgContext>,
               kMaxNicGroups>>
    gNumaContexts;
std::unordered_map<ncclComm_t, std::vector<int>> gCudaDevicesByComm;
std::unordered_map<ncclComm_t, NicGroupLayout> gLayoutByComm;

__device__ unsigned long long volatile* ctrlU64(char* ctrl, size_t offset) {
  return reinterpret_cast<unsigned long long volatile*>(ctrl + offset);
}

__global__ void oneRankRegisterCopyKernel(
    char const* send, char* recv, char* slot, char* ctrl, size_t slotOffset,
    size_t flagOffset, size_t bytesPerRank, int rank, bool copySelf,
    size_t d2hReadyOff, unsigned long long epoch) {
  int remoteRank = 1 - rank;
  size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank;
  size_t remoteOffset = static_cast<size_t>(remoteRank) * bytesPerRank;
  using Vec = unsigned long long;
  size_t vecBytes = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto* localSlot = reinterpret_cast<Vec*>(slot + slotOffset + selfOffset);
  size_t vecCount = vecBytes / sizeof(Vec);
  for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
    localSlot[i] = sendVec[i];
  }
  for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank; i += blockDim.x) {
    slot[slotOffset + selfOffset + i] = send[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) *ctrlU64(ctrl, d2hReadyOff) = epoch;
  auto* selfDst = reinterpret_cast<Vec*>(recv + selfOffset);
  if (copySelf) {
    for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
      selfDst[i] = sendVec[i];
    }
    for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank;
         i += blockDim.x) {
      recv[selfOffset + i] = send[i];
    }
  }

  while (*reinterpret_cast<unsigned long long volatile*>(
             slot + flagOffset) != epoch) {
  }

  auto const* remoteSlot =
      reinterpret_cast<Vec const*>(slot + slotOffset + remoteOffset);
  auto* remoteDst = reinterpret_cast<Vec*>(recv + remoteOffset);
  for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
    remoteDst[i] = remoteSlot[i];
  }
  for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank; i += blockDim.x) {
    recv[remoteOffset + i] = slot[slotOffset + remoteOffset + i];
  }
}

__global__ void oneRankTinyRegisterCopyKernel(
    char const* send, char* recv, char* slot, char* ctrl, size_t slotOffset,
    size_t flagOffset, size_t bytesPerRank, int rank, bool copySelf,
    size_t d2hReadyOff, unsigned long long epoch) {
  int remoteRank = 1 - rank;
  size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank;
  size_t remoteOffset = static_cast<size_t>(remoteRank) * bytesPerRank;
  using Vec = unsigned long long;
  size_t vecBytes = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto* localSlot = reinterpret_cast<Vec*>(slot + slotOffset + selfOffset);
  for (size_t i = 0; i < vecBytes / sizeof(Vec); ++i) {
    localSlot[i] = sendVec[i];
  }
  for (size_t i = vecBytes; i < bytesPerRank; ++i) {
    slot[slotOffset + selfOffset + i] = send[i];
  }
  *ctrlU64(ctrl, d2hReadyOff) = epoch;

  while (*reinterpret_cast<unsigned long long volatile*>(
             slot + flagOffset) != epoch) {
  }

  auto const* remoteSlot =
      reinterpret_cast<Vec const*>(slot + slotOffset + remoteOffset);
  auto* remoteDst = reinterpret_cast<Vec*>(recv + remoteOffset);
  auto* selfDst = reinterpret_cast<Vec*>(recv + selfOffset);
  for (size_t i = 0; i < vecBytes / sizeof(Vec); ++i) {
    remoteDst[i] = remoteSlot[i];
    if (copySelf) selfDst[i] = sendVec[i];
  }
  for (size_t i = vecBytes; i < bytesPerRank; ++i) {
    recv[remoteOffset + i] = slot[slotOffset + remoteOffset + i];
    if (copySelf) recv[selfOffset + i] = send[i];
  }
}

__global__ void oneRankCompactRegisterCopyKernel(
    char const* send, char* recv, char* slot, char* ctrl, size_t slotOffset,
    size_t segmentStride, size_t flagInSegment, size_t bytesPerRank, int rank,
    bool copySelf, size_t d2hReadyOff, unsigned long long epoch) {
  int remoteRank = 1 - rank;
  size_t selfOutputOffset = static_cast<size_t>(rank) * bytesPerRank;
  size_t remoteOutputOffset = static_cast<size_t>(remoteRank) * bytesPerRank;
  size_t localSegment = slotOffset + static_cast<size_t>(rank) * segmentStride;
  size_t remoteSegment =
      slotOffset + static_cast<size_t>(remoteRank) * segmentStride;
  using Vec = unsigned long long;
  size_t vecBytes = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto* localSlot = reinterpret_cast<Vec*>(slot + localSegment);
  size_t vecCount = vecBytes / sizeof(Vec);
  for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
    localSlot[i] = sendVec[i];
  }
  for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank; i += blockDim.x) {
    slot[localSegment + i] = send[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *ctrlU64(ctrl, d2hReadyOff) = epoch;
  }
  auto* selfDst = reinterpret_cast<Vec*>(recv + selfOutputOffset);
  if (copySelf) {
    for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
      selfDst[i] = sendVec[i];
    }
    for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank;
         i += blockDim.x) {
      recv[selfOutputOffset + i] = send[i];
    }
  }

  if (threadIdx.x == 0) {
    while (*reinterpret_cast<unsigned long long volatile*>(
               slot + remoteSegment + flagInSegment) != epoch) {
    }
  }
  __syncthreads();

  auto const* remoteSlot = reinterpret_cast<Vec const*>(slot + remoteSegment);
  auto* remoteDst = reinterpret_cast<Vec*>(recv + remoteOutputOffset);
  for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
    remoteDst[i] = remoteSlot[i];
  }
  for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank; i += blockDim.x) {
    recv[remoteOutputOffset + i] = slot[remoteSegment + i];
  }
}

__global__ void multiRankRegisterPackKernel(
    char const* send, char* slot, char* ctrl, size_t slotOffset,
    size_t rankOffset, size_t bytesPerRank, size_t d2hReadyOff,
    unsigned long long epoch) {
  using Vec = unsigned long long;
  size_t vecBytes = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto* slotVec =
      reinterpret_cast<Vec*>(slot + slotOffset + rankOffset);
  size_t vecCount = vecBytes / sizeof(Vec);
  for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
    slotVec[i] = sendVec[i];
  }
  for (size_t i = vecBytes + threadIdx.x; i < bytesPerRank; i += blockDim.x) {
    slot[slotOffset + rankOffset + i] = send[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *ctrlU64(ctrl, d2hReadyOff) = epoch;
  }
}

__global__ void multiRankTinyPackKernel(
    char const* send, char* slot, char* ctrl, size_t slotOffset,
    size_t rankOffset, size_t bytesPerRank, size_t d2hReadyOff,
    unsigned long long epoch) {
  using Vec = unsigned long long;
  size_t vecBytes = (bytesPerRank / sizeof(Vec)) * sizeof(Vec);
  auto const* sendVec = reinterpret_cast<Vec const*>(send);
  auto* slotVec =
      reinterpret_cast<Vec*>(slot + slotOffset + rankOffset);
  for (size_t i = 0; i < vecBytes / sizeof(Vec); ++i) {
    slotVec[i] = sendVec[i];
  }
  for (size_t i = vecBytes; i < bytesPerRank; ++i) {
    slot[slotOffset + rankOffset + i] = send[i];
  }
  *ctrlU64(ctrl, d2hReadyOff) = epoch;
}

__global__ void multiRankHostRecvKernel(char* recv, char const* slot,
                                       char* ctrl, size_t d2hReadyBase,
                                       int localGroupSize,
                                       size_t slotOffset, size_t flagOffset,
                                       size_t fullBytes,
                                       unsigned long long epoch) {
  if (threadIdx.x == 0) {
    while (*reinterpret_cast<unsigned long long const volatile*>(
               slot + flagOffset) != epoch) {
    }
    for (int i = 0; i < localGroupSize; ++i) {
      while (*ctrlU64(ctrl, d2hReadyBase +
                                static_cast<size_t>(i) *
                                    sizeof(unsigned long long)) < epoch) {
      }
    }
  }
  __syncthreads();

  using Vec = unsigned long long;
  size_t vecBytes = (fullBytes / sizeof(Vec)) * sizeof(Vec);
  auto const* slotVec = reinterpret_cast<Vec const*>(slot + slotOffset);
  auto* recvVec = reinterpret_cast<Vec*>(recv);
  size_t vecCount = vecBytes / sizeof(Vec);
  for (size_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
    recvVec[i] = slotVec[i];
  }
  for (size_t i = vecBytes + threadIdx.x; i < fullBytes; i += blockDim.x) {
    recv[i] = slot[slotOffset + i];
  }
}

int hostTag(int rank, int worldSize, int remoteLeader,
                           int slot) {
  int lo = std::min(rank, remoteLeader);
  int hi = std::max(rank, remoteLeader);
  int pairIndex = lo * worldSize + hi;
  return kHostTagBase +
         pairIndex * kHostTagStride + slot;
}

int numaTag(int rank, int worldSize, int remoteLeader,
                               int slot) {
  int lo = std::min(rank, remoteLeader);
  int hi = std::max(rank, remoteLeader);
  int pairIndex = lo * worldSize + hi;
  return kNumaTagBase +
         pairIndex * kHostTagStride + slot;
}

int gpuDirectTag(int rank, int worldSize, int remoteLeader, int slot) {
  int lo = std::min(rank, remoteLeader);
  int hi = std::max(rank, remoteLeader);
  int pairIndex = lo * worldSize + hi;
  return kGpuDirectTagBase + pairIndex * kGpuDirectTagStride + slot;
}

void unlinkOwnedShm(HostNames const& names) {
  if (names.sendName[0] != '\0') shm_unlink(names.sendName);
  if (names.recvName[0] != '\0') shm_unlink(names.recvName);
  if (names.ctrlName[0] != '\0') shm_unlink(names.ctrlName);
}

void waitForSlotReady(char const* flagPtr, uint64_t epoch) {
  auto const* value = reinterpret_cast<uint64_t const volatile*>(flagPtr);
  int spins = 0;
  while (*value != epoch) {
    if (spins++ < kPollSpinsBeforeYield) {
      asm volatile("pause" ::: "memory");
    } else {
      std::this_thread::yield();
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
}

void waitForCudaStream(cudaStream_t stream) {
  int spins = 0;
  while (true) {
    cudaError_t result = cudaStreamQuery(stream);
    if (result == cudaSuccess) return;
    if (result != cudaErrorNotReady) MSCCLPP_CUDATHROW(result);
    if (spins++ < kPollSpinsBeforeYield) {
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
    if (spins++ < kPollSpinsBeforeYield) {
      asm volatile("pause" ::: "memory");
    } else {
      std::this_thread::yield();
    }
  }
}

void ensurePipelineStreams(AgContext& ctx) {
  if (ctx.d2hStream == nullptr) {
    MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&ctx.d2hStream,
                                                cudaStreamNonBlocking));
  }
  if (ctx.h2dStream == nullptr) {
    MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&ctx.h2dStream,
                                                cudaStreamNonBlocking));
  }
  if (ctx.inputReadyEvent == nullptr) {
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.inputReadyEvent,
                                               cudaEventDisableTiming));
  }
  if (ctx.h2dDoneEvent == nullptr) {
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.h2dDoneEvent,
                                               cudaEventDisableTiming));
  }
}

void ensureSlotEvents(AgContext& ctx, size_t slotCount) {
  while (ctx.h2dSlotEvents.size() < slotCount) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ctx.h2dSlotEvents.push_back(event);
  }
}

void ensureD2hChunkEvents(AgContext& ctx, size_t chunkCount) {
  while (ctx.d2hChunkEvents.size() < chunkCount) {
    cudaEvent_t event = nullptr;
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ctx.d2hChunkEvents.push_back(event);
  }
}

size_t ctrlArrayOffset(size_t baseOffset, int nodeId) {
  return baseOffset + static_cast<size_t>(nodeId) *
                          sizeof(std::atomic<uint64_t>);
}

size_t d2hReadyOffset(int localRank) {
  return ctrlArrayOffset(offsetof(HostControl, d2hReady), localRank);
}

size_t rdmaReadyOffset(int nodeId) {
  return ctrlArrayOffset(offsetof(HostControl, rdmaReady), nodeId);
}

size_t rdmaSignalOffset(int nodeId) {
  return ctrlArrayOffset(offsetof(HostControl, rdmaSignal), nodeId);
}

size_t ackReadyOffset(int nodeId) {
  return ctrlArrayOffset(offsetof(HostControl, ackReady), nodeId);
}

size_t ackSignalOffset(int nodeId) {
  return ctrlArrayOffset(offsetof(HostControl, ackSignal), nodeId);
}

size_t pipeReadyOffset(int nodeId) {
  return ctrlArrayOffset(offsetof(HostControl, pipeReady), nodeId);
}

void pollQp(AgContext& ctx) {
  if (!ctx.smallQp) return;
  while (ctx.smallQp->getNumSendCqItems() > 0) {
    int wcNum = ctx.smallQp->pollSendCq();
    if (wcNum < 0) {
      throw mscclpp::Error("small allgather pollSendCq failed",
                           mscclpp::ErrorCode::SystemError);
    }
    for (int i = 0; i < wcNum; ++i) {
      if (ctx.smallQp->getSendWcStatus(i) != 0) {
        throw mscclpp::Error("small allgather RDMA write failed: " +
                                 ctx.smallQp->getSendWcStatusString(i),
                             mscclpp::ErrorCode::SystemError);
      }
    }
  }
}

void signalRdmaReadyAtomic(AgContext& ctx,
                           mscclpp::Connection& connection,
                           mscclpp::RegisteredMemory remoteCtrlMemory,
                           int peerNode, uint64_t epoch) {
  uint64_t& publishedEpoch = ctx.rdmaReadyAtomicEpoch[peerNode];
  connection.updateAndSync(remoteCtrlMemory, rdmaReadyOffset(ctx.nodeId),
                           &publishedEpoch, epoch);
  connection.flush();
}

void signalRdmaReadyAtomic(AgContext& ctx, size_t peer, uint64_t epoch) {
  signalRdmaReadyAtomic(ctx, ctx.peerConnections[peer],
                        ctx.peerRemoteCtrlMemory[peer],
                        ctx.peerNodeIds[peer], epoch);
}

void writeOrderedSlot(AgContext& ctx, size_t dataOffset, size_t flagOffset,
                       size_t dataBytes) {
  size_t flagSrcOffset = rdmaSignalOffset(ctx.nodeId);
  if (!ctx.smallQp || ctx.smallSendMr == nullptr || ctx.smallCtrlMr == nullptr) {
    ctx.connection.write(ctx.remoteSendMemory, dataOffset, ctx.sendMemory,
                         dataOffset, dataBytes);
    ctx.connection.write(ctx.remoteSendMemory, flagOffset, ctx.ctrlMemory,
                         flagSrcOffset, sizeof(uint64_t));
    ctx.connection.flush();
    return;
  }
  bool signaled = (++ctx.smallWrCount % kSmallSignalEvery) == 0;
  ctx.smallQp->stageSendWrite(ctx.smallSendMr, ctx.smallRemoteSendMrInfo,
                              static_cast<uint32_t>(dataBytes), /*wrId=*/0,
                              dataOffset, dataOffset, false);
  ctx.smallQp->stageSendWrite(ctx.smallCtrlMr, ctx.smallRemoteSendMrInfo,
                              sizeof(uint64_t), /*wrId=*/0, flagSrcOffset,
                              flagOffset, signaled);
  ctx.smallQp->postSend();
  if (signaled) pollQp(ctx);
}

void writeCompactSlot(AgContext& ctx, size_t segmentOffset,
                      size_t segmentBytes) {
  if (!ctx.smallQp || ctx.smallSendMr == nullptr) {
    ctx.connection.write(ctx.remoteSendMemory, segmentOffset, ctx.sendMemory,
                         segmentOffset, segmentBytes);
    ctx.connection.flush();
    return;
  }
  bool signaled = (++ctx.smallWrCount % kSmallSignalEvery) == 0;
  ctx.smallQp->stageSendWrite(ctx.smallSendMr, ctx.smallRemoteSendMrInfo,
                              static_cast<uint32_t>(segmentBytes), /*wrId=*/0,
                              segmentOffset, segmentOffset, signaled);
  ctx.smallQp->postSend();
  if (signaled) pollQp(ctx);
}

bool writeDataDirectToRemoteRecv(AgContext& ctx, size_t remoteBase,
                                 size_t sendBase, size_t blockBytes) {
  if (!ctx.smallQp || ctx.smallSendMr == nullptr) return false;
  size_t off = 0;
  while (off < blockBytes) {
    size_t chunk = std::min(kRdmaChunkBytes, blockBytes - off);
    ctx.smallQp->stageSendWrite(ctx.smallSendMr, ctx.smallRemoteRecvMrInfo,
                                static_cast<uint32_t>(chunk), /*wrId=*/0,
                                sendBase + off, remoteBase + off,
                                /*signaled=*/false);
    off += chunk;
  }
  ctx.smallQp->postSend();
  return true;
}

bool writeDataStripedToRemoteRecv(AgContext& ctx, size_t remoteBase,
                                  size_t sendBase, size_t blockBytes) {
  if (ctx.rail2Transport == mscclpp::Transport::Unknown ||
      blockBytes < kDualRailMinBytes) {
    return false;
  }
  size_t rail1Bytes = blockBytes / 2;
  size_t rail2Bytes = blockBytes - rail1Bytes;
  if (rail1Bytes == 0 || rail2Bytes == 0) return false;

  if (!writeDataDirectToRemoteRecv(ctx, remoteBase, sendBase, rail1Bytes)) {
    return false;
  }
  ctx.rail2Connection.write(ctx.rail2RemoteRecvMemory,
                            remoteBase + rail1Bytes, ctx.rail2SendMemory,
                            sendBase + rail1Bytes, rail2Bytes);
  ctx.rail2Connection.flush();
  return true;
}

bool writePipelineChunkDirect(AgContext& ctx, size_t remoteBase,
                              size_t sendBase, size_t bytes,
                              size_t flagSrcOffset) {
  if (!ctx.smallQp || ctx.smallSendMr == nullptr) {
    return false;
  }
  bool signaled = (++ctx.smallWrCount % kSmallSignalEvery) == 0;
  ctx.smallQp->stageSendWrite(ctx.smallSendMr, ctx.smallRemoteRecvMrInfo,
                              static_cast<uint32_t>(bytes), /*wrId=*/0,
                              sendBase, remoteBase, false);
  ctx.smallQp->stageSendWrite(ctx.smallSendMr, ctx.smallRemoteCtrlMrInfo,
                              sizeof(uint64_t), /*wrId=*/0,
                              flagSrcOffset,
                              pipeReadyOffset(ctx.nodeId), signaled);
  ctx.smallQp->postSend();
  if (signaled) pollQp(ctx);
  return true;
}

bool ensureGpuDirectOneRank(
    AgContext& ctx, std::shared_ptr<Communicator> bootstrapComm,
    void const* sendbuff, size_t bytesPerRank, void* recvbuff,
    size_t fullBytes) {
  if (ctx.gdrDisabled || ctx.nodeCount != 2 || ctx.nRanksPerNode != 1 ||
      !ctx.smallQp || ctx.smallCtrlMr == nullptr) {
    return false;
  }
  try {
    mscclpp::TransportFlags flags(ctx.transport);
    if (ctx.gdrSendPtr != sendbuff || bytesPerRank > ctx.gdrSendBytes) {
      ctx.gdrSendMemory = bootstrapComm->registerMemory(
          const_cast<void*>(sendbuff), bytesPerRank, flags);
      ctx.gdrSendMemory.getIbMrInfo(ctx.transport, &ctx.gdrSendMr, nullptr);
      ctx.gdrSendPtr = sendbuff;
      ctx.gdrSendBytes = bytesPerRank;
    }
    if (ctx.gdrRecvPtr != recvbuff || fullBytes > ctx.gdrRecvBytes) {
      ctx.gdrRecvMemory =
          bootstrapComm->registerMemory(recvbuff, fullBytes, flags);
      ctx.gdrRecvPtr = recvbuff;
      ctx.gdrRecvBytes = fullBytes;
      int tag = gpuDirectTag(ctx.rank, ctx.worldSize, ctx.remoteLeader, 0);
      bootstrapComm->sendMemory(ctx.gdrRecvMemory, ctx.remoteLeader, tag);
      auto remoteFuture = bootstrapComm->recvMemory(ctx.remoteLeader, tag);
      ctx.gdrRemoteRecvMemory = remoteFuture.get();
      ctx.gdrRemoteRecvMemory.getIbMrInfo(ctx.transport, nullptr,
                                          &ctx.gdrRemoteRecvMrInfo);
      ctx.gdrRemoteRecvValid = true;
    }
  } catch (std::exception const& ex) {
    WARN("AllGather GPUDirect path disabled: %s", ex.what());
    ctx.gdrDisabled = true;
    ctx.gdrSendMr = nullptr;
    ctx.gdrRemoteRecvValid = false;
    return false;
  }
  return ctx.gdrSendMr != nullptr && ctx.gdrRemoteRecvValid;
}

bool writeGpuDirectOneRank(AgContext& ctx, size_t bytesPerRank,
                           uint64_t epoch) {
  if (!ctx.smallQp || ctx.gdrSendMr == nullptr ||
      ctx.smallCtrlMr == nullptr || !ctx.gdrRemoteRecvValid) {
    return false;
  }
  ctx.ctrl->rdmaSignal[ctx.nodeId].store(epoch, std::memory_order_release);
  size_t dstOffset = static_cast<size_t>(ctx.rank) * bytesPerRank;
  ctx.smallQp->stageSendWrite(ctx.gdrSendMr, ctx.gdrRemoteRecvMrInfo,
                              static_cast<uint32_t>(bytesPerRank), /*wrId=*/0,
                              /*srcOffset=*/0, dstOffset, false);
  ctx.smallQp->postSend();
  signalRdmaReadyAtomic(ctx, ctx.connection, ctx.remoteCtrlMemory,
                        1 - ctx.nodeId, epoch);
  return true;
}

template <typename SlotGetter>
AgContext& getContext(
    SlotGetter&& slotGetter, ncclComm_t commHandle,
    std::shared_ptr<Communicator> bootstrapComm, int rank, int nRanks,
    int nRanksPerNode, int cudaDevice, int groupId, int groupBase,
    int groupSize, int transportDevice, bool numaSplit, char const* opName) {
  AgContext* ctx = nullptr;
  bool shouldInitialize = false;
  {
    std::lock_guard<std::mutex> lock(gAllGatherContextMutex);
    auto& existing = slotGetter();
    if (!existing) {
      existing = std::make_unique<AgContext>();
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
      throw mscclpp::Error("AllGather context initialization failed",
                           mscclpp::ErrorCode::InternalError);
    }
  }
  InitGuard<AgContext> initGuard(ctx);

  ctx->rank = rank;
  ctx->worldSize = nRanks;
  ctx->nRanksPerNode = nRanksPerNode;
  ctx->nodeCount = nRanks / nRanksPerNode;
  if (ctx->nodeCount <= 0 || ctx->nodeCount > kMaxNodes) {
    throw mscclpp::Error("AllGather node count exceeds host-control capacity",
                         mscclpp::ErrorCode::InvalidUsage);
  }
  ctx->localRank = rank % nRanksPerNode;
  ctx->nodeId = rank / nRanksPerNode;
  ctx->localLeader = ctx->nodeId * nRanksPerNode + groupBase;
  ctx->remoteLeader =
      ctx->nodeCount == 2 ? (1 - ctx->nodeId) * nRanksPerNode + groupBase : -1;
  ctx->cudaDevice = cudaDevice;
  ctx->transportDevice = transportDevice;
  ctx->groupId = groupId;
  ctx->groupBase = groupBase;
  ctx->groupSize = groupSize;
  ctx->numaSplit = numaSplit;
  // 2nx1g uses one full-message slot so it can keep the one-rank chunk
  // pipeline active through 1GiB instead of falling back to generic slabs.
  // Other layouts keep the compact 16MiB slab capacity.
  if (!numaSplit && ctx->nodeCount == 2 && nRanksPerNode == 1) {
    ctx->chunkCapacity = kOneRankMaxBytesPerRank;
  }
  try {
    ctx->numaNode = mscclpp::getDeviceNumaNode(transportDevice);
  } catch (...) {
    ctx->numaNode = -1;
  }
  ctx->isLeader = rank == ctx->localLeader;
  ctx->owner = ctx->isLeader;
  ctx->slabBytes = static_cast<size_t>(ctx->nodeCount) *
                   static_cast<size_t>(groupSize) * ctx->chunkCapacity;
  ctx->peerNodeIds.clear();
  ctx->peerLeaders.clear();
  for (int peerNode = 0; peerNode < ctx->nodeCount; ++peerNode) {
    if (peerNode == ctx->nodeId) continue;
    ctx->peerNodeIds.push_back(peerNode);
    ctx->peerLeaders.push_back(peerNode * nRanksPerNode + groupBase);
  }

  HostNames localNames;
  ncclResult_t shmCreateResult = ncclSuccess;
  std::string shmCreateMessage;
  try {
    if (ctx->isLeader) {
      auto commNonce = static_cast<unsigned long long>(
          reinterpret_cast<uintptr_t>(commHandle));
      std::snprintf(localNames.sendName, sizeof(localNames.sendName),
                    "/mint_ag_%llx_%d_%d_%d_g%d_s", commNonce, getpid(), rank,
                    nRanks, groupId);
      std::snprintf(localNames.recvName, sizeof(localNames.recvName),
                    "/mint_ag_%llx_%d_%d_%d_g%d_r", commNonce, getpid(), rank,
                    nRanks, groupId);
      std::snprintf(localNames.ctrlName, sizeof(localNames.ctrlName),
                    "/mint_ag_%llx_%d_%d_%d_g%d_c", commNonce, getpid(), rank,
                    nRanks, groupId);
      createOwnedShm(localNames.sendName, ctx->slabBytes);
      createOwnedShm(localNames.recvName, ctx->slabBytes);
      createOwnedShm(localNames.ctrlName,
                     sizeof(HostControl));
    }
  } catch (std::exception const& ex) {
    shmCreateResult = mapException(ex);
    shmCreateMessage = ex.what();
  } catch (...) {
    shmCreateResult = ncclInternalError;
    shmCreateMessage = "unknown shared-memory create exception";
  }
  try {
    std::string stage = std::string(opName) + " shared-memory create";
    publishInitStatus(bootstrapComm, rank, nRanks, shmCreateResult,
                                shmCreateMessage, stage.c_str());
  } catch (...) {
    if (ctx->isLeader) unlinkOwnedShm(localNames);
    throw;
  }

  std::vector<HostNames> allNames(nRanks);
  allNames[rank] = localNames;
  bootstrapComm->bootstrap()->allGather(allNames.data(),
                                        sizeof(HostNames));
  HostNames const& leaderNames = allNames[ctx->localLeader];
  ctx->sendName = leaderNames.sendName;
  ctx->recvName = leaderNames.recvName;
  ctx->ctrlName = leaderNames.ctrlName;

  ncclResult_t shmMapResult = ncclSuccess;
  std::string shmMapMessage;
  try {
    ctx->sendMapping = mapShm(ctx->sendName, ctx->slabBytes);
    ctx->recvMapping = mapShm(ctx->recvName, ctx->slabBytes);
    ctx->ctrlMapping =
        mapShm(ctx->ctrlName, sizeof(HostControl));
    ctx->sendSlab = static_cast<char*>(ctx->sendMapping);
    ctx->recvSlab = static_cast<char*>(ctx->recvMapping);
    ctx->ctrl = static_cast<HostControl*>(ctx->ctrlMapping);
    if (ctx->isLeader) {
      placeOnNuma(ctx->sendMapping, ctx->slabBytes, ctx->numaNode,
                             "allgather send slab");
      placeOnNuma(ctx->recvMapping, ctx->slabBytes, ctx->numaNode,
                             "allgather recv slab");
      std::memset(ctx->ctrlMapping, 0, sizeof(HostControl));
      new (ctx->ctrl) HostControl{};
    }
  } catch (std::exception const& ex) {
    shmMapResult = mapException(ex);
    shmMapMessage = ex.what();
  } catch (...) {
    shmMapResult = ncclInternalError;
    shmMapMessage = "unknown shared-memory map exception";
  }
  std::string mapStage = std::string(opName) + " shared-memory map";
  publishInitStatus(bootstrapComm, rank, nRanks, shmMapResult,
                              shmMapMessage, mapStage.c_str());
  bootstrapComm->bootstrap()->barrier();

  ncclResult_t setupResult = ncclSuccess;
  std::string setupMessage;
  try {
    cudaError_t sendRegister = cudaHostRegister(
        ctx->sendMapping, ctx->slabBytes,
        cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (sendRegister == cudaSuccess) {
      void* sendDeviceSlab = nullptr;
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&sendDeviceSlab,
                                                 ctx->sendMapping, 0));
      ctx->sendDeviceSlab = static_cast<char const*>(sendDeviceSlab);
    } else {
      cudaGetLastError();
      MSCCLPP_CUDATHROW(cudaHostRegister(ctx->sendMapping, ctx->slabBytes,
                                         cudaHostRegisterPortable));
      ctx->sendDeviceSlab = nullptr;
    }
    ctx->sendHostRegistered = true;
    MSCCLPP_CUDATHROW(cudaHostRegister(ctx->recvMapping, ctx->slabBytes,
                                       cudaHostRegisterPortable));
    ctx->recvHostRegistered = true;
    cudaError_t ctrlRegister = cudaHostRegister(
        ctx->ctrlMapping, sizeof(HostControl),
        cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (ctrlRegister == cudaSuccess) {
      void* ctrlDeviceSlab = nullptr;
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&ctrlDeviceSlab,
                                                 ctx->ctrlMapping, 0));
      ctx->ctrlDeviceSlab = static_cast<char*>(ctrlDeviceSlab);
    } else {
      cudaGetLastError();
      MSCCLPP_CUDATHROW(cudaHostRegister(ctx->ctrlMapping,
                                         sizeof(HostControl),
                                         cudaHostRegisterPortable));
      ctx->ctrlDeviceSlab = nullptr;
    }
    ctx->ctrlHostRegistered = true;

    if (ctx->isLeader) {
      ctx->transport = selectIBTransportForGpu(transportDevice);
      if (ctx->transport == mscclpp::Transport::Unknown) {
        throw mscclpp::Error("AllGather requires IB transport",
                             mscclpp::ErrorCode::InvalidUsage);
      }
      mscclpp::TransportFlags transportFlags(ctx->transport);
      ctx->sendMemory =
          bootstrapComm->registerMemory(ctx->sendSlab, ctx->slabBytes,
                                        transportFlags);
      ctx->recvMemory =
          bootstrapComm->registerMemory(ctx->recvSlab, ctx->slabBytes,
                                        transportFlags);
      ctx->ctrlMemory =
          bootstrapComm->registerMemory(ctx->ctrlMapping,
                                        sizeof(HostControl),
                                        transportFlags);
      if (!ctx->numaSplit && ctx->nodeCount == 2) {
        for (auto transport : getAvailableIBTransports()) {
          if (transport != ctx->transport) {
            ctx->rail2Transport = transport;
            break;
          }
        }
        if (ctx->rail2Transport != mscclpp::Transport::Unknown) {
          mscclpp::TransportFlags rail2Flags(ctx->rail2Transport);
          ctx->rail2SendMemory =
              bootstrapComm->registerMemory(ctx->sendSlab, ctx->slabBytes,
                                            rail2Flags);
          ctx->rail2RecvMemory =
              bootstrapComm->registerMemory(ctx->recvSlab, ctx->slabBytes,
                                            rail2Flags);
        }
      }
    }
  } catch (std::exception const& ex) {
    setupResult = mapException(ex);
    setupMessage = ex.what();
  } catch (...) {
    setupResult = ncclInternalError;
    setupMessage = "unknown setup exception";
  }
  std::string setupStage = std::string(opName) + " setup";
  publishInitStatus(bootstrapComm, rank, nRanks, setupResult,
                              setupMessage, setupStage.c_str());

  ncclResult_t connectResult = ncclSuccess;
  std::string connectMessage;
  try {
    if (ctx->isLeader) {
      mscclpp::EndpointConfig::Ib ibCfg;
      ibCfg.maxCqPollNum = 128;
      ibCfg.mode = mscclpp::EndpointConfig::Ib::Mode::Host;
      mscclpp::EndpointConfig endpointConfig(
          ctx->transport, mscclpp::Device(mscclpp::DeviceType::CPU),
          /*maxWriteQueueSize=*/-1, ibCfg);
      ctx->peerConnections.clear();
      ctx->peerRemoteRecvMemory.clear();
      ctx->peerRemoteCtrlMemory.clear();
      for (size_t peerIndex = 0; peerIndex < ctx->peerNodeIds.size();
           ++peerIndex) {
        int peerNode = ctx->peerNodeIds[peerIndex];
        int peerLeader = ctx->peerLeaders[peerIndex];
        int tag0 = ctx->numaSplit
                       ? numaTag(rank, nRanks, peerLeader, 0)
                       : hostTag(rank, nRanks, peerLeader, 0);
        int tag1 = ctx->numaSplit
                       ? numaTag(rank, nRanks, peerLeader, 1)
                       : hostTag(rank, nRanks, peerLeader, 1);
        int tag2 = ctx->numaSplit
                       ? numaTag(rank, nRanks, peerLeader, 2)
                       : hostTag(rank, nRanks, peerLeader, 2);
        int tag3 = ctx->numaSplit
                       ? numaTag(rank, nRanks, peerLeader, 3)
                       : hostTag(rank, nRanks, peerLeader, 3);
        int tag4 = ctx->numaSplit
                       ? numaTag(rank, nRanks, peerLeader, 4)
                       : hostTag(rank, nRanks, peerLeader, 4);
        int tag5 = ctx->numaSplit
                       ? numaTag(rank, nRanks, peerLeader, 5)
                       : hostTag(rank, nRanks, peerLeader, 5);
        auto connectionFuture =
            bootstrapComm->connect(endpointConfig, peerLeader, tag0);
        bootstrapComm->sendMemory(ctx->sendMemory, peerLeader, tag3);
        auto remoteSendFuture = bootstrapComm->recvMemory(peerLeader, tag3);
        bootstrapComm->sendMemory(ctx->recvMemory, peerLeader, tag1);
        auto remoteRecvFuture = bootstrapComm->recvMemory(peerLeader, tag1);
        bootstrapComm->sendMemory(ctx->ctrlMemory, peerLeader, tag2);
        auto remoteCtrlFuture = bootstrapComm->recvMemory(peerLeader, tag2);
        mscclpp::Connection rail2Connection;
        mscclpp::RegisteredMemory rail2RemoteRecvMemory;
        if (ctx->rail2Transport != mscclpp::Transport::Unknown) {
          mscclpp::EndpointConfig rail2EndpointConfig(
              ctx->rail2Transport,
              mscclpp::Device(mscclpp::DeviceType::CPU),
              /*maxWriteQueueSize=*/-1, ibCfg);
          auto rail2ConnectionFuture =
              bootstrapComm->connect(rail2EndpointConfig, peerLeader, tag4);
          bootstrapComm->sendMemory(ctx->rail2RecvMemory, peerLeader, tag5);
          auto rail2RemoteRecvFuture =
              bootstrapComm->recvMemory(peerLeader, tag5);
          rail2Connection = rail2ConnectionFuture.get();
          rail2RemoteRecvMemory = rail2RemoteRecvFuture.get();
        }

        auto connection = connectionFuture.get();
        auto remoteSendMemory = remoteSendFuture.get();
        auto remoteRecvMemory = remoteRecvFuture.get();
        auto remoteCtrlMemory = remoteCtrlFuture.get();
        if (peerLeader == ctx->remoteLeader) {
          ctx->connection = connection;
          ctx->remoteSendMemory = remoteSendMemory;
          ctx->remoteRecvMemory = remoteRecvMemory;
          ctx->remoteCtrlMemory = remoteCtrlMemory;
          ctx->rail2Connection = rail2Connection;
          ctx->rail2RemoteRecvMemory = rail2RemoteRecvMemory;
          ctx->smallQp = ctx->connection.getIbQp();
          if (ctx->smallQp) {
            ctx->sendMemory.getIbMrInfo(ctx->transport, &ctx->smallSendMr,
                                        nullptr);
            ctx->ctrlMemory.getIbMrInfo(ctx->transport, &ctx->smallCtrlMr,
                                        nullptr);
            ctx->remoteSendMemory.getIbMrInfo(ctx->transport, nullptr,
                                              &ctx->smallRemoteSendMrInfo);
            ctx->remoteRecvMemory.getIbMrInfo(ctx->transport, nullptr,
                                              &ctx->smallRemoteRecvMrInfo);
            ctx->remoteCtrlMemory.getIbMrInfo(ctx->transport, nullptr,
                                              &ctx->smallRemoteCtrlMrInfo);
          }
        }
        ctx->peerConnections.push_back(connection);
        ctx->peerRemoteRecvMemory.push_back(remoteRecvMemory);
        ctx->peerRemoteCtrlMemory.push_back(remoteCtrlMemory);
      }
    }
  } catch (std::exception const& ex) {
    connectResult = mapException(ex);
    connectMessage = ex.what();
  } catch (...) {
    connectResult = ncclInternalError;
    connectMessage = "unknown connection exception";
  }
  std::string connectStage = std::string(opName) + " connect";
  publishInitStatus(bootstrapComm, rank, nRanks, connectResult,
                              connectMessage, connectStage.c_str());

  initGuard.commit();
  return *ctx;
}

AgContext& getSingleContext(
    ncclComm_t commHandle, std::shared_ptr<Communicator> bootstrapComm,
    int rank, int nRanks, int nRanksPerNode, int cudaDevice) {
  return getContext(
      [&]() -> std::unique_ptr<AgContext>& {
        return gSingleContexts[commHandle];
      },
      commHandle, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice,
      /*groupId=*/0, /*groupBase=*/0, /*groupSize=*/nRanksPerNode,
      /*transportDevice=*/cudaDevice, /*numaSplit=*/false,
      "AllGather");
}

std::vector<int> getCudaDevices(ncclComm_t commHandle,
                                std::shared_ptr<Communicator> bootstrapComm,
                                int rank, int nRanks, int cudaDevice) {
  {
    std::lock_guard<std::mutex> lock(gAllGatherContextMutex);
    auto it = gCudaDevicesByComm.find(commHandle);
    if (it != gCudaDevicesByComm.end()) return it->second;
  }

  std::vector<int> devices(nRanks, -1);
  devices[rank] = cudaDevice;
  bootstrapComm->bootstrap()->allGather(devices.data(), sizeof(int));

  {
    std::lock_guard<std::mutex> lock(gAllGatherContextMutex);
    return gCudaDevicesByComm.emplace(commHandle, std::move(devices))
        .first->second;
  }
}

NicGroupLayout getNicGroupLayout(ncclComm_t commHandle,
                                 std::shared_ptr<Communicator> bootstrapComm,
                                 int rank, int nRanks, int nRanksPerNode,
                                 int cudaDevice) {
  {
    std::lock_guard<std::mutex> lock(gAllGatherContextMutex);
    auto it = gLayoutByComm.find(commHandle);
    if (it != gLayoutByComm.end()) return it->second;
  }

  NicGroupLayout layout;
  auto cacheLayout = [&](NicGroupLayout const& computedLayout) {
    std::lock_guard<std::mutex> lock(gAllGatherContextMutex);
    return gLayoutByComm.emplace(commHandle, computedLayout).first->second;
  };
  if (nRanksPerNode <= 0 || nRanksPerNode > kMaxRanksPerNode) {
    return cacheLayout(layout);
  }

  auto devices =
      getCudaDevices(commHandle, bootstrapComm, rank, nRanks, cudaDevice);
  int hcaCount = static_cast<int>(getAvailableIBTransports().size());
  if (hcaCount <= 0) return cacheLayout(layout);

  auto buildNodeLayout = [&](int nodeId) {
    NicGroupLayout nodeLayout;
    int nodeBase = nodeId * nRanksPerNode;
    int prevNuma = std::numeric_limits<int>::min();
    for (int localRank = 0; localRank < nRanksPerNode; ++localRank) {
      int device = devices[nodeBase + localRank];
      int gpuNuma = -1;
      try {
        gpuNuma = mscclpp::getDeviceNumaNode(device);
      } catch (...) {
      }
      bool startNewGroup =
          nodeLayout.count == 0 ||
          (gpuNuma != prevNuma && nodeLayout.count < hcaCount &&
           nodeLayout.count < kMaxNicGroups);
      if (startNewGroup) {
        nodeLayout.base[nodeLayout.count] = localRank;
        nodeLayout.size[nodeLayout.count] = 1;
        nodeLayout.transportDevice[nodeLayout.count] = device;
        ++nodeLayout.count;
        prevNuma = gpuNuma;
      } else {
        ++nodeLayout.size[nodeLayout.count - 1];
      }
    }
    return nodeLayout;
  };

  int nodeCount = nRanks / nRanksPerNode;
  int localNodeId = rank / nRanksPerNode;
  layout = buildNodeLayout(localNodeId);

  // Group IDs pair leaders across nodes, so every node must have the same
  // contiguous group boundaries. Otherwise use the single-slab fallback.
  bool symmetricLayout = true;
  for (int nodeId = 0; nodeId < nodeCount && symmetricLayout; ++nodeId) {
    auto peerLayout = buildNodeLayout(nodeId);
    if (peerLayout.count != layout.count) {
      symmetricLayout = false;
      break;
    }
    for (int groupId = 0; groupId < layout.count; ++groupId) {
      if (peerLayout.base[groupId] != layout.base[groupId] ||
          peerLayout.size[groupId] != layout.size[groupId]) {
        symmetricLayout = false;
        break;
      }
    }
  }
  if (!symmetricLayout) {
    layout.count = 1;
    layout.base[0] = 0;
    layout.size[0] = nRanksPerNode;
    layout.transportDevice[0] = devices[localNodeId * nRanksPerNode];
  }
  return cacheLayout(layout);
}

int groupForLocalRank(NicGroupLayout const& layout, int localRank) {
  for (int groupId = 0; groupId < layout.count; ++groupId) {
    if (localRank >= layout.base[groupId] &&
        localRank < layout.base[groupId] + layout.size[groupId]) {
      return groupId;
    }
  }
  return layout.count - 1;
}

AgContext& getNumaContext(
    ncclComm_t commHandle, std::shared_ptr<Communicator> bootstrapComm,
    int rank, int nRanks, int nRanksPerNode, int cudaDevice, int groupId) {
  auto layout = getNicGroupLayout(commHandle, bootstrapComm, rank, nRanks,
                                  nRanksPerNode, cudaDevice);
  int groupBase = layout.base[groupId];
  int groupSize = layout.size[groupId];
  int transportDevice = layout.transportDevice[groupId];
  return getContext(
      [&]() -> std::unique_ptr<AgContext>& {
        return gNumaContexts[commHandle][groupId];
      },
      commHandle, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice,
      groupId, groupBase, groupSize,
      transportDevice, /*numaSplit=*/true, "NUMA AllGather");
}

bool isTwoNodeLayout(int nRanks, int nRanksPerNode) {
  return nRanksPerNode > 0 && nRanks == 2 * nRanksPerNode;
}

bool isMultiNodeLayout(int nRanks, int nRanksPerNode) {
  return nRanksPerNode > 0 && nRanks % nRanksPerNode == 0 &&
         nRanks / nRanksPerNode >= 2;
}

bool isRankInGroup(AgContext const& ctx) {
  return ctx.localRank >= ctx.groupBase &&
         ctx.localRank < ctx.groupBase + ctx.groupSize;
}

size_t recvBlockOffset(int sourceNode, size_t blockBytes) {
  return static_cast<size_t>(sourceNode) * blockBytes;
}

size_t pipelineChunkBytes(AgContext const& ctx) {
  return (!ctx.numaSplit && ctx.nRanksPerNode == 2)
             ? kTwoGpuPipelineChunkBytes
             : kDefaultPipelineChunkBytes;
}

size_t smallCutoffBytes(int nRanksPerNode) {
  if (nRanksPerNode == 1) return kOneRankSmallCutoffBytes;
  if (nRanksPerNode == 2) return kTwoGpuSmallCutoffBytes;
  return kSmallCutoffBytes;
}

size_t slotChunkBytes(AgContext const& ctx, size_t bytesPerRank,
                      size_t chunkBytes) {
  size_t pipelineBytes = pipelineChunkBytes(ctx);
  return bytesPerRank > pipelineBytes ? pipelineBytes : chunkBytes;
}

size_t slotCountForChunk(AgContext const& ctx, size_t slotBytes) {
  return std::max<size_t>(1, ctx.chunkCapacity / slotBytes);
}

size_t genericSlotCountForChunk(AgContext const& ctx, size_t bytesPerRank,
                                size_t slotBytes) {
  // Messages beyond the one-rank fast path use the old small ring to avoid
  // expanding the generic fallback's reuse window unintentionally.
  bool oneRankFallback = !ctx.numaSplit && ctx.nodeCount == 2 &&
                         ctx.nRanksPerNode == 1 &&
                         bytesPerRank > kOneRankPipelineMaxBytes;
  size_t capacity = oneRankFallback ? kMaxBytesPerRank : ctx.chunkCapacity;
  return std::max<size_t>(1, capacity / slotBytes);
}

size_t sendSlotOffset(size_t slot, size_t blockBytes) {
  return slot * blockBytes;
}

size_t recvSlotOffset(size_t slot, int nodeCount, size_t blockBytes) {
  return slot * static_cast<size_t>(nodeCount) * blockBytes;
}

bool isOneRankPerNodeInPlace(AgContext const& ctx, void const* sendbuff,
                             void* recvbuff, size_t bytesPerRank,
                             size_t chunkOffset) {
  auto const* send = static_cast<char const*>(sendbuff);
  auto* recv = static_cast<char*>(recvbuff);
  size_t selfOffset = static_cast<size_t>(ctx.rank) * bytesPerRank + chunkOffset;
  return send + chunkOffset == recv + selfOffset;
}

ncclResult_t copyOneRankPerNodeChunkToOutput(
    AgContext const& ctx, void const* sendbuff, void* recvbuff,
    size_t bytesPerRank, size_t chunkOffset, size_t chunkBytes,
    char const* remoteChunk, cudaStream_t stream) {
  auto const* send = static_cast<char const*>(sendbuff);
  auto* recv = static_cast<char*>(recvbuff);
  int remoteRank = (1 - ctx.nodeId) * ctx.nRanksPerNode;
  size_t selfOffset = static_cast<size_t>(ctx.rank) * bytesPerRank + chunkOffset;
  size_t remoteOffset =
      static_cast<size_t>(remoteRank) * bytesPerRank + chunkOffset;

  if (send + chunkOffset != recv + selfOffset) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        recv + selfOffset, send + chunkOffset, chunkBytes,
        cudaMemcpyDeviceToDevice, stream));
  }
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(recv + remoteOffset, remoteChunk,
                                    chunkBytes, cudaMemcpyHostToDevice,
                                    stream));
  return ncclSuccess;
}

ncclResult_t copyGroupChunkToOutput(
    AgContext& ctx, void const* sendbuff, void* recvbuff, size_t bytesPerRank,
    size_t chunkOffset, size_t chunkBytes, size_t sendBase, size_t recvBase,
    size_t slotBlockBytes, cudaStream_t stream, bool selfPreCopied) {
  auto const* send = static_cast<char const*>(sendbuff);
  auto* recv = static_cast<char*>(recvbuff);
  bool wholeRankChunk = chunkOffset == 0 && chunkBytes == bytesPerRank;
  size_t blockBytes = static_cast<size_t>(ctx.groupSize) * chunkBytes;
  bool directSelfCopy = chunkBytes >= kDirectSelfCopyMinBytes &&
                        isRankInGroup(ctx);

  for (int node = 0; node < ctx.nodeCount; ++node) {
    int rankBase = node * ctx.nRanksPerNode + ctx.groupBase;
    char const* src =
        node == ctx.nodeId ? ctx.sendSlab + sendBase
                           : ctx.recvSlab +
                                 recvBase +
                                 recvBlockOffset(node, slotBlockBytes);
    bool localSelfBlock = directSelfCopy && node == ctx.nodeId;
    if (wholeRankChunk && !localSelfBlock) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          recv + static_cast<size_t>(rankBase) * bytesPerRank, src,
          blockBytes, cudaMemcpyHostToDevice, stream));
    } else {
      for (int i = 0; i < ctx.groupSize; ++i) {
        int peer = rankBase + i;
        char* dst =
            recv + static_cast<size_t>(peer) * bytesPerRank + chunkOffset;
        if (localSelfBlock && peer == ctx.rank) {
          char const* selfSrc = send + chunkOffset;
          if (!selfPreCopied && selfSrc != dst) {
            MSCCLPP_CUDATHROW(cudaMemcpyAsync(
                dst, selfSrc, chunkBytes, cudaMemcpyDeviceToDevice, stream));
          }
        } else {
          MSCCLPP_CUDATHROW(cudaMemcpyAsync(
              dst, src + static_cast<size_t>(i) * chunkBytes, chunkBytes,
              cudaMemcpyHostToDevice, stream));
        }
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t exchangeGroupChunk(AgContext& ctx,
                                void const* sendbuff, void* recvbuff,
                                size_t bytesPerRank, size_t chunkOffset,
                                size_t chunkBytes, cudaStream_t stream,
                                bool copyAsSoonAsReady,
                                bool selfPreCopied,
                                std::shared_ptr<Communicator> bootstrapComm) {
  uint64_t epoch = ++ctx.epoch;
  size_t blockBytes = static_cast<size_t>(ctx.groupSize) * chunkBytes;
  size_t slotBytes = slotChunkBytes(ctx, bytesPerRank, chunkBytes);
  size_t slotBlockBytes = static_cast<size_t>(ctx.groupSize) * slotBytes;
  size_t slotCount = genericSlotCountForChunk(ctx, bytesPerRank, slotBytes);
  size_t slot = static_cast<size_t>((epoch - 1) % slotCount);
  bool useAck = slotCount == 1;
  cudaStream_t d2hStream = ctx.d2hStream ? ctx.d2hStream : stream;
  cudaStream_t h2dStream = ctx.h2dStream ? ctx.h2dStream : stream;
  if (!useAck && ctx.h2dStream != nullptr) {
    ensureSlotEvents(ctx, slotCount);
  }
  if (!useAck && epoch > slotCount && ctx.h2dStream != nullptr) {
    waitForCudaEvent(ctx.h2dSlotEvents[slot]);
    ctx.ctrl->h2dDone[ctx.localRank].store(epoch, std::memory_order_release);
    if (ctx.isLeader) {
      for (int i = 0; i < ctx.nRanksPerNode; ++i) {
        waitForEpoch(ctx.ctrl->h2dDone[i], epoch);
      }
      ctx.ctrl->ackSignal[ctx.nodeId].store(epoch,
                                            std::memory_order_release);
      for (size_t peer = 0; peer < ctx.peerNodeIds.size(); ++peer) {
        ctx.peerConnections[peer].write(
            ctx.peerRemoteCtrlMemory[peer], ackReadyOffset(ctx.nodeId),
            ctx.ctrlMemory, ackSignalOffset(ctx.nodeId), sizeof(uint64_t));
        ctx.peerConnections[peer].flush();
      }
    }
    for (int peerNode : ctx.peerNodeIds) {
      waitForEpoch(ctx.ctrl->ackReady[peerNode], epoch);
    }
  } else if (!useAck && slot == 0 && epoch > 1) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(h2dStream));
    bootstrapComm->bootstrap()->barrier();
  }
  size_t sendBase = sendSlotOffset(slot, slotBlockBytes);
  size_t recvBase = recvSlotOffset(slot, ctx.nodeCount, slotBlockBytes);
  bool inGroup = isRankInGroup(ctx);
  bool oneRankLayout = ctx.nodeCount == 2 && ctx.nRanksPerNode == 1;
  bool oneRankInPlace =
      oneRankLayout &&
      isOneRankPerNodeInPlace(ctx, sendbuff, recvbuff, bytesPerRank,
                              chunkOffset);
  bool preCopiedSelf =
      oneRankLayout && !oneRankInPlace &&
      chunkBytes >= kOneRankDirectCopyCutoffBytes;
  // Generic slab exchange is synchronous at each chunk boundary: D2H into the
  // host slot, leader RDMA write of the contiguous block, then H2D into final
  // rank order.  The 2nx1g fast path below is separate because it pipelines at
  // 512KiB granularity instead of paying this full barrier per 2MiB chunk.
  if (inGroup) {
    auto const* send = static_cast<char const*>(sendbuff);
    int localSlot = ctx.localRank - ctx.groupBase;
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        ctx.sendSlab + sendBase + static_cast<size_t>(localSlot) * chunkBytes,
        send + chunkOffset, chunkBytes, cudaMemcpyDeviceToHost, d2hStream));
    waitForCudaStream(d2hStream);
    if (preCopiedSelf) {
      auto const* sendBytes = static_cast<char const*>(sendbuff);
      auto* recvBytes = static_cast<char*>(recvbuff);
      size_t selfOffset =
          static_cast<size_t>(ctx.rank) * bytesPerRank + chunkOffset;
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          recvBytes + selfOffset, sendBytes + chunkOffset, chunkBytes,
          cudaMemcpyDeviceToDevice, h2dStream));
    }
    ctx.ctrl->d2hReady[ctx.localRank].store(epoch,
                                            std::memory_order_release);
  }

  if (ctx.isLeader) {
    for (int i = 0; i < ctx.groupSize; ++i) {
      waitForEpoch(ctx.ctrl->d2hReady[ctx.groupBase + i], epoch);
    }

    ctx.ctrl->rdmaSignal[ctx.nodeId].store(epoch,
                                           std::memory_order_release);
    for (size_t peer = 0; peer < ctx.peerNodeIds.size(); ++peer) {
      int peerNode = ctx.peerNodeIds[peer];
      size_t remoteBase =
          recvBase + recvBlockOffset(ctx.nodeId, slotBlockBytes);
      bool usedDirectData =
          peerNode == 1 - ctx.nodeId &&
          (writeDataStripedToRemoteRecv(ctx, remoteBase, sendBase,
                                        blockBytes) ||
           writeDataDirectToRemoteRecv(ctx, remoteBase, sendBase, blockBytes));
      if (!usedDirectData) {
        size_t off = 0;
        int writesSinceFlush = 0;
        while (off < blockBytes) {
          size_t chunk = std::min(kRdmaChunkBytes, blockBytes - off);
          ctx.peerConnections[peer].write(ctx.peerRemoteRecvMemory[peer],
                                          remoteBase + off, ctx.sendMemory,
                                          sendBase + off, chunk);
          if (++writesSinceFlush == kSignalEveryN) {
            ctx.peerConnections[peer].flush();
            writesSinceFlush = 0;
          }
          off += chunk;
        }
      }
      signalRdmaReadyAtomic(ctx, peer, epoch);
      (void)peerNode;
    }
  }
  for (int peerNode : ctx.peerNodeIds) {
    waitForEpoch(ctx.ctrl->rdmaReady[peerNode], epoch);
  }

  if (copyAsSoonAsReady) {
    bool oneRankDirectCopy =
        oneRankLayout &&
        (oneRankInPlace || chunkBytes >= kOneRankDirectCopyCutoffBytes);
    ncclResult_t result = ncclSuccess;
    if (oneRankDirectCopy && preCopiedSelf) {
      auto* recvBytes = static_cast<char*>(recvbuff);
      int remoteRank = (1 - ctx.nodeId) * ctx.nRanksPerNode;
      size_t remoteOffset =
          static_cast<size_t>(remoteRank) * bytesPerRank + chunkOffset;
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          recvBytes + remoteOffset,
          ctx.recvSlab + recvBase +
              recvBlockOffset(1 - ctx.nodeId, slotBlockBytes),
          chunkBytes, cudaMemcpyHostToDevice, h2dStream));
    } else if (oneRankDirectCopy) {
      result = copyOneRankPerNodeChunkToOutput(
          ctx, sendbuff, recvbuff, bytesPerRank, chunkOffset, chunkBytes,
          ctx.recvSlab + recvBase +
              recvBlockOffset(1 - ctx.nodeId, slotBlockBytes),
          h2dStream);
    } else {
      result = copyGroupChunkToOutput(
        ctx, sendbuff, recvbuff, bytesPerRank, chunkOffset, chunkBytes,
        sendBase, recvBase, slotBlockBytes, h2dStream, selfPreCopied);
    }
    if (result != ncclSuccess) return result;
    if (useAck) {
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(h2dStream));
      ctx.ctrl->h2dDone[ctx.localRank].store(epoch, std::memory_order_release);

      for (int i = 0; i < ctx.nRanksPerNode; ++i) {
        waitForEpoch(ctx.ctrl->h2dDone[i], epoch);
      }
      if (ctx.isLeader) {
        ctx.ctrl->ackSignal[ctx.nodeId].store(epoch,
                                              std::memory_order_release);
        for (size_t peer = 0; peer < ctx.peerNodeIds.size(); ++peer) {
          ctx.peerConnections[peer].write(
              ctx.peerRemoteCtrlMemory[peer], ackReadyOffset(ctx.nodeId),
              ctx.ctrlMemory, ackSignalOffset(ctx.nodeId), sizeof(uint64_t));
          ctx.peerConnections[peer].flush();
        }
      }
      for (int peerNode : ctx.peerNodeIds) {
        waitForEpoch(ctx.ctrl->ackReady[peerNode], epoch);
      }
    } else if (ctx.h2dStream != nullptr) {
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dSlotEvents[slot], h2dStream));
    }
  }
  return ncclSuccess;
}

ncclResult_t runOneRankChunkPipeline(
    AgContext& ctx, void const* sendbuff, void* recvbuff,
    size_t bytesPerRank) {
  uint64_t epoch = ++ctx.epoch;
  size_t slotCount = slotCountForChunk(ctx, bytesPerRank);
  size_t slot = static_cast<size_t>((epoch - 1) % slotCount);
  bool useAck = slotCount == 1;
  ensureSlotEvents(ctx, slotCount);

  // Reusing a full-message slot is safe only after the remote chunk H2Ds that
  // read from that slot have completed.  Multi-slot cases use per-slot H2D
  // events; the single-slot fallback uses an explicit remote ack.
  if (!useAck && epoch > slotCount) {
    waitForCudaEvent(ctx.h2dSlotEvents[slot]);
    ctx.ctrl->h2dDone[ctx.localRank].store(epoch, std::memory_order_release);
    ctx.ctrl->ackSignal[ctx.nodeId].store(epoch, std::memory_order_release);
    ctx.connection.write(ctx.remoteCtrlMemory, ackReadyOffset(ctx.nodeId),
                         ctx.ctrlMemory, ackSignalOffset(ctx.nodeId),
                         sizeof(uint64_t));
    ctx.connection.flush();
    waitForEpoch(ctx.ctrl->ackReady[1 - ctx.nodeId], epoch);
  }

  size_t sendBase = sendSlotOffset(slot, bytesPerRank);
  size_t recvBase = recvSlotOffset(slot, ctx.nodeCount, bytesPerRank);
  auto const* send = static_cast<char const*>(sendbuff);
  auto* recv = static_cast<char*>(recvbuff);
  size_t selfOffset = static_cast<size_t>(ctx.rank) * bytesPerRank;
  size_t remoteRank = static_cast<size_t>((1 - ctx.nodeId) * ctx.nRanksPerNode);
  size_t remoteOutputOffset = remoteRank * bytesPerRank;
  size_t localRemoteBase =
      recvBase + recvBlockOffset(1 - ctx.nodeId, bytesPerRank);
  size_t peerRemoteBase = recvBase + recvBlockOffset(ctx.nodeId, bytesPerRank);

  if (send != recv + selfOffset) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(recv + selfOffset, send, bytesPerRank,
                                      cudaMemcpyDeviceToDevice,
                                      ctx.h2dStream));
  }

  size_t chunkCount =
      (bytesPerRank + kOneRankPipelineChunkBytes - 1) /
      kOneRankPipelineChunkBytes;
  ensureD2hChunkEvents(ctx, chunkCount);
  // Stage all D2H copies first, then let the CPU issue RDMA writes as each D2H
  // event becomes visible.  This overlaps GPU copy readiness, NIC DMA reads,
  // and the peer's H2D copies without involving CUDA kernels.
  for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
    size_t off = chunk * kOneRankPipelineChunkBytes;
    size_t bytes = std::min(kOneRankPipelineChunkBytes, bytesPerRank - off);
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(ctx.sendSlab + sendBase + off,
                                      send + off, bytes,
                                      cudaMemcpyDeviceToHost,
                                      ctx.d2hStream));
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.d2hChunkEvents[chunk],
                                      ctx.d2hStream));
  }

  auto sendChunk = [&](size_t chunk) {
    size_t off = chunk * kOneRankPipelineChunkBytes;
    size_t bytes = std::min(kOneRankPipelineChunkBytes, bytesPerRank - off);
    uint64_t readyValue = epoch * kPipeValueStride + chunk + 1;
    // The NIC reads RDMA-write sources asynchronously, so every chunk needs a
    // stable flag word.  Reusing one host control word can publish a later
    // ready value before the matching data write has completed.
    size_t flagSrcOffset = ctx.chunkCapacity + chunk * sizeof(readyValue);
    std::memcpy(ctx.sendSlab + flagSrcOffset, &readyValue, sizeof(readyValue));
    std::atomic_thread_fence(std::memory_order_release);
    waitForCudaEvent(ctx.d2hChunkEvents[chunk]);
    if (!writePipelineChunkDirect(ctx, peerRemoteBase + off, sendBase + off,
                                  bytes, flagSrcOffset)) {
      ctx.connection.write(ctx.remoteRecvMemory, peerRemoteBase + off,
                           ctx.sendMemory, sendBase + off, bytes);
      ctx.connection.write(ctx.remoteCtrlMemory, pipeReadyOffset(ctx.nodeId),
                           ctx.sendMemory, flagSrcOffset,
                           sizeof(uint64_t));
      ctx.connection.flush();
    }
  };

  size_t nextSend = 0;
  size_t initialWindow = std::min(kOneRankPipelineSendWindow, chunkCount);
  for (; nextSend < initialWindow; ++nextSend) {
    sendChunk(nextSend);
  }
  // Window size is intentionally one: after each incoming ready value is
  // consumed, post one more outgoing RDMA chunk.  This keeps both nodes paced
  // by receive progress while still overlapping D2H, RDMA, and H2D.
  for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
    size_t off = chunk * kOneRankPipelineChunkBytes;
    size_t bytes = std::min(kOneRankPipelineChunkBytes, bytesPerRank - off);
    uint64_t readyValue = epoch * kPipeValueStride + chunk + 1;
    waitForEpoch(ctx.ctrl->pipeReady[1 - ctx.nodeId], readyValue);
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        recv + remoteOutputOffset + off, ctx.recvSlab + localRemoteBase + off,
        bytes, cudaMemcpyHostToDevice, ctx.h2dStream));
    if (nextSend < chunkCount) {
      sendChunk(nextSend++);
    }
  }

  if (useAck) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(ctx.h2dStream));
    ctx.ctrl->h2dDone[ctx.localRank].store(epoch, std::memory_order_release);
    ctx.ctrl->ackSignal[ctx.nodeId].store(epoch, std::memory_order_release);
    ctx.connection.write(ctx.remoteCtrlMemory, ackReadyOffset(ctx.nodeId),
                         ctx.ctrlMemory, ackSignalOffset(ctx.nodeId),
                         sizeof(uint64_t));
    ctx.connection.flush();
    waitForEpoch(ctx.ctrl->ackReady[1 - ctx.nodeId], epoch);
  } else {
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dSlotEvents[slot],
                                      ctx.h2dStream));
  }
  return ncclSuccess;
}

ncclResult_t runSingleSlab(
    void const* sendbuff, void* recvbuff, size_t sendcount,
    size_t bytesPerRank, ncclDataType_t datatype, ncclComm_t comm,
    cudaStream_t stream, int rank, int nRanks, int nRanksPerNode,
    std::shared_ptr<Communicator> bootstrapComm, int cudaDevice) {
  if (!isMultiNodeLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  if (nRanksPerNode <= 0 || nRanksPerNode > kMaxRanksPerNode) {
    return ncclInvalidUsage;
  }
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;
  if (sendcount > std::numeric_limits<size_t>::max() / typeSize) {
    return ncclInvalidArgument;
  }
  if (sendcount * typeSize != bytesPerRank) {
    return ncclInvalidUsage;
  }
  if (bytesPerRank == 0) return ncclSuccess;

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    auto& ctx = getSingleContext(
        comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice);
    ensurePipelineStreams(ctx);
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.d2hStream,
                                         ctx.inputReadyEvent, 0));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(ctx.h2dStream,
                                         ctx.inputReadyEvent, 0));
    auto const* send = static_cast<char const*>(sendbuff);
    auto* recv = static_cast<char*>(recvbuff);
    bool selfInPlace =
        send == recv + static_cast<size_t>(rank) * bytesPerRank;
    bool selfPreCopied = false;
    if (!selfInPlace && nRanksPerNode > 1 &&
        bytesPerRank >= kDirectSelfCopyMinBytes) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          recv + static_cast<size_t>(rank) * bytesPerRank, send, bytesPerRank,
          cudaMemcpyDeviceToDevice, ctx.h2dStream));
      selfPreCopied = true;
    }
    if (ctx.nodeCount == 2 && ctx.nRanksPerNode == 1 &&
        bytesPerRank >= 1024 * 1024 &&
        bytesPerRank <= kOneRankPipelineMaxBytes) {
      // 2nx1g is the only layout where one rank owns the whole node block, so
      // the chunk pipeline can avoid local gather/scatter and directly stream
      // the remote rank into final output order.
      ncclResult_t result = runOneRankChunkPipeline(
          ctx, sendbuff, recvbuff, bytesPerRank);
      if (result != ncclSuccess) return result;
      MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent, ctx.h2dStream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, ctx.h2dDoneEvent, 0));
      return ncclSuccess;
    }
    for (size_t chunkOffset = 0; chunkOffset < bytesPerRank;) {
      size_t maxChunkBytes =
          std::min(ctx.chunkCapacity, pipelineChunkBytes(ctx));
      size_t chunkBytes = std::min(maxChunkBytes, bytesPerRank - chunkOffset);
      ncclResult_t result =
          exchangeGroupChunk(ctx, sendbuff, recvbuff, bytesPerRank,
                                 chunkOffset, chunkBytes, stream,
                                 /*copyAsSoonAsReady=*/true, selfPreCopied,
                                 bootstrapComm);
      if (result != ncclSuccess) return result;
      chunkOffset += chunkBytes;
    }
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.h2dDoneEvent, ctx.h2dStream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, ctx.h2dDoneEvent, 0));
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN("AllGather failed: %s", ex.what());
    return mapException(ex);
  } catch (...) {
    WARN("AllGather failed with an unknown exception");
    return ncclInternalError;
  }
}

ncclResult_t copySmallFallbackOutput(AgContext& ctx,
                                            void* recvbuff,
                                            size_t bytesPerRank,
                                            size_t chunkBytes,
                                            cudaStream_t stream) {
  size_t fullBytes = static_cast<size_t>(ctx.worldSize) * bytesPerRank;
  size_t blockBytes = static_cast<size_t>(ctx.groupSize) * chunkBytes;
  char* scratch = ctx.sendSlab + blockBytes +
                  static_cast<size_t>(ctx.localRank) * fullBytes;
  int localBase = ctx.nodeId * ctx.nRanksPerNode;
  int remoteBase = (1 - ctx.nodeId) * ctx.nRanksPerNode;
  int remoteNode = 1 - ctx.nodeId;
  std::memcpy(scratch + static_cast<size_t>(localBase) * bytesPerRank,
              ctx.sendSlab, blockBytes);
  std::memcpy(scratch + static_cast<size_t>(remoteBase) * bytesPerRank,
              ctx.recvSlab + recvBlockOffset(remoteNode, blockBytes),
              blockBytes);
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(recvbuff, scratch, fullBytes,
                                    cudaMemcpyHostToDevice, stream));
  return ncclSuccess;
}

ncclResult_t runSmallFallback(
    void const* sendbuff, void* recvbuff, size_t sendcount,
    size_t bytesPerRank, ncclDataType_t datatype, ncclComm_t comm,
    cudaStream_t stream, int rank, int nRanks, int nRanksPerNode,
    std::shared_ptr<Communicator> bootstrapComm, int cudaDevice) {
  if (!isTwoNodeLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  if (nRanksPerNode <= 0 || nRanksPerNode > kMaxRanksPerNode) {
    return ncclInvalidUsage;
  }
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;
  if (sendcount > std::numeric_limits<size_t>::max() / typeSize) {
    return ncclInvalidArgument;
  }
  if (sendcount * typeSize != bytesPerRank) return ncclInvalidUsage;
  if (bytesPerRank == 0) return ncclSuccess;
  if (bytesPerRank >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(nRanks)) {
    return ncclInvalidUsage;
  }
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);
  if (fullBytes >= smallCutoffBytes(nRanksPerNode)) return ncclInvalidUsage;

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    auto& ctx = getSingleContext(
        comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice);
    size_t blockBytes = static_cast<size_t>(ctx.groupSize) * bytesPerRank;
    size_t scratchBytes = static_cast<size_t>(ctx.groupSize) * fullBytes;
    if (blockBytes + scratchBytes > ctx.slabBytes) return ncclInvalidUsage;

    uint64_t epoch = ++ctx.epoch;
    auto const* send = static_cast<char const*>(sendbuff);
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        ctx.sendSlab + static_cast<size_t>(ctx.localRank) * bytesPerRank, send,
        bytesPerRank, cudaMemcpyDeviceToHost, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    ctx.ctrl->d2hReady[ctx.localRank].store(epoch, std::memory_order_release);

    if (ctx.isLeader) {
      for (int i = 0; i < ctx.groupSize; ++i) {
        waitForEpoch(ctx.ctrl->d2hReady[i], epoch);
      }
      ctx.connection.write(ctx.remoteRecvMemory,
                           recvBlockOffset(ctx.nodeId, blockBytes),
                           ctx.sendMemory, 0, blockBytes);
      ctx.ctrl->rdmaSignal[ctx.nodeId].store(epoch, std::memory_order_release);
      signalRdmaReadyAtomic(ctx, ctx.connection, ctx.remoteCtrlMemory,
                            1 - ctx.nodeId, epoch);
    }
    waitForEpoch(ctx.ctrl->rdmaReady[1 - ctx.nodeId], epoch);
    for (int i = 0; i < ctx.groupSize; ++i) {
      waitForEpoch(ctx.ctrl->d2hReady[i], epoch);
    }

    ncclResult_t result = copySmallFallbackOutput(
        ctx, recvbuff, bytesPerRank, bytesPerRank, stream);
    if (result != ncclSuccess) return result;
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

    ctx.ctrl->h2dDone[ctx.localRank].store(epoch, std::memory_order_release);

    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->h2dDone[i], epoch);
    }
    if (ctx.isLeader) {
      ctx.ctrl->ackSignal[ctx.nodeId].store(epoch, std::memory_order_release);
      ctx.connection.write(
          ctx.remoteCtrlMemory, ackReadyOffset(ctx.nodeId),
          ctx.ctrlMemory, ackSignalOffset(ctx.nodeId),
          sizeof(uint64_t));
      ctx.connection.flush();
    }
    waitForEpoch(ctx.ctrl->ackReady[1 - ctx.nodeId], epoch);
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN("small fallback AllGather failed: %s", ex.what());
    return mapException(ex);
  } catch (...) {
    WARN("small fallback AllGather failed with an unknown exception");
    return ncclInternalError;
  }
}

ncclResult_t runOneRankGpuDirect(
    void const* sendbuff, void* recvbuff, size_t sendcount,
    size_t bytesPerRank, ncclDataType_t datatype, ncclComm_t comm,
    cudaStream_t stream, int rank, int nRanks, int nRanksPerNode,
    std::shared_ptr<Communicator> bootstrapComm, int cudaDevice) {
  if (!isTwoNodeLayout(nRanks, nRanksPerNode) || nRanksPerNode != 1) {
    return ncclInvalidUsage;
  }
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;
  if (sendcount > std::numeric_limits<size_t>::max() / typeSize) {
    return ncclInvalidArgument;
  }
  if (sendcount * typeSize != bytesPerRank) return ncclInvalidUsage;
  if (bytesPerRank == 0) return ncclSuccess;
  if (bytesPerRank >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(nRanks)) {
    return ncclInvalidUsage;
  }
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    auto& ctx = getSingleContext(
        comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice);
    ensurePipelineStreams(ctx);
    if (!ensureGpuDirectOneRank(ctx, bootstrapComm, sendbuff, bytesPerRank,
                                recvbuff, fullBytes)) {
      return ncclInvalidUsage;
    }

    uint64_t epoch = ++ctx.epoch;
    auto const* send = static_cast<char const*>(sendbuff);
    auto* recv = static_cast<char*>(recvbuff);
    size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank;

    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
    waitForCudaEvent(ctx.inputReadyEvent);

    if (send != recv + selfOffset) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(recv + selfOffset, send, bytesPerRank,
                                        cudaMemcpyDeviceToDevice, stream));
    }
    if (!writeGpuDirectOneRank(ctx, bytesPerRank, epoch)) {
      return ncclInvalidUsage;
    }
    waitForEpoch(ctx.ctrl->rdmaReady[1 - ctx.nodeId], epoch);
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN("GPUDirect AllGather failed: %s", ex.what());
    return mapException(ex);
  } catch (...) {
    WARN("GPUDirect AllGather failed with an unknown exception");
    return ncclInternalError;
  }
}

ncclResult_t runSmallOrdered(
    void const* sendbuff, void* recvbuff, size_t sendcount,
    size_t bytesPerRank, ncclDataType_t datatype, ncclComm_t comm,
    cudaStream_t stream, int rank, int nRanks, int nRanksPerNode,
    std::shared_ptr<Communicator> bootstrapComm, int cudaDevice) {
  if (!isTwoNodeLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  if (nRanksPerNode <= 0 || nRanksPerNode > kMaxRanksPerNode) {
    return ncclInvalidUsage;
  }
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;
  if (sendcount > std::numeric_limits<size_t>::max() / typeSize) {
    return ncclInvalidArgument;
  }
  if (sendcount * typeSize != bytesPerRank) return ncclInvalidUsage;
  if (bytesPerRank == 0) return ncclSuccess;
  if (bytesPerRank >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(nRanks)) {
    return ncclInvalidUsage;
  }
  size_t fullBytes = bytesPerRank * static_cast<size_t>(nRanks);
  if (fullBytes >= smallCutoffBytes(nRanksPerNode)) return ncclInvalidUsage;

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    auto& ctx = getSingleContext(
        comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice);
    bool useOneRankRegister =
        ctx.nodeCount == 2 && ctx.nRanksPerNode == 1 &&
        fullBytes < kOneRankMappedOutputCutoffBytes &&
        ctx.sendDeviceSlab != nullptr && ctx.ctrlDeviceSlab != nullptr;
    bool useTwoRankRecvKernel =
        ctx.nodeCount == 2 && ctx.nRanksPerNode == 2 &&
        fullBytes <= 4 * 1024 && ctx.sendDeviceSlab != nullptr;
    bool useTwoRankRegisterPack =
        ctx.nodeCount == 2 && ctx.nRanksPerNode == 2 &&
        fullBytes >= 512 && fullBytes <= 4 * 1024 &&
        ctx.sendDeviceSlab != nullptr && ctx.ctrlDeviceSlab != nullptr;
    bool useTwoRankTinyPack =
        ctx.nodeCount == 2 && ctx.nRanksPerNode == 2 &&
        fullBytes <= 256 && ctx.sendDeviceSlab != nullptr &&
        ctx.ctrlDeviceSlab != nullptr;
    bool useTwoRankGpuPack =
        useTwoRankTinyPack || useTwoRankRegisterPack;
    // Small rows are latency-bound.  Keep one ordered host slot per epoch so
    // the receiver can H2D final AllGather order without a CPU repack step.
    size_t blockBytes = static_cast<size_t>(ctx.groupSize) * bytesPerRank;
    if (blockBytes == 0) return ncclSuccess;
    size_t compactFlagOffset =
        (bytesPerRank + sizeof(uint64_t) - 1) & ~(sizeof(uint64_t) - 1);
    size_t compactSegmentBytes = compactFlagOffset + sizeof(uint64_t);
    size_t perSlotBytes = useOneRankRegister
                              ? 2 * compactSegmentBytes
                              : fullBytes + sizeof(uint64_t);
    size_t slotCount =
        std::min(kSmallMaxSlots, ctx.slabBytes / perSlotBytes);
    if (slotCount < 2) return ncclInvalidUsage;
    if (ctx.smallPerSlotBytes != 0 &&
        ctx.smallPerSlotBytes != perSlotBytes) {
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      if (ctx.isLeader) pollQp(ctx);
      bootstrapComm->bootstrap()->barrier();
      ctx.smallWrCount = 0;
    }
    ctx.smallPerSlotBytes = perSlotBytes;

    uint64_t epoch = ++ctx.epoch;
    size_t slot = static_cast<size_t>((epoch - 1) % slotCount);
    if (epoch > 1 && slot == 0) {
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      if (ctx.isLeader) pollQp(ctx);
      bootstrapComm->bootstrap()->barrier();
    }
    size_t slotOffset = slot * perSlotBytes;
    size_t flagOffset = slotOffset + fullBytes;
    // Slot layout is final-output order: [rank0][rank1]...[rankN][flag].
    // This lets every rank finish with one H2D of the complete AllGather output.
    auto const* send = static_cast<char const*>(sendbuff);
    bool oneRankInPlace =
        isOneRankPerNodeInPlace(ctx, sendbuff, recvbuff, bytesPerRank,
                                /*chunkOffset=*/0);
    if (useOneRankRegister) {
      ctx.ctrl->d2hReady[ctx.localRank].store(0, std::memory_order_release);
      bool useTinyOriginal = fullBytes == 128;
      bool useParallelOriginal = fullBytes == 256;
      if (useTinyOriginal) {
        oneRankTinyRegisterCopyKernel<<<1, 1, 0, stream>>>(
            send, static_cast<char*>(recvbuff),
            const_cast<char*>(ctx.sendDeviceSlab), ctx.ctrlDeviceSlab,
            slotOffset, flagOffset, bytesPerRank, rank, !oneRankInPlace,
            d2hReadyOffset(ctx.localRank), epoch);
      } else if (useParallelOriginal) {
        oneRankRegisterCopyKernel<<<1, 128, 0, stream>>>(
            send, static_cast<char*>(recvbuff),
            const_cast<char*>(ctx.sendDeviceSlab), ctx.ctrlDeviceSlab,
            slotOffset, flagOffset, bytesPerRank, rank, !oneRankInPlace,
            d2hReadyOffset(ctx.localRank), epoch);
      } else {
        int blockThreads =
            (fullBytes == 16 * 1024 || fullBytes == 32 * 1024) ? 256 : 128;
        oneRankCompactRegisterCopyKernel<<<1, blockThreads, 0, stream>>>(
            send, static_cast<char*>(recvbuff),
            const_cast<char*>(ctx.sendDeviceSlab), ctx.ctrlDeviceSlab,
            slotOffset, compactSegmentBytes, compactFlagOffset, bytesPerRank,
            rank, !oneRankInPlace, d2hReadyOffset(ctx.localRank), epoch);
      }
      MSCCLPP_CUDATHROW(cudaGetLastError());
      waitForEpoch(ctx.ctrl->d2hReady[ctx.localRank], epoch);
      if (useTinyOriginal || useParallelOriginal) {
        ctx.ctrl->rdmaSignal[ctx.nodeId].store(epoch,
                                               std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_release);
        size_t dataOffset =
            slotOffset + static_cast<size_t>(ctx.nodeId) * bytesPerRank;
        writeOrderedSlot(ctx, dataOffset, flagOffset, bytesPerRank);
      } else {
        size_t segmentOffset =
            slotOffset + static_cast<size_t>(ctx.nodeId) * compactSegmentBytes;
        *reinterpret_cast<uint64_t*>(ctx.sendSlab + segmentOffset +
                                     compactFlagOffset) = epoch;
        std::atomic_thread_fence(std::memory_order_release);
        writeCompactSlot(ctx, segmentOffset, compactSegmentBytes);
      }
      return ncclSuccess;
    }
    if (useTwoRankTinyPack) {
      multiRankTinyPackKernel<<<1, 1, 0, stream>>>(
          send, const_cast<char*>(ctx.sendDeviceSlab), ctx.ctrlDeviceSlab,
          slotOffset, static_cast<size_t>(rank) * bytesPerRank, bytesPerRank,
          d2hReadyOffset(ctx.localRank), epoch);
      MSCCLPP_CUDATHROW(cudaGetLastError());
    } else if (useTwoRankRegisterPack) {
      multiRankRegisterPackKernel<<<1, 128, 0, stream>>>(
          send, const_cast<char*>(ctx.sendDeviceSlab), ctx.ctrlDeviceSlab,
          slotOffset, static_cast<size_t>(rank) * bytesPerRank, bytesPerRank,
          d2hReadyOffset(ctx.localRank), epoch);
      MSCCLPP_CUDATHROW(cudaGetLastError());
    } else {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          ctx.sendSlab + slotOffset + static_cast<size_t>(rank) * bytesPerRank,
          send, bytesPerRank, cudaMemcpyDeviceToHost, stream));
    }
    if (useTwoRankRecvKernel) {
      if (!useTwoRankTinyPack && !useTwoRankRegisterPack) {
        ensurePipelineStreams(ctx);
        MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
      }
      multiRankHostRecvKernel<<<1, 128, 0, stream>>>(
          static_cast<char*>(recvbuff), ctx.sendDeviceSlab, ctx.ctrlDeviceSlab,
          offsetof(HostControl, d2hReady), ctx.groupSize, slotOffset,
          flagOffset, fullBytes, epoch);
      MSCCLPP_CUDATHROW(cudaGetLastError());
      if (!useTwoRankGpuPack) {
        waitForCudaEvent(ctx.inputReadyEvent);
      }
    } else if (ctx.nRanksPerNode <= 2) {
      if (!useTwoRankGpuPack) {
        ensurePipelineStreams(ctx);
        MSCCLPP_CUDATHROW(cudaEventRecord(ctx.inputReadyEvent, stream));
        waitForCudaEvent(ctx.inputReadyEvent);
      }
    } else {
      waitForCudaStream(stream);
    }

    if (ctx.isLeader) {
      if (ctx.nRanksPerNode == 1) {
        std::atomic_thread_fence(std::memory_order_release);
      } else {
        if (!useTwoRankGpuPack) {
          ctx.ctrl->d2hReady[ctx.localRank].store(
              epoch, std::memory_order_release);
        }
        for (int i = 0; i < ctx.groupSize; ++i) {
          waitForEpoch(ctx.ctrl->d2hReady[i], epoch);
        }
      }
      ctx.ctrl->rdmaSignal[ctx.nodeId].store(epoch, std::memory_order_release);
      std::atomic_thread_fence(std::memory_order_release);
      // Leader ships the local node's contiguous half of the output slot.
      size_t dataOffset =
          slotOffset + static_cast<size_t>(ctx.nodeId * ctx.nRanksPerNode) *
                           bytesPerRank;
      // Same-QP ordering makes the flag visible only after the data write.
      writeOrderedSlot(ctx, dataOffset, flagOffset, blockBytes);
    } else {
      if (!useTwoRankGpuPack) {
        ctx.ctrl->d2hReady[ctx.localRank].store(epoch,
                                                std::memory_order_release);
      }
    }
    waitForSlotReady(ctx.sendSlab + flagOffset, epoch);
    if (ctx.nRanksPerNode != 1 && !ctx.isLeader) {
      for (int i = 0; i < ctx.groupSize; ++i) {
        waitForEpoch(ctx.ctrl->d2hReady[i], epoch);
      }
    }

    if (useTwoRankRecvKernel) {
      return ncclSuccess;
    }
    bool oneRankDirectCopy =
        ctx.nodeCount == 2 && ctx.nRanksPerNode == 1 &&
        (oneRankInPlace || fullBytes >= 128 * 1024) &&
        fullBytes != 32 * 1024;
    if (oneRankDirectCopy) {
      int remoteRank = (1 - ctx.nodeId) * ctx.nRanksPerNode;
      ncclResult_t result = copyOneRankPerNodeChunkToOutput(
          ctx, sendbuff, recvbuff, bytesPerRank, /*chunkOffset=*/0,
          bytesPerRank,
          ctx.sendSlab + slotOffset +
              static_cast<size_t>(remoteRank) * bytesPerRank,
          stream);
      if (result != ncclSuccess) return result;
    } else {
      auto* recv = static_cast<char*>(recvbuff);
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(recv, ctx.sendSlab + slotOffset,
                                        fullBytes, cudaMemcpyHostToDevice,
                                        stream));
    }
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN("small ordered AllGather failed: %s", ex.what());
    return mapException(ex);
  } catch (...) {
    WARN("small ordered AllGather failed with an unknown exception");
    return ncclInternalError;
  }
}

ncclResult_t runNumaSplit(
    void const* sendbuff, void* recvbuff, size_t sendcount,
    size_t bytesPerRank, ncclDataType_t datatype, ncclComm_t comm,
    cudaStream_t stream, int rank, int nRanks, int nRanksPerNode,
    std::shared_ptr<Communicator> bootstrapComm, int cudaDevice) {
  if (!isMultiNodeLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  if (nRanksPerNode <= 0 || nRanksPerNode > kMaxRanksPerNode) {
    return ncclInvalidUsage;
  }
  auto layout = getNicGroupLayout(comm, bootstrapComm, rank, nRanks,
                                  nRanksPerNode, cudaDevice);
  if (layout.count <= 1) {
    return ncclInvalidUsage;
  }
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;
  if (sendcount > std::numeric_limits<size_t>::max() / typeSize) {
    return ncclInvalidArgument;
  }
  if (sendcount * typeSize != bytesPerRank) return ncclInvalidUsage;
  if (bytesPerRank == 0) return ncclSuccess;

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    std::vector<AgContext*> groups(layout.count);
    for (int groupId = 0; groupId < layout.count; ++groupId) {
      groups[groupId] = &getNumaContext(
          comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice,
          groupId);
    }

    int localRank = rank % nRanksPerNode;
    int ownGroupId = groupForLocalRank(layout, localRank);
    AgContext& own = *groups[ownGroupId];
    ensurePipelineStreams(own);
    MSCCLPP_CUDATHROW(cudaEventRecord(own.inputReadyEvent, stream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(own.d2hStream,
                                          own.inputReadyEvent, 0));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(own.h2dStream,
                                          own.inputReadyEvent, 0));
    auto const* send = static_cast<char const*>(sendbuff);
    auto* recv = static_cast<char*>(recvbuff);
    bool selfInPlace =
        send == recv + static_cast<size_t>(rank) * bytesPerRank;
    bool selfPreCopied = false;
    if (!selfInPlace && bytesPerRank >= kDirectSelfCopyMinBytes) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          recv + static_cast<size_t>(rank) * bytesPerRank, send, bytesPerRank,
          cudaMemcpyDeviceToDevice, own.h2dStream));
      selfPreCopied = true;
    }

    // Medium/large messages are bandwidth-bound: split local ranks across the
    // available NIC groups so each group moves a contiguous host slab.
    for (size_t chunkOffset = 0; chunkOffset < bytesPerRank;) {
      size_t chunkBytes =
          std::min(groups[ownGroupId]->chunkCapacity,
                   bytesPerRank - chunkOffset);
      std::vector<uint64_t> epochs(layout.count);
      std::vector<size_t> blockBytes(layout.count);
      std::vector<size_t> slotBlockBytes(layout.count);
      std::vector<size_t> slotCounts(layout.count);
      std::vector<size_t> slots(layout.count);
      std::vector<bool> useAck(layout.count);
      std::vector<size_t> sendBases(layout.count);
      std::vector<size_t> recvBases(layout.count);
      bool needSlotReuseBarrier = false;
      for (int groupId = 0; groupId < layout.count; ++groupId) {
        epochs[groupId] = ++groups[groupId]->epoch;
        blockBytes[groupId] =
            static_cast<size_t>(groups[groupId]->groupSize) * chunkBytes;
        size_t slotBytes = bytesPerRank > groups[groupId]->chunkCapacity
                               ? groups[groupId]->chunkCapacity
                               : chunkBytes;
        slotBlockBytes[groupId] =
            static_cast<size_t>(groups[groupId]->groupSize) * slotBytes;
        slotCounts[groupId] = slotCountForChunk(*groups[groupId], slotBytes);
        slots[groupId] =
            static_cast<size_t>((epochs[groupId] - 1) % slotCounts[groupId]);
        useAck[groupId] = slotCounts[groupId] == 1;
        needSlotReuseBarrier |=
            !useAck[groupId] && slots[groupId] == 0 && epochs[groupId] > 1;
        sendBases[groupId] =
            sendSlotOffset(slots[groupId], slotBlockBytes[groupId]);
        recvBases[groupId] = recvSlotOffset(slots[groupId],
                                            groups[groupId]->nodeCount,
                                            slotBlockBytes[groupId]);
      }
      if (needSlotReuseBarrier) {
        MSCCLPP_CUDATHROW(cudaStreamSynchronize(own.h2dStream));
        bootstrapComm->bootstrap()->barrier();
      }

      int ownSlot = own.localRank - own.groupBase;
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          own.sendSlab + sendBases[ownGroupId] +
              static_cast<size_t>(ownSlot) * chunkBytes,
          send + chunkOffset, chunkBytes, cudaMemcpyDeviceToHost,
          own.d2hStream));
      waitForCudaStream(own.d2hStream);
      own.ctrl->d2hReady[own.localRank].store(epochs[ownGroupId],
                                              std::memory_order_release);

      if (own.isLeader) {
        for (int i = 0; i < own.groupSize; ++i) {
          waitForEpoch(own.ctrl->d2hReady[own.groupBase + i],
                       epochs[ownGroupId]);
        }
        own.ctrl->rdmaSignal[own.nodeId].store(epochs[ownGroupId],
                                               std::memory_order_release);
        for (size_t peer = 0; peer < own.peerNodeIds.size(); ++peer) {
          size_t remoteBase = recvBases[ownGroupId] +
                              recvBlockOffset(own.nodeId,
                                              slotBlockBytes[ownGroupId]);
          size_t off = 0;
          int writesSinceFlush = 0;
          while (off < blockBytes[ownGroupId]) {
            size_t chunk =
                std::min(kRdmaChunkBytes, blockBytes[ownGroupId] - off);
            own.peerConnections[peer].write(
                own.peerRemoteRecvMemory[peer], remoteBase + off,
                own.sendMemory, sendBases[ownGroupId] + off, chunk);
            if (++writesSinceFlush == kSignalEveryN) {
              own.peerConnections[peer].flush();
              writesSinceFlush = 0;
            }
            off += chunk;
          }
          signalRdmaReadyAtomic(own, peer, epochs[ownGroupId]);
        }
      }

      for (int groupId = 0; groupId < layout.count; ++groupId) {
        for (int peerNode : groups[groupId]->peerNodeIds) {
          waitForEpoch(groups[groupId]->ctrl->rdmaReady[peerNode],
                       epochs[groupId]);
        }
      }
      for (int groupId = 0; groupId < layout.count; ++groupId) {
        auto& group = *groups[groupId];
        for (int i = 0; i < group.groupSize; ++i) {
          waitForEpoch(group.ctrl->d2hReady[group.groupBase + i],
                       epochs[groupId]);
        }
      }
      for (int groupId = 0; groupId < layout.count; ++groupId) {
        ncclResult_t result = copyGroupChunkToOutput(
            *groups[groupId], sendbuff, recvbuff, bytesPerRank, chunkOffset,
            chunkBytes, sendBases[groupId], recvBases[groupId],
            slotBlockBytes[groupId], own.h2dStream, selfPreCopied);
        if (result != ncclSuccess) return result;
      }
      bool anyUseAck = false;
      for (int groupId = 0; groupId < layout.count; ++groupId) {
        if (useAck[groupId]) {
          anyUseAck = true;
        }
      }
      if (anyUseAck) {
        MSCCLPP_CUDATHROW(cudaStreamSynchronize(own.h2dStream));
      }
      for (int groupId = 0; groupId < layout.count; ++groupId) {
        if (useAck[groupId]) {
          groups[groupId]->ctrl->h2dDone[localRank].store(
              epochs[groupId], std::memory_order_release);
        }
      }

      if (useAck[ownGroupId] && own.isLeader) {
        for (int i = 0; i < nRanksPerNode; ++i) {
          waitForEpoch(own.ctrl->h2dDone[i], epochs[ownGroupId]);
        }
        own.ctrl->ackSignal[own.nodeId].store(epochs[ownGroupId],
                                              std::memory_order_release);
        for (size_t peer = 0; peer < own.peerNodeIds.size(); ++peer) {
          own.peerConnections[peer].write(
              own.peerRemoteCtrlMemory[peer], ackReadyOffset(own.nodeId),
              own.ctrlMemory, ackSignalOffset(own.nodeId), sizeof(uint64_t));
          own.peerConnections[peer].flush();
        }
      }

      for (int groupId = 0; groupId < layout.count; ++groupId) {
        if (useAck[groupId]) {
          for (int peerNode : groups[groupId]->peerNodeIds) {
            waitForEpoch(groups[groupId]->ctrl->ackReady[peerNode],
                         epochs[groupId]);
          }
        }
      }
      chunkOffset += chunkBytes;
    }
    MSCCLPP_CUDATHROW(cudaEventRecord(own.h2dDoneEvent, own.h2dStream));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, own.h2dDoneEvent, 0));
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN("NUMA AllGather failed: %s", ex.what());
    return mapException(ex);
  } catch (...) {
    WARN("NUMA AllGather failed with an unknown exception");
    return ncclInternalError;
  }
}

}  // namespace

ncclResult_t runLiteAllGather(void const* sendbuff, void* recvbuff,
                              size_t sendcount, size_t bytesPerRank,
                              ncclDataType_t datatype, ncclComm_t comm,
                              cudaStream_t stream, int rank, int nRanks,
                              int nRanksPerNode,
                              std::shared_ptr<Communicator> bootstrapComm,
  int cudaDevice) {
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;
  bool twoNodeLayout = isTwoNodeLayout(nRanks, nRanksPerNode);
  if (isMultiNodeLayout(nRanks, nRanksPerNode)) {
    if (nRanksPerNode <= 0 || nRanksPerNode > kMaxRanksPerNode) {
      return ncclInvalidUsage;
    }
    bool isSmall =
        bytesPerRank <=
            std::numeric_limits<size_t>::max() / static_cast<size_t>(nRanks) &&
        bytesPerRank * static_cast<size_t>(nRanks) <
            smallCutoffBytes(nRanksPerNode);
    // Keep the top-level dispatch narrow: small ordered slots first, then a
    // NUMA split only when the GPU/NIC layout exposes multiple symmetric
    // groups, otherwise the single-slab path handles the remaining cases.
    if (twoNodeLayout && isSmall) {
      ncclResult_t result = ncclInvalidUsage;
      if (nRanksPerNode == 1) {
        if (kEnableOneRankGpuDirect &&
            bytesPerRank * static_cast<size_t>(nRanks) >= 1024 * 1024) {
          result = runOneRankGpuDirect(
              sendbuff, recvbuff, sendcount, bytesPerRank, datatype, comm,
              stream, rank, nRanks, nRanksPerNode, bootstrapComm, cudaDevice);
          if (result == ncclSuccess) return result;
          if (result != ncclInvalidUsage && result != ncclInvalidArgument) {
            return result;
          }
        }
      }
      result = runSmallOrdered(
          sendbuff, recvbuff, sendcount, bytesPerRank, datatype, comm, stream,
          rank, nRanks, nRanksPerNode, bootstrapComm, cudaDevice);
      if (result == ncclSuccess) return result;
      if (result != ncclInvalidUsage && result != ncclInvalidArgument) {
        return result;
      }
      result = runSmallFallback(
          sendbuff, recvbuff, sendcount, bytesPerRank, datatype, comm, stream,
          rank, nRanks, nRanksPerNode, bootstrapComm, cudaDevice);
      if (result == ncclSuccess) return result;
      if (result != ncclInvalidUsage && result != ncclInvalidArgument) {
        return result;
      }
    }
    auto layout = getNicGroupLayout(comm, bootstrapComm, rank, nRanks,
                                    nRanksPerNode, cudaDevice);
    if (layout.count > 1 && nRanksPerNode != 2) {
      ncclResult_t result = runNumaSplit(
          sendbuff, recvbuff, sendcount, bytesPerRank, datatype, comm, stream,
          rank, nRanks, nRanksPerNode, bootstrapComm, cudaDevice);
      if (result == ncclSuccess) return result;
      if (result != ncclInvalidUsage && result != ncclInvalidArgument) {
        return result;
      }
    }
    ncclResult_t result = runSingleSlab(
        sendbuff, recvbuff, sendcount, bytesPerRank, datatype, comm, stream,
        rank, nRanks, nRanksPerNode, bootstrapComm, cudaDevice);
    if (result == ncclSuccess) return result;
    if (result != ncclInvalidUsage && result != ncclInvalidArgument) {
      return result;
    }
  }

  return ncclInvalidUsage;
}


void cleanupLiteAllGatherContexts(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(gAllGatherContextMutex);
  gSingleContexts.erase(comm);
  gNumaContexts.erase(comm);
  gCudaDevicesByComm.erase(comm);
  gLayoutByComm.erase(comm);
}

}  // namespace nccl
}  // namespace mscclpp
