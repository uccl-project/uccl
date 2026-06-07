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

// Multi-node AllGather has two host-memory data paths:
// - <128KiB: one ordered host slot per epoch, direct-QP data+flag write, one H2D.
// - >=128KiB: split local ranks into contiguous NIC groups and move host slabs.


static constexpr int kHostTagBase = 0x560000;
static constexpr int kHostTagStride = 4;
static constexpr int kNumaTagBase = 0x565000;
static constexpr int kMaxRanksPerNode = 8;
static constexpr int kMaxNodes = 16;
static constexpr int kMaxNicGroups = kMaxRanksPerNode;
static constexpr size_t kSmallCutoffBytes = 128 * 1024;
static constexpr size_t kOneRankDirectCopyCutoffBytes = 128 * 1024;
static constexpr size_t kSmallMaxSlots = 1024;
static constexpr int kSmallSignalEvery = 64;
static constexpr size_t kMaxBytesPerRank = 16 * 1024 * 1024;
static constexpr size_t kRdmaChunkBytes = 256 * 1024;
static constexpr int kSignalEveryN = 256;

using lite::createOwnedShm;
using lite::cudaResult;
using lite::getAvailableIBTransports;
using lite::InitGuard;
using lite::mapException;
using lite::mapShm;
using lite::placeOnNuma;
using lite::publishInitStatus;
using lite::selectIBTransportForGpu;
using lite::waitForCudaEvent;
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

  std::string sendName;
  std::string recvName;
  std::string ctrlName;
  void* sendMapping = nullptr;
  void* recvMapping = nullptr;
  void* ctrlMapping = nullptr;
  char* sendSlab = nullptr;
  char* recvSlab = nullptr;
  HostControl* ctrl = nullptr;
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
  int smallWrCount = 0;
  cudaEvent_t smallD2hDoneEvent = nullptr;
  std::mutex initMutex;
  std::condition_variable initCv;
  std::exception_ptr initException = nullptr;

  ~AgContext() {
    smallQp.reset();
    smallSendMr = nullptr;
    smallCtrlMr = nullptr;
    if (smallD2hDoneEvent) cudaEventDestroy(smallD2hDoneEvent);
    connection = mscclpp::Connection{};
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

void unlinkOwnedShm(HostNames const& names) {
  if (names.sendName[0] != '\0') shm_unlink(names.sendName);
  if (names.recvName[0] != '\0') shm_unlink(names.recvName);
  if (names.ctrlName[0] != '\0') shm_unlink(names.ctrlName);
}

void waitForSlotReady(char const* flagPtr, uint64_t epoch) {
  auto const* value = reinterpret_cast<uint64_t const volatile*>(flagPtr);
  while (*value < epoch) {
    std::this_thread::yield();
  }
}

size_t ctrlArrayOffset(size_t baseOffset, int nodeId) {
  return baseOffset + static_cast<size_t>(nodeId) *
                          sizeof(std::atomic<uint64_t>);
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
  bool signaled =
      (++ctx.smallWrCount % kSmallSignalEvery) == 0;
  ctx.smallQp->stageSendWrite(ctx.smallSendMr, ctx.smallRemoteSendMrInfo,
                              static_cast<uint32_t>(dataBytes), /*wrId=*/0,
                              dataOffset, dataOffset, false);
  ctx.smallQp->stageSendWrite(ctx.smallCtrlMr, ctx.smallRemoteSendMrInfo,
                              sizeof(uint64_t), /*wrId=*/0, flagSrcOffset,
                              flagOffset, signaled);
  ctx.smallQp->postSend();
  if (signaled) pollQp(ctx);
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
    MSCCLPP_CUDATHROW(cudaHostRegister(ctx->sendMapping, ctx->slabBytes,
                                       cudaHostRegisterPortable));
    ctx->sendHostRegistered = true;
    MSCCLPP_CUDATHROW(cudaHostRegister(ctx->recvMapping, ctx->slabBytes,
                                       cudaHostRegisterPortable));
    ctx->recvHostRegistered = true;
    MSCCLPP_CUDATHROW(cudaHostRegister(ctx->ctrlMapping,
                                       sizeof(HostControl),
                                       cudaHostRegisterPortable));
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
        auto connectionFuture =
            bootstrapComm->connect(endpointConfig, peerLeader, tag0);
        bootstrapComm->sendMemory(ctx->sendMemory, peerLeader, tag3);
        auto remoteSendFuture = bootstrapComm->recvMemory(peerLeader, tag3);
        bootstrapComm->sendMemory(ctx->recvMemory, peerLeader, tag1);
        auto remoteRecvFuture = bootstrapComm->recvMemory(peerLeader, tag1);
        bootstrapComm->sendMemory(ctx->ctrlMemory, peerLeader, tag2);
        auto remoteCtrlFuture = bootstrapComm->recvMemory(peerLeader, tag2);

        auto connection = connectionFuture.get();
        auto remoteSendMemory = remoteSendFuture.get();
        auto remoteRecvMemory = remoteRecvFuture.get();
        auto remoteCtrlMemory = remoteCtrlFuture.get();
        if (peerLeader == ctx->remoteLeader) {
          ctx->connection = connection;
          ctx->remoteSendMemory = remoteSendMemory;
          ctx->remoteRecvMemory = remoteRecvMemory;
          ctx->remoteCtrlMemory = remoteCtrlMemory;
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

size_t recvBlockOffset(AgContext const& ctx, int sourceNode,
                       size_t blockBytes) {
  return ctx.nodeCount == 2 ? 0 : static_cast<size_t>(sourceNode) * blockBytes;
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
    AgContext& ctx, void* recvbuff, size_t bytesPerRank,
    size_t chunkOffset, size_t chunkBytes, cudaStream_t stream) {
  auto* recv = static_cast<char*>(recvbuff);
  bool wholeRankChunk = chunkOffset == 0 && chunkBytes == bytesPerRank;
  size_t blockBytes = static_cast<size_t>(ctx.groupSize) * chunkBytes;

  for (int node = 0; node < ctx.nodeCount; ++node) {
    int rankBase = node * ctx.nRanksPerNode + ctx.groupBase;
    char const* src =
        node == ctx.nodeId ? ctx.sendSlab
                           : ctx.recvSlab +
                                 recvBlockOffset(ctx, node, blockBytes);
    if (wholeRankChunk) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          recv + static_cast<size_t>(rankBase) * bytesPerRank, src,
          blockBytes, cudaMemcpyHostToDevice, stream));
    } else {
      for (int i = 0; i < ctx.groupSize; ++i) {
        int peer = rankBase + i;
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(
            recv + static_cast<size_t>(peer) * bytesPerRank + chunkOffset,
            src + static_cast<size_t>(i) * chunkBytes, chunkBytes,
            cudaMemcpyHostToDevice, stream));
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t exchangeGroupChunk(AgContext& ctx,
                                    void const* sendbuff, void* recvbuff,
                                    size_t bytesPerRank, size_t chunkOffset,
                                    size_t chunkBytes, cudaStream_t stream,
                                    bool copyAsSoonAsReady) {
  uint64_t epoch = ++ctx.epoch;
  bool inGroup = isRankInGroup(ctx);
  if (inGroup) {
    auto const* send = static_cast<char const*>(sendbuff);
    int slot = ctx.localRank - ctx.groupBase;
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        ctx.sendSlab + static_cast<size_t>(slot) * chunkBytes,
        send + chunkOffset, chunkBytes, cudaMemcpyDeviceToHost, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    ctx.ctrl->d2hReady[ctx.localRank].store(epoch,
                                            std::memory_order_release);
  }

  if (ctx.isLeader) {
    for (int i = 0; i < ctx.groupSize; ++i) {
      waitForEpoch(ctx.ctrl->d2hReady[ctx.groupBase + i], epoch);
    }

    size_t blockBytes = static_cast<size_t>(ctx.groupSize) * chunkBytes;
    ctx.ctrl->rdmaSignal[ctx.nodeId].store(epoch,
                                           std::memory_order_release);
    for (size_t peer = 0; peer < ctx.peerNodeIds.size(); ++peer) {
      int peerNode = ctx.peerNodeIds[peer];
      size_t remoteBase = recvBlockOffset(ctx, ctx.nodeId, blockBytes);
      size_t off = 0;
      int writesSinceFlush = 0;
      while (off < blockBytes) {
        size_t chunk = std::min(kRdmaChunkBytes, blockBytes - off);
        ctx.peerConnections[peer].write(ctx.peerRemoteRecvMemory[peer],
                                        remoteBase + off, ctx.sendMemory, off,
                                        chunk);
        if (++writesSinceFlush == kSignalEveryN) {
          ctx.peerConnections[peer].flush();
          writesSinceFlush = 0;
        }
        off += chunk;
      }
      ctx.peerConnections[peer].write(
          ctx.peerRemoteCtrlMemory[peer], rdmaReadyOffset(ctx.nodeId),
          ctx.ctrlMemory, rdmaSignalOffset(ctx.nodeId), sizeof(uint64_t));
      ctx.peerConnections[peer].flush();
      (void)peerNode;
    }
  }
  for (int peerNode : ctx.peerNodeIds) {
    waitForEpoch(ctx.ctrl->rdmaReady[peerNode], epoch);
  }

  if (copyAsSoonAsReady) {
    bool oneRankDirectCopy =
        ctx.nodeCount == 2 && ctx.nRanksPerNode == 1 &&
        (isOneRankPerNodeInPlace(ctx, sendbuff, recvbuff, bytesPerRank,
                                 chunkOffset) ||
         chunkBytes >= kOneRankDirectCopyCutoffBytes);
    ncclResult_t result =
        oneRankDirectCopy
            ? copyOneRankPerNodeChunkToOutput(
                  ctx, sendbuff, recvbuff, bytesPerRank, chunkOffset,
                  chunkBytes,
                  ctx.recvSlab +
                      recvBlockOffset(ctx, 1 - ctx.nodeId, chunkBytes),
                  stream)
            : copyGroupChunkToOutput(
                  ctx, recvbuff, bytesPerRank, chunkOffset, chunkBytes,
                  stream);
    if (result != ncclSuccess) return result;
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    ctx.ctrl->h2dDone[ctx.localRank].store(epoch, std::memory_order_release);

    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->h2dDone[i], epoch);
    }
    if (ctx.isLeader) {
      ctx.ctrl->ackSignal[ctx.nodeId].store(epoch, std::memory_order_release);
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
    for (size_t chunkOffset = 0; chunkOffset < bytesPerRank;) {
      size_t chunkBytes =
          std::min(ctx.chunkCapacity, bytesPerRank - chunkOffset);
      ncclResult_t result =
          exchangeGroupChunk(ctx, sendbuff, recvbuff, bytesPerRank,
                                 chunkOffset, chunkBytes, stream,
                                 /*copyAsSoonAsReady=*/true);
      if (result != ncclSuccess) return result;
      chunkOffset += chunkBytes;
    }
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
              ctx.recvSlab + recvBlockOffset(ctx, remoteNode, blockBytes),
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
  if (fullBytes >= kSmallCutoffBytes) return ncclInvalidUsage;

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
                           recvBlockOffset(ctx, ctx.nodeId, blockBytes),
                           ctx.sendMemory, 0, blockBytes);
      ctx.ctrl->rdmaSignal[ctx.nodeId].store(epoch, std::memory_order_release);
      ctx.connection.write(
          ctx.remoteCtrlMemory, rdmaReadyOffset(ctx.nodeId),
          ctx.ctrlMemory, rdmaSignalOffset(ctx.nodeId),
          sizeof(uint64_t));
      ctx.connection.flush();
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
  if (fullBytes >= kSmallCutoffBytes) return ncclInvalidUsage;

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    auto& ctx = getSingleContext(
        comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice);
    size_t blockBytes = static_cast<size_t>(ctx.groupSize) * bytesPerRank;
    if (blockBytes == 0) return ncclSuccess;
    size_t perSlotBytes = fullBytes + sizeof(uint64_t);
    size_t slotCount =
        std::min(kSmallMaxSlots, ctx.slabBytes / perSlotBytes);
    if (slotCount < 2) return ncclInvalidUsage;

    uint64_t epoch = ++ctx.epoch;
    size_t slot = static_cast<size_t>((epoch - 1) % slotCount);
    if (epoch > 1 && slot == 0) {
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      if (ctx.isLeader) pollQp(ctx);
      bootstrapComm->bootstrap()->barrier();
    }
    size_t slotOffset = slot * perSlotBytes;
    size_t flagOffset = slotOffset + fullBytes;
    if (ctx.smallD2hDoneEvent == nullptr) {
      MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.smallD2hDoneEvent,
                                                 cudaEventDisableTiming));
    }

    // Slot layout is final-output order: [rank0][rank1]...[rankN][flag].
    // This lets every rank finish with one H2D of the complete AllGather output.
    auto const* send = static_cast<char const*>(sendbuff);
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        ctx.sendSlab + slotOffset + static_cast<size_t>(rank) * bytesPerRank,
        send, bytesPerRank, cudaMemcpyDeviceToHost, stream));
    MSCCLPP_CUDATHROW(cudaEventRecord(ctx.smallD2hDoneEvent, stream));
    waitForCudaEvent(ctx.smallD2hDoneEvent);

    if (ctx.isLeader) {
      if (ctx.nRanksPerNode == 1) {
        std::atomic_thread_fence(std::memory_order_release);
      } else {
        ctx.ctrl->d2hReady[ctx.localRank].store(epoch,
                                                std::memory_order_release);
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
      ctx.ctrl->d2hReady[ctx.localRank].store(epoch, std::memory_order_release);
    }
    waitForSlotReady(ctx.sendSlab + flagOffset, epoch);
    if (ctx.nRanksPerNode != 1) {
      for (int i = 0; i < ctx.groupSize; ++i) {
        waitForEpoch(ctx.ctrl->d2hReady[i], epoch);
      }
    }

    bool oneRankDirectCopy =
        ctx.nodeCount == 2 && ctx.nRanksPerNode == 1 &&
        isOneRankPerNodeInPlace(ctx, sendbuff, recvbuff, bytesPerRank,
                                /*chunkOffset=*/0);
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
    auto const* send = static_cast<char const*>(sendbuff);

    // Medium/large messages are bandwidth-bound: split local ranks across the
    // available NIC groups so each group moves a contiguous host slab.
    for (size_t chunkOffset = 0; chunkOffset < bytesPerRank;) {
      size_t chunkBytes =
          std::min(groups[ownGroupId]->chunkCapacity,
                   bytesPerRank - chunkOffset);
      std::vector<uint64_t> epochs(layout.count);
      for (int groupId = 0; groupId < layout.count; ++groupId) {
        epochs[groupId] = ++groups[groupId]->epoch;
      }

      AgContext& own = *groups[ownGroupId];
      int ownSlot = own.localRank - own.groupBase;
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          own.sendSlab + static_cast<size_t>(ownSlot) * chunkBytes,
          send + chunkOffset, chunkBytes, cudaMemcpyDeviceToHost, stream));
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      own.ctrl->d2hReady[own.localRank].store(epochs[ownGroupId],
                                              std::memory_order_release);

      if (own.isLeader) {
        for (int i = 0; i < own.groupSize; ++i) {
          waitForEpoch(own.ctrl->d2hReady[own.groupBase + i],
                       epochs[ownGroupId]);
        }
        size_t blockBytes = static_cast<size_t>(own.groupSize) * chunkBytes;
        own.ctrl->rdmaSignal[own.nodeId].store(epochs[ownGroupId],
                                               std::memory_order_release);
        for (size_t peer = 0; peer < own.peerNodeIds.size(); ++peer) {
          size_t remoteBase = recvBlockOffset(own, own.nodeId, blockBytes);
          size_t off = 0;
          int writesSinceFlush = 0;
          while (off < blockBytes) {
            size_t chunk = std::min(kRdmaChunkBytes, blockBytes - off);
            own.peerConnections[peer].write(
                own.peerRemoteRecvMemory[peer], remoteBase + off,
                own.sendMemory, off, chunk);
            if (++writesSinceFlush == kSignalEveryN) {
              own.peerConnections[peer].flush();
              writesSinceFlush = 0;
            }
            off += chunk;
          }
          own.peerConnections[peer].write(
              own.peerRemoteCtrlMemory[peer], rdmaReadyOffset(own.nodeId),
              own.ctrlMemory, rdmaSignalOffset(own.nodeId), sizeof(uint64_t));
          own.peerConnections[peer].flush();
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
            *groups[groupId], recvbuff, bytesPerRank, chunkOffset, chunkBytes,
            stream);
        if (result != ncclSuccess) return result;
      }
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      for (int groupId = 0; groupId < layout.count; ++groupId) {
        groups[groupId]->ctrl->h2dDone[localRank].store(
            epochs[groupId], std::memory_order_release);
      }

      if (own.isLeader) {
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
        for (int peerNode : groups[groupId]->peerNodeIds) {
          waitForEpoch(groups[groupId]->ctrl->ackReady[peerNode],
                       epochs[groupId]);
        }
      }
      chunkOffset += chunkBytes;
    }
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
            kSmallCutoffBytes;
    if (twoNodeLayout && isSmall) {
      ncclResult_t result = runSmallOrdered(
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
    if (layout.count > 1) {
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
