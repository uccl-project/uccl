#include "alltoall.hpp"
#include "env.hpp"
#include "gpu_utils.hpp"
#include "ib.hpp"
#include "logger.hpp"
#include "numa.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <cuda.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define NCCL_API extern "C" __attribute__((visibility("default")))

static constexpr auto MSCCLPP_NCCL = mscclpp::LogSubsys::NCCL;

namespace {

static constexpr int kNcclAllToAllOptTagBase = 0x550000;
static constexpr int kNcclAllToAllOptTagStride = 8;
static constexpr size_t kAllToAllRdmaChunkBytes = 256 * 1024;
static constexpr int kAllToAllSignalEveryN = 256;
static constexpr int kOptimizedAllToAllRanksPerNode = 4;
static constexpr int kOptimizedAllToAllNumaGroups = 2;
static constexpr int kOptimizedAllToAllNumaGroupSize =
    kOptimizedAllToAllRanksPerNode / kOptimizedAllToAllNumaGroups;

inline int allToAllOptTag(int rank, int worldSize, int remoteLeader, int slot) {
  int lo = std::min(rank, remoteLeader);
  int hi = std::max(rank, remoteLeader);
  int pairIndex = lo * worldSize + hi;
  return kNcclAllToAllOptTagBase + pairIndex * kNcclAllToAllOptTagStride + slot;
}

struct AllToAllOptSharedControl {
  alignas(64) std::atomic<uint64_t> packReady[kOptimizedAllToAllRanksPerNode];
  alignas(64) std::atomic<uint64_t> rdmaReady{0};
  alignas(64) std::atomic<uint64_t> fanoutDone[kOptimizedAllToAllRanksPerNode];
};

struct AllToAllOptShmNames {
  char sendName[96] = {};
  char recvName[96] = {};
  char ctrlName[96] = {};
};

struct AllToAllOptContext {
  bool initialized = false;
  bool owner = false;
  bool isLeader = false;
  int rank = -1;
  int worldSize = -1;
  int localRank = -1;
  int nodeId = -1;
  int localLeader = -1;
  int remoteLeader = -1;
  size_t maxTableBytes = 1ULL << 30;
  size_t chunkCapacity = 0;
  size_t remoteBytesCapacity = 0;
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
  AllToAllOptSharedControl* ctrl = nullptr;
  bool sendHostRegistered = false;
  bool recvHostRegistered = false;

  mscclpp::Transport transport = mscclpp::Transport::Unknown;
  mscclpp::RegisteredMemory sendMemory;
  mscclpp::RegisteredMemory recvMemory;
  mscclpp::RegisteredMemory remoteRecvMemory;
  mscclpp::Connection connection;

  ~AllToAllOptContext() {
    if (sendHostRegistered) cudaHostUnregister(sendMapping);
    if (recvHostRegistered) cudaHostUnregister(recvMapping);
    if (sendMapping) munmap(sendMapping, slabBytes);
    if (recvMapping) munmap(recvMapping, slabBytes);
    if (ctrlMapping) munmap(ctrlMapping, sizeof(AllToAllOptSharedControl));
    if (owner) {
      if (!sendName.empty()) shm_unlink(sendName.c_str());
      if (!recvName.empty()) shm_unlink(recvName.c_str());
      if (!ctrlName.empty()) shm_unlink(ctrlName.c_str());
    }
  }
};

struct AllToAllNumaOptGroupContext {
  bool owner = false;
  bool isLeader = false;
  int groupId = -1;
  int groupBase = -1;
  int rank = -1;
  int worldSize = -1;
  int localRank = -1;
  int localLeader = -1;
  int remoteLeader = -1;
  size_t maxTableBytes = 1ULL << 30;
  size_t chunkCapacity = 0;
  size_t groupRemoteBytesCapacity = 0;
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
  AllToAllOptSharedControl* ctrl = nullptr;
  bool sendHostRegistered = false;
  bool recvHostRegistered = false;

  mscclpp::Transport transport = mscclpp::Transport::Unknown;
  mscclpp::RegisteredMemory sendMemory;
  mscclpp::RegisteredMemory recvMemory;
  mscclpp::RegisteredMemory remoteRecvMemory;
  mscclpp::Connection connection;

  ~AllToAllNumaOptGroupContext() {
    if (sendHostRegistered) cudaHostUnregister(sendMapping);
    if (recvHostRegistered) cudaHostUnregister(recvMapping);
    if (sendMapping) munmap(sendMapping, slabBytes);
    if (recvMapping) munmap(recvMapping, slabBytes);
    if (ctrlMapping) munmap(ctrlMapping, sizeof(AllToAllOptSharedControl));
    if (owner) {
      if (!sendName.empty()) shm_unlink(sendName.c_str());
      if (!recvName.empty()) shm_unlink(recvName.c_str());
      if (!ctrlName.empty()) shm_unlink(ctrlName.c_str());
    }
  }
};

struct AllToAllNumaOptContext {
  bool initialized = false;
  int rank = -1;
  int localRank = -1;
  int nodeId = -1;
  AllToAllNumaOptGroupContext groups[kOptimizedAllToAllNumaGroups];
};

std::mutex gAllToAllContextMutex;
std::unordered_map<ncclComm_t, std::unique_ptr<AllToAllOptContext>>
    gAllToAllOptContexts;
std::unordered_map<ncclComm_t, std::unique_ptr<AllToAllNumaOptContext>>
    gAllToAllNumaOptContexts;

static int gpuNumaNode(int cudaDeviceId) {
  try {
    return mscclpp::getDeviceNumaNode(cudaDeviceId);
  } catch (...) {
    return -1;
  }
}

static int getIBDeviceNumaNode(std::string const& ibDevName) {
  std::string path = "/sys/class/infiniband/" + ibDevName + "/device/numa_node";
  std::ifstream f(path);
  int node = -1;
  if (f.is_open()) f >> node;
  return node;
}

static std::vector<mscclpp::Transport> getAvailableIBTransports() {
  std::string hcaEnv = mscclpp::env()->hcaDevices;
  int count;
  if (!hcaEnv.empty()) {
    count = 0;
    std::stringstream ss(hcaEnv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      ++count;
    }
  } else {
    count = mscclpp::getIBDeviceCount();
  }

  static const mscclpp::Transport transports[] = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
      mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
      mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  std::vector<mscclpp::Transport> result;
  for (int i = 0; i < count && i < 8; ++i) result.push_back(transports[i]);
  return result;
}

static mscclpp::Transport selectAllToAllIBTransportForGpu(int cudaDeviceId) {
  static std::mutex cacheMu;
  static std::unordered_map<int, mscclpp::Transport> cache;
  {
    std::lock_guard<std::mutex> lk(cacheMu);
    auto it = cache.find(cudaDeviceId);
    if (it != cache.end()) return it->second;
  }

  auto available = getAvailableIBTransports();
  if (available.empty()) return mscclpp::Transport::Unknown;

  int gpuNuma = gpuNumaNode(cudaDeviceId);
  std::vector<mscclpp::Transport> sameNuma;
  for (auto transport : available) {
    try {
      std::string name = mscclpp::getIBDeviceName(transport);
      if (gpuNuma >= 0 && getIBDeviceNumaNode(name) == gpuNuma) {
        sameNuma.push_back(transport);
      }
    } catch (...) {
    }
  }

  auto const& choices = sameNuma.empty() ? available : sameNuma;
  auto chosen = choices[static_cast<size_t>(cudaDeviceId) % choices.size()];
  {
    std::lock_guard<std::mutex> lk(cacheMu);
    cache[cudaDeviceId] = chosen;
  }
  return chosen;
}

inline ncclResult_t mapMscclppException(std::exception const& ex) {
  if (auto const* err = dynamic_cast<mscclpp::Error const*>(&ex)) {
    switch (err->getErrorCode()) {
      case mscclpp::ErrorCode::InvalidUsage:
        return ncclInvalidUsage;
      case mscclpp::ErrorCode::Timeout:
      case mscclpp::ErrorCode::SystemError:
        return ncclSystemError;
      default:
        return ncclInternalError;
    }
  }
  if (dynamic_cast<mscclpp::CudaError const*>(&ex) != nullptr ||
      dynamic_cast<mscclpp::CuError const*>(&ex) != nullptr) {
    return ncclUnhandledCudaError;
  }
  return ncclInternalError;
}

template <typename Fn>
ncclResult_t runAllToAllGuarded(char const* opName, Fn&& fn) {
  try {
    fn();
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN(MSCCLPP_NCCL, std::string(opName),
         " failed: ", std::string(ex.what()));
    return mapMscclppException(ex);
  } catch (...) {
    WARN(MSCCLPP_NCCL, std::string(opName),
         " failed with an unknown exception");
    return ncclInternalError;
  }
}

static bool optimizedAllToAllEnabled() {
  char const* env = std::getenv("MSCCLPP_NCCL_ALLTOALL_OPT");
  return env != nullptr && std::strcmp(env, "0") != 0;
}

static bool optimizedAllToAllUseNodeMode() {
  char const* env = std::getenv("MSCCLPP_NCCL_ALLTOALL_OPT");
  return env != nullptr && std::strcmp(env, "node") == 0;
}

static void waitForSharedEpoch(std::atomic<uint64_t> const& value,
                               uint64_t epoch) {
  while (value.load(std::memory_order_acquire) < epoch) {
    std::this_thread::yield();
  }
}

static void leaderBootstrapExchange(mscclpp::Bootstrap* bootstrap,
                                    int remoteLeader, int tag, uint64_t epoch) {
  std::vector<char> sendData(sizeof(epoch));
  std::memcpy(sendData.data(), &epoch, sizeof(epoch));
  bootstrap->send(sendData, remoteLeader, tag);
  std::vector<char> recvData;
  bootstrap->recv(recvData, remoteLeader, tag);
}

static void createOwnedShm(std::string const& name, size_t size) {
  shm_unlink(name.c_str());
  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) {
    throw mscclpp::Error("shm_open failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  if (ftruncate(fd, size) < 0) {
    close(fd);
    throw mscclpp::Error("ftruncate failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  close(fd);
}

static void* mapShm(std::string const& name, size_t size) {
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
  return mapping;
}

static AllToAllOptContext& getAllToAllOptContext(
    AllToAllCommView const& commView) {
  std::lock_guard<std::mutex> lock(gAllToAllContextMutex);
  auto& existing = gAllToAllOptContexts[commView.handle];
  if (existing && existing->initialized) return *existing;

  auto ctx = std::make_unique<AllToAllOptContext>();
  ctx->rank = commView.comm->bootstrap()->getRank();
  ctx->worldSize = commView.worldSize;
  ctx->localRank = ctx->rank % commView.nRanksPerNode;
  ctx->nodeId = ctx->rank / commView.nRanksPerNode;
  ctx->localLeader = ctx->nodeId * commView.nRanksPerNode;
  ctx->remoteLeader = (1 - ctx->nodeId) * commView.nRanksPerNode;
  ctx->isLeader = (ctx->rank == ctx->localLeader);
  ctx->owner = ctx->isLeader;
  ctx->chunkCapacity = ctx->maxTableBytes / static_cast<size_t>(ctx->worldSize);
  ctx->remoteBytesCapacity =
      static_cast<size_t>(commView.nRanksPerNode) * ctx->chunkCapacity;
  ctx->slabBytes =
      static_cast<size_t>(commView.nRanksPerNode) * ctx->remoteBytesCapacity;

  AllToAllOptShmNames localNames;
  if (ctx->isLeader) {
    std::snprintf(localNames.sendName, sizeof(localNames.sendName),
                  "/mint_a2a_%d_%d_%d_s", getpid(), ctx->rank, ctx->worldSize);
    std::snprintf(localNames.recvName, sizeof(localNames.recvName),
                  "/mint_a2a_%d_%d_%d_r", getpid(), ctx->rank, ctx->worldSize);
    std::snprintf(localNames.ctrlName, sizeof(localNames.ctrlName),
                  "/mint_a2a_%d_%d_%d_c", getpid(), ctx->rank, ctx->worldSize);
    createOwnedShm(localNames.sendName, ctx->slabBytes);
    createOwnedShm(localNames.recvName, ctx->slabBytes);
    createOwnedShm(localNames.ctrlName, sizeof(AllToAllOptSharedControl));
  }

  std::vector<AllToAllOptShmNames> allNames(ctx->worldSize);
  allNames[ctx->rank] = localNames;
  commView.comm->bootstrap()->allGather(allNames.data(),
                                        sizeof(AllToAllOptShmNames));

  AllToAllOptShmNames const& leaderNames = allNames[ctx->localLeader];
  ctx->sendName = leaderNames.sendName;
  ctx->recvName = leaderNames.recvName;
  ctx->ctrlName = leaderNames.ctrlName;

  ctx->sendMapping = mapShm(ctx->sendName, ctx->slabBytes);
  ctx->recvMapping = mapShm(ctx->recvName, ctx->slabBytes);
  ctx->ctrlMapping = mapShm(ctx->ctrlName, sizeof(AllToAllOptSharedControl));
  ctx->sendSlab = static_cast<char*>(ctx->sendMapping);
  ctx->recvSlab = static_cast<char*>(ctx->recvMapping);
  ctx->ctrl = static_cast<AllToAllOptSharedControl*>(ctx->ctrlMapping);
  if (ctx->isLeader) {
    std::memset(ctx->ctrlMapping, 0, sizeof(AllToAllOptSharedControl));
    new (ctx->ctrl) AllToAllOptSharedControl{};
  }
  commView.comm->bootstrap()->barrier();

  MSCCLPP_CUDATHROW(cudaHostRegister(ctx->sendMapping, ctx->slabBytes,
                                     cudaHostRegisterPortable));
  ctx->sendHostRegistered = true;
  MSCCLPP_CUDATHROW(cudaHostRegister(ctx->recvMapping, ctx->slabBytes,
                                     cudaHostRegisterPortable));
  ctx->recvHostRegistered = true;

  if (ctx->isLeader) {
    ctx->transport = selectAllToAllIBTransportForGpu(commView.cudaDevice);
    if (ctx->transport == mscclpp::Transport::Unknown) {
      throw mscclpp::Error("optimized alltoall requires IB transport",
                           mscclpp::ErrorCode::InvalidUsage);
    }
    mscclpp::TransportFlags transportFlags(ctx->transport);
    ctx->sendMemory = commView.comm->registerMemory(
        ctx->sendSlab, ctx->slabBytes, transportFlags);
    ctx->recvMemory = commView.comm->registerMemory(
        ctx->recvSlab, ctx->slabBytes, transportFlags);

    mscclpp::EndpointConfig::Ib ibCfg;
    ibCfg.maxCqPollNum = 128;
    mscclpp::EndpointConfig endpointConfig(
        ctx->transport, mscclpp::Device(mscclpp::DeviceType::CPU),
        /*maxWriteQueueSize=*/-1, ibCfg);
    int tag0 = allToAllOptTag(ctx->rank, ctx->worldSize, ctx->remoteLeader, 0);
    int tag1 = allToAllOptTag(ctx->rank, ctx->worldSize, ctx->remoteLeader, 1);
    auto connectionFuture =
        commView.comm->connect(endpointConfig, ctx->remoteLeader, tag0);
    commView.comm->sendMemory(ctx->recvMemory, ctx->remoteLeader, tag1);
    auto remoteRecvFuture = commView.comm->recvMemory(ctx->remoteLeader, tag1);
    ctx->connection = connectionFuture.get();
    ctx->remoteRecvMemory = remoteRecvFuture.get();
  }

  ctx->initialized = true;
  existing = std::move(ctx);
  return *existing;
}

static AllToAllNumaOptContext& getAllToAllNumaOptContext(
    AllToAllCommView const& commView) {
  std::lock_guard<std::mutex> lock(gAllToAllContextMutex);
  auto& existing = gAllToAllNumaOptContexts[commView.handle];
  if (existing && existing->initialized) return *existing;

  auto ctx = std::make_unique<AllToAllNumaOptContext>();
  ctx->rank = commView.comm->bootstrap()->getRank();
  ctx->localRank = ctx->rank % commView.nRanksPerNode;
  ctx->nodeId = ctx->rank / commView.nRanksPerNode;

  for (int groupId = 0; groupId < kOptimizedAllToAllNumaGroups; ++groupId) {
    auto& group = ctx->groups[groupId];
    group.groupId = groupId;
    group.groupBase = groupId * kOptimizedAllToAllNumaGroupSize;
    group.rank = ctx->rank;
    group.worldSize = commView.worldSize;
    group.localRank = ctx->localRank;
    group.localLeader = ctx->nodeId * commView.nRanksPerNode + group.groupBase;
    group.remoteLeader =
        (1 - ctx->nodeId) * commView.nRanksPerNode + group.groupBase;
    group.isLeader = (ctx->rank == group.localLeader);
    group.owner = group.isLeader;
    group.chunkCapacity =
        group.maxTableBytes / static_cast<size_t>(commView.worldSize);
    group.groupRemoteBytesCapacity =
        static_cast<size_t>(kOptimizedAllToAllNumaGroupSize) *
        group.chunkCapacity;
    group.slabBytes = static_cast<size_t>(commView.nRanksPerNode) *
                      group.groupRemoteBytesCapacity;

    AllToAllOptShmNames localNames;
    if (group.isLeader) {
      std::snprintf(localNames.sendName, sizeof(localNames.sendName),
                    "/mint_a2an_%d_%d_%d_s", getpid(), ctx->rank, groupId);
      std::snprintf(localNames.recvName, sizeof(localNames.recvName),
                    "/mint_a2an_%d_%d_%d_r", getpid(), ctx->rank, groupId);
      std::snprintf(localNames.ctrlName, sizeof(localNames.ctrlName),
                    "/mint_a2an_%d_%d_%d_c", getpid(), ctx->rank, groupId);
      createOwnedShm(localNames.sendName, group.slabBytes);
      createOwnedShm(localNames.recvName, group.slabBytes);
      createOwnedShm(localNames.ctrlName, sizeof(AllToAllOptSharedControl));
    }

    std::vector<AllToAllOptShmNames> allNames(commView.worldSize);
    allNames[ctx->rank] = localNames;
    commView.comm->bootstrap()->allGather(allNames.data(),
                                          sizeof(AllToAllOptShmNames));

    AllToAllOptShmNames const& leaderNames = allNames[group.localLeader];
    group.sendName = leaderNames.sendName;
    group.recvName = leaderNames.recvName;
    group.ctrlName = leaderNames.ctrlName;

    group.sendMapping = mapShm(group.sendName, group.slabBytes);
    group.recvMapping = mapShm(group.recvName, group.slabBytes);
    group.ctrlMapping =
        mapShm(group.ctrlName, sizeof(AllToAllOptSharedControl));
    group.sendSlab = static_cast<char*>(group.sendMapping);
    group.recvSlab = static_cast<char*>(group.recvMapping);
    group.ctrl = static_cast<AllToAllOptSharedControl*>(group.ctrlMapping);
    if (group.isLeader) {
      std::memset(group.ctrlMapping, 0, sizeof(AllToAllOptSharedControl));
      new (group.ctrl) AllToAllOptSharedControl{};
    }
  }

  commView.comm->bootstrap()->barrier();

  for (int groupId = 0; groupId < kOptimizedAllToAllNumaGroups; ++groupId) {
    auto& group = ctx->groups[groupId];
    MSCCLPP_CUDATHROW(cudaHostRegister(group.sendMapping, group.slabBytes,
                                       cudaHostRegisterPortable));
    group.sendHostRegistered = true;
    MSCCLPP_CUDATHROW(cudaHostRegister(group.recvMapping, group.slabBytes,
                                       cudaHostRegisterPortable));
    group.recvHostRegistered = true;

    if (group.isLeader) {
      group.transport = selectAllToAllIBTransportForGpu(commView.cudaDevice);
      if (group.transport == mscclpp::Transport::Unknown) {
        throw mscclpp::Error("per-NUMA alltoall requires IB transport",
                             mscclpp::ErrorCode::InvalidUsage);
      }
      mscclpp::TransportFlags transportFlags(group.transport);
      group.sendMemory = commView.comm->registerMemory(
          group.sendSlab, group.slabBytes, transportFlags);
      group.recvMemory = commView.comm->registerMemory(
          group.recvSlab, group.slabBytes, transportFlags);

      mscclpp::EndpointConfig::Ib ibCfg;
      ibCfg.maxCqPollNum = 32;
      mscclpp::EndpointConfig endpointConfig(
          group.transport, mscclpp::Device(mscclpp::DeviceType::CPU),
          /*maxWriteQueueSize=*/-1, ibCfg);
      int tag0 =
          allToAllOptTag(group.rank, group.worldSize, group.remoteLeader, 4);
      int tag1 =
          allToAllOptTag(group.rank, group.worldSize, group.remoteLeader, 5);
      auto connectionFuture =
          commView.comm->connect(endpointConfig, group.remoteLeader, tag0);
      commView.comm->sendMemory(group.recvMemory, group.remoteLeader, tag1);
      auto remoteRecvFuture =
          commView.comm->recvMemory(group.remoteLeader, tag1);
      group.connection = connectionFuture.get();
      group.remoteRecvMemory = remoteRecvFuture.get();
    }
  }

  ctx->initialized = true;
  existing = std::move(ctx);
  return *existing;
}

static ncclResult_t executePerNumaOptimizedAllToAllRemote(
    AllToAllCommView const& commView,
    std::vector<GroupedP2POp const*> const& sends,
    std::vector<GroupedP2POp const*> const& recvs, size_t chunkBytes) {
  return runAllToAllGuarded("per-NUMA optimized ncclAllToAll", [&]() {
    mscclpp::CudaDeviceGuard deviceGuard(commView.cudaDevice);
    auto& ctx = getAllToAllNumaOptContext(commView);
    int localSize = commView.nRanksPerNode;
    int rank = commView.comm->bootstrap()->getRank();
    int nodeId = rank / commView.nRanksPerNode;
    int remoteBase = (1 - nodeId) * commView.nRanksPerNode;

    for (int groupId = 0; groupId < kOptimizedAllToAllNumaGroups; ++groupId) {
      auto& group = ctx.groups[groupId];
      if (chunkBytes > group.chunkCapacity) {
        throw mscclpp::Error(
            "per-NUMA optimized alltoall message exceeds 1GB table cap",
            mscclpp::ErrorCode::InvalidUsage);
      }

      uint64_t epoch = ++group.epoch;
      if (ctx.localRank >= group.groupBase &&
          ctx.localRank < group.groupBase + kOptimizedAllToAllNumaGroupSize) {
        size_t groupRemoteBytes = static_cast<size_t>(localSize) * chunkBytes;
        int srcInGroup = ctx.localRank - group.groupBase;
        char* localPackDst =
            group.sendSlab + static_cast<size_t>(srcInGroup) * groupRemoteBytes;
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(
            localPackDst, sends[remoteBase]->sendbuff, groupRemoteBytes,
            cudaMemcpyDeviceToHost, sends[remoteBase]->stream));
        MSCCLPP_CUDATHROW(cudaStreamSynchronize(sends[remoteBase]->stream));
        group.ctrl->packReady[ctx.localRank].store(epoch,
                                                   std::memory_order_release);
      }
    }

    std::vector<std::thread> groupThreads;
    std::vector<ncclResult_t> groupResults(kOptimizedAllToAllNumaGroups,
                                           ncclSuccess);
    groupThreads.reserve(kOptimizedAllToAllNumaGroups);
    for (int groupId = 0; groupId < kOptimizedAllToAllNumaGroups; ++groupId) {
      groupThreads.emplace_back([&, groupId]() {
        try {
          mscclpp::CudaDeviceGuard threadDeviceGuard(commView.cudaDevice);
          auto& group = ctx.groups[groupId];
          uint64_t epoch = group.epoch;
          size_t groupRemoteBytes = static_cast<size_t>(localSize) * chunkBytes;

          if (group.isLeader) {
            for (int localRank = group.groupBase;
                 localRank < group.groupBase + kOptimizedAllToAllNumaGroupSize;
                 ++localRank) {
              waitForSharedEpoch(group.ctrl->packReady[localRank], epoch);
            }
            size_t aggregateBytes =
                static_cast<size_t>(kOptimizedAllToAllNumaGroupSize) *
                groupRemoteBytes;
            size_t off = 0;
            int writesSinceFlush = 0;
            while (off < aggregateBytes) {
              size_t chunk =
                  std::min(kAllToAllRdmaChunkBytes, aggregateBytes - off);
              group.connection.write(group.remoteRecvMemory, off,
                                     group.sendMemory, off, chunk);
              if (++writesSinceFlush == kAllToAllSignalEveryN) {
                group.connection.flush();
                writesSinceFlush = 0;
              }
              off += chunk;
            }
            group.connection.flush();

            int doneTag = allToAllOptTag(group.rank, group.worldSize,
                                         group.remoteLeader, 6);
            leaderBootstrapExchange(commView.comm->bootstrap().get(),
                                    group.remoteLeader, doneTag, epoch);
            group.ctrl->rdmaReady.store(epoch, std::memory_order_release);
          } else {
            waitForSharedEpoch(group.ctrl->rdmaReady, epoch);
          }

          for (int srcInGroup = 0; srcInGroup < kOptimizedAllToAllNumaGroupSize;
               ++srcInGroup) {
            char* src = group.recvSlab +
                        static_cast<size_t>(srcInGroup) * groupRemoteBytes +
                        static_cast<size_t>(ctx.localRank) * chunkBytes;
            GroupedP2POp const* recvOp =
                recvs[remoteBase + group.groupBase + srcInGroup];
            MSCCLPP_CUDATHROW(cudaMemcpyAsync(recvOp->recvbuff, src, chunkBytes,
                                              cudaMemcpyHostToDevice,
                                              recvOp->stream));
          }
          MSCCLPP_CUDATHROW(cudaStreamSynchronize(
              recvs[remoteBase + group.groupBase]->stream));
          group.ctrl->fanoutDone[ctx.localRank].store(
              epoch, std::memory_order_release);

          if (group.isLeader) {
            for (int localRank = 0; localRank < localSize; ++localRank) {
              waitForSharedEpoch(group.ctrl->fanoutDone[localRank], epoch);
            }
            int ackTag = allToAllOptTag(group.rank, group.worldSize,
                                        group.remoteLeader, 7);
            leaderBootstrapExchange(commView.comm->bootstrap().get(),
                                    group.remoteLeader, ackTag, epoch);
          }
        } catch (std::exception const& ex) {
          WARN(MSCCLPP_NCCL, "per-NUMA alltoall group failed: ", ex.what());
          groupResults[groupId] = mapMscclppException(ex);
        } catch (...) {
          WARN(MSCCLPP_NCCL,
               "per-NUMA alltoall group failed with unknown exception");
          groupResults[groupId] = ncclInternalError;
        }
      });
    }
    for (auto& thread : groupThreads) thread.join();
    for (auto result : groupResults) {
      if (result != ncclSuccess) {
        throw mscclpp::Error("per-NUMA alltoall group failed",
                             mscclpp::ErrorCode::InternalError);
      }
    }
  });
}

static ncclResult_t executeOptimizedAllToAllRemote(
    AllToAllCommView const& commView, GroupedP2POp const& firstRemoteSend,
    std::vector<GroupedP2POp const*> const& remoteRecvs, size_t chunkBytes) {
  return runAllToAllGuarded("optimized ncclAllToAll", [&]() {
    mscclpp::CudaDeviceGuard deviceGuard(commView.cudaDevice);
    auto& ctx = getAllToAllOptContext(commView);
    if (chunkBytes > ctx.chunkCapacity) {
      throw mscclpp::Error("optimized alltoall message exceeds 1GB table cap",
                           mscclpp::ErrorCode::InvalidUsage);
    }

    uint64_t epoch = ++ctx.epoch;
    int localSize = commView.nRanksPerNode;
    size_t remoteBytes = static_cast<size_t>(localSize) * chunkBytes;
    char* localPackDst =
        ctx.sendSlab + static_cast<size_t>(ctx.localRank) * remoteBytes;

    MSCCLPP_CUDATHROW(cudaMemcpyAsync(localPackDst, firstRemoteSend.sendbuff,
                                      remoteBytes, cudaMemcpyDeviceToHost,
                                      firstRemoteSend.stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(firstRemoteSend.stream));
    ctx.ctrl->packReady[ctx.localRank].store(epoch, std::memory_order_release);

    if (ctx.isLeader) {
      for (int localRank = 0; localRank < localSize; ++localRank) {
        waitForSharedEpoch(ctx.ctrl->packReady[localRank], epoch);
      }
      size_t aggregateBytes = static_cast<size_t>(localSize) * remoteBytes;
      size_t off = 0;
      int writesSinceFlush = 0;
      while (off < aggregateBytes) {
        size_t chunk = std::min(kAllToAllRdmaChunkBytes, aggregateBytes - off);
        ctx.connection.write(ctx.remoteRecvMemory, off, ctx.sendMemory, off,
                             chunk);
        if (++writesSinceFlush == kAllToAllSignalEveryN) {
          ctx.connection.flush();
          writesSinceFlush = 0;
        }
        off += chunk;
      }
      ctx.connection.flush();

      int doneTag =
          allToAllOptTag(ctx.rank, ctx.worldSize, ctx.remoteLeader, 2);
      leaderBootstrapExchange(commView.comm->bootstrap().get(),
                              ctx.remoteLeader, doneTag, epoch);
      ctx.ctrl->rdmaReady.store(epoch, std::memory_order_release);
    } else {
      waitForSharedEpoch(ctx.ctrl->rdmaReady, epoch);
    }

    for (int srcLocal = 0; srcLocal < localSize; ++srcLocal) {
      char* src = ctx.recvSlab + static_cast<size_t>(srcLocal) * remoteBytes +
                  static_cast<size_t>(ctx.localRank) * chunkBytes;
      GroupedP2POp const* recvOp = remoteRecvs[srcLocal];
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(recvOp->recvbuff, src, chunkBytes,
                                        cudaMemcpyHostToDevice,
                                        recvOp->stream));
    }
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(remoteRecvs[0]->stream));
    ctx.ctrl->fanoutDone[ctx.localRank].store(epoch, std::memory_order_release);

    if (ctx.isLeader) {
      for (int localRank = 0; localRank < localSize; ++localRank) {
        waitForSharedEpoch(ctx.ctrl->fanoutDone[localRank], epoch);
      }
      int ackTag = allToAllOptTag(ctx.rank, ctx.worldSize, ctx.remoteLeader, 3);
      leaderBootstrapExchange(commView.comm->bootstrap().get(),
                              ctx.remoteLeader, ackTag, epoch);
    }
  });
}

}  // namespace

ncclResult_t executeSelfGroupedP2POp(GroupedP2POp const& sendOp,
                                     GroupedP2POp const& recvOp) {
  if (sendOp.kind != GroupedP2POpKind::Send ||
      recvOp.kind != GroupedP2POpKind::Recv || sendOp.comm != recvOp.comm ||
      sendOp.count != recvOp.count || sendOp.datatype != recvOp.datatype ||
      sendOp.sendbuff == nullptr || recvOp.recvbuff == nullptr) {
    WARN(MSCCLPP_NCCL, "invalid self ncclSend/ncclRecv pair in group");
    return ncclInvalidUsage;
  }

  size_t typeSize = ncclTypeSize(sendOp.datatype);
  if (typeSize == 0) return ncclInvalidArgument;
  size_t bytes = sendOp.count * typeSize;
  if (bytes == 0 || sendOp.sendbuff == recvOp.recvbuff) return ncclSuccess;

  cudaEvent_t sendReady = nullptr;
  if (sendOp.stream != recvOp.stream) {
    cudaError_t eventResult =
        cudaEventCreateWithFlags(&sendReady, cudaEventDisableTiming);
    if (eventResult != cudaSuccess) return ncclUnhandledCudaError;
    eventResult = cudaEventRecord(sendReady, sendOp.stream);
    if (eventResult == cudaSuccess) {
      eventResult = cudaStreamWaitEvent(recvOp.stream, sendReady, 0);
    }
    cudaEventDestroy(sendReady);
    if (eventResult != cudaSuccess) return ncclUnhandledCudaError;
  }

  cudaError_t copyResult =
      cudaMemcpyAsync(recvOp.recvbuff, sendOp.sendbuff, bytes,
                      cudaMemcpyDeviceToDevice, recvOp.stream);
  return copyResult == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

bool tryExecuteOptimizedGroupedAllToAll(AllToAllCommView const& commView,
                                        std::vector<GroupedP2POp>& ops,
                                        ncclResult_t& result) {
  result = ncclSuccess;
  if (!optimizedAllToAllEnabled() || ops.empty()) return false;

  if (commView.handle == nullptr || !commView.hasIB ||
      commView.worldSize != 8 ||
      commView.nRanksPerNode != kOptimizedAllToAllRanksPerNode ||
      ops.size() != static_cast<size_t>(commView.worldSize * 2)) {
    return false;
  }

  size_t count = ops[0].count;
  ncclDataType_t datatype = ops[0].datatype;
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0 || count == 0) return false;
  size_t chunkBytes = count * typeSize;

  std::vector<GroupedP2POp const*> sends(commView.worldSize, nullptr);
  std::vector<GroupedP2POp const*> recvs(commView.worldSize, nullptr);
  for (auto const& op : ops) {
    if (op.comm != commView.handle || op.count != count ||
        op.datatype != datatype || op.peer < 0 ||
        op.peer >= commView.worldSize) {
      return false;
    }
    if (op.kind == GroupedP2POpKind::Send) {
      if (sends[op.peer] != nullptr || op.sendbuff == nullptr) return false;
      sends[op.peer] = &op;
    } else {
      if (recvs[op.peer] != nullptr || op.recvbuff == nullptr) return false;
      recvs[op.peer] = &op;
    }
  }
  for (int peer = 0; peer < commView.worldSize; ++peer) {
    if (sends[peer] == nullptr || recvs[peer] == nullptr) return false;
  }

  int rank = commView.comm->bootstrap()->getRank();
  int nodeId = rank / commView.nRanksPerNode;
  int remoteBase = (1 - nodeId) * commView.nRanksPerNode;
  auto const* firstRemoteSend = sends[remoteBase];
  char const* remoteSendBase =
      static_cast<char const*>(firstRemoteSend->sendbuff);
  for (int i = 1; i < commView.nRanksPerNode; ++i) {
    if (static_cast<char const*>(sends[remoteBase + i]->sendbuff) !=
        remoteSendBase + static_cast<size_t>(i) * chunkBytes) {
      return false;
    }
  }

  std::vector<GroupedP2POp const*> remoteRecvs(commView.nRanksPerNode);
  for (int i = 0; i < commView.nRanksPerNode; ++i) {
    remoteRecvs[i] = recvs[remoteBase + i];
  }

  if (optimizedAllToAllUseNodeMode()) {
    result = executeOptimizedAllToAllRemote(commView, *firstRemoteSend,
                                            remoteRecvs, chunkBytes);
  } else {
    result = executePerNumaOptimizedAllToAllRemote(commView, sends, recvs,
                                                   chunkBytes);
  }
  if (result != ncclSuccess) return true;

  std::vector<GroupedP2POp> remaining;
  remaining.reserve(ops.size() -
                    static_cast<size_t>(commView.nRanksPerNode * 2));
  for (auto const& op : ops) {
    if (op.peer >= remoteBase &&
        op.peer < remoteBase + commView.nRanksPerNode) {
      continue;
    }
    remaining.push_back(op);
  }
  ops = std::move(remaining);
  return true;
}

void cleanupAllToAllContexts(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(gAllToAllContextMutex);
  gAllToAllOptContexts.erase(comm);
  gAllToAllNumaOptContexts.erase(comm);
}

NCCL_API ncclResult_t ncclAllToAll(void const* sendbuff, void* recvbuff,
                                   size_t count, ncclDataType_t datatype,
                                   ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) {
    WARN(MSCCLPP_NCCL,
         "ncclAllToAll received invalid arguments: sendbuff=%p recvbuff=%p "
         "comm=%p",
         sendbuff, recvbuff, comm);
    return ncclInvalidArgument;
  }

  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) {
    WARN(MSCCLPP_NCCL, "ncclAllToAll got an invalid datatype %d", datatype);
    return ncclInvalidArgument;
  }
  if (count == 0 || count > std::numeric_limits<size_t>::max() / typeSize) {
    WARN(MSCCLPP_NCCL, "ncclAllToAll got an invalid count %zu for datatype %d",
         count, datatype);
    return ncclInvalidArgument;
  }

  int worldSize = 0;
  ncclResult_t result = ncclCommCount(comm, &worldSize);
  if (result != ncclSuccess) return result;
  int rank = 0;
  result = ncclCommUserRank(comm, &rank);
  if (result != ncclSuccess) return result;

  size_t bytes = count * typeSize;
  if (worldSize <= 0 || bytes > std::numeric_limits<size_t>::max() /
                                    static_cast<size_t>(worldSize)) {
    WARN(MSCCLPP_NCCL,
         "ncclAllToAll message size overflows addressable buffer space");
    return ncclInvalidArgument;
  }

  char const* send = static_cast<char const*>(sendbuff);
  char* recv = static_cast<char*>(recvbuff);
  size_t selfOffset = static_cast<size_t>(rank) * bytes;
  if (send + selfOffset != recv + selfOffset) {
    cudaError_t copyResult =
        cudaMemcpyAsync(recv + selfOffset, send + selfOffset, bytes,
                        cudaMemcpyDeviceToDevice, stream);
    if (copyResult != cudaSuccess) {
      WARN(MSCCLPP_NCCL, "ncclAllToAll self-copy failed: %s",
           cudaGetErrorString(copyResult));
      return ncclUnhandledCudaError;
    }
  }

  if (worldSize == 1) return ncclSuccess;

  result = ncclGroupStart();
  if (result != ncclSuccess) return result;

  ncclResult_t enqueueResult = ncclSuccess;
  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer == rank) continue;
    size_t peerOffset = static_cast<size_t>(peer) * bytes;
    enqueueResult =
        ncclSend(send + peerOffset, count, datatype, peer, comm, stream);
    if (enqueueResult != ncclSuccess) break;
  }
  if (enqueueResult == ncclSuccess) {
    for (int peer = 0; peer < worldSize; ++peer) {
      if (peer == rank) continue;
      size_t peerOffset = static_cast<size_t>(peer) * bytes;
      enqueueResult =
          ncclRecv(recv + peerOffset, count, datatype, peer, comm, stream);
      if (enqueueResult != ncclSuccess) break;
    }
  }

  ncclResult_t groupResult = ncclGroupEnd();
  if (enqueueResult != ncclSuccess) return enqueueResult;
  return groupResult;
}

NCCL_API ncclResult_t ncclAlltoAll(void const* sendbuff, void* recvbuff,
                                   size_t count, ncclDataType_t datatype,
                                   ncclComm_t comm, cudaStream_t stream) {
  return ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

NCCL_API ncclResult_t ncclAllToAllv(void const* sendbuff,
                                    [[maybe_unused]] const size_t sendcounts[],
                                    const size_t sdispls[], void* recvbuff,
                                    const size_t recvcounts[],
                                    const size_t rdispls[],
                                    ncclDataType_t datatype, ncclComm_t comm,
                                    cudaStream_t stream) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr ||
      recvcounts == nullptr || rdispls == nullptr || sdispls == nullptr) {
    return ncclInvalidArgument;
  }
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;

  int worldSize = 0;
  ncclResult_t result = ncclCommCount(comm, &worldSize);
  if (result != ncclSuccess) return result;

  size_t bytes = recvcounts[0] * typeSize;
  if (worldSize == 1) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        static_cast<char*>(recvbuff) + rdispls[0] * typeSize,
        static_cast<char const*>(sendbuff) + sdispls[0] * typeSize, bytes,
        cudaMemcpyDeviceToDevice, stream));
    return ncclSuccess;
  }
  WARN(MSCCLPP_NCCL, "ncclAllToAllv is currently unavailable");
  return ncclInternalError;
}
