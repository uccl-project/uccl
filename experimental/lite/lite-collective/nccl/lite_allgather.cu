#include "native_collectives.hpp"
#include "debug.h"
#include "env.hpp"
#include "gpu_utils.hpp"
#include "ib.hpp"
#include "numa.hpp"
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

inline void* operator new(std::size_t, std::align_val_t, void* ptr) noexcept {
  return ptr;
}

namespace mscclpp {
namespace nccl {
namespace {

// 2-node AllGather has two data paths:
// - <128KiB: one ordered host slot per epoch, direct-QP data+flag write, one H2D.
// - >=128KiB on 2x4G: split local ranks by NUMA/NIC and move two host slabs.


static constexpr int kHostTagBase = 0x560000;
static constexpr int kHostTagStride = 4;
static constexpr int kNumaTagBase = 0x565000;
static constexpr int kNumaGroups = 2;
static constexpr int kNumaGroupSize = 2;
static constexpr int kMaxRanksPerNode = 8;
static constexpr size_t kSmallCutoffBytes = 128 * 1024;
static constexpr size_t kSmallMaxSlots = 1024;
static constexpr int kSmallSignalEvery = 64;
static constexpr size_t kMaxBytesPerRank = 16 * 1024 * 1024;
static constexpr size_t kRdmaChunkBytes = 256 * 1024;
static constexpr int kSignalEveryN = 256;

ncclResult_t cudaResult(cudaError_t error, char const* operation) {
  if (error == cudaSuccess) return ncclSuccess;
  WARN("%s failed with CUDA error: %s", operation, cudaGetErrorString(error));
  if (error == cudaErrorInvalidValue) return ncclInvalidArgument;
  return ncclUnhandledCudaError;
}

struct HostControl {
  alignas(64) std::atomic<uint64_t>
      d2hReady[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> rdmaReady{0};
  alignas(64) std::atomic<uint64_t> rdmaSignal{0};
  alignas(64) std::atomic<uint64_t>
      h2dDone[kMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> ackReady{0};
  alignas(64) std::atomic<uint64_t> ackSignal{0};
};

struct HostNames {
  char sendName[96] = {};
  char recvName[96] = {};
  char ctrlName[96] = {};
};

struct InitStatus {
  int result = static_cast<int>(ncclSuccess);
  char message[160] = {};
};

struct AgContext {
  bool initialized = false;
  bool initializing = false;
  bool owner = false;
  bool isLeader = false;
  int rank = -1;
  int worldSize = -1;
  int nRanksPerNode = -1;
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

std::mutex gAllGatherContextMutex;
std::unordered_map<ncclComm_t, std::unique_ptr<AgContext>>
    gSingleContexts;
std::unordered_map<
    ncclComm_t,
    std::array<std::unique_ptr<AgContext>,
               kNumaGroups>>
    gNumaContexts;

template <typename Context>
class InitGuard {
 public:
  explicit InitGuard(Context* ctx) : ctx_(ctx) {}
  ~InitGuard() {
    if (committed_) return;
    auto failure = std::current_exception();
    if (!failure) {
      failure = std::make_exception_ptr(mscclpp::Error(
          "native collective context initialization did not complete",
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
  Context* ctx_;
  bool committed_ = false;
};

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

ncclResult_t mapException(std::exception const& ex) {
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

mscclpp::ErrorCode initErrorCode(ncclResult_t result) {
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

void publishInitStatus(std::shared_ptr<Communicator> bootstrapComm,
                                 int rank, int nRanks, ncclResult_t result,
                                 std::string const& message,
                                 char const* stage) {
  std::vector<InitStatus> statuses(nRanks);
  auto& local = statuses[rank];
  local.result = static_cast<int>(result);
  if (!message.empty()) {
    std::snprintf(local.message, sizeof(local.message), "%s", message.c_str());
  }
  bootstrapComm->bootstrap()->allGather(statuses.data(),
                                        sizeof(InitStatus));

  for (int peer = 0; peer < nRanks; ++peer) {
    auto peerResult = static_cast<ncclResult_t>(statuses[peer].result);
    if (peerResult != ncclSuccess) {
      std::string detail(statuses[peer].message);
      if (detail.empty()) detail = "unknown initialization error";
      throw mscclpp::Error(std::string(stage) + " failed on rank " +
                               std::to_string(peer) + ": " + detail,
                           initErrorCode(peerResult));
    }
  }
}

int getIBDeviceNumaNode(std::string const& ibDevName) {
  std::string path = "/sys/class/infiniband/" + ibDevName + "/device/numa_node";
  std::ifstream f(path);
  int node = -1;
  if (f.is_open()) f >> node;
  return node;
}

std::vector<mscclpp::Transport> getAvailableIBTransports() {
  static const mscclpp::Transport transports[] = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1,
      mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5,
      mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  int count = 0;
  std::string hcaEnv = mscclpp::env()->hcaDevices;
  if (!hcaEnv.empty()) {
    std::stringstream ss(hcaEnv);
    std::string tok;
    while (std::getline(ss, tok, ',')) ++count;
  } else {
    count = mscclpp::getIBDeviceCount();
  }
  count = std::min(count,
                   static_cast<int>(sizeof(transports) / sizeof(transports[0])));
  std::vector<mscclpp::Transport> result;
  result.reserve(count);
  for (int i = 0; i < count; ++i) result.push_back(transports[i]);
  return result;
}

mscclpp::Transport selectIBTransportForGpu(int cudaDeviceId) {
  auto available = getAvailableIBTransports();
  if (available.empty()) return mscclpp::Transport::Unknown;

  int gpuNuma = -1;
  try {
    gpuNuma = mscclpp::getDeviceNumaNode(cudaDeviceId);
  } catch (...) {
  }

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
  return choices[static_cast<size_t>(cudaDeviceId) % choices.size()];
}

void placeOnNuma(void* mapping, size_t size, int numaNode,
                            char const* name) {
  if (mapping == nullptr || size == 0 || numaNode < 0) return;
  if (numa_available() < 0) {
    WARN("NUMA placement unavailable for %s", name);
    return;
  }
  std::memset(mapping, 0, size);
  numa_tonode_memory(mapping, size, numaNode);
}

void createOwnedShm(std::string const& name, size_t size) {
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

void unlinkOwnedShm(HostNames const& names) {
  if (names.sendName[0] != '\0') shm_unlink(names.sendName);
  if (names.recvName[0] != '\0') shm_unlink(names.recvName);
  if (names.ctrlName[0] != '\0') shm_unlink(names.ctrlName);
}

void* mapShm(std::string const& name, size_t size) {
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

void waitForEpoch(std::atomic<uint64_t> const& value, uint64_t epoch) {
  while (value.load(std::memory_order_acquire) < epoch) {
    std::this_thread::yield();
  }
}

void waitForSlotReady(char const* flagPtr, uint64_t epoch) {
  auto const* value = reinterpret_cast<uint64_t const volatile*>(flagPtr);
  while (*value < epoch) {
    std::this_thread::yield();
  }
}

void waitForCudaEvent(cudaEvent_t event) {
  while (true) {
    cudaError_t result = cudaEventQuery(event);
    if (result == cudaSuccess) return;
    if (result != cudaErrorNotReady) MSCCLPP_CUDATHROW(result);
    std::this_thread::yield();
  }
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
  size_t flagSrcOffset = offsetof(HostControl, rdmaSignal);
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
  ctx->localRank = rank % nRanksPerNode;
  ctx->nodeId = rank / nRanksPerNode;
  ctx->localLeader = ctx->nodeId * nRanksPerNode + groupBase;
  ctx->remoteLeader = (1 - ctx->nodeId) * nRanksPerNode + groupBase;
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
  ctx->slabBytes = static_cast<size_t>(groupSize) * ctx->chunkCapacity;

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
      int tag0 = ctx->numaSplit
                     ? numaTag(rank, nRanks,
                                                  ctx->remoteLeader, 0)
                     : hostTag(rank, nRanks, ctx->remoteLeader,
                                              0);
      int tag1 = ctx->numaSplit
                     ? numaTag(rank, nRanks,
                                                  ctx->remoteLeader, 1)
                     : hostTag(rank, nRanks, ctx->remoteLeader,
                                              1);
      int tag2 = ctx->numaSplit
                     ? numaTag(rank, nRanks,
                                                  ctx->remoteLeader, 2)
                     : hostTag(rank, nRanks, ctx->remoteLeader,
                                              2);
      int tag3 = ctx->numaSplit
                     ? numaTag(rank, nRanks,
                                                  ctx->remoteLeader, 3)
                     : hostTag(rank, nRanks, ctx->remoteLeader,
                                              3);
      auto connectionFuture =
          bootstrapComm->connect(endpointConfig, ctx->remoteLeader, tag0);
      bootstrapComm->sendMemory(ctx->sendMemory, ctx->remoteLeader, tag3);
      auto remoteSendFuture =
          bootstrapComm->recvMemory(ctx->remoteLeader, tag3);
      bootstrapComm->sendMemory(ctx->recvMemory, ctx->remoteLeader, tag1);
      auto remoteRecvFuture =
          bootstrapComm->recvMemory(ctx->remoteLeader, tag1);
      bootstrapComm->sendMemory(ctx->ctrlMemory, ctx->remoteLeader, tag2);
      auto remoteCtrlFuture =
          bootstrapComm->recvMemory(ctx->remoteLeader, tag2);
      ctx->connection = connectionFuture.get();
      ctx->remoteSendMemory = remoteSendFuture.get();
      ctx->remoteRecvMemory = remoteRecvFuture.get();
      ctx->remoteCtrlMemory = remoteCtrlFuture.get();
      ctx->smallQp = ctx->connection.getIbQp();
      if (ctx->smallQp) {
        ctx->sendMemory.getIbMrInfo(ctx->transport, &ctx->smallSendMr, nullptr);
        ctx->ctrlMemory.getIbMrInfo(ctx->transport, &ctx->smallCtrlMr, nullptr);
        ctx->remoteSendMemory.getIbMrInfo(ctx->transport, nullptr,
                                          &ctx->smallRemoteSendMrInfo);
        ctx->remoteRecvMemory.getIbMrInfo(ctx->transport, nullptr,
                                          &ctx->smallRemoteRecvMrInfo);
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

AgContext& getNumaContext(
    ncclComm_t commHandle, std::shared_ptr<Communicator> bootstrapComm,
    int rank, int nRanks, int nRanksPerNode, int cudaDevice, int groupId) {
  int groupBase = groupId * kNumaGroupSize;
  return getContext(
      [&]() -> std::unique_ptr<AgContext>& {
        return gNumaContexts[commHandle][groupId];
      },
      commHandle, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice,
      groupId, groupBase, kNumaGroupSize,
      /*transportDevice=*/groupBase, /*numaSplit=*/true,
      "NUMA AllGather");
}

ncclResult_t finishGroup(ncclResult_t enqueueResult) {
  ncclResult_t groupResult = ncclGroupEnd();
  if (enqueueResult != ncclSuccess) return enqueueResult;
  return groupResult;
}

bool isTwoNodeLayout(int nRanks, int nRanksPerNode) {
  return nRanksPerNode > 1 && nRanks == 2 * nRanksPerNode;
}

bool isRankInGroup(AgContext const& ctx) {
  return ctx.localRank >= ctx.groupBase &&
         ctx.localRank < ctx.groupBase + ctx.groupSize;
}

ncclResult_t copyGroupChunkToOutput(
    AgContext& ctx, void* recvbuff, size_t bytesPerRank,
    size_t chunkOffset, size_t chunkBytes, cudaStream_t stream) {
  auto* recv = static_cast<char*>(recvbuff);
  int localBase = ctx.nodeId * ctx.nRanksPerNode;
  int remoteBase = (1 - ctx.nodeId) * ctx.nRanksPerNode;
  bool wholeRankChunk = chunkOffset == 0 && chunkBytes == bytesPerRank;

  if (wholeRankChunk) {
    size_t blockBytes = static_cast<size_t>(ctx.groupSize) * chunkBytes;
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        recv + static_cast<size_t>(localBase + ctx.groupBase) * bytesPerRank,
        ctx.sendSlab, blockBytes, cudaMemcpyHostToDevice, stream));
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        recv + static_cast<size_t>(remoteBase + ctx.groupBase) * bytesPerRank,
        ctx.recvSlab, blockBytes, cudaMemcpyHostToDevice, stream));
    return ncclSuccess;
  }

  for (int i = 0; i < ctx.groupSize; ++i) {
    int localPeer = localBase + ctx.groupBase + i;
    int remotePeer = remoteBase + ctx.groupBase + i;
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        recv + static_cast<size_t>(localPeer) * bytesPerRank + chunkOffset,
        ctx.sendSlab + static_cast<size_t>(i) * chunkBytes, chunkBytes,
        cudaMemcpyHostToDevice, stream));
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        recv + static_cast<size_t>(remotePeer) * bytesPerRank + chunkOffset,
        ctx.recvSlab + static_cast<size_t>(i) * chunkBytes, chunkBytes,
        cudaMemcpyHostToDevice, stream));
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
    size_t off = 0;
    int writesSinceFlush = 0;
    while (off < blockBytes) {
      size_t chunk = std::min(kRdmaChunkBytes, blockBytes - off);
      ctx.connection.write(ctx.remoteRecvMemory, off, ctx.sendMemory, off,
                           chunk);
      if (++writesSinceFlush == kSignalEveryN) {
        ctx.connection.flush();
        writesSinceFlush = 0;
      }
      off += chunk;
    }
    ctx.ctrl->rdmaSignal.store(epoch, std::memory_order_release);
    ctx.connection.write(
        ctx.remoteCtrlMemory, offsetof(HostControl, rdmaReady),
        ctx.ctrlMemory, offsetof(HostControl, rdmaSignal),
        sizeof(uint64_t));
    ctx.connection.flush();
  }
  waitForEpoch(ctx.ctrl->rdmaReady, epoch);

  if (copyAsSoonAsReady) {
    ncclResult_t result = copyGroupChunkToOutput(
        ctx, recvbuff, bytesPerRank, chunkOffset, chunkBytes, stream);
    if (result != ncclSuccess) return result;
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    ctx.ctrl->h2dDone[ctx.localRank].store(epoch, std::memory_order_release);

    for (int i = 0; i < ctx.nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->h2dDone[i], epoch);
    }
    if (ctx.isLeader) {
      ctx.ctrl->ackSignal.store(epoch, std::memory_order_release);
      ctx.connection.write(
          ctx.remoteCtrlMemory, offsetof(HostControl, ackReady),
          ctx.ctrlMemory, offsetof(HostControl, ackSignal),
          sizeof(uint64_t));
      ctx.connection.flush();
    }
    waitForEpoch(ctx.ctrl->ackReady, epoch);
  }
  return ncclSuccess;
}

ncclResult_t runSingleSlab(
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
  std::memcpy(scratch + static_cast<size_t>(localBase) * bytesPerRank,
              ctx.sendSlab, blockBytes);
  std::memcpy(scratch + static_cast<size_t>(remoteBase) * bytesPerRank,
              ctx.recvSlab, blockBytes);
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
      ctx.connection.write(ctx.remoteRecvMemory, 0, ctx.sendMemory, 0,
                           blockBytes);
      ctx.ctrl->rdmaSignal.store(epoch, std::memory_order_release);
      ctx.connection.write(
          ctx.remoteCtrlMemory, offsetof(HostControl, rdmaReady),
          ctx.ctrlMemory, offsetof(HostControl, rdmaSignal),
          sizeof(uint64_t));
      ctx.connection.flush();
    }
    waitForEpoch(ctx.ctrl->rdmaReady, epoch);
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
      ctx.ctrl->ackSignal.store(epoch, std::memory_order_release);
      ctx.connection.write(
          ctx.remoteCtrlMemory, offsetof(HostControl, ackReady),
          ctx.ctrlMemory, offsetof(HostControl, ackSignal),
          sizeof(uint64_t));
      ctx.connection.flush();
    }
    waitForEpoch(ctx.ctrl->ackReady, epoch);
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
    ctx.ctrl->d2hReady[ctx.localRank].store(epoch, std::memory_order_release);

    if (ctx.isLeader) {
      for (int i = 0; i < ctx.groupSize; ++i) {
        waitForEpoch(ctx.ctrl->d2hReady[i], epoch);
      }
      ctx.ctrl->rdmaSignal.store(epoch, std::memory_order_release);
      std::atomic_thread_fence(std::memory_order_release);
      // Leader ships the local node's contiguous half of the output slot.
      size_t dataOffset =
          slotOffset + static_cast<size_t>(ctx.nodeId * ctx.nRanksPerNode) *
                           bytesPerRank;
      // Same-QP ordering makes the flag visible only after the data write.
      writeOrderedSlot(ctx, dataOffset, flagOffset, blockBytes);
    }
    waitForSlotReady(ctx.sendSlab + flagOffset, epoch);
    for (int i = 0; i < ctx.groupSize; ++i) {
      waitForEpoch(ctx.ctrl->d2hReady[i], epoch);
    }

    auto* recv = static_cast<char*>(recvbuff);
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(recv, ctx.sendSlab + slotOffset,
                                      fullBytes, cudaMemcpyHostToDevice,
                                      stream));
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
  if (!isTwoNodeLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  if (nRanksPerNode !=
      kNumaGroups * kNumaGroupSize) {
    return ncclInvalidUsage;
  }
  if (getAvailableIBTransports().size() < kNumaGroups) {
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
    std::array<AgContext*, kNumaGroups> groups{};
    for (int groupId = 0; groupId < kNumaGroups; ++groupId) {
      groups[groupId] = &getNumaContext(
          comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice,
          groupId);
    }

    int localRank = rank % nRanksPerNode;
    int ownGroupId = localRank / kNumaGroupSize;
    auto const* send = static_cast<char const*>(sendbuff);

    // Medium/large messages are bandwidth-bound: use both NICs by splitting
    // local ranks 0/1 and 2/3 into independent NUMA-local host slabs.
    for (size_t chunkOffset = 0; chunkOffset < bytesPerRank;) {
      size_t chunkBytes =
          std::min(groups[ownGroupId]->chunkCapacity,
                   bytesPerRank - chunkOffset);
      std::array<uint64_t, kNumaGroups> epochs{};
      for (int groupId = 0; groupId < kNumaGroups; ++groupId) {
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
        size_t off = 0;
        int writesSinceFlush = 0;
        while (off < blockBytes) {
          size_t chunk =
              std::min(kRdmaChunkBytes, blockBytes - off);
          own.connection.write(own.remoteRecvMemory, off, own.sendMemory, off,
                               chunk);
          if (++writesSinceFlush == kSignalEveryN) {
            own.connection.flush();
            writesSinceFlush = 0;
          }
          off += chunk;
        }
        own.ctrl->rdmaSignal.store(epochs[ownGroupId],
                                   std::memory_order_release);
        own.connection.write(
            own.remoteCtrlMemory,
            offsetof(HostControl, rdmaReady), own.ctrlMemory,
            offsetof(HostControl, rdmaSignal),
            sizeof(uint64_t));
        own.connection.flush();
      }

      for (int groupId = 0; groupId < kNumaGroups; ++groupId) {
        waitForEpoch(groups[groupId]->ctrl->rdmaReady, epochs[groupId]);
      }
      for (int groupId = 0; groupId < kNumaGroups; ++groupId) {
        ncclResult_t result = copyGroupChunkToOutput(
            *groups[groupId], recvbuff, bytesPerRank, chunkOffset, chunkBytes,
            stream);
        if (result != ncclSuccess) return result;
      }
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      for (int groupId = 0; groupId < kNumaGroups; ++groupId) {
        groups[groupId]->ctrl->h2dDone[localRank].store(
            epochs[groupId], std::memory_order_release);
      }

      if (own.isLeader) {
        for (int i = 0; i < nRanksPerNode; ++i) {
          waitForEpoch(own.ctrl->h2dDone[i], epochs[ownGroupId]);
        }
        own.ctrl->ackSignal.store(epochs[ownGroupId],
                                  std::memory_order_release);
        own.connection.write(
            own.remoteCtrlMemory,
            offsetof(HostControl, ackReady), own.ctrlMemory,
            offsetof(HostControl, ackSignal), sizeof(uint64_t));
        own.connection.flush();
      }

      for (int groupId = 0; groupId < kNumaGroups; ++groupId) {
        waitForEpoch(groups[groupId]->ctrl->ackReady, epochs[groupId]);
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
  if (isTwoNodeLayout(nRanks, nRanksPerNode)) {
    bool isSmall =
        bytesPerRank <=
            std::numeric_limits<size_t>::max() / static_cast<size_t>(nRanks) &&
        bytesPerRank * static_cast<size_t>(nRanks) <
            kSmallCutoffBytes;
    if (isSmall) {
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
    if (nRanksPerNode ==
        kNumaGroups * kNumaGroupSize) {
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

  size_t maxChunkCount = std::max(static_cast<size_t>(1),
                                  static_cast<size_t>(2 * 1024 * 1024) /
                                     typeSize);
  auto const* send = static_cast<char const*>(sendbuff);
  auto* recv = static_cast<char*>(recvbuff);
  for (size_t elemOffset = 0; elemOffset < sendcount;
       elemOffset += maxChunkCount) {
    size_t chunkCount = std::min(maxChunkCount, sendcount - elemOffset);
    size_t chunkBytes = chunkCount * typeSize;
    size_t offsetBytes = elemOffset * typeSize;
    size_t selfOffset = static_cast<size_t>(rank) * bytesPerRank + offsetBytes;
    if (send + offsetBytes != recv + selfOffset) {
      ncclResult_t result =
          cudaResult(cudaMemcpyAsync(recv + selfOffset, send + offsetBytes,
                                     chunkBytes, cudaMemcpyDeviceToDevice,
                                     stream),
                     "allgather self copy");
      if (result != ncclSuccess) return result;
    }

    ncclResult_t result = ncclGroupStart();
    if (result != ncclSuccess) return result;

    ncclResult_t enqueueResult = ncclSuccess;
    for (int peer = 0; peer < nRanks; ++peer) {
      if (peer == rank) continue;
      enqueueResult =
          ncclSend(send + offsetBytes, chunkCount, datatype, peer, comm, stream);
      if (enqueueResult != ncclSuccess) break;
    }
    if (enqueueResult == ncclSuccess) {
      for (int peer = 0; peer < nRanks; ++peer) {
        if (peer == rank) continue;
        enqueueResult =
            ncclRecv(recv + static_cast<size_t>(peer) * bytesPerRank +
                         offsetBytes,
                     chunkCount, datatype, peer, comm, stream);
        if (enqueueResult != ncclSuccess) break;
      }
    }
    result = finishGroup(enqueueResult);
    if (result != ncclSuccess) return result;
    if (elemOffset + chunkCount < sendcount) {
      result =
          cudaResult(cudaStreamSynchronize(stream), "allgather chunk sync");
      if (result != ncclSuccess) return result;
      bootstrapComm->bootstrap()->barrier();
    }
  }
  ncclResult_t result = cudaResult(cudaStreamSynchronize(stream),
                                   "allgather final synchronization");
  if (result != ncclSuccess) return result;
  bootstrapComm->bootstrap()->barrier();
  return ncclSuccess;
}


void cleanupLiteAllGatherContexts(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(gAllGatherContextMutex);
  gSingleContexts.erase(comm);
  gNumaContexts.erase(comm);
}

}  // namespace nccl
}  // namespace mscclpp
