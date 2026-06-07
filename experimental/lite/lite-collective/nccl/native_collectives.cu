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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

inline void* operator new(std::size_t, std::align_val_t, void* ptr) noexcept {
  return ptr;
}

namespace mscclpp {
namespace nccl {
namespace {

static constexpr int kNativeReduceScatterHostTagBase = 0x570000;
static constexpr int kNativeReduceScatterHostTagStride = 8;
static constexpr int kNativeHostMaxRanksPerNode = 8;
static constexpr size_t kNativeReduceScatterHostMaxShardBytes = 2 * 1024 * 1024;
static constexpr size_t kNativeReduceScatterRdmaChunkBytes = 4 * 1024 * 1024;
static constexpr int kNativeReduceScatterSignalEveryN = 256;
static constexpr int kNativeReduceScatterNumaPairScratchRows = 21;

ncclResult_t cudaResult(cudaError_t error, char const* operation) {
  if (error == cudaSuccess) return ncclSuccess;
  WARN("%s failed with CUDA error: %s", operation, cudaGetErrorString(error));
  if (error == cudaErrorInvalidValue) return ncclInvalidArgument;
  return ncclUnhandledCudaError;
}


struct NativeReduceScatterHostControl {
  alignas(64) std::atomic<uint64_t>
      d2hReady[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t>
      partialReady[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> rdmaReady{0};
  alignas(64) std::atomic<uint64_t> rdmaSignal{0};
  alignas(64) std::atomic<uint64_t>
      h2dDone[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t> ackReady{0};
  alignas(64) std::atomic<uint64_t> ackSignal{0};
  alignas(64) std::atomic<uint64_t>
      pairRdmaReady[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t>
      pairRdmaSignal[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t>
      pairAckReady[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t>
      pairAckSignal[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t>
      localCopyReady[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t>
      localCrossReady[kNativeHostMaxRanksPerNode];
  alignas(64) std::atomic<uint64_t>
      localDoneReady[kNativeHostMaxRanksPerNode];
};

struct NativeReduceScatterHostNames {
  char workName[96] = {};
  char ctrlName[96] = {};
};

struct NativeHostInitStatus {
  int result = static_cast<int>(ncclSuccess);
  char message[160] = {};
};


struct NativeReduceScatterHostContext {
  bool initialized = false;
  bool initializing = false;
  bool owner = false;
  bool isLeader = false;
  int rank = -1;
  int worldSize = -1;
  int localRank = -1;
  int nodeId = -1;
  int localLeader = -1;
  int remoteLeader = -1;
  int cudaDevice = -1;
  size_t maxShardBytes = kNativeReduceScatterHostMaxShardBytes;
  size_t inputCapacity = 0;
  size_t partialBlockCapacity = 0;
  size_t sendSlabOffset = 0;
  size_t localPartialOffset = 0;
  size_t sendPartialOffset = 0;
  size_t recvPartialOffset = 0;
  size_t workBytes = 0;
  uint64_t blockEpoch = 0;
  uint64_t pairEpoch = 0;

  std::string workName;
  std::string ctrlName;
  void* workMapping = nullptr;
  void* ctrlMapping = nullptr;
  char* workSlab = nullptr;
  NativeReduceScatterHostControl* ctrl = nullptr;
  bool workHostRegistered = false;
  bool ctrlHostRegistered = false;

  mscclpp::Transport transport = mscclpp::Transport::Unknown;
  mscclpp::RegisteredMemory sendPartialMemory;
  mscclpp::RegisteredMemory recvPartialMemory;
  mscclpp::RegisteredMemory ctrlMemory;
  mscclpp::RegisteredMemory remoteRecvPartialMemory;
  mscclpp::RegisteredMemory remoteCtrlMemory;
  mscclpp::Connection connection;
  mscclpp::RegisteredMemory pairRemoteRecvPartialMemory;
  mscclpp::RegisteredMemory pairRemoteCtrlMemory;
  mscclpp::Connection pairConnection;
  bool localScratchIpcReady = false;
  void* localScratchBuffer = nullptr;
  size_t localScratchBufferSize = 0;
  mscclpp::RegisteredMemory localScratchMemory;
  std::vector<mscclpp::RegisteredMemory> remoteScratchMemories;
  std::vector<char*> remoteScratchPtrs;
  std::mutex initMutex;
  std::condition_variable initCv;
  std::exception_ptr initException = nullptr;

  char* sendSlab() { return workSlab + sendSlabOffset; }
  char* localPartialSlab() { return workSlab + localPartialOffset; }
  char* sendPartialSlab() { return workSlab + sendPartialOffset; }
  char* recvPartialSlab() { return workSlab + recvPartialOffset; }

  ~NativeReduceScatterHostContext() {
    pairConnection = mscclpp::Connection{};
    connection = mscclpp::Connection{};
    remoteScratchPtrs.clear();
    remoteScratchMemories.clear();
    localScratchMemory = mscclpp::RegisteredMemory{};
    pairRemoteCtrlMemory = mscclpp::RegisteredMemory{};
    pairRemoteRecvPartialMemory = mscclpp::RegisteredMemory{};
    remoteCtrlMemory = mscclpp::RegisteredMemory{};
    remoteRecvPartialMemory = mscclpp::RegisteredMemory{};
    ctrlMemory = mscclpp::RegisteredMemory{};
    recvPartialMemory = mscclpp::RegisteredMemory{};
    sendPartialMemory = mscclpp::RegisteredMemory{};
    if (workHostRegistered) cudaHostUnregister(workMapping);
    if (ctrlHostRegistered) cudaHostUnregister(ctrlMapping);
    if (workMapping) munmap(workMapping, workBytes);
    if (ctrlMapping) munmap(ctrlMapping,
                            sizeof(NativeReduceScatterHostControl));
    if (owner) {
      if (!workName.empty()) shm_unlink(workName.c_str());
      if (!ctrlName.empty()) shm_unlink(ctrlName.c_str());
    }
  }
};

std::mutex gNativeCollectiveContextMutex;
std::unordered_map<ncclComm_t,
                   std::unique_ptr<NativeReduceScatterHostContext>>
    gReduceScatterHostContexts;

template <typename Context>
class NativeContextInitGuard {
 public:
  explicit NativeContextInitGuard(Context* ctx) : ctx_(ctx) {}
  ~NativeContextInitGuard() {
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


int nativeReduceScatterHostTag(int rank, int worldSize, int remoteLeader,
                               int slot) {
  int lo = std::min(rank, remoteLeader);
  int hi = std::max(rank, remoteLeader);
  int pairIndex = lo * worldSize + hi;
  return kNativeReduceScatterHostTagBase +
         pairIndex * kNativeReduceScatterHostTagStride + slot;
}

ncclResult_t mapNativeException(std::exception const& ex) {
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

mscclpp::ErrorCode nativeInitErrorCode(ncclResult_t result) {
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

void publishNativeHostInitStatus(std::shared_ptr<Communicator> bootstrapComm,
                                 int rank, int nRanks, ncclResult_t result,
                                 std::string const& message,
                                 char const* stage) {
  std::vector<NativeHostInitStatus> statuses(nRanks);
  auto& local = statuses[rank];
  local.result = static_cast<int>(result);
  if (!message.empty()) {
    std::snprintf(local.message, sizeof(local.message), "%s", message.c_str());
  }
  bootstrapComm->bootstrap()->allGather(statuses.data(),
                                        sizeof(NativeHostInitStatus));

  for (int peer = 0; peer < nRanks; ++peer) {
    auto peerResult = static_cast<ncclResult_t>(statuses[peer].result);
    if (peerResult != ncclSuccess) {
      std::string detail(statuses[peer].message);
      if (detail.empty()) detail = "unknown initialization error";
      throw mscclpp::Error(std::string(stage) + " failed on rank " +
                               std::to_string(peer) + ": " + detail,
                           nativeInitErrorCode(peerResult));
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

mscclpp::Transport selectNativeIBTransportForGpu(int cudaDeviceId) {
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


void unlinkOwnedShm(NativeReduceScatterHostNames const& names) {
  if (names.workName[0] != '\0') shm_unlink(names.workName);
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


NativeReduceScatterHostContext& getReduceScatterHostContext(
    ncclComm_t commHandle, std::shared_ptr<Communicator> bootstrapComm,
    int rank, int nRanks, int nRanksPerNode, int cudaDevice) {
  NativeReduceScatterHostContext* ctx = nullptr;
  bool shouldInitialize = false;
  {
    std::lock_guard<std::mutex> lock(gNativeCollectiveContextMutex);
    auto& existing = gReduceScatterHostContexts[commHandle];
    if (!existing) {
      existing = std::make_unique<NativeReduceScatterHostContext>();
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
      throw mscclpp::Error("host-staged reducescatter initialization failed",
                           mscclpp::ErrorCode::InternalError);
    }
  }
  NativeContextInitGuard<NativeReduceScatterHostContext> initGuard(ctx);

  ctx->rank = rank;
  ctx->worldSize = nRanks;
  ctx->localRank = rank % nRanksPerNode;
  ctx->nodeId = rank / nRanksPerNode;
  ctx->localLeader = ctx->nodeId * nRanksPerNode;
  ctx->remoteLeader = (1 - ctx->nodeId) * nRanksPerNode;
  ctx->cudaDevice = cudaDevice;
  ctx->isLeader = rank == ctx->localLeader;
  ctx->owner = ctx->isLeader;
  ctx->inputCapacity = static_cast<size_t>(nRanks) * ctx->maxShardBytes;
  ctx->partialBlockCapacity =
      static_cast<size_t>(nRanksPerNode) * ctx->maxShardBytes;
  ctx->sendSlabOffset = 0;
  ctx->localPartialOffset =
      static_cast<size_t>(nRanksPerNode) * ctx->inputCapacity;
  ctx->sendPartialOffset =
      ctx->localPartialOffset + ctx->partialBlockCapacity;
  ctx->recvPartialOffset =
      ctx->sendPartialOffset + ctx->partialBlockCapacity;
  ctx->workBytes = ctx->recvPartialOffset + ctx->partialBlockCapacity;

  NativeReduceScatterHostNames localNames;
  ncclResult_t shmCreateResult = ncclSuccess;
  std::string shmCreateMessage;
  try {
    if (ctx->isLeader) {
      auto commNonce = static_cast<unsigned long long>(
          reinterpret_cast<uintptr_t>(commHandle));
      std::snprintf(localNames.workName, sizeof(localNames.workName),
                    "/mint_rs_%llx_%d_%d_%d_w", commNonce, getpid(), rank,
                    nRanks);
      std::snprintf(localNames.ctrlName, sizeof(localNames.ctrlName),
                    "/mint_rs_%llx_%d_%d_%d_c", commNonce, getpid(), rank,
                    nRanks);
      createOwnedShm(localNames.workName, ctx->workBytes);
      createOwnedShm(localNames.ctrlName,
                     sizeof(NativeReduceScatterHostControl));
    }
  } catch (std::exception const& ex) {
    shmCreateResult = mapNativeException(ex);
    shmCreateMessage = ex.what();
  } catch (...) {
    shmCreateResult = ncclInternalError;
    shmCreateMessage = "unknown shared-memory create exception";
  }
  try {
    publishNativeHostInitStatus(
        bootstrapComm, rank, nRanks, shmCreateResult, shmCreateMessage,
        "host-staged reducescatter shared-memory create");
  } catch (...) {
    if (ctx->isLeader) unlinkOwnedShm(localNames);
    throw;
  }

  std::vector<NativeReduceScatterHostNames> allNames(nRanks);
  allNames[rank] = localNames;
  bootstrapComm->bootstrap()->allGather(
      allNames.data(), sizeof(NativeReduceScatterHostNames));
  NativeReduceScatterHostNames const& leaderNames = allNames[ctx->localLeader];
  ctx->workName = leaderNames.workName;
  ctx->ctrlName = leaderNames.ctrlName;

  ncclResult_t shmMapResult = ncclSuccess;
  std::string shmMapMessage;
  try {
    ctx->workMapping = mapShm(ctx->workName, ctx->workBytes);
    ctx->ctrlMapping =
        mapShm(ctx->ctrlName, sizeof(NativeReduceScatterHostControl));
    ctx->workSlab = static_cast<char*>(ctx->workMapping);
    ctx->ctrl = static_cast<NativeReduceScatterHostControl*>(ctx->ctrlMapping);
    if (ctx->isLeader) {
      std::memset(ctx->ctrlMapping, 0,
                  sizeof(NativeReduceScatterHostControl));
      new (ctx->ctrl) NativeReduceScatterHostControl{};
    }
  } catch (std::exception const& ex) {
    shmMapResult = mapNativeException(ex);
    shmMapMessage = ex.what();
  } catch (...) {
    shmMapResult = ncclInternalError;
    shmMapMessage = "unknown shared-memory map exception";
  }
  publishNativeHostInitStatus(bootstrapComm, rank, nRanks, shmMapResult,
                              shmMapMessage,
                              "host-staged reducescatter shared-memory map");
  bootstrapComm->bootstrap()->barrier();

  ncclResult_t setupResult = ncclSuccess;
  std::string setupMessage;
  try {
    MSCCLPP_CUDATHROW(cudaHostRegister(ctx->workMapping, ctx->workBytes,
                                       cudaHostRegisterPortable));
    ctx->workHostRegistered = true;
    MSCCLPP_CUDATHROW(cudaHostRegister(ctx->ctrlMapping,
                                       sizeof(NativeReduceScatterHostControl),
                                       cudaHostRegisterPortable));
    ctx->ctrlHostRegistered = true;
    ctx->transport = selectNativeIBTransportForGpu(cudaDevice);
    if (ctx->transport == mscclpp::Transport::Unknown) {
      throw mscclpp::Error("host-staged reducescatter requires IB transport",
                           mscclpp::ErrorCode::InvalidUsage);
    }
    mscclpp::TransportFlags transportFlags(ctx->transport);
    ctx->sendPartialMemory = bootstrapComm->registerMemory(
        ctx->sendPartialSlab(), ctx->partialBlockCapacity, transportFlags);
    ctx->recvPartialMemory = bootstrapComm->registerMemory(
        ctx->recvPartialSlab(), ctx->partialBlockCapacity, transportFlags);
    ctx->ctrlMemory =
        bootstrapComm->registerMemory(ctx->ctrlMapping,
                                      sizeof(NativeReduceScatterHostControl),
                                      transportFlags);
  } catch (std::exception const& ex) {
    setupResult = mapNativeException(ex);
    setupMessage = ex.what();
  } catch (...) {
    setupResult = ncclInternalError;
    setupMessage = "unknown setup exception";
  }
  publishNativeHostInitStatus(bootstrapComm, rank, nRanks, setupResult,
                              setupMessage, "host-staged reducescatter setup");

  ncclResult_t connectResult = ncclSuccess;
  std::string connectMessage;
  try {
    if (ctx->isLeader) {
      mscclpp::EndpointConfig::Ib ibCfg;
      ibCfg.maxCqPollNum = 128;
      mscclpp::EndpointConfig endpointConfig(
          ctx->transport, mscclpp::Device(mscclpp::DeviceType::CPU),
          /*maxWriteQueueSize=*/-1, ibCfg);
      int tag0 =
          nativeReduceScatterHostTag(rank, nRanks, ctx->remoteLeader, 0);
      int tag1 =
          nativeReduceScatterHostTag(rank, nRanks, ctx->remoteLeader, 1);
      int tag2 =
          nativeReduceScatterHostTag(rank, nRanks, ctx->remoteLeader, 2);
      auto connectionFuture =
          bootstrapComm->connect(endpointConfig, ctx->remoteLeader, tag0);
      bootstrapComm->sendMemory(ctx->recvPartialMemory, ctx->remoteLeader,
                                tag1);
      auto remoteRecvFuture =
          bootstrapComm->recvMemory(ctx->remoteLeader, tag1);
      bootstrapComm->sendMemory(ctx->ctrlMemory, ctx->remoteLeader, tag2);
      auto remoteCtrlFuture =
          bootstrapComm->recvMemory(ctx->remoteLeader, tag2);
      ctx->connection = connectionFuture.get();
      ctx->remoteRecvPartialMemory = remoteRecvFuture.get();
      ctx->remoteCtrlMemory = remoteCtrlFuture.get();
    }

    int remotePeer = ctx->remoteLeader + ctx->localRank;
    mscclpp::EndpointConfig::Ib pairIbCfg;
    pairIbCfg.maxCqPollNum = 128;
    mscclpp::EndpointConfig pairEndpointConfig(
        ctx->transport, mscclpp::Device(mscclpp::DeviceType::CPU),
        /*maxWriteQueueSize=*/-1, pairIbCfg);
    int pairTag0 = nativeReduceScatterHostTag(rank, nRanks, remotePeer, 3);
    int pairTag1 = nativeReduceScatterHostTag(rank, nRanks, remotePeer, 4);
    int pairTag2 = nativeReduceScatterHostTag(rank, nRanks, remotePeer, 5);
    auto pairConnectionFuture =
        bootstrapComm->connect(pairEndpointConfig, remotePeer, pairTag0);
    bootstrapComm->sendMemory(ctx->recvPartialMemory, remotePeer, pairTag1);
    auto pairRemoteRecvFuture =
        bootstrapComm->recvMemory(remotePeer, pairTag1);
    bootstrapComm->sendMemory(ctx->ctrlMemory, remotePeer, pairTag2);
    auto pairRemoteCtrlFuture =
        bootstrapComm->recvMemory(remotePeer, pairTag2);
    ctx->pairConnection = pairConnectionFuture.get();
    ctx->pairRemoteRecvPartialMemory = pairRemoteRecvFuture.get();
    ctx->pairRemoteCtrlMemory = pairRemoteCtrlFuture.get();
  } catch (std::exception const& ex) {
    connectResult = mapNativeException(ex);
    connectMessage = ex.what();
  } catch (...) {
    connectResult = ncclInternalError;
    connectMessage = "unknown connection exception";
  }
  publishNativeHostInitStatus(bootstrapComm, rank, nRanks, connectResult,
                              connectMessage,
                              "host-staged reducescatter connect");

  initGuard.commit();
  return *ctx;
}

void ensureReduceScatterLocalScratchIpc(
    NativeReduceScatterHostContext& ctx,
    std::shared_ptr<Communicator> bootstrapComm, int rank, int nRanks,
    int nRanksPerNode, void* scratchBuffer, size_t scratchBufferSize) {
  if (ctx.localScratchIpcReady && ctx.localScratchBuffer == scratchBuffer &&
      ctx.localScratchBufferSize == scratchBufferSize) {
    return;
  }

  mscclpp::TransportFlags ipcFlags(mscclpp::Transport::CudaIpc);
  ctx.localScratchMemory =
      bootstrapComm->registerMemory(scratchBuffer, scratchBufferSize, ipcFlags);

  ctx.remoteScratchMemories.clear();
  ctx.remoteScratchPtrs.clear();
  ctx.remoteScratchMemories.reserve(nRanksPerNode - 1);
  ctx.remoteScratchPtrs.reserve(nRanksPerNode - 1);

  int localBase = (rank / nRanksPerNode) * nRanksPerNode;
  std::vector<decltype(bootstrapComm->recvMemory(0, 0))> memoryFutures;
  memoryFutures.reserve(nRanksPerNode - 1);
  for (int i = 0; i < nRanksPerNode; ++i) {
    int peer = localBase + i;
    if (peer == rank) continue;
    int tag = nativeReduceScatterHostTag(rank, nRanks, peer, 6);
    bootstrapComm->sendMemory(ctx.localScratchMemory, peer, tag);
    memoryFutures.push_back(bootstrapComm->recvMemory(peer, tag));
  }

  for (auto& future : memoryFutures) {
    ctx.remoteScratchMemories.push_back(future.get());
  }
  for (auto const& memory : ctx.remoteScratchMemories) {
    ctx.remoteScratchPtrs.push_back(static_cast<char*>(memory.data()));
  }

  ctx.localScratchBuffer = scratchBuffer;
  ctx.localScratchBufferSize = scratchBufferSize;
  ctx.localScratchIpcReady = true;
}

void reduceFloatShardFromHost(char const* __restrict__ sendSlab,
                              size_t inputStride, size_t bytesPerRank,
                              int nRanksPerNode, int targetRank,
                              size_t recvcount, float* __restrict__ dst) {
  size_t targetOffset = static_cast<size_t>(targetRank) * bytesPerRank;
  if (nRanksPerNode == 4) {
    auto const* __restrict__ src0 =
        reinterpret_cast<float const*>(sendSlab + targetOffset);
    auto const* __restrict__ src1 = reinterpret_cast<float const*>(
        sendSlab + inputStride + targetOffset);
    auto const* __restrict__ src2 = reinterpret_cast<float const*>(
        sendSlab + 2 * inputStride + targetOffset);
    auto const* __restrict__ src3 = reinterpret_cast<float const*>(
        sendSlab + 3 * inputStride + targetOffset);
#pragma GCC ivdep
    for (size_t i = 0; i < recvcount; ++i) {
      dst[i] = src0[i] + src1[i] + src2[i] + src3[i];
    }
    return;
  }

  auto const* first = reinterpret_cast<float const*>(sendSlab + targetOffset);
  std::memcpy(dst, first, recvcount * sizeof(float));
  for (int local = 1; local < nRanksPerNode; ++local) {
    auto const* src = reinterpret_cast<float const*>(
        sendSlab + static_cast<size_t>(local) * inputStride + targetOffset);
#pragma GCC ivdep
    for (size_t i = 0; i < recvcount; ++i) dst[i] += src[i];
  }
}

template <typename T>
__global__ void reduceRowsKernel(char const* rows, void* output, size_t count,
                                 size_t rowBytes, int nRows, ncclRedOp_t op) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  auto const* first = reinterpret_cast<T const*>(rows);
  T acc = first[idx];
  for (int row = 1; row < nRows; ++row) {
    auto const* shard =
        reinterpret_cast<T const*>(rows + static_cast<size_t>(row) * rowBytes);
    T value = shard[idx];
    if (op == ncclSum) {
      acc = acc + value;
    } else if (op == ncclProd) {
      acc = acc * value;
    } else if (op == ncclMax) {
      acc = value > acc ? value : acc;
    } else if (op == ncclMin) {
      acc = value < acc ? value : acc;
    }
  }
  reinterpret_cast<T*>(output)[idx] = acc;
}

__forceinline__ __device__ int4 addFloat4Vec(int4 a, int4 b) {
  int4 out;
  out.x = __float_as_int(__int_as_float(a.x) + __int_as_float(b.x));
  out.y = __float_as_int(__int_as_float(a.y) + __int_as_float(b.y));
  out.z = __float_as_int(__int_as_float(a.z) + __int_as_float(b.z));
  out.w = __float_as_int(__int_as_float(a.w) + __int_as_float(b.w));
  return out;
}

__global__ void packReduceScatterPairFloat4Kernel(
    float const* send, char* selfRows, char* packRows, size_t nVec,
    size_t rowBytes, int localRank, int nRanksPerNode, int localBase,
    int remoteBase) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto* self4 = reinterpret_cast<int4*>(selfRows);
  auto* pack4 = reinterpret_cast<int4*>(packRows);
  for (int targetLocal = 0; targetLocal < nRanksPerNode; ++targetLocal) {
    int4 localValue =
        send4[(static_cast<size_t>(localBase + targetLocal) * nVec) + vecIdx];
    int4 remoteValue =
        send4[(static_cast<size_t>(remoteBase + targetLocal) * nVec) + vecIdx];
    if (targetLocal == localRank) {
      self4[vecIdx] = localValue;
      self4[rowVecs + vecIdx] = remoteValue;
    } else {
      int peerIdx = targetLocal < localRank ? targetLocal : targetLocal - 1;
      auto* peerRows = pack4 + static_cast<size_t>(2 * peerIdx) * rowVecs;
      peerRows[vecIdx] = localValue;
      peerRows[rowVecs + vecIdx] = remoteValue;
    }
  }
}

__global__ void reducePackedReduceScatterPairFloat4Kernel(
    char const* selfRows, char const* packedRecvRows, float* localOutput,
    float* remoteOutput, size_t nVec, size_t rowBytes, int localRank,
    int nRanksPerNode) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* self4 = reinterpret_cast<int4 const*>(selfRows);
  auto const* packed4 = reinterpret_cast<int4 const*>(packedRecvRows);
  int4 localAcc = self4[vecIdx];
  int4 remoteAcc = self4[rowVecs + vecIdx];
  for (int targetLocal = 0; targetLocal < nRanksPerNode; ++targetLocal) {
    if (targetLocal == localRank) continue;
    int peerIdx = targetLocal < localRank ? targetLocal : targetLocal - 1;
    auto const* peerRows =
        packed4 + static_cast<size_t>(2 * peerIdx) * rowVecs;
    localAcc = addFloat4Vec(peerRows[vecIdx], localAcc);
    remoteAcc = addFloat4Vec(peerRows[rowVecs + vecIdx], remoteAcc);
  }
  reinterpret_cast<int4*>(localOutput)[vecIdx] = localAcc;
  reinterpret_cast<int4*>(remoteOutput)[vecIdx] = remoteAcc;
}

__global__ void packReduceScatterPairFloatKernel(
    float const* send, char* selfRows, char* packRows, size_t recvcount,
    size_t rowBytes, int localRank, int nRanksPerNode, int localBase,
    int remoteBase) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= recvcount) return;

  size_t rowStride = rowBytes / sizeof(float);
  auto* self = reinterpret_cast<float*>(selfRows);
  auto* pack = reinterpret_cast<float*>(packRows);
  for (int targetLocal = 0; targetLocal < nRanksPerNode; ++targetLocal) {
    float localValue =
        send[(static_cast<size_t>(localBase + targetLocal) * recvcount) + idx];
    float remoteValue =
        send[(static_cast<size_t>(remoteBase + targetLocal) * recvcount) + idx];
    if (targetLocal == localRank) {
      self[idx] = localValue;
      self[rowStride + idx] = remoteValue;
    } else {
      int peerIdx = targetLocal < localRank ? targetLocal : targetLocal - 1;
      auto* peerRows = pack + static_cast<size_t>(2 * peerIdx) * rowStride;
      peerRows[idx] = localValue;
      peerRows[rowStride + idx] = remoteValue;
    }
  }
}

__global__ void reducePackedReduceScatterPairFloatKernel(
    char const* selfRows, char const* packedRecvRows, float* localOutput,
    float* remoteOutput, size_t recvcount, size_t rowBytes, int localRank,
    int nRanksPerNode) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= recvcount) return;

  size_t rowStride = rowBytes / sizeof(float);
  auto const* self = reinterpret_cast<float const*>(selfRows);
  auto const* packed = reinterpret_cast<float const*>(packedRecvRows);
  float localAcc = self[idx];
  float remoteAcc = self[rowStride + idx];
  for (int targetLocal = 0; targetLocal < nRanksPerNode; ++targetLocal) {
    if (targetLocal == localRank) continue;
    int peerIdx = targetLocal < localRank ? targetLocal : targetLocal - 1;
    auto const* peerRows =
        packed + static_cast<size_t>(2 * peerIdx) * rowStride;
    localAcc += peerRows[idx];
    remoteAcc += peerRows[rowStride + idx];
  }
  localOutput[idx] = localAcc;
  remoteOutput[idx] = remoteAcc;
}

__global__ void packReduceScatterNumaPairFloat4Kernel(
    float const* send, char* selfRows, char* partnerSendRows, size_t nVec,
    size_t rowBytes, int localRank, int localBase, int remoteBase) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  int partnerLocal = localRank ^ 1;
  int crossPairBase = localRank < 2 ? 2 : 0;
  int crossAssigned = crossPairBase + (localRank & 1);
  int partnerCrossAssigned = crossPairBase + (partnerLocal & 1);
  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* send4 = reinterpret_cast<int4 const*>(send);
  auto* self4 = reinterpret_cast<int4*>(selfRows);
  auto* partner4 = reinterpret_cast<int4*>(partnerSendRows);

  int4 ownLocal =
      send4[(static_cast<size_t>(localBase + localRank) * nVec) + vecIdx];
  int4 ownRemote =
      send4[(static_cast<size_t>(remoteBase + localRank) * nVec) + vecIdx];
  int4 crossLocal =
      send4[(static_cast<size_t>(localBase + crossAssigned) * nVec) + vecIdx];
  int4 crossRemote =
      send4[(static_cast<size_t>(remoteBase + crossAssigned) * nVec) + vecIdx];
  self4[vecIdx] = ownLocal;
  self4[rowVecs + vecIdx] = ownRemote;
  self4[2 * rowVecs + vecIdx] = crossLocal;
  self4[3 * rowVecs + vecIdx] = crossRemote;

  int4 partnerLocalValue =
      send4[(static_cast<size_t>(localBase + partnerLocal) * nVec) + vecIdx];
  int4 partnerRemoteValue =
      send4[(static_cast<size_t>(remoteBase + partnerLocal) * nVec) + vecIdx];
  int4 partnerCrossLocal = send4[
      (static_cast<size_t>(localBase + partnerCrossAssigned) * nVec) + vecIdx];
  int4 partnerCrossRemote = send4[
      (static_cast<size_t>(remoteBase + partnerCrossAssigned) * nVec) + vecIdx];
  partner4[vecIdx] = partnerLocalValue;
  partner4[rowVecs + vecIdx] = partnerRemoteValue;
  partner4[2 * rowVecs + vecIdx] = partnerCrossLocal;
  partner4[3 * rowVecs + vecIdx] = partnerCrossRemote;
}

__global__ void reduceNumaPairPartialsFloat4Kernel(
    char const* selfRows, char const* partnerRecvRows, char* ownPartialRows,
    char* crossSendRows, size_t nVec, size_t rowBytes) {
  size_t vecIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= nVec) return;

  size_t rowVecs = rowBytes / sizeof(int4);
  auto const* self4 = reinterpret_cast<int4 const*>(selfRows);
  auto const* partner4 = reinterpret_cast<int4 const*>(partnerRecvRows);
  auto* own4 = reinterpret_cast<int4*>(ownPartialRows);
  auto* cross4 = reinterpret_cast<int4*>(crossSendRows);
  own4[vecIdx] = addFloat4Vec(partner4[vecIdx], self4[vecIdx]);
  own4[rowVecs + vecIdx] =
      addFloat4Vec(partner4[rowVecs + vecIdx], self4[rowVecs + vecIdx]);
  cross4[vecIdx] =
      addFloat4Vec(partner4[2 * rowVecs + vecIdx], self4[2 * rowVecs + vecIdx]);
  cross4[rowVecs + vecIdx] =
      addFloat4Vec(partner4[3 * rowVecs + vecIdx], self4[3 * rowVecs + vecIdx]);
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

__global__ void packReduceScatterNumaPairFloatKernel(
    float const* send, char* selfRows, char* partnerSendRows, size_t recvcount,
    size_t rowBytes, int localRank, int localBase, int remoteBase) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= recvcount) return;

  int partnerLocal = localRank ^ 1;
  int crossPairBase = localRank < 2 ? 2 : 0;
  int crossAssigned = crossPairBase + (localRank & 1);
  int partnerCrossAssigned = crossPairBase + (partnerLocal & 1);
  size_t rowStride = rowBytes / sizeof(float);
  auto* self = reinterpret_cast<float*>(selfRows);
  auto* partner = reinterpret_cast<float*>(partnerSendRows);

  self[idx] =
      send[(static_cast<size_t>(localBase + localRank) * recvcount) + idx];
  self[rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + localRank) * recvcount) + idx];
  self[2 * rowStride + idx] =
      send[(static_cast<size_t>(localBase + crossAssigned) * recvcount) + idx];
  self[3 * rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + crossAssigned) * recvcount) + idx];

  partner[idx] =
      send[(static_cast<size_t>(localBase + partnerLocal) * recvcount) + idx];
  partner[rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + partnerLocal) * recvcount) + idx];
  partner[2 * rowStride + idx] =
      send[(static_cast<size_t>(localBase + partnerCrossAssigned) * recvcount) +
           idx];
  partner[3 * rowStride + idx] =
      send[(static_cast<size_t>(remoteBase + partnerCrossAssigned) * recvcount) +
           idx];
}

__global__ void reduceNumaPairPartialsFloatKernel(
    char const* selfRows, char const* partnerRecvRows, char* ownPartialRows,
    char* crossSendRows, size_t recvcount, size_t rowBytes) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= recvcount) return;

  size_t rowStride = rowBytes / sizeof(float);
  auto const* self = reinterpret_cast<float const*>(selfRows);
  auto const* partner = reinterpret_cast<float const*>(partnerRecvRows);
  auto* own = reinterpret_cast<float*>(ownPartialRows);
  auto* cross = reinterpret_cast<float*>(crossSendRows);
  own[idx] = self[idx] + partner[idx];
  own[rowStride + idx] = self[rowStride + idx] + partner[rowStride + idx];
  cross[idx] = self[2 * rowStride + idx] + partner[2 * rowStride + idx];
  cross[rowStride + idx] =
      self[3 * rowStride + idx] + partner[3 * rowStride + idx];
}

__global__ void finalNumaPairReduceFloatKernel(
    char const* ownPartialRows, char const* crossRecvRows, float* localOutput,
    float* remoteOutput, size_t recvcount, size_t rowBytes) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= recvcount) return;

  size_t rowStride = rowBytes / sizeof(float);
  auto const* own = reinterpret_cast<float const*>(ownPartialRows);
  auto const* cross = reinterpret_cast<float const*>(crossRecvRows);
  localOutput[idx] = own[idx] + cross[idx];
  remoteOutput[idx] = own[rowStride + idx] + cross[rowStride + idx];
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

ncclResult_t launchReduceRows(void* rows, void* output, size_t elemOffset,
                              size_t count, size_t rowBytes,
                              ncclDataType_t datatype, ncclRedOp_t op,
                              int nRows, cudaStream_t stream) {
  if (op != ncclSum && op != ncclProd && op != ncclMax && op != ncclMin) {
    WARN("unsupported native reduction op %d", op);
    return ncclInvalidArgument;
  }

  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;

  switch (datatype) {
    case ncclFloat32:
      reduceRowsKernel<float><<<blocks, threads, 0, stream>>>(
          static_cast<char const*>(rows),
          static_cast<float*>(output) + elemOffset, count, rowBytes, nRows, op);
      break;
    case ncclFloat64:
      reduceRowsKernel<double><<<blocks, threads, 0, stream>>>(
          static_cast<char const*>(rows),
          static_cast<double*>(output) + elemOffset, count, rowBytes, nRows, op);
      break;
    case ncclInt32:
      reduceRowsKernel<int32_t><<<blocks, threads, 0, stream>>>(
          static_cast<char const*>(rows),
          static_cast<int32_t*>(output) + elemOffset, count, rowBytes, nRows,
          op);
      break;
    case ncclUint32:
      reduceRowsKernel<uint32_t><<<blocks, threads, 0, stream>>>(
          static_cast<char const*>(rows),
          static_cast<uint32_t*>(output) + elemOffset, count, rowBytes, nRows,
          op);
      break;
    default:
      WARN("unsupported native reduction datatype %d", datatype);
      return ncclInvalidArgument;
  }

  return cudaResult(cudaGetLastError(), "native reduction kernel launch");
}

ncclResult_t launchPackReduceScatterPairFloat(
    void const* sendbuff, void* selfRows, void* packRows, size_t recvcount,
    size_t rowBytes, int localRank, int nRanksPerNode, int localBase,
    int remoteBase, cudaStream_t stream) {
  constexpr int threads = 256;
  if (recvcount % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = recvcount / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    packReduceScatterPairFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff), static_cast<char*>(selfRows),
        static_cast<char*>(packRows), nVec, rowBytes, localRank, nRanksPerNode,
        localBase, remoteBase);
    return cudaResult(cudaGetLastError(),
                      "packed reduce-scatter local vector pack kernel launch");
  }
  int blocks = static_cast<int>((recvcount + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  packReduceScatterPairFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff), static_cast<char*>(selfRows),
      static_cast<char*>(packRows), recvcount, rowBytes, localRank,
      nRanksPerNode, localBase, remoteBase);
  return cudaResult(cudaGetLastError(),
                    "packed reduce-scatter local pack kernel launch");
}

ncclResult_t launchPackReduceScatterNumaPairFloat(
    void const* sendbuff, void* selfRows, void* partnerSendRows,
    size_t recvcount, size_t rowBytes, int localRank, int localBase,
    int remoteBase, cudaStream_t stream) {
  constexpr int threads = 256;
  if (recvcount % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = recvcount / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    packReduceScatterNumaPairFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(sendbuff), static_cast<char*>(selfRows),
        static_cast<char*>(partnerSendRows), nVec, rowBytes, localRank,
        localBase, remoteBase);
    return cudaResult(cudaGetLastError(),
                      "numa-pair reduce-scatter pack kernel launch");
  }
  int blocks = static_cast<int>((recvcount + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  packReduceScatterNumaPairFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(sendbuff), static_cast<char*>(selfRows),
      static_cast<char*>(partnerSendRows), recvcount, rowBytes, localRank,
      localBase, remoteBase);
  return cudaResult(cudaGetLastError(),
                    "numa-pair reduce-scatter scalar pack kernel launch");
}

ncclResult_t launchReduceNumaPairPartialsFloat(
    void* selfRows, void* partnerRecvRows, void* ownPartialRows,
    void* crossSendRows, size_t recvcount, size_t rowBytes,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (recvcount % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = recvcount / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    reduceNumaPairPartialsFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(selfRows),
        static_cast<char const*>(partnerRecvRows),
        static_cast<char*>(ownPartialRows), static_cast<char*>(crossSendRows),
        nVec, rowBytes);
    return cudaResult(cudaGetLastError(),
                      "numa-pair partial reduce kernel launch");
  }
  int blocks = static_cast<int>((recvcount + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  reduceNumaPairPartialsFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(selfRows),
      static_cast<char const*>(partnerRecvRows),
      static_cast<char*>(ownPartialRows), static_cast<char*>(crossSendRows),
      recvcount, rowBytes);
  return cudaResult(cudaGetLastError(),
                    "numa-pair scalar partial reduce kernel launch");
}

ncclResult_t launchFinalNumaPairReduceFloat(
    void* ownPartialRows, void* crossRecvRows, void* localOutput,
    void* remoteOutput, size_t recvcount, size_t rowBytes,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (recvcount % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = recvcount / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    finalNumaPairReduceFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(ownPartialRows),
        static_cast<char const*>(crossRecvRows), static_cast<float*>(localOutput),
        static_cast<float*>(remoteOutput), nVec, rowBytes);
    return cudaResult(cudaGetLastError(),
                      "numa-pair final reduce kernel launch");
  }
  int blocks = static_cast<int>((recvcount + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  finalNumaPairReduceFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(ownPartialRows),
      static_cast<char const*>(crossRecvRows), static_cast<float*>(localOutput),
      static_cast<float*>(remoteOutput), recvcount, rowBytes);
  return cudaResult(cudaGetLastError(),
                    "numa-pair scalar final reduce kernel launch");
}

ncclResult_t launchAddFloat(void const* lhs, void const* rhs, void* output,
                            size_t recvcount, cudaStream_t stream) {
  constexpr int threads = 256;
  if (recvcount % 4 == 0) {
    size_t nVec = recvcount / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    addFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(lhs), static_cast<float const*>(rhs),
        static_cast<float*>(output), nVec);
    return cudaResult(cudaGetLastError(), "float vector add kernel launch");
  }
  int blocks = static_cast<int>((recvcount + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  addFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<float const*>(lhs), static_cast<float const*>(rhs),
      static_cast<float*>(output), recvcount);
  return cudaResult(cudaGetLastError(), "float scalar add kernel launch");
}

ncclResult_t launchReducePackedReduceScatterPairFloat(
    void* selfRows, void* packedRecvRows, void* localOutput, void* remoteOutput,
    size_t recvcount, size_t rowBytes, int localRank, int nRanksPerNode,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (recvcount % 4 == 0 && rowBytes % sizeof(int4) == 0) {
    size_t nVec = recvcount / 4;
    int blocks = static_cast<int>((nVec + threads - 1) / threads);
    if (blocks == 0) return ncclSuccess;
    reducePackedReduceScatterPairFloat4Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<char const*>(selfRows),
        static_cast<char const*>(packedRecvRows),
        static_cast<float*>(localOutput), static_cast<float*>(remoteOutput),
        nVec, rowBytes, localRank, nRanksPerNode);
    return cudaResult(
        cudaGetLastError(),
        "packed reduce-scatter local vector reduce kernel launch");
  }
  int blocks = static_cast<int>((recvcount + threads - 1) / threads);
  if (blocks == 0) return ncclSuccess;
  reducePackedReduceScatterPairFloatKernel<<<blocks, threads, 0, stream>>>(
      static_cast<char const*>(selfRows),
      static_cast<char const*>(packedRecvRows), static_cast<float*>(localOutput),
      static_cast<float*>(remoteOutput), recvcount, rowBytes, localRank,
      nRanksPerNode);
  return cudaResult(cudaGetLastError(),
                    "packed reduce-scatter local reduce kernel launch");
}

size_t maxChunkBytes(size_t scratchBufferSize, int nRows, size_t typeSize) {
  if (nRows <= 0 || typeSize == 0) return 0;
  size_t rowBytes = scratchBufferSize / static_cast<size_t>(nRows);
  rowBytes = std::min(rowBytes, static_cast<size_t>(2 * 1024 * 1024));
  return rowBytes / typeSize * typeSize;
}

ncclResult_t finishGroup(ncclResult_t enqueueResult) {
  ncclResult_t groupResult = ncclGroupEnd();
  if (enqueueResult != ncclSuccess) return enqueueResult;
  return groupResult;
}

bool isTwoNodeLayout(int nRanks, int nRanksPerNode) {
  return nRanksPerNode > 1 && nRanks == 2 * nRanksPerNode;
}

ncclResult_t syncAndBarrier(cudaStream_t stream,
                            std::shared_ptr<Communicator> bootstrapComm,
                            char const* operation) {
  ncclResult_t result = cudaResult(cudaStreamSynchronize(stream), operation);
  if (result != ncclSuccess) return result;
  bootstrapComm->bootstrap()->barrier();
  return ncclSuccess;
}

ncclResult_t runHierarchicalAllReduce2Node(
    void const* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream, int rank, int nRanks, void* scratchBuffer,
    size_t scratchBufferSize, int nRanksPerNode,
    std::shared_ptr<Communicator> bootstrapComm) {
  if (!isTwoNodeLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;

  int localRank = rank % nRanksPerNode;
  int nodeId = rank / nRanksPerNode;
  int localBase = nodeId * nRanksPerNode;
  int remoteBase = (1 - nodeId) * nRanksPerNode;
  int remotePeer = remoteBase + localRank;
  int localPartialRow = nRanksPerNode;
  int remoteIncomingRow = localPartialRow + 1;
  int nScratchRows = remoteIncomingRow + 1;
  size_t rowBytes = maxChunkBytes(scratchBufferSize, nScratchRows, typeSize);
  if (rowBytes == 0) return ncclInvalidUsage;

  size_t maxChunkCount = rowBytes / typeSize;
  auto const* send = static_cast<char const*>(sendbuff);
  auto* scratch = static_cast<char*>(scratchBuffer);
  for (size_t elemOffset = 0; elemOffset < count; elemOffset += maxChunkCount) {
    size_t chunkCount = std::min(maxChunkCount, count - elemOffset);
    size_t chunkBytes = chunkCount * typeSize;
    size_t offsetBytes = elemOffset * typeSize;
    size_t scratchRowBytes = chunkBytes;

    ncclResult_t result = cudaResult(
        cudaMemcpyAsync(scratch + static_cast<size_t>(localRank) *
                                      scratchRowBytes,
                        send + offsetBytes, chunkBytes,
                        cudaMemcpyDeviceToDevice, stream),
        "hierarchical allreduce self chunk copy");
    if (result != ncclSuccess) return result;

    result = ncclGroupStart();
    if (result != ncclSuccess) return result;
    ncclResult_t enqueueResult = ncclSuccess;
    for (int i = 0; i < nRanksPerNode; ++i) {
      int peer = localBase + i;
      if (peer == rank) continue;
      enqueueResult =
          ncclSend(send + offsetBytes, chunkCount, datatype, peer, comm, stream);
      if (enqueueResult != ncclSuccess) break;
    }
    if (enqueueResult == ncclSuccess) {
      for (int i = 0; i < nRanksPerNode; ++i) {
        int peer = localBase + i;
        if (peer == rank) continue;
        enqueueResult =
            ncclRecv(scratch + static_cast<size_t>(i) * scratchRowBytes,
                     chunkCount, datatype, peer, comm, stream);
        if (enqueueResult != ncclSuccess) break;
      }
    }
    result = finishGroup(enqueueResult);
    if (result != ncclSuccess) return result;

    result = launchReduceRows(
        scratch, scratch + static_cast<size_t>(localPartialRow) *
                             scratchRowBytes,
        0, chunkCount, scratchRowBytes, datatype, op, nRanksPerNode, stream);
    if (result != ncclSuccess) return result;

    result = ncclGroupStart();
    if (result != ncclSuccess) return result;
    enqueueResult =
        ncclSend(scratch + static_cast<size_t>(localPartialRow) *
                             scratchRowBytes,
                 chunkCount, datatype, remotePeer, comm, stream);
    if (enqueueResult == ncclSuccess) {
      enqueueResult =
          ncclRecv(scratch + static_cast<size_t>(remoteIncomingRow) *
                               scratchRowBytes,
                   chunkCount, datatype, remotePeer, comm, stream);
    }
    result = finishGroup(enqueueResult);
    if (result != ncclSuccess) return result;

    result = launchReduceRows(
        scratch + static_cast<size_t>(localPartialRow) * scratchRowBytes,
        recvbuff, elemOffset, chunkCount, scratchRowBytes, datatype, op, 2,
        stream);
    if (result != ncclSuccess) return result;

    if (elemOffset + chunkCount < count) {
      result = syncAndBarrier(stream, bootstrapComm,
                              "hierarchical allreduce chunk sync");
      if (result != ncclSuccess) return result;
    }
  }

  return syncAndBarrier(stream, bootstrapComm,
                        "hierarchical allreduce final synchronization");
}


ncclResult_t runHostStagedReduceScatter2Node(
    void const* sendbuff, void* recvbuff, size_t recvcount,
    size_t bytesPerRank, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream, int rank, int nRanks,
    int nRanksPerNode, std::shared_ptr<Communicator> bootstrapComm,
    int cudaDevice) {
  if (!isTwoNodeLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  if (datatype != ncclFloat32 || op != ncclSum) return ncclInvalidUsage;
  if (nRanksPerNode <= 0 || nRanksPerNode > kNativeHostMaxRanksPerNode) {
    return ncclInvalidUsage;
  }
  if (recvcount > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return ncclInvalidArgument;
  }
  if (recvcount * sizeof(float) != bytesPerRank ||
      bytesPerRank > kNativeReduceScatterHostMaxShardBytes) {
    return ncclInvalidUsage;
  }

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    auto& ctx = getReduceScatterHostContext(
        comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice);
    uint64_t epoch = ++ctx.blockEpoch;
    int localBase = ctx.nodeId * nRanksPerNode;
    int remoteBase = (1 - ctx.nodeId) * nRanksPerNode;
    size_t inputBytes = static_cast<size_t>(nRanks) * bytesPerRank;
    size_t partialBlockBytes = static_cast<size_t>(nRanksPerNode) * bytesPerRank;

    char* localInput =
        ctx.sendSlab() + static_cast<size_t>(ctx.localRank) * ctx.inputCapacity;
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(localInput, sendbuff, inputBytes,
                                     cudaMemcpyDeviceToHost, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    ctx.ctrl->d2hReady[ctx.localRank].store(epoch,
                                            std::memory_order_release);

    for (int i = 0; i < nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->d2hReady[i], epoch);
    }

    auto* localPartial = reinterpret_cast<float*>(
        ctx.localPartialSlab() +
        static_cast<size_t>(ctx.localRank) * bytesPerRank);
    auto* remotePartial = reinterpret_cast<float*>(
        ctx.sendPartialSlab() +
        static_cast<size_t>(ctx.localRank) * bytesPerRank);
    reduceFloatShardFromHost(ctx.sendSlab(), ctx.inputCapacity, bytesPerRank,
                             nRanksPerNode, localBase + ctx.localRank,
                             recvcount, localPartial);
    reduceFloatShardFromHost(ctx.sendSlab(), ctx.inputCapacity, bytesPerRank,
                             nRanksPerNode, remoteBase + ctx.localRank,
                             recvcount, remotePartial);
    ctx.ctrl->partialReady[ctx.localRank].store(epoch,
                                                std::memory_order_release);

    if (ctx.isLeader) {
      for (int i = 0; i < nRanksPerNode; ++i) {
        waitForEpoch(ctx.ctrl->partialReady[i], epoch);
      }
      if (ctx.pairEpoch > 0) {
        for (int i = 0; i < nRanksPerNode; ++i) {
          waitForEpoch(ctx.ctrl->pairAckReady[i], ctx.pairEpoch);
        }
      }
      if (epoch > 1) {
        waitForEpoch(ctx.ctrl->ackReady, epoch - 1);
      }

      size_t off = 0;
      int writesSinceFlush = 0;
      while (off < partialBlockBytes) {
        size_t chunk =
            std::min(kNativeReduceScatterRdmaChunkBytes,
                     partialBlockBytes - off);
        ctx.connection.write(ctx.remoteRecvPartialMemory, off,
                             ctx.sendPartialMemory, off, chunk);
        if (++writesSinceFlush == kNativeReduceScatterSignalEveryN) {
          ctx.connection.flush();
          writesSinceFlush = 0;
        }
        off += chunk;
      }
      ctx.ctrl->rdmaSignal.store(epoch, std::memory_order_release);
      ctx.connection.write(
          ctx.remoteCtrlMemory,
          offsetof(NativeReduceScatterHostControl, rdmaReady), ctx.ctrlMemory,
          offsetof(NativeReduceScatterHostControl, rdmaSignal),
          sizeof(uint64_t));
      ctx.connection.flush();
    }
    waitForEpoch(ctx.ctrl->rdmaReady, epoch);

    auto* remoteIncoming = reinterpret_cast<float*>(
        ctx.recvPartialSlab() +
        static_cast<size_t>(ctx.localRank) * bytesPerRank);
    for (size_t i = 0; i < recvcount; ++i) {
      localPartial[i] += remoteIncoming[i];
    }

    MSCCLPP_CUDATHROW(cudaMemcpyAsync(recvbuff, localPartial, bytesPerRank,
                                     cudaMemcpyHostToDevice, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    ctx.ctrl->h2dDone[ctx.localRank].store(epoch, std::memory_order_release);

    for (int i = 0; i < nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->h2dDone[i], epoch);
    }
    if (ctx.isLeader) {
      ctx.ctrl->ackSignal.store(epoch, std::memory_order_release);
      ctx.connection.write(
          ctx.remoteCtrlMemory,
          offsetof(NativeReduceScatterHostControl, ackReady), ctx.ctrlMemory,
          offsetof(NativeReduceScatterHostControl, ackSignal),
          sizeof(uint64_t));
      ctx.connection.flush();
    }
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN("host-staged reducescatter failed: %s", ex.what());
    return mapNativeException(ex);
  } catch (...) {
    WARN("host-staged reducescatter failed with an unknown exception");
    return ncclInternalError;
  }
}

ncclResult_t runGpuLocalReduceScatter2Node(
    void const* sendbuff, void* recvbuff, size_t recvcount,
    size_t bytesPerRank, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream, int rank, int nRanks,
    void* scratchBuffer, size_t scratchBufferSize, int nRanksPerNode,
    std::shared_ptr<Communicator> bootstrapComm, int cudaDevice) {
  if (!isTwoNodeLayout(nRanks, nRanksPerNode)) return ncclInvalidUsage;
  if (datatype != ncclFloat32 || op != ncclSum) return ncclInvalidUsage;
  if (nRanksPerNode <= 0 || nRanksPerNode > kNativeHostMaxRanksPerNode) {
    return ncclInvalidUsage;
  }
  if (recvcount > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return ncclInvalidArgument;
  }
  if (recvcount * sizeof(float) != bytesPerRank ||
      bytesPerRank > kNativeReduceScatterHostMaxShardBytes) {
    return ncclInvalidUsage;
  }

  int localRank = rank % nRanksPerNode;
  int nodeId = rank / nRanksPerNode;
  int localBase = nodeId * nRanksPerNode;
  int remoteBase = (1 - nodeId) * nRanksPerNode;
  size_t typeSize = ncclTypeSize(datatype);
  int defaultPackedScratchRows = 4 * nRanksPerNode + 1;
  size_t numaPairRowBytes = maxChunkBytes(
      scratchBufferSize, kNativeReduceScatterNumaPairScratchRows, typeSize);
  bool useNumaPairLocal = nRanksPerNode == 4 &&
                          numaPairRowBytes >= bytesPerRank;
  int packedScratchRows = useNumaPairLocal
                              ? kNativeReduceScatterNumaPairScratchRows
                              : defaultPackedScratchRows;
  size_t packedRowBytes =
      maxChunkBytes(scratchBufferSize, packedScratchRows, typeSize);
  bool usePackedLocal = packedRowBytes >= bytesPerRank;
  int nScratchRows =
      usePackedLocal ? packedScratchRows : (2 * nRanksPerNode + 3);
  size_t rowBytes = usePackedLocal
                        ? bytesPerRank
                        : maxChunkBytes(scratchBufferSize, nScratchRows,
                                        typeSize);
  if (rowBytes < bytesPerRank) return ncclInvalidUsage;

  try {
    mscclpp::CudaDeviceGuard deviceGuard(cudaDevice);
    auto& ctx = getReduceScatterHostContext(
        comm, bootstrapComm, rank, nRanks, nRanksPerNode, cudaDevice);
    uint64_t epoch = ++ctx.pairEpoch;
    auto const* send = static_cast<char const*>(sendbuff);
    auto* scratch = static_cast<char*>(scratchBuffer);
    char* localPartialGpu = nullptr;
    char* remoteIncomingGpu = nullptr;
    char* remotePartialGpu = nullptr;
    ncclResult_t result = ncclSuccess;

    if (usePackedLocal) {
      char* selfRows = scratch;
      char* packSendRows = selfRows + 2 * rowBytes;
      char* packRecvRows = packSendRows +
                           static_cast<size_t>(2 * (nRanksPerNode - 1)) *
                               rowBytes;
      localPartialGpu = packRecvRows +
                        static_cast<size_t>(2 * (nRanksPerNode - 1)) *
                            rowBytes;
      remoteIncomingGpu = localPartialGpu + rowBytes;
      remotePartialGpu = remoteIncomingGpu + rowBytes;

      if (useNumaPairLocal) {
        char* numaSelfRows = scratch;
        char* partnerSendRows = numaSelfRows + 4 * rowBytes;
        char* partnerRecvRows = partnerSendRows + 4 * rowBytes;
        char* ownPartialRows = partnerRecvRows + 4 * rowBytes;
        char* crossSendRows = ownPartialRows + 2 * rowBytes;
        char* crossRecvRows = crossSendRows + 2 * rowBytes;
        localPartialGpu = crossRecvRows + 2 * rowBytes;
        remoteIncomingGpu = localPartialGpu + rowBytes;
        remotePartialGpu = remoteIncomingGpu + rowBytes;

        result = launchPackReduceScatterNumaPairFloat(
            sendbuff, numaSelfRows, partnerSendRows, recvcount, rowBytes,
            localRank, localBase, remoteBase, stream);
        if (result != ncclSuccess) return result;

        ensureReduceScatterLocalScratchIpc(ctx, bootstrapComm, rank, nRanks,
                                           nRanksPerNode, scratchBuffer,
                                           scratchBufferSize);
        int partnerLocal = localRank ^ 1;
        int partnerPeerIdx =
            partnerLocal < localRank ? partnerLocal : partnerLocal - 1;
        size_t partnerRecvOffset = static_cast<size_t>(8) * rowBytes;
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(
            ctx.remoteScratchPtrs[partnerPeerIdx] + partnerRecvOffset,
            partnerSendRows, 4 * bytesPerRank, cudaMemcpyDeviceToDevice,
            stream));
        MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
        ctx.ctrl->localCopyReady[localRank].store(epoch,
                                                  std::memory_order_release);
        waitForEpoch(ctx.ctrl->localCopyReady[partnerLocal], epoch);

        result = launchReduceNumaPairPartialsFloat(
            numaSelfRows, partnerRecvRows, ownPartialRows, crossSendRows,
            recvcount, rowBytes, stream);
        if (result != ncclSuccess) return result;

        int crossTargetLocal = localRank < 2 ? localRank + 2 : localRank - 2;
        int crossPeerIdx =
            crossTargetLocal < localRank ? crossTargetLocal : crossTargetLocal - 1;
        size_t crossRecvOffset = static_cast<size_t>(16) * rowBytes;
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(
            ctx.remoteScratchPtrs[crossPeerIdx] + crossRecvOffset,
            crossSendRows, 2 * bytesPerRank, cudaMemcpyDeviceToDevice,
            stream));
        MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
        ctx.ctrl->localCrossReady[localRank].store(epoch,
                                                   std::memory_order_release);
        waitForEpoch(ctx.ctrl->localCrossReady[crossTargetLocal], epoch);

        result = launchFinalNumaPairReduceFloat(
            ownPartialRows, crossRecvRows, localPartialGpu, remotePartialGpu,
            recvcount, rowBytes, stream);
        if (result != ncclSuccess) return result;
      } else {
        result = launchPackReduceScatterPairFloat(
            sendbuff, selfRows, packSendRows, recvcount, rowBytes, localRank,
            nRanksPerNode, localBase, remoteBase, stream);
        if (result != ncclSuccess) return result;

        ensureReduceScatterLocalScratchIpc(ctx, bootstrapComm, rank, nRanks,
                                           nRanksPerNode, scratchBuffer,
                                           scratchBufferSize);
        for (int i = 0; i < nRanksPerNode; ++i) {
          int peer = localBase + i;
          if (peer == rank) continue;
          int peerIdx = i < localRank ? i : i - 1;
          int remotePeerIdx = localRank < i ? localRank : localRank - 1;
          size_t localOffset = static_cast<size_t>(2 * peerIdx) * rowBytes;
          size_t remoteOffset =
              (2 + static_cast<size_t>(2 * (nRanksPerNode - 1)) +
               static_cast<size_t>(2 * remotePeerIdx)) *
              rowBytes;
          MSCCLPP_CUDATHROW(cudaMemcpyAsync(
              ctx.remoteScratchPtrs[peerIdx] + remoteOffset,
              packSendRows + localOffset, 2 * bytesPerRank,
              cudaMemcpyDeviceToDevice, stream));
        }
      }
      if (!useNumaPairLocal) {
        MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
        ctx.ctrl->localCopyReady[localRank].store(epoch,
                                                  std::memory_order_release);
        for (int i = 0; i < nRanksPerNode; ++i) {
          waitForEpoch(ctx.ctrl->localCopyReady[i], epoch);
        }

        result = launchReducePackedReduceScatterPairFloat(
            selfRows, packRecvRows, localPartialGpu, remotePartialGpu,
            recvcount, rowBytes, localRank, nRanksPerNode, stream);
        if (result != ncclSuccess) return result;
      }
    } else {
      char* localRows = scratch;
      char* remoteRows =
          localRows + static_cast<size_t>(nRanksPerNode) * rowBytes;
      localPartialGpu =
          remoteRows + static_cast<size_t>(nRanksPerNode) * rowBytes;
      remoteIncomingGpu = localPartialGpu + rowBytes;
      remotePartialGpu = remoteIncomingGpu + rowBytes;

      size_t localTargetOffset = static_cast<size_t>(rank) * bytesPerRank;
      size_t remoteTargetOffset =
          static_cast<size_t>(remoteBase + localRank) * bytesPerRank;
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          localRows + static_cast<size_t>(localRank) * rowBytes,
          send + localTargetOffset, bytesPerRank, cudaMemcpyDeviceToDevice,
          stream));
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          remoteRows + static_cast<size_t>(localRank) * rowBytes,
          send + remoteTargetOffset, bytesPerRank, cudaMemcpyDeviceToDevice,
          stream));

      result = ncclGroupStart();
      if (result != ncclSuccess) return result;
      ncclResult_t enqueueResult = ncclSuccess;
      for (int i = 0; i < nRanksPerNode; ++i) {
        int peer = localBase + i;
        if (peer == rank) continue;
        size_t peerLocalOffset = static_cast<size_t>(peer) * bytesPerRank;
        enqueueResult =
            ncclSend(send + peerLocalOffset, recvcount, datatype, peer, comm,
                     stream);
        if (enqueueResult != ncclSuccess) break;
      }
      if (enqueueResult == ncclSuccess) {
        for (int i = 0; i < nRanksPerNode; ++i) {
          int peer = localBase + i;
          if (peer == rank) continue;
          enqueueResult =
              ncclRecv(localRows + static_cast<size_t>(i) * rowBytes, recvcount,
                       datatype, peer, comm, stream);
          if (enqueueResult != ncclSuccess) break;
        }
      }
      result = finishGroup(enqueueResult);
      if (result != ncclSuccess) return result;

      result = launchReduceRows(localRows, localPartialGpu, 0, recvcount,
                                rowBytes, datatype, op, nRanksPerNode, stream);
      if (result != ncclSuccess) return result;

      result = ncclGroupStart();
      if (result != ncclSuccess) return result;
      enqueueResult = ncclSuccess;
      for (int i = 0; i < nRanksPerNode; ++i) {
        int peer = localBase + i;
        if (peer == rank) continue;
        size_t peerRemoteOffset =
            static_cast<size_t>(remoteBase + i) * bytesPerRank;
        enqueueResult =
            ncclSend(send + peerRemoteOffset, recvcount, datatype, peer, comm,
                     stream);
        if (enqueueResult != ncclSuccess) break;
      }
      if (enqueueResult == ncclSuccess) {
        for (int i = 0; i < nRanksPerNode; ++i) {
          int peer = localBase + i;
          if (peer == rank) continue;
          enqueueResult =
              ncclRecv(remoteRows + static_cast<size_t>(i) * rowBytes,
                       recvcount, datatype, peer, comm, stream);
          if (enqueueResult != ncclSuccess) break;
        }
      }
      result = finishGroup(enqueueResult);
      if (result != ncclSuccess) return result;

      result = launchReduceRows(remoteRows, remotePartialGpu, 0, recvcount,
                                rowBytes, datatype, op, nRanksPerNode, stream);
      if (result != ncclSuccess) return result;
    }

    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        ctx.sendPartialSlab() + static_cast<size_t>(localRank) * bytesPerRank,
        remotePartialGpu, bytesPerRank, cudaMemcpyDeviceToHost, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

    if (epoch > 1) {
      waitForEpoch(ctx.ctrl->pairAckReady[localRank], epoch - 1);
    }
    if (ctx.blockEpoch > 0) {
      waitForEpoch(ctx.ctrl->ackReady, ctx.blockEpoch);
    }

    size_t pairSlotOffset = static_cast<size_t>(localRank) * bytesPerRank;
    size_t pairRdmaReadyOffset =
        offsetof(NativeReduceScatterHostControl, pairRdmaReady) +
        static_cast<size_t>(localRank) * sizeof(ctx.ctrl->pairRdmaReady[0]);
    size_t pairRdmaSignalOffset =
        offsetof(NativeReduceScatterHostControl, pairRdmaSignal) +
        static_cast<size_t>(localRank) * sizeof(ctx.ctrl->pairRdmaSignal[0]);
    size_t pairAckReadyOffset =
        offsetof(NativeReduceScatterHostControl, pairAckReady) +
        static_cast<size_t>(localRank) * sizeof(ctx.ctrl->pairAckReady[0]);
    size_t pairAckSignalOffset =
        offsetof(NativeReduceScatterHostControl, pairAckSignal) +
        static_cast<size_t>(localRank) * sizeof(ctx.ctrl->pairAckSignal[0]);

    ctx.pairConnection.write(ctx.pairRemoteRecvPartialMemory, pairSlotOffset,
                             ctx.sendPartialMemory, pairSlotOffset,
                             bytesPerRank);
    ctx.ctrl->pairRdmaSignal[localRank].store(epoch,
                                              std::memory_order_release);
    ctx.pairConnection.write(ctx.pairRemoteCtrlMemory, pairRdmaReadyOffset,
                             ctx.ctrlMemory, pairRdmaSignalOffset,
                             sizeof(uint64_t));
    ctx.pairConnection.flush();

    waitForEpoch(ctx.ctrl->pairRdmaReady[localRank], epoch);

    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        remoteIncomingGpu,
        ctx.recvPartialSlab() + static_cast<size_t>(localRank) * bytesPerRank,
        bytesPerRank, cudaMemcpyHostToDevice, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    result = launchAddFloat(localPartialGpu, remoteIncomingGpu, recvbuff,
                            recvcount, stream);
    bool localDonePublished = false;
    auto publishLocalDone = [&] {
      if (localDonePublished) return;
      ctx.ctrl->localDoneReady[localRank].store(epoch,
                                                std::memory_order_release);
      localDonePublished = true;
    };
    if (result != ncclSuccess) {
      publishLocalDone();
      return result;
    }
    result = cudaResult(cudaStreamSynchronize(stream),
                        "gpu-local reducescatter final reduction");
    publishLocalDone();
    if (result != ncclSuccess) return result;

    ctx.ctrl->pairAckSignal[localRank].store(epoch, std::memory_order_release);
    ctx.pairConnection.write(ctx.pairRemoteCtrlMemory, pairAckReadyOffset,
                             ctx.ctrlMemory, pairAckSignalOffset,
                             sizeof(uint64_t));
    ctx.pairConnection.flush();
    for (int i = 0; i < nRanksPerNode; ++i) {
      waitForEpoch(ctx.ctrl->localDoneReady[i], epoch);
    }
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN("gpu-local reducescatter failed: %s", ex.what());
    return mapNativeException(ex);
  } catch (...) {
    WARN("gpu-local reducescatter failed with an unknown exception");
    return ncclInternalError;
  }
}

}  // namespace


ncclResult_t runSendRecvReduceScatter(void const* sendbuff, void* recvbuff,
                                      size_t recvcount, size_t bytesPerRank,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream,
                                      int rank, int nRanks, void* scratchBuffer,
                                      size_t scratchBufferSize,
                                      int nRanksPerNode,
                                      std::shared_ptr<Communicator> bootstrapComm,
                                      int cudaDevice) {
                                      size_t typeSize = ncclTypeSize(datatype);
                                      if (typeSize == 0) return ncclInvalidArgument;
                                      if (isTwoNodeLayout(nRanks, nRanksPerNode)) {
    ncclResult_t result = runGpuLocalReduceScatter2Node(
        sendbuff, recvbuff, recvcount, bytesPerRank, datatype, op, comm, stream,
        rank, nRanks, scratchBuffer, scratchBufferSize, nRanksPerNode,
        bootstrapComm, cudaDevice);
    if (result == ncclSuccess) return result;
    if (result != ncclInvalidUsage && result != ncclInvalidArgument) {
      return result;
    }

    result = runHostStagedReduceScatter2Node(
        sendbuff, recvbuff, recvcount, bytesPerRank, datatype, op, comm, stream,
        rank, nRanks, nRanksPerNode, bootstrapComm, cudaDevice);
    if (result == ncclSuccess) return result;
    if (result != ncclInvalidUsage && result != ncclInvalidArgument) {
      return result;
    }
  }

  size_t rowBytes = maxChunkBytes(scratchBufferSize, nRanks, typeSize);
  if (rowBytes == 0) return ncclInvalidUsage;

  size_t maxChunkCount = rowBytes / typeSize;
  auto const* send = static_cast<char const*>(sendbuff);
  auto* scratch = static_cast<char*>(scratchBuffer);
  for (size_t elemOffset = 0; elemOffset < recvcount;
       elemOffset += maxChunkCount) {
    size_t chunkCount = std::min(maxChunkCount, recvcount - elemOffset);
    size_t chunkBytes = chunkCount * typeSize;
    size_t offsetBytes = elemOffset * typeSize;

    ncclResult_t result = cudaResult(
        cudaMemcpyAsync(scratch + static_cast<size_t>(rank) * rowBytes,
                        send + static_cast<size_t>(rank) * bytesPerRank +
                            offsetBytes,
                        chunkBytes, cudaMemcpyDeviceToDevice, stream),
        "reducescatter self shard copy");
    if (result != ncclSuccess) return result;

    result = ncclGroupStart();
    if (result != ncclSuccess) return result;

    ncclResult_t enqueueResult = ncclSuccess;
    for (int peer = 0; peer < nRanks; ++peer) {
      if (peer == rank) continue;
      enqueueResult =
          ncclSend(send + static_cast<size_t>(peer) * bytesPerRank + offsetBytes,
                   chunkCount, datatype, peer, comm, stream);
      if (enqueueResult != ncclSuccess) break;
    }
    if (enqueueResult == ncclSuccess) {
      for (int peer = 0; peer < nRanks; ++peer) {
        if (peer == rank) continue;
        enqueueResult = ncclRecv(scratch + static_cast<size_t>(peer) * rowBytes,
                                 chunkCount, datatype, peer, comm, stream);
        if (enqueueResult != ncclSuccess) break;
      }
    }

    result = finishGroup(enqueueResult);
    if (result != ncclSuccess) return result;
    result = launchReduceRows(scratch, recvbuff, elemOffset, chunkCount,
                              rowBytes, datatype, op, nRanks, stream);
    if (result != ncclSuccess) return result;
    if (elemOffset + chunkCount < recvcount) {
      result = cudaResult(cudaStreamSynchronize(stream),
                          "reducescatter chunk synchronization");
      if (result != ncclSuccess) return result;
      bootstrapComm->bootstrap()->barrier();
    }
  }
  ncclResult_t result = cudaResult(cudaStreamSynchronize(stream),
                                   "reducescatter final synchronization");
  if (result != ncclSuccess) return result;
  bootstrapComm->bootstrap()->barrier();
  return ncclSuccess;
}

ncclResult_t runSendRecvAllReduce(void const* sendbuff, void* recvbuff,
                                  size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm_t comm,
                                  cudaStream_t stream, int rank, int nRanks,
                                  void* scratchBuffer,
                                  size_t scratchBufferSize,
                                  int nRanksPerNode,
                                  std::shared_ptr<Communicator> bootstrapComm,
                                  int cudaDevice) {
  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) return ncclInvalidArgument;

  if (isTwoNodeLayout(nRanks, nRanksPerNode)) {
    if (count % static_cast<size_t>(nRanks) == 0) {
      size_t shardCount = count / static_cast<size_t>(nRanks);
      size_t shardBytes = shardCount * typeSize;
      auto* recv = static_cast<char*>(recvbuff);
      void* localShard = recv + static_cast<size_t>(rank) * shardBytes;

      ncclResult_t result = runSendRecvReduceScatter(
          sendbuff, localShard, shardCount, shardBytes, datatype, op, comm,
          stream, rank, nRanks, scratchBuffer, scratchBufferSize,
          nRanksPerNode, bootstrapComm, cudaDevice);
      if (result == ncclSuccess) {
        return runLiteAllGather(localShard, recvbuff, shardCount, shardBytes,
                                datatype, comm, stream, rank, nRanks,
                                nRanksPerNode, bootstrapComm, cudaDevice);
      }
      if (result != ncclInvalidUsage && result != ncclInvalidArgument) {
        return result;
      }
    }

    ncclResult_t result = runHierarchicalAllReduce2Node(
        sendbuff, recvbuff, count, datatype, op, comm, stream, rank, nRanks,
        scratchBuffer, scratchBufferSize, nRanksPerNode, bootstrapComm);
    if (result != ncclInvalidUsage) return result;
  }

  if (count % static_cast<size_t>(nRanks) != 0) return ncclInvalidArgument;

  if (sendbuff != recvbuff) {
    size_t bytes = count * typeSize;
    ncclResult_t result = cudaResult(cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                                        cudaMemcpyDeviceToDevice, stream),
                                      "allreduce out-of-place input copy");
    if (result != ncclSuccess) return result;
    return runSendRecvAllReduce(recvbuff, recvbuff, count, datatype, op, comm,
                                stream, rank, nRanks, scratchBuffer,
                                scratchBufferSize, nRanksPerNode,
                                bootstrapComm, cudaDevice);
  }

  size_t shardCount = count / static_cast<size_t>(nRanks);
  size_t shardBytes = shardCount * typeSize;
  auto* recv = static_cast<char*>(recvbuff);
  void* localShard = recv + static_cast<size_t>(rank) * shardBytes;

  ncclResult_t result = runSendRecvReduceScatter(
      sendbuff, localShard, shardCount, shardBytes, datatype, op, comm, stream,
      rank, nRanks, scratchBuffer, scratchBufferSize, nRanksPerNode,
      bootstrapComm, cudaDevice);
  if (result != ncclSuccess) return result;

  return runLiteAllGather(localShard, recvbuff, shardCount, shardBytes, datatype,
                          comm, stream, rank, nRanks, nRanksPerNode,
                          bootstrapComm, cudaDevice);
}

void cleanupNativeCollectiveContexts(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(gNativeCollectiveContextMutex);
  gReduceScatterHostContexts.erase(comm);
  cleanupLiteAllGatherContexts(comm);
}

}  // namespace nccl
}  // namespace mscclpp
