// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm_collection_builder.hpp"
#include "core.hpp"
#include "env.hpp"
#include "executor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <filesystem>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>
#if defined(ENABLE_NPKIT)
#include "npkit.hpp"
#endif
#include "algorithm.hpp"
#include "algorithm_selector.hpp"
#include "allgather.hpp"
#include "allreduce.hpp"
#include "broadcast.hpp"
#include "datatype_conversion.hpp"
#include "gpu_utils.hpp"
#include "logger.hpp"
#include "nccl.h"
#include "semaphore.hpp"
#include <atomic>
#include <cuda.h>
#include <dlfcn.h>
#include <mutex>
#include <thread>

static constexpr auto MSCCLPP_NCCL = mscclpp::LogSubsys::NCCL;

namespace {

static constexpr int kNcclSendRecvInitTagBase = 0x530000;
static constexpr int kNcclSendRecvInitTagStride = 8;

static const mscclpp::Transport kIBTransports[] = {
    mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
    mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
    mscclpp::Transport::IB6, mscclpp::Transport::IB7};

// Async worker state for CPU-driven IB send/recv.  Lives on the heap via
// unique_ptr so that NcclSendRecvPeerContext stays movable (std::thread and
// std::atomic are not).
static constexpr size_t kSendChunkBytes = 1024 * 1024;  // 1MB pipeline chunk

struct SendRecvWorkerState {
  struct WorkItem {
    enum Type : uint8_t { SEND = 0, SEND_CHUNK = 1 } type;
    size_t bytes;
    size_t offset;      // SEND_CHUNK: offset into staging buffer
    int eventIndex;     // SEND_CHUNK: index into chunkEvents[]
    bool lastChunk;     // SEND_CHUNK: signal + flush after this chunk
  };

  // Lock-free SPSC ring buffer (one producer = caller, one consumer = worker).
  static constexpr uint32_t kCapacity = 256;
  WorkItem ring[kCapacity];
  alignas(64) std::atomic<uint32_t> head{0};
  alignas(64) std::atomic<uint32_t> tail{0};

  std::atomic<bool> stopFlag{false};
  std::thread thread;

  // Send: CUDA event tracks D2H completion on the user stream.
  cudaEvent_t d2hDoneEvent = nullptr;

  // Chunk events pool for pipelined sends (grown on demand).
  std::vector<cudaEvent_t> chunkEvents;

  void ensureChunkEvents(int count) {
    while (static_cast<int>(chunkEvents.size()) < count) {
      cudaEvent_t e;
      MSCCLPP_CUDATHROW(
          cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      chunkEvents.push_back(e);
    }
  }

  void push(WorkItem const& item) {
    uint32_t h = head.load(std::memory_order_relaxed);
    while (h - tail.load(std::memory_order_acquire) >= kCapacity) {
      // Queue full — spin (should not happen in practice).
    }
    ring[h % kCapacity] = item;
    head.store(h + 1, std::memory_order_release);
  }

  bool pop(WorkItem& item) {
    uint32_t t = tail.load(std::memory_order_relaxed);
    if (t >= head.load(std::memory_order_acquire)) return false;
    item = ring[t % kCapacity];
    tail.store(t + 1, std::memory_order_release);
    return true;
  }

  ~SendRecvWorkerState() {
    stopFlag.store(true, std::memory_order_release);
    if (thread.joinable()) thread.join();
    if (d2hDoneEvent) cudaEventDestroy(d2hDoneEvent);
    for (auto e : chunkEvents) cudaEventDestroy(e);
  }
};

struct NcclSendRecvPeerContext {
  int localDevice = -1;
  size_t stagingBytes = 0;
  mscclpp::Transport transport = mscclpp::Transport::Unknown;
  std::shared_ptr<char> sendStagingBuffer;
  std::shared_ptr<char> recvStagingBuffer;
  mscclpp::RegisteredMemory sendStagingMemory;
  mscclpp::RegisteredMemory recvStagingMemory;
  mscclpp::RegisteredMemory remoteRecvStagingMemory;
  mscclpp::Connection connection;
  std::shared_ptr<mscclpp::Host2HostSemaphore> h2hSemaphore;

  // Async worker — destroyed first (declared last) so the thread is joined
  // before the connection / semaphore members are destroyed.
  std::unique_ptr<SendRecvWorkerState> worker;
};

inline bool hasIBDevices() { return mscclpp::getIBDeviceCount() > 0; }

inline int sendRecvInitTag(int rank, int worldSize, int peer, int slot) {
  int lo = std::min(rank, peer);
  int hi = std::max(rank, peer);
  int pairIndex = lo * worldSize + hi;
  return kNcclSendRecvInitTagBase + pairIndex * kNcclSendRecvInitTagStride +
         slot;
}

inline size_t sendRecvStagingBytes() {
  int bytes = mscclpp::env()->ncclSendRecvStagingBytes;
  if (bytes <= 0) {
    throw mscclpp::Error(
        "MSCCLPP_NCCL_SENDRECV_STAGING_BYTES must be positive",
        mscclpp::ErrorCode::InvalidUsage);
  }
  return static_cast<size_t>(bytes);
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
ncclResult_t runNcclGuarded(char const* opName, Fn&& fn) {
  try {
    fn();
    return ncclSuccess;
  } catch (std::exception const& ex) {
    WARN(MSCCLPP_NCCL, std::string(opName), " failed: ",
         std::string(ex.what()));
    return mapMscclppException(ex);
  } catch (...) {
    WARN(MSCCLPP_NCCL, std::string(opName),
         " failed with an unknown exception");
    return ncclInternalError;
  }
}

// Worker thread: spins on the SPSC queue, issues IB operations on the CPU.
static void sendRecvWorkerLoop(NcclSendRecvPeerContext* ctx) {
  auto* ws = ctx->worker.get();
  if (ctx->localDevice >= 0) {
    cudaSetDevice(ctx->localDevice);
  }

  SendRecvWorkerState::WorkItem item;
  while (!ws->stopFlag.load(std::memory_order_acquire)) {
    if (!ws->pop(item)) continue;  // spin

    try {
      if (item.type == SendRecvWorkerState::WorkItem::SEND) {
        // Small message: single write, same as before.
        while (cudaEventQuery(ws->d2hDoneEvent) == cudaErrorNotReady) {
        }
        ctx->connection.write(ctx->remoteRecvStagingMemory, 0,
                              ctx->sendStagingMemory, 0, item.bytes);
        ctx->h2hSemaphore->signal();
        ctx->connection.flush();
      } else {
        // SEND_CHUNK: pipelined — wait for this chunk's D2H, then post
        // the IB write.  write() posts to the NIC immediately, so the
        // transfer overlaps with subsequent D2H copies on the stream.
        while (cudaEventQuery(ws->chunkEvents[item.eventIndex]) ==
               cudaErrorNotReady) {
        }
        ctx->connection.write(ctx->remoteRecvStagingMemory, item.offset,
                              ctx->sendStagingMemory, item.offset,
                              item.bytes);
        if (item.lastChunk) {
          ctx->h2hSemaphore->signal();
          ctx->connection.flush();
        }
      }
    } catch (std::exception const& ex) {
      WARN(MSCCLPP_NCCL, "sendRecvWorker failed: ", ex.what());
    }
  }
}

void registerMigratedAppNcclAlgorithms(uintptr_t scratchBuffer,
                                       size_t scratchBufferSize) {
  auto b = mscclpp::collective::AlgorithmCollectionBuilder::getInstance();
  b->addAlgorithmBuilder(
      std::make_shared<BroadcastAlgo6>(scratchBuffer, scratchBufferSize));
  b->addAlgorithmBuilder(std::make_shared<AllgatherAlgo6>());
  b->addAlgorithmBuilder(
      std::make_shared<AllgatherAlgo8>(scratchBuffer, scratchBufferSize));
  b->addAlgorithmBuilder(
      std::make_shared<AllreducePacket>(scratchBuffer, scratchBufferSize));
  b->addAlgorithmBuilder(std::make_shared<AllreduceNvls>());
  b->addAlgorithmBuilder(std::make_shared<AllreduceNvlsWithCopy>(
      scratchBuffer, scratchBufferSize));
  b->addAlgorithmBuilder(
      std::make_shared<Allreduce8>(scratchBuffer, scratchBufferSize));
  b->addAlgorithmBuilder(
      std::make_shared<AllreduceNvlsPacket>(scratchBuffer, scratchBufferSize));
}

}  // namespace

#define NCCL_API extern "C" __attribute__((visibility("default")))

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

typedef enum mscclppNcclDlopenErr {
  dlopenSuccess = 0,
  dlopenError = 1,
} mscclppNcclDlopenErr_t;

typedef struct _mscclppNcclOps_t {
  ncclResult_t (*CommInitRank)(ncclComm_t* comm, int nranks,
                               ncclUniqueId commId, int rank);
  ncclResult_t (*GetUniqueId)(ncclUniqueId* uniqueId);
  ncclResult_t (*CommFinalize)(ncclComm_t comm);
  ncclResult_t (*CommDestroy)(ncclComm_t comm);
  ncclResult_t (*CommUserRank)(const ncclComm_t, int* rank);
  ncclResult_t (*AllReduce)(void const* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*AllGather)(void const* sendbuff, void* recvbuff,
                            size_t sendcount, ncclDataType_t datatype,
                            ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*Broadcast)(void const* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int root, ncclComm_t comm,
                            cudaStream_t stream);
  ncclResult_t (*ReduceScatter)(void const* sendbuff, void* recvbuff,
                                size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm,
                                cudaStream_t stream);
  ncclResult_t (*Reduce)(void const* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int root,
                         ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*Send)(void const* sendbuff, size_t count,
                       ncclDataType_t datatype, int peer, ncclComm_t comm,
                       cudaStream_t stream);
  ncclResult_t (*Recv)(void* recvbuff, size_t count, ncclDataType_t datatype,
                       int peer, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*GroupStart)();
  ncclResult_t (*GroupEnd)();
} mscclppNcclOps_t;

mscclppNcclOps_t mscclppNcclOps;
void* mscclppNcclDlHandle = NULL;
bool mscclppNcclDlopenSharedLib = false;

#define QUOTE(symbol) #symbol

#define NCCL_DLSYM(_struct_, _handle_, _prefix_, _function_, _type_) \
  do {                                                               \
    _struct_._function_ =                                            \
        (_type_)dlsym((_handle_), QUOTE(_prefix_##_function_));      \
    if (_struct_._function_ == NULL) {                               \
      printf("Failed: dlsym error: Cannot open %s: %s\n",            \
             QUOTE(_prefix_##_function_), dlerror());                \
      exit(dlopenError);                                             \
    }                                                                \
  } while (0)

static inline int mscclppNcclDlopenInit() {
  char const* ncclLibPath = mscclpp::env()->ncclSharedLibPath.c_str();
  if (ncclLibPath != nullptr && ncclLibPath[0] != '\0') {
    if (std::filesystem::is_directory(ncclLibPath)) {
      WARN(MSCCLPP_NCCL,
           "The value of the environment variable %s is a directory",
           ncclLibPath);
      return dlopenError;
    }

    mscclppNcclDlHandle =
        dlopen(ncclLibPath, RTLD_LAZY | RTLD_NODELETE | RTLD_DEEPBIND);
    if (!mscclppNcclDlHandle) {
      WARN(MSCCLPP_NCCL,
           "Cannot open the shared library specified by MSCCLPP_NCCL_LIB_PATH: "
           "%s\n",
           dlerror());
      return dlopenError;
    }
  } else {
    WARN(MSCCLPP_NCCL, "The value of MSCCLPP_NCCL_LIB_PATH is empty!\n");
    return dlopenError;
  }

  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, CommInitRank,
             ncclResult_t(*)(ncclComm_t*, int, ncclUniqueId, int));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, GetUniqueId,
             ncclResult_t(*)(ncclUniqueId*));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, CommFinalize,
             ncclResult_t(*)(ncclComm_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, CommDestroy,
             ncclResult_t(*)(ncclComm_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, CommUserRank,
             ncclResult_t(*)(ncclComm_t, int*));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, AllReduce,
             ncclResult_t(*)(void const*, void*, size_t, ncclDataType_t,
                             ncclRedOp_t, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, AllGather,
             ncclResult_t(*)(void const*, void*, size_t, ncclDataType_t,
                             ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, Broadcast,
             ncclResult_t(*)(void const*, void*, size_t, ncclDataType_t, int,
                             ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, ReduceScatter,
             ncclResult_t(*)(void const*, void*, size_t, ncclDataType_t,
                             ncclRedOp_t, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, Reduce,
             ncclResult_t(*)(void const*, void*, size_t, ncclDataType_t,
                             ncclRedOp_t, int, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, Send,
             ncclResult_t(*)(void const*, size_t, ncclDataType_t, int,
                             ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, Recv,
             ncclResult_t(*)(void*, size_t, ncclDataType_t, int, ncclComm_t,
                             cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, GroupStart,
             ncclResult_t(*)());
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, GroupEnd,
             ncclResult_t(*)());

  return dlopenSuccess;
}

// No need to call this function, handle will be closed at program exit
[[maybe_unused]] static inline void mscclppNcclDlopenFinalize() {
  if (mscclppNcclDlHandle) {
    dlclose(mscclppNcclDlHandle);
  }
}

static inline int mscclppNcclInFallbackList(char const* collOps,
                                            char const* fallbackList) {
  if (strcmp(fallbackList, "all") == 0) {
    return 1;
  }

  char* fallbackListCopy = strdup(fallbackList);
  char* token = strtok(fallbackListCopy, ",");
  while (token != NULL) {
    if (strcmp(collOps, token) == 0) {
      free(fallbackListCopy);
      return 1;
    }
    token = strtok(NULL, ",");
  }

  free(fallbackListCopy);
  return 0;
}

static bool tryLoadNcclSharedLib() {
  if (mscclppNcclDlopenSharedLib) return true;
  if (!mscclpp::env()->ncclSharedLibPath.empty()) {
    if (mscclppNcclDlopenInit() == dlopenSuccess) {
      mscclppNcclDlopenSharedLib = true;
      return true;
    }
  }
  return false;
}

// Declare the global map to store associations between raw pointer and shared
// pointer
static std::unordered_map<void*, std::shared_ptr<char>> ptrMap;

struct splitCommInfo {
  int color;
  int key;
  int originalRank;
};

enum class GroupedP2POpKind { Send, Recv };

struct GroupedP2POp {
  GroupedP2POpKind kind;
  void const* sendbuff = nullptr;
  void* recvbuff = nullptr;
  size_t count = 0;
  ncclDataType_t datatype = ncclFloat32;
  int peer = -1;
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
};

thread_local int gNcclGroupDepth = 0;
thread_local std::vector<GroupedP2POp> gNcclGroupedP2POps;

struct ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::shared_ptr<mscclpp::Executor> executor;
  mscclpp::AlgorithmCollection algorithmCollection;
  std::shared_ptr<char> scratchBuffer_;
  std::shared_ptr<void> flagBuffer_;
  size_t flagBufferSize_;
  const size_t scratchBufferSize_ = (1 << 27);  // 128MB
  int nRanksPerNode;
  int worldSize;
  int cudaDevice = -1;  // cached from ncclCommInitRank

  void* mscclppNcclComm;
  std::mutex sendRecvMutex;
  std::unordered_map<int, NcclSendRecvPeerContext> sendRecvPeerContexts;

  // Cached at init time to avoid per-call overhead.
  bool hasIB = false;
  size_t sendRecvStagingBytesCached = 0;
};

static bool peersShareNode(ncclComm_t comm, int peer) {
  return comm->nRanksPerNode > 0 &&
         (comm->comm->bootstrap()->getRank() / comm->nRanksPerNode ==
          peer / comm->nRanksPerNode);
}

static mscclpp::Transport selectSendRecvTransport(ncclComm_t comm, int peer) {
  if (!hasIBDevices() || peersShareNode(comm, peer)) {
    return mscclpp::Transport::Unknown;
  }
  int localIndex = comm->comm->bootstrap()->getRank() % comm->nRanksPerNode;
  if (localIndex < 0 ||
      localIndex >= static_cast<int>(sizeof(kIBTransports) /
                                     sizeof(kIBTransports[0]))) {
    throw mscclpp::Error("Local rank index is out of supported IB range",
                         mscclpp::ErrorCode::InvalidUsage);
  }
  return kIBTransports[localIndex];
}

static int deviceForBuffer(void const* ptr) {
  int device = mscclpp::detail::gpuIdFromAddress(const_cast<void*>(ptr));
  if (device < 0) {
    throw mscclpp::Error("Failed to infer GPU device from buffer pointer",
                         mscclpp::ErrorCode::InvalidUsage);
  }
  return device;
}

static void initializeSendRecvPeerContext(ncclComm_t comm, int peer,
                                          int localDevice,
                                          NcclSendRecvPeerContext& ctx) {
  mscclpp::CudaDeviceGuard deviceGuard(localDevice);
  int rank = comm->comm->bootstrap()->getRank();

  ctx.localDevice = localDevice;
  ctx.stagingBytes = sendRecvStagingBytes();
  ctx.transport = selectSendRecvTransport(comm, peer);
  if (ctx.transport == mscclpp::Transport::Unknown) {
    throw mscclpp::Error(
        "CPU-driven ncclSend/ncclRecv currently requires inter-node IB "
        "transport",
        mscclpp::ErrorCode::InvalidUsage);
  }

  // The staging buffers only participate in D2H/H2D copies and IB host MR
  // registration; they do not need a device alias. Using plain pinned host
  // memory avoids routing them into the GPU DMABUF registration path.
  ctx.sendStagingBuffer =
      mscclpp::detail::gpuCallocHostShared<char>(ctx.stagingBytes, 0);
  ctx.recvStagingBuffer =
      mscclpp::detail::gpuCallocHostShared<char>(ctx.stagingBytes, 0);

  mscclpp::TransportFlags transportFlags(ctx.transport);
  ctx.sendStagingMemory = comm->comm->registerMemory(
      ctx.sendStagingBuffer.get(), ctx.stagingBytes, transportFlags);
  ctx.recvStagingMemory = comm->comm->registerMemory(
      ctx.recvStagingBuffer.get(), ctx.stagingBytes, transportFlags);

  // Use a CPU-device endpoint so that Host2HostSemaphore tokens live in
  // CPU-pinned memory and RDMA atomics target host memory (always supported).
  mscclpp::EndpointConfig endpointConfig(
      ctx.transport, mscclpp::Device(mscclpp::DeviceType::CPU));
  auto connectionFuture =
      comm->comm->connect(endpointConfig, peer,
                          sendRecvInitTag(rank, comm->worldSize, peer, 0));
  comm->comm->sendMemory(ctx.recvStagingMemory, peer,
                         sendRecvInitTag(rank, comm->worldSize, peer, 1));
  auto remoteRecvStagingFuture =
      comm->comm->recvMemory(peer,
                             sendRecvInitTag(rank, comm->worldSize, peer, 1));

  ctx.connection = connectionFuture.get();
  auto semaphore =
      comm->comm
          ->buildSemaphore(ctx.connection, peer,
                           sendRecvInitTag(rank, comm->worldSize, peer, 2))
          .get();
  ctx.remoteRecvStagingMemory = remoteRecvStagingFuture.get();
  ctx.h2hSemaphore =
      std::make_shared<mscclpp::Host2HostSemaphore>(semaphore);

  // Start async worker thread for IB operations.
  ctx.worker = std::make_unique<SendRecvWorkerState>();
  MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.worker->d2hDoneEvent,
                                              cudaEventDisableTiming));
  ctx.worker->thread = std::thread(sendRecvWorkerLoop, &ctx);
}

static NcclSendRecvPeerContext& getSendRecvPeerContext(ncclComm_t comm,
                                                       int peer,
                                                       int localDevice) {
  std::lock_guard<std::mutex> lock(comm->sendRecvMutex);
  auto [it, inserted] = comm->sendRecvPeerContexts.try_emplace(peer);
  if (!it->second.worker) {
    try {
      initializeSendRecvPeerContext(comm, peer, localDevice, it->second);
    } catch (...) {
      if (inserted) {
        comm->sendRecvPeerContexts.erase(it);
      }
      throw;
    }
  } else if (it->second.localDevice != localDevice) {
    throw mscclpp::Error(
        "ncclSend/ncclRecv staging context was initialized on a different GPU",
        mscclpp::ErrorCode::InvalidUsage);
  }
  return it->second;
}

static ncclResult_t executeNcclSendImpl(void const* sendbuff, size_t count,
                                        ncclDataType_t datatype, int peer,
                                        ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr || sendbuff == nullptr || peer < 0 ||
      peer >= comm->worldSize) {
    WARN(MSCCLPP_NCCL,
         "ncclSend received invalid arguments: sendbuff=%p peer=%d comm=%p",
         sendbuff, peer, comm);
    return ncclInvalidArgument;
  }
  if (peer == comm->comm->bootstrap()->getRank()) {
    WARN(MSCCLPP_NCCL, "ncclSend does not support self-send");
    return ncclInvalidUsage;
  }

  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) {
    WARN(MSCCLPP_NCCL, "ncclSend got an invalid datatype %d", datatype);
    return ncclInvalidArgument;
  }
  size_t bytes = count * typeSize;
  if (bytes == 0) return ncclSuccess;
  int localDevice = comm->cudaDevice;
  size_t stagingBytes = comm->sendRecvStagingBytesCached;

  if (!comm->hasIB || peersShareNode(comm, peer) || bytes > stagingBytes) {
    if (mscclppNcclDlopenSharedLib == true) {
      return mscclppNcclOps.Send(
          sendbuff, count, datatype, peer,
          *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
    }
    WARN(MSCCLPP_NCCL,
         "CPU-driven ncclSend requires inter-node IB and message size "
         "<= ", stagingBytes, " bytes");
    return ncclInvalidUsage;
  }

  return runNcclGuarded("ncclSend", [&]() {
    mscclpp::CudaDeviceGuard deviceGuard(localDevice);
    auto& peerCtx = getSendRecvPeerContext(comm, peer, localDevice);
    auto* ws = peerCtx.worker.get();

    if (bytes <= kSendChunkBytes) {
      // Small message: single D2H + single IB write (no pipeline overhead).
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(
          peerCtx.sendStagingBuffer.get(), sendbuff, bytes,
          cudaMemcpyDeviceToHost, stream));
      MSCCLPP_CUDATHROW(cudaEventRecord(ws->d2hDoneEvent, stream));
      ws->push({SendRecvWorkerState::WorkItem::SEND, bytes});
    } else {
      // Large message: pipeline D2H copies with IB writes chunk by chunk.
      int numChunks = static_cast<int>((bytes + kSendChunkBytes - 1) /
                                       kSendChunkBytes);
      ws->ensureChunkEvents(numChunks);
      char const* src = static_cast<char const*>(sendbuff);
      char* staging = peerCtx.sendStagingBuffer.get();
      for (int i = 0; i < numChunks; ++i) {
        size_t off = static_cast<size_t>(i) * kSendChunkBytes;
        size_t chunkBytes = std::min(kSendChunkBytes, bytes - off);
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(
            staging + off, src + off, chunkBytes,
            cudaMemcpyDeviceToHost, stream));
        MSCCLPP_CUDATHROW(cudaEventRecord(ws->chunkEvents[i], stream));
        ws->push({SendRecvWorkerState::WorkItem::SEND_CHUNK, chunkBytes,
                  off, i, i == numChunks - 1});
      }
    }
  });
}

static ncclResult_t executeNcclRecvImpl(void* recvbuff, size_t count,
                                        ncclDataType_t datatype, int peer,
                                        ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr || recvbuff == nullptr || peer < 0 ||
      peer >= comm->worldSize) {
    WARN(MSCCLPP_NCCL,
         "ncclRecv received invalid arguments: recvbuff=%p peer=%d comm=%p",
         recvbuff, peer, comm);
    return ncclInvalidArgument;
  }
  if (peer == comm->comm->bootstrap()->getRank()) {
    WARN(MSCCLPP_NCCL, "ncclRecv does not support self-recv");
    return ncclInvalidUsage;
  }

  size_t typeSize = ncclTypeSize(datatype);
  if (typeSize == 0) {
    WARN(MSCCLPP_NCCL, "ncclRecv got an invalid datatype %d", datatype);
    return ncclInvalidArgument;
  }
  size_t bytes = count * typeSize;
  if (bytes == 0) return ncclSuccess;
  int localDevice = comm->cudaDevice;
  size_t stagingBytes = comm->sendRecvStagingBytesCached;

  if (!comm->hasIB || peersShareNode(comm, peer) || bytes > stagingBytes) {
    if (mscclppNcclDlopenSharedLib == true) {
      return mscclppNcclOps.Recv(
          recvbuff, count, datatype, peer,
          *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
    }
    WARN(MSCCLPP_NCCL,
         "CPU-driven ncclRecv requires inter-node IB and message size "
         "<= ", stagingBytes, " bytes");
    return ncclInvalidUsage;
  }

  return runNcclGuarded("ncclRecv", [&]() {
    mscclpp::CudaDeviceGuard deviceGuard(localDevice);
    auto& peerCtx = getSendRecvPeerContext(comm, peer, localDevice);
    peerCtx.h2hSemaphore->wait(/*maxSpinCount=*/-1);
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(recvbuff, peerCtx.recvStagingBuffer.get(),
                                      bytes, cudaMemcpyHostToDevice, stream));
  });
}

NCCL_API ncclResult_t ncclGetVersion(int* version) {
  if (version == nullptr) {
    WARN(MSCCLPP_NCCL, "version is nullptr");
    return ncclInvalidArgument;
  }
  *version = MSCCLPP_VERSION;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  if (uniqueId == nullptr) {
    WARN(MSCCLPP_NCCL, "uniqueId is nullptr");
    return ncclInvalidArgument;
  }
  if (mscclpp::UniqueIdBytes != NCCL_UNIQUE_ID_BYTES) return ncclInternalError;
  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();
  memcpy(uniqueId, &id, sizeof(ncclUniqueId));
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks,
                                             ncclUniqueId commId, int rank,
                                             ncclConfig_t*) {
  // TODO: implement config
  return ncclCommInitRank(comm, nranks, commId, rank);
}

static std::pair<int, int> getDeviceComputeCapability() {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  int major = 0, minor = 0;
  CUDACHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   device));
  CUDACHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                   device));
  return std::make_pair(major, minor);
}

static std::shared_ptr<mscclpp::Algorithm> algoSelector(
    std::unordered_map<
        std::string,
        std::unordered_map<std::string,
                           std::shared_ptr<mscclpp::Algorithm>>> const&
        algoMapByCollective,
    mscclpp::CollectiveRequest const& request) {
  if (algoMapByCollective.find(request.collective) ==
      algoMapByCollective.end()) {
    return nullptr;
  }

  for (auto const& pair : algoMapByCollective.at(request.collective)) {
    auto const& algo = pair.second;
    if (algo->type() == mscclpp::AlgorithmType::DSL) {
      if (mscclpp::nccl::matchExecutionPlan(
              std::static_pointer_cast<mscclpp::DslAlgorithm>(algo), request)) {
        return algo;
      }
    }
  }

  // Prepare algorithm selector configuration
  static bool const isNvlsSupported = mscclpp::isNvlsSupported();
  static const std::pair<int, int> deviceComputeCapability =
      getDeviceComputeCapability();
  static bool const ncclSymmetricMemory = mscclpp::env()->ncclSymmetricMemory;

  bool const isCuMemMapAllocated =
      mscclpp::isCuMemMapAllocated(const_cast<void*>(request.inputBuffer)) &&
      mscclpp::isCuMemMapAllocated(request.outputBuffer);

  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
  CUDACHECK(cudaStreamIsCapturing(request.stream, &captureStatus));
  bool const inCaptureMode = (captureStatus == cudaStreamCaptureStatusActive);

  mscclpp::nccl::AlgorithmSelectorConfig config{
      .symmetricMemory = ncclSymmetricMemory,
      .nvlsSupported = isNvlsSupported,
      .isCuMemMapAllocated = isCuMemMapAllocated,
      .inCaptureMode = inCaptureMode,
      .computeCapability = deviceComputeCapability,
      .ncclDlopenSharedLib = mscclppNcclDlopenSharedLib};

  auto const& algoMap = algoMapByCollective.at(request.collective);

  // Check if this is a multi-node scenario
  if (request.nRanksPerNode != request.worldSize) {
    return mscclpp::nccl::selectMultiNodeAlgorithm(algoMap, request, config);
  }

  // Single-node scenarios
  if (request.collective == "allgather") {
    return mscclpp::nccl::selectSingleNodeAllgather(algoMap, request, config);
  }

  if (request.collective == "allreduce") {
    return mscclpp::nccl::selectSingleNodeAllreduce(algoMap, request, config);
  }

  INFO(MSCCLPP_NCCL,
       "No suitable algorithm found for collective '%s', fallback to nccl/rccl",
       request.collective.c_str());
  return nullptr;
}

NCCL_API ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                                       ncclUniqueId commId, int rank) {
  INFO(MSCCLPP_NCCL,
       "Initializing NCCL communicator for rank %d, world_size=%d", rank,
       nranks);
  if (comm == nullptr) {
    WARN(MSCCLPP_NCCL, "comm is nullptr");
    return ncclInvalidArgument;
  }
  if (nranks < 0 || rank < 0 || rank >= nranks) {
    WARN(MSCCLPP_NCCL, "nranks is %d, rank is %d", nranks, rank);
    return ncclInvalidArgument;
  }
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap =
      std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
  mscclpp::UniqueId id;
  memcpy(id.data(), &commId, sizeof(ncclUniqueId));
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> mscclppComm =
      std::make_shared<mscclpp::Communicator>(bootstrap);
  ncclComm* commPtr = new ncclComm();

  commPtr->comm = mscclppComm;
  commPtr->scratchBuffer_ =
      mscclpp::GpuBuffer<char>(commPtr->scratchBufferSize_).memory();
  commPtr->executor =
      std::make_shared<mscclpp::Executor>(mscclppComm, commPtr->scratchBuffer_);

  auto [buffer, size] = mscclpp::getFlagBuffer();
  commPtr->flagBuffer_ = buffer;
  commPtr->flagBufferSize_ = size;

  commPtr->nRanksPerNode = mscclppComm->bootstrap()->getNranksPerNode();
  commPtr->worldSize = mscclppComm->bootstrap()->getNranks();
  commPtr->hasIB = hasIBDevices();
  MSCCLPP_CUDATHROW(cudaGetDevice(&commPtr->cudaDevice));
  try {
    commPtr->sendRecvStagingBytesCached = sendRecvStagingBytes();
  } catch (...) {
    commPtr->sendRecvStagingBytesCached = 0;
  }
  auto algoBuilder =
      mscclpp::collective::AlgorithmCollectionBuilder::getInstance();
  algoBuilder->setFallbackAlgorithmSelector(algoSelector);
  commPtr->algorithmCollection = algoBuilder->buildDefaultAlgorithms(
      reinterpret_cast<uintptr_t>(commPtr->scratchBuffer_.get()),
      commPtr->scratchBufferSize_,
      reinterpret_cast<uintptr_t>(commPtr->flagBuffer_.get()),
      commPtr->flagBufferSize_, rank);
  // Algorithms from this directory (same registration pattern as
  // mscclpp/apps/nccl).
  registerMigratedAppNcclAlgorithms(
      reinterpret_cast<uintptr_t>(commPtr->scratchBuffer_.get()),
      commPtr->scratchBufferSize_);
  // Extend with user-defined algorithms (singleton builders)
  commPtr->algorithmCollection.extend(algoBuilder->build());

  *comm = commPtr;
#if defined(ENABLE_NPKIT)
  if (mscclpp::env()->npkitDumpDir != "") {
    NpKit::Init(rank);
  }
#endif

  const std::string ncclLibPath = mscclpp::env()->ncclSharedLibPath;
  if (!ncclLibPath.empty() && !mscclppNcclDlopenSharedLib) {
    if (!tryLoadNcclSharedLib()) {
      WARN(MSCCLPP_NCCL, "Failed to load the shared library for nccl/rccl");
      return ncclInternalError;
    }
  }

  if (mscclppNcclDlopenSharedLib == true) {
    ncclUniqueId mscclppNcclUniqueId;
    if (rank == 0) {
      mscclppNcclOps.GetUniqueId(&mscclppNcclUniqueId);
    }
    // After broadcast, mscclppNcclUniqueId on each rank has the same
    // ncclUniqueId
    bootstrap->broadcast(&mscclppNcclUniqueId, sizeof(ncclUniqueId), 0);

    commPtr->mscclppNcclComm = new ncclComm_t();
    if (commPtr->mscclppNcclComm == nullptr) {
      WARN(MSCCLPP_NCCL, "Failed to allocate memory for mscclppNcclComm");
      return ncclInternalError;
    }
    mscclppNcclOps.CommInitRank(
        reinterpret_cast<ncclComm_t*>(commPtr->mscclppNcclComm), nranks,
        mscclppNcclUniqueId, rank);
  }

  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, int const*) {
  if (ndev == 1) {
    ncclUniqueId Id;
    ncclGetUniqueId(&Id);
    return ncclCommInitRank(comm, ndev, Id, 0);
  }
  // TODO: implement this function
  WARN(MSCCLPP_NCCL, "ncclCommInitAll is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  if (comm == nullptr) {
    WARN(MSCCLPP_NCCL, "comm is nullptr");
    return ncclInvalidArgument;
  }
  ncclComm_t* mscclppNcclCommPtr =
      reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm);
  if (mscclppNcclCommPtr != nullptr && mscclppNcclOps.CommFinalize != nullptr) {
    ncclResult_t result = mscclppNcclOps.CommFinalize(*mscclppNcclCommPtr);
    if (result != ncclSuccess) {
      return result;
    }
  }
  comm->comm->bootstrap()->barrier();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == nullptr) {
    WARN(MSCCLPP_NCCL, "comm is nullptr");
    return ncclInvalidArgument;
  }
#if defined(ENABLE_NPKIT)
  std::string const& npkitDumpDir = mscclpp::env()->npkitDumpDir;
  if (npkitDumpDir != "") {
    NpKit::Dump(npkitDumpDir);
    NpKit::Shutdown();
  }
#endif

  ncclComm_t* mscclppNcclCommPtr =
      reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm);
  if (mscclppNcclCommPtr != nullptr) {
    mscclppNcclOps.CommDestroy(*mscclppNcclCommPtr);
    delete mscclppNcclCommPtr;
  }
  delete comm;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommAbort(ncclComm_t) {
  // TODO: implement this function
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                                    ncclComm_t* newcomm, ncclConfig_t*) {
  *newcomm = NCCL_COMM_NULL;
  int nRanks = comm->comm->bootstrap()->getNranks();
  int rank = comm->comm->bootstrap()->getRank();
  splitCommInfo info{color, key, comm->comm->bootstrap()->getRank()};
  std::vector<splitCommInfo> infos(nRanks);
  infos[rank] = info;
  comm->comm->bootstrap()->allGather(infos.data(), sizeof(splitCommInfo));
  comm->comm->bootstrap()->barrier();
  std::vector<splitCommInfo> group;
  std::copy_if(
      infos.begin(), infos.end(), std::back_inserter(group),
      [color](splitCommInfo const& info) { return info.color == color; });
  std::sort(group.begin(), group.end(),
            [](splitCommInfo const& a, splitCommInfo const& b) {
              return a.key < b.key;
            });
  int newRank = std::distance(group.begin(),
                              std::find_if(group.begin(), group.end(),
                                           [rank](splitCommInfo const& info) {
                                             return info.originalRank == rank;
                                           }));
  int groupSize = group.size();
  ncclUniqueId uniqueId;
  if (newRank == 0) {
    ncclGetUniqueId(&uniqueId);
  }
  std::vector<ncclUniqueId> uniqueIds(nRanks);
  uniqueIds[rank] = uniqueId;
  comm->comm->bootstrap()->allGather(uniqueIds.data(), sizeof(ncclUniqueId));
  comm->comm->bootstrap()->barrier();
  uniqueId = uniqueIds[group.front().originalRank];
  if (color == NCCL_SPLIT_NOCOLOR) {
    return ncclSuccess;
  }
  return ncclCommInitRankConfig(newcomm, groupSize, uniqueId, newRank, nullptr);
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t*, int, int, int, ncclUniqueId*,
                                      ncclConfig_t*) {
  WARN(MSCCLPP_NCCL, "ncclCommInitRankScalable is currently unavailable");
  return ncclInternalError;
}

NCCL_API char const* ncclGetErrorString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return "no error";
    case ncclUnhandledCudaError:
      return "unhandled cuda error (run with MSCCLPP_DEBUG=INFO for details)";
    case ncclSystemError:
      return "unhandled system error (run with MSCCLPP_DEBUG=INFO for details)";
    case ncclInternalError:
      return "internal error (run with MSCCLPP_DEBUG=WARN for details)";
    case ncclInvalidArgument:
      return "invalid argument (run with MSCCLPP_DEBUG=WARN for details)";
    case ncclInvalidUsage:
      return "invalid usage (run with MSCCLPP_DEBUG=WARN for details)";
    case ncclRemoteError:
      return "remote process exited or there was a network error";
    case ncclInProgress:
      return "NCCL operation in progress";
    default:
      return "unknown result code";
  }
}

NCCL_API char const* ncclGetLastError(ncclComm_t) {
  // TODO: implement this function
  return "";
}

NCCL_API ncclResult_t ncclCommGetAsyncError(ncclComm_t,
                                            ncclResult_t* asyncError) {
  if (asyncError == nullptr) {
    WARN(MSCCLPP_NCCL, "asyncError is nullptr");
    return ncclInvalidArgument;
  }
  *asyncError = ncclSuccess;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  if (comm == nullptr || count == nullptr) {
    WARN(MSCCLPP_NCCL, "comm is nullptr or count is nullptr");
    return ncclInvalidArgument;
  }
  *count = comm->comm->bootstrap()->getNranks();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
  if (comm == nullptr || device == nullptr) {
    WARN(MSCCLPP_NCCL, "comm is nullptr or device is nullptr");
    return ncclInvalidArgument;
  }
  *device = comm->comm->bootstrap()->getRank();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  if (comm == nullptr || rank == nullptr) {
    WARN(MSCCLPP_NCCL, "comm is nullptr or rank is nullptr");
    return ncclInvalidArgument;
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.CommUserRank(
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), rank);
  }

  *rank = comm->comm->bootstrap()->getRank();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommWindowRegister(ncclComm_t, void*, size_t,
                                             ncclWindow_t*, int) {
  WARN(MSCCLPP_NCCL, "ncclCommWindowRegister is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t) {
  WARN(MSCCLPP_NCCL, "ncclCommWindowDeregister is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t*, void*,
                                               ncclDataType_t,
                                               ncclScalarResidence_t,
                                               ncclComm_t) {
  // TODO: implement this function
  WARN(MSCCLPP_NCCL, "ncclRedOpCreatePreMulSum is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRedOpDestroy(ncclRedOp_t, ncclComm_t) {
  // TODO: implement this function
  WARN(MSCCLPP_NCCL, "ncclRedOpDestroy is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclReduce(void const* sendbuff, void* recvbuff,
                                 size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, int root, ncclComm_t comm,
                                 cudaStream_t stream) {
  // TODO: implement this function
  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.Reduce(
        sendbuff, recvbuff, count, datatype, op, root,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }
  WARN(MSCCLPP_NCCL, "ncclReduce is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclBcast(void* buff, size_t count,
                                ncclDataType_t datatype, int root,
                                ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API ncclResult_t ncclBroadcast(void const* sendbuff, void* recvbuff,
                                    size_t count, ncclDataType_t datatype,
                                    int root, ncclComm_t comm,
                                    cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                                cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }
  int rank = comm->comm->bootstrap()->getRank();
  if ((sendbuff == nullptr && root == rank) || recvbuff == nullptr ||
      bytes == 0 || comm == nullptr) {
    WARN(MSCCLPP_NCCL,
         "One or more of the following conditions is met: sendbuff or recvbuff "
         "pointer is nullptr, bytes is 0, "
         "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  INFO(
      MSCCLPP_NCCL,
      "rank %d broadcast sendbuff %p recvbuff %p count %ld, dtype %d, comm: %p",
      rank, sendbuff, recvbuff, count, datatype, comm);

  char const* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  if (mscclppNcclDlopenSharedLib == true &&
      mscclppNcclInFallbackList("broadcast", fallbackList)) {
    return mscclppNcclOps.Broadcast(
        sendbuff, recvbuff, count, datatype, root,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  mscclpp::DataType dtype = ncclDataTypeToMscclpp(datatype);
  static std::unordered_map<std::string, std::vector<uint64_t>> hints{
      {"root", {static_cast<uint64_t>(root)}}};
  hints["root"][0] = static_cast<uint64_t>(root);

  bool const symmetricMemory = mscclpp::env()->ncclSymmetricMemory;
  mscclpp::CollectiveRequest request = {.worldSize = comm->worldSize,
                                        .nRanksPerNode = comm->nRanksPerNode,
                                        .rank = rank,
                                        .inputBuffer = sendbuff,
                                        .outputBuffer = recvbuff,
                                        .messageSize = bytes,
                                        .stream = stream,
                                        .collective = "broadcast",
                                        .dtype = dtype,
                                        .hints = hints};
  auto algo = comm->algorithmCollection.selectAlgorithm(request);
  if (algo != nullptr) {
    std::unordered_map<std::string, uintptr_t> extras{
        {"root", reinterpret_cast<uintptr_t>(&root)}};
    return static_cast<ncclResult_t>(
        algo->execute(comm->comm, sendbuff, recvbuff, bytes, bytes, dtype,
                      mscclpp::ReduceOp::NOP, stream, comm->executor, 0, 0,
                      symmetricMemory, extras));
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.Broadcast(
        sendbuff, recvbuff, count, datatype, root,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  WARN(MSCCLPP_NCCL, "No FallBack implementation for broadcast");
  return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclAllReduce(void const* sendbuff, void* recvbuff,
                                    size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t reductionOperation,
                                    ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                                cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }
  // Checking if the parameters are valids
  if (sendbuff == nullptr || recvbuff == nullptr || count == 0 ||
      ncclTypeSize(datatype) == 0 || comm == nullptr) {
    WARN(MSCCLPP_NCCL,
         "One or more of the following conditions is met: sendbuff or recvbuff "
         "pointer is nullptr, count is 0, "
         "datatype is invalid, or comm is nullptr.");
    return ncclInvalidArgument;
  }
  // Declarating variables
  int rank = comm->comm->bootstrap()->getRank();
  INFO(MSCCLPP_NCCL,
       "rank %d allreduce sendbuff %p recvbuff %p count %ld, dtype %d comm is "
       "%p",
       rank, sendbuff, recvbuff, count, datatype, comm);

  char const* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  if (mscclppNcclDlopenSharedLib &&
      mscclppNcclInFallbackList("allreduce", fallbackList)) {
    return mscclppNcclOps.AllReduce(
        sendbuff, recvbuff, count, datatype, reductionOperation,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }
  mscclpp::DataType dtype = ncclDataTypeToMscclpp(datatype);
  bool const symmetricMemory = mscclpp::env()->ncclSymmetricMemory;
  mscclpp::CollectiveRequest request = {.worldSize = comm->worldSize,
                                        .nRanksPerNode = comm->nRanksPerNode,
                                        .rank = rank,
                                        .inputBuffer = sendbuff,
                                        .outputBuffer = recvbuff,
                                        .messageSize = bytes,
                                        .stream = stream,
                                        .collective = "allreduce",
                                        .dtype = dtype,
                                        .hints = {}};

  auto algo = comm->algorithmCollection.selectAlgorithm(request);
  if (algo != nullptr) {
    return static_cast<ncclResult_t>(
        algo->execute(comm->comm, sendbuff, recvbuff, bytes, bytes, dtype,
                      ncclRedOpToMscclpp(reductionOperation), stream,
                      comm->executor, 0, 0, symmetricMemory));
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.AllReduce(
        sendbuff, recvbuff, count, datatype, reductionOperation,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  WARN(MSCCLPP_NCCL, "No FallBack implementation for AllReduce");
  return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclReduceScatter(void const* sendbuff, void* recvbuff,
                                        size_t recvcount,
                                        ncclDataType_t datatype, ncclRedOp_t op,
                                        ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = recvcount * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                                cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }

  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 ||
      comm == nullptr) {
    WARN(MSCCLPP_NCCL,
         "One or more of the following conditions is met: sendbuff or recvbuff "
         "pointer is nullptr, bytes is 0, "
         "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  INFO(MSCCLPP_NCCL,
       "ReduceScatter recvcount: %ld, datatype: %d, op: %d, messageSize: %ld",
       recvcount, datatype, op, bytes * comm->comm->bootstrap()->getNranks());

  char const* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  if (mscclppNcclDlopenSharedLib == true &&
      mscclppNcclInFallbackList("reducescatter", fallbackList)) {
    return mscclppNcclOps.ReduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();
  mscclpp::DataType dtype = ncclDataTypeToMscclpp(datatype);
  bool const symmetricMemory = mscclpp::env()->ncclSymmetricMemory;
  mscclpp::CollectiveRequest request = {.worldSize = comm->worldSize,
                                        .nRanksPerNode = comm->nRanksPerNode,
                                        .rank = rank,
                                        .inputBuffer = sendbuff,
                                        .outputBuffer = recvbuff,
                                        .messageSize = bytes * nRank,
                                        .stream = stream,
                                        .collective = "reducescatter",
                                        .dtype = dtype,
                                        .hints = {}};
  auto algo = comm->algorithmCollection.selectAlgorithm(request);
  if (algo != nullptr) {
    return static_cast<ncclResult_t>(algo->execute(
        comm->comm, sendbuff, recvbuff, bytes * nRank, bytes, dtype,
        ncclRedOpToMscclpp(op), stream, comm->executor, 0, 0, symmetricMemory));
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.ReduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  WARN(MSCCLPP_NCCL, "No FallBack implementation for ReduceScatter");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllGather(void const* sendbuff, void* recvbuff,
                                    size_t sendcount, ncclDataType_t datatype,
                                    ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = sendcount * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                                cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 ||
      comm == nullptr) {
    WARN(MSCCLPP_NCCL,
         "One or more of the following conditions is met: sendbuff or recvbuff "
         "pointer is nullptr, bytes is 0, "
         "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();
  INFO(MSCCLPP_NCCL,
       "rank %d allgather sendbuff %p recvbuff %p count %ld, dtype %d, comm %p",
       rank, sendbuff, recvbuff, sendcount, datatype, comm);

  char const* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  if (mscclppNcclDlopenSharedLib == true &&
      mscclppNcclInFallbackList("allgather", fallbackList)) {
    return mscclppNcclOps.AllGather(
        sendbuff, recvbuff, sendcount, datatype,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  mscclpp::DataType dtype = ncclDataTypeToMscclpp(datatype);
  bool const symmetricMemory = mscclpp::env()->ncclSymmetricMemory;
  mscclpp::CollectiveRequest request = {.worldSize = comm->worldSize,
                                        .nRanksPerNode = comm->nRanksPerNode,
                                        .rank = rank,
                                        .inputBuffer = sendbuff,
                                        .outputBuffer = recvbuff,
                                        .messageSize = bytes,
                                        .stream = stream,
                                        .collective = "allgather",
                                        .dtype = dtype,
                                        .hints = {}};

  auto algo = comm->algorithmCollection.selectAlgorithm(request);
  if (algo != nullptr) {
    return static_cast<ncclResult_t>(algo->execute(
        comm->comm, sendbuff, recvbuff, bytes, bytes * nRank, dtype,
        mscclpp::ReduceOp::NOP, stream, comm->executor, 0, 0, symmetricMemory));
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.AllGather(
        sendbuff, recvbuff, sendcount, datatype,
        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  WARN(MSCCLPP_NCCL, "No FallBack implementation for AllGather");
  return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclSend(void const* sendbuff, size_t count,
                               ncclDataType_t datatype, int peer,
                               ncclComm_t comm, cudaStream_t stream) {
  if (!mscclppNcclDlopenSharedLib && gNcclGroupDepth > 0) {
    gNcclGroupedP2POps.push_back({.kind = GroupedP2POpKind::Send,
                                  .sendbuff = sendbuff,
                                  .recvbuff = nullptr,
                                  .count = count,
                                  .datatype = datatype,
                                  .peer = peer,
                                  .comm = comm,
                                  .stream = stream});
    return ncclSuccess;
  }
  return executeNcclSendImpl(sendbuff, count, datatype, peer, comm, stream);
}

NCCL_API ncclResult_t ncclRecv(void* recvbuff, size_t count,
                               ncclDataType_t datatype, int peer,
                               ncclComm_t comm, cudaStream_t stream) {
  if (!mscclppNcclDlopenSharedLib && gNcclGroupDepth > 0) {
    gNcclGroupedP2POps.push_back({.kind = GroupedP2POpKind::Recv,
                                  .sendbuff = nullptr,
                                  .recvbuff = recvbuff,
                                  .count = count,
                                  .datatype = datatype,
                                  .peer = peer,
                                  .comm = comm,
                                  .stream = stream});
    return ncclSuccess;
  }
  return executeNcclRecvImpl(recvbuff, count, datatype, peer, comm, stream);
}

NCCL_API ncclResult_t ncclAllToAll(void const* sendbuff, void* recvbuff,
                                   size_t count, ncclDataType_t datatype,
                                   ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes,
                                cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }
  // TODO: implement this function
  WARN(MSCCLPP_NCCL, "ncclAllToAll is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllToAllv(void const* sendbuff,
                                    [[maybe_unused]] const size_t sendcounts[],
                                    const size_t sdispls[], void* recvbuff,
                                    const size_t recvcounts[],
                                    const size_t rdispls[],
                                    ncclDataType_t datatype, ncclComm_t comm,
                                    cudaStream_t stream) {
  size_t bytes = recvcounts[0] * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        (char*)recvbuff + rdispls[0] * ncclTypeSize(datatype),
        (char const*)sendbuff + sdispls[0] * ncclTypeSize(datatype), bytes,
        cudaMemcpyDeviceToDevice, stream));
    return ncclSuccess;
  }
  WARN(MSCCLPP_NCCL, "ncclAllToAllv is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclGroupStart() {
  tryLoadNcclSharedLib();
  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.GroupStart();
  }
  gNcclGroupDepth++;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGroupEnd() {
  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.GroupEnd();
  }
  if (gNcclGroupDepth <= 0) {
    WARN(MSCCLPP_NCCL, "ncclGroupEnd called without a matching ncclGroupStart");
    return ncclInvalidUsage;
  }
  gNcclGroupDepth--;
  if (gNcclGroupDepth > 0) {
    return ncclSuccess;
  }

  auto ops = std::move(gNcclGroupedP2POps);
  gNcclGroupedP2POps.clear();
  for (auto const& op : ops) {
    ncclResult_t result =
        (op.kind == GroupedP2POpKind::Send)
            ? executeNcclSendImpl(op.sendbuff, op.count, op.datatype, op.peer,
                                  op.comm, op.stream)
            : executeNcclRecvImpl(op.recvbuff, op.count, op.datatype, op.peer,
                                  op.comm, op.stream);
    if (result != ncclSuccess) {
      WARN(MSCCLPP_NCCL, "ncclGroupEnd queued ",
           (op.kind == GroupedP2POpKind::Send ? "send" : "recv"),
           " failed for peer ", op.peer, " count ", op.count, " dtype ",
           static_cast<int>(op.datatype), " with result ",
           static_cast<int>(result));
      return result;
    }
  }
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t*) {
  // TODO: implement this function
  WARN(MSCCLPP_NCCL, "ncclGroupSimulateEnd is not implemented");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommRegister(const ncclComm_t, void*, size_t,
                                       void**) {
  // TODO: Implementation
  WARN(MSCCLPP_NCCL, "ncclCommRegister is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommDeregister(const ncclComm_t, void*) {
  // TODO: Implementation
  WARN(MSCCLPP_NCCL, "ncclCommDeregister is currently unavailable");
  return ncclInternalError;
}

ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    WARN(MSCCLPP_NCCL, "ptr is nullptr or size is 0");
    return ncclInvalidArgument;
  }
  std::shared_ptr<char> sharedPtr;
  try {
    sharedPtr = mscclpp::GpuBuffer(size).memory();
    if (sharedPtr == nullptr) {
      WARN(MSCCLPP_NCCL, "Failed to allocate memory via ncclMemAlloc");
      return ncclSystemError;
    }
  } catch (mscclpp::Error const& e) {
    if (e.getErrorCode() == mscclpp::ErrorCode::InvalidUsage) {
      WARN(MSCCLPP_NCCL, "Invalid usage: %s", e.what());
      return ncclInvalidUsage;
    } else {
      WARN(MSCCLPP_NCCL, "Internal error: %s", e.what());
      return ncclInternalError;
    }
  } catch (mscclpp::CudaError const& e) {
    WARN(MSCCLPP_NCCL, "Cuda error: %s", e.what());
    return ncclUnhandledCudaError;
  } catch (mscclpp::CuError const& e) {
    WARN(MSCCLPP_NCCL, "Cu error: %s", e.what());
    return ncclUnhandledCudaError;
  } catch (mscclpp::BaseError const& e) {
    WARN(MSCCLPP_NCCL, "Base error: %s", e.what());
    return ncclInternalError;
  }
  ptrMap[sharedPtr.get()] = sharedPtr;

  // Return the pointer
  *ptr = sharedPtr.get();
  return ncclSuccess;
}

ncclResult_t ncclMemFree(void* ptr) {
  auto ptrIt = ptrMap.find(ptr);
  if (ptrIt != ptrMap.end()) {
    ptrMap.erase(ptrIt);
    return ncclSuccess;
  }

  // Pointer not found
  WARN(MSCCLPP_NCCL, "Pointer not found");
  return ncclInvalidUsage;
}
