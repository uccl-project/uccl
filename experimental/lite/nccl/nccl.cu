// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "algorithm_collection_builder.hpp"
#include "core.hpp"
#include "env.hpp"
#include "executor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
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
#include "ib.hpp"
#include "logger.hpp"
#include "memory_channel.hpp"
#include "nccl.h"
#include "numa.hpp"
#include "registered_memory.hpp"
#include "semaphore.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#include <cuda.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static constexpr auto MSCCLPP_NCCL = mscclpp::LogSubsys::NCCL;

namespace {

static constexpr int kNcclSendRecvInitTagBase = 0x530000;
static constexpr int kNcclSendRecvInitTagStride = 8;
// Runtime tag base for per-call recvbuff IPC handle exchange (GroupEnd).
// Uses a distinct range to avoid collisions with init-time tags.
static constexpr int kNcclRecvbuffExchangeTagBase = 0x540000;

static const mscclpp::Transport kIBTransports[] = {
    mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
    mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
    mscclpp::Transport::IB6, mscclpp::Transport::IB7};

// ---------------------------------------------------------------------------
// Topology-aware IB transport selection.
//
// Matches each GPU to an IB HCA on the same NUMA node for optimal PCIe
// locality.  When MSCCLPP_HCA_DEVICES is set, only those devices are
// considered.  Falls back to round-robin when NUMA information is unavailable
// or no same-NUMA device exists.
// ---------------------------------------------------------------------------

static int getIBDeviceNumaNode(std::string const& ibDevName) {
  std::string path = "/sys/class/infiniband/" + ibDevName + "/device/numa_node";
  std::ifstream f(path);
  int node = -1;
  if (f.is_open()) f >> node;
  return node;
}

// Returns the list of IB transports we are allowed to use.
// If MSCCLPP_HCA_DEVICES is set, the list is limited to those devices (in
// order).  Otherwise all hardware IB devices are included.
static std::vector<mscclpp::Transport> getAvailableIBTransports() {
  std::string hcaEnv = mscclpp::env()->hcaDevices;
  int count;
  if (!hcaEnv.empty()) {
    count = 0;
    std::stringstream ss(hcaEnv);
    std::string tok;
    while (std::getline(ss, tok, ',')) ++count;
  } else {
    count = mscclpp::getIBDeviceCount();
  }
  count = std::min(count, static_cast<int>(sizeof(kIBTransports) /
                                           sizeof(kIBTransports[0])));
  std::vector<mscclpp::Transport> result;
  result.reserve(count);
  for (int i = 0; i < count; ++i) result.push_back(kIBTransports[i]);
  return result;
}

// Select the best IB transport for a given CUDA device, preferring HCAs on
// the same NUMA node.  The result is deterministic and symmetric across nodes
// with identical hardware, so the peer's transport can be computed locally.
static mscclpp::Transport selectIBTransportForGpu(int cudaDeviceId) {
  static std::mutex cacheMu;
  static std::unordered_map<int, mscclpp::Transport> cache;
  {
    std::lock_guard<std::mutex> lk(cacheMu);
    auto it = cache.find(cudaDeviceId);
    if (it != cache.end()) return it->second;
  }

  auto available = getAvailableIBTransports();
  if (available.empty()) {
    return mscclpp::Transport::Unknown;
  }

  int gpuNuma = -1;
  try {
    gpuNuma = mscclpp::getDeviceNumaNode(cudaDeviceId);
  } catch (...) {
    // NUMA info unavailable — fall through to round-robin.
  }

  // Collect NUMA nodes for each available IB device.
  std::vector<mscclpp::Transport> sameNuma;
  for (auto t : available) {
    try {
      std::string name = mscclpp::getIBDeviceName(t);
      int ibNuma = getIBDeviceNumaNode(name);
      if (gpuNuma >= 0 && ibNuma == gpuNuma) {
        sameNuma.push_back(t);
      }
    } catch (...) {
      // Skip devices whose names can't be resolved.
    }
  }

  mscclpp::Transport chosen;
  std::string chosenName;
  if (!sameNuma.empty()) {
    // Among same-NUMA HCAs, round-robin by device ID.
    chosen = sameNuma[cudaDeviceId % sameNuma.size()];
  } else {
    // No NUMA match — round-robin across all available.
    chosen = available[cudaDeviceId % available.size()];
  }
  try {
    chosenName = mscclpp::getIBDeviceName(chosen);
  } catch (...) {
    chosenName = "?";
  }
  std::string reason = sameNuma.empty() ? " (round-robin)" : " (NUMA-local)";
  INFO(MSCCLPP_NCCL, "GPU ", cudaDeviceId, " (NUMA ", gpuNuma, ") -> IB ",
       chosenName, reason);

  {
    std::lock_guard<std::mutex> lk(cacheMu);
    cache[cudaDeviceId] = chosen;
  }
  return chosen;
}

// ---------------------------------------------------------------------------
// POSIX-shm based control block for intra-node CudaIpc signaling.
//
// Each side allocates a shm segment containing a ShmBlock.  The peer maps it
// at init time via bootstrap name exchange.  The block carries:
//   - counter: data-ready semaphore (signal/wait)
//   - recvbuffAddr + addrGeneration: per-iteration recvbuff pointer exchange
//     (avoids bootstrap TCP per iteration — only ~10ns shm r/w)
//   - allocBase + allocGeneration: allocation base for IPC offset computation
//     (triggers IPC handle re-exchange only on new cudaMalloc, not on offset
//      changes within the same allocation)
// ---------------------------------------------------------------------------
struct ShmBlock {
  alignas(64) std::atomic<uint64_t> counter{0};
  alignas(64) std::atomic<uint64_t> recvbuffAddr{0};
  alignas(64) std::atomic<uint64_t> addrGeneration{0};
  alignas(64) std::atomic<uint64_t> allocBase{0};
  alignas(64) std::atomic<uint64_t> allocGeneration{0};
};

struct ShmSemaphore {
  ShmBlock* local = nullptr;   // our shm segment (peer maps it)
  ShmBlock* remote = nullptr;  // peer's shm segment (we mapped it)

  std::string localShmName;
  void* localMapping = nullptr;
  void* remoteMapping = nullptr;
  uint64_t outbound = 0;
  uint64_t expectedInbound = 0;
  uint64_t localAddrGen = 0;
  uint64_t expectedAddrGen = 0;

  void signal() {
    ++outbound;
    remote->counter.store(outbound, std::memory_order_release);
  }

  void wait() {
    ++expectedInbound;
    while (local->counter.load(std::memory_order_acquire) < expectedInbound) {
      // Pure host memory spin — no CUDA API, no SM.
    }
  }

  // Called by receiver in GroupEnd: publish current recvbuff address.
  // The addrGeneration release-store ensures all prior relaxed writes
  // (recvbuffAddr, allocBase, allocGeneration) are visible to the sender.
  void publishRecvbuff(uintptr_t addr) {
    local->recvbuffAddr.store(addr, std::memory_order_relaxed);
    local->addrGeneration.store(++localAddrGen, std::memory_order_release);
  }

  // Called by receiver when allocation base changes (new cudaMalloc).
  void publishAllocBase(uintptr_t base) {
    local->allocBase.store(base, std::memory_order_relaxed);
    local->allocGeneration.fetch_add(1, std::memory_order_relaxed);
  }

  // Called by sender in GroupEnd: wait for receiver to publish, then read.
  uintptr_t readPeerRecvbuff() {
    ++expectedAddrGen;
    while (remote->addrGeneration.load(std::memory_order_acquire) <
           expectedAddrGen) {
      // Spin — typically < 1μs (both sides enter GroupEnd ~simultaneously).
    }
    return remote->recvbuffAddr.load(std::memory_order_relaxed);
  }

  uint64_t readPeerAllocGeneration() const {
    return remote->allocGeneration.load(std::memory_order_acquire);
  }
  uintptr_t readPeerAllocBase() const {
    return remote->allocBase.load(std::memory_order_relaxed);
  }

  ~ShmSemaphore() {
    if (remoteMapping) munmap(remoteMapping, sizeof(ShmBlock));
    if (localMapping) {
      munmap(localMapping, sizeof(ShmBlock));
      if (!localShmName.empty()) shm_unlink(localShmName.c_str());
    }
  }
};

// Create a ShmSemaphore between rank and peer. Both sides call this
// simultaneously (coordinated via bootstrap send/recv).
static std::unique_ptr<ShmSemaphore> createShmSemaphore(
    mscclpp::Bootstrap* bootstrap, int rank, int peer, int tag) {
  auto sem = std::make_unique<ShmSemaphore>();

  // Create local shm segment.
  sem->localShmName = "/mint_sem_" + std::to_string(getpid()) + "_" +
                      std::to_string(rank) + "_" + std::to_string(peer);
  shm_unlink(sem->localShmName.c_str());  // clean up any stale segment
  int fd = shm_open(sem->localShmName.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) {
    throw mscclpp::Error("shm_open failed for " + sem->localShmName,
                         mscclpp::ErrorCode::SystemError);
  }
  if (ftruncate(fd, sizeof(ShmBlock)) < 0) {
    close(fd);
    throw mscclpp::Error("ftruncate failed", mscclpp::ErrorCode::SystemError);
  }
  sem->localMapping = mmap(nullptr, sizeof(ShmBlock), PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
  close(fd);
  if (sem->localMapping == MAP_FAILED) {
    sem->localMapping = nullptr;
    throw mscclpp::Error("mmap failed", mscclpp::ErrorCode::SystemError);
  }
  sem->local = new (sem->localMapping) ShmBlock{};

  // Exchange shm names with peer.
  std::vector<char> nameData(sem->localShmName.begin(),
                             sem->localShmName.end());
  bootstrap->send(nameData, peer, tag);

  std::vector<char> peerNameData;
  bootstrap->recv(peerNameData, peer, tag);
  std::string peerShmName(peerNameData.begin(), peerNameData.end());

  // Map peer's shm segment.
  int peerFd = shm_open(peerShmName.c_str(), O_RDWR, 0600);
  if (peerFd < 0) {
    throw mscclpp::Error("shm_open failed for peer " + peerShmName,
                         mscclpp::ErrorCode::SystemError);
  }
  sem->remoteMapping = mmap(nullptr, sizeof(ShmBlock), PROT_READ | PROT_WRITE,
                            MAP_SHARED, peerFd, 0);
  close(peerFd);
  if (sem->remoteMapping == MAP_FAILED) {
    sem->remoteMapping = nullptr;
    throw mscclpp::Error("mmap peer failed", mscclpp::ErrorCode::SystemError);
  }
  sem->remote = reinterpret_cast<ShmBlock*>(sem->remoteMapping);

  return sem;
}

// Async worker state for CPU-driven send/recv (both IB and CudaIpc paths).
// Lives on the heap via unique_ptr so that NcclSendRecvPeerContext stays
// movable (std::thread and std::atomic are not).
static constexpr size_t kSendChunkBytes = 256 * 1024;  // 256KB RDMA WR size
// Number of RDMA chunks per D2H copy batch.  A single cudaMemcpyAsync covers
// kD2HBatchChunks * kSendChunkBytes.  Larger batches reduce per-copy overhead
// (~8μs per cudaMemcpyAsync call on the CUDA DMA engine) at the cost of
// coarser pipeline granularity.  16 × 256KB = 4MB per D2H batch.
static constexpr int kD2HBatchChunks = 16;  // 4MB per D2H copy
// Extra bytes appended to staging buffers for an RDMA write progress counter.
static constexpr size_t kProgressCounterPad = 64;
// Signal every Nth IB work request so the CQ doesn't overflow. Unsignaled WRs
// skip the completion queue entirely, dramatically reducing flush() cost.
static constexpr int kSignalEveryN = 64;

struct SendRecvWorkerState {
  struct WorkItem {
    enum Type : uint8_t { SEND = 0, SEND_BATCH = 1 } type;
    size_t bytes;        // SEND: total bytes; SEND_BATCH: unused
    size_t batchOffset;  // SEND_BATCH: byte offset of first chunk in batch
    size_t batchBytes;   // SEND_BATCH: total bytes in this D2H batch
    int numRdmaChunks;  // SEND_BATCH: how many kSendChunkBytes RDMA WRs to post
    int eventIndex;  // index into completionEvents[] (both SEND and SEND_BATCH)
    bool lastBatch;  // SEND_BATCH: flush after this batch
    uint64_t progressVal;  // SEND_BATCH: counter value to write after batch (0
                           // = skip)
  };

  // Lock-free SPSC ring buffer (one producer = caller, one consumer = worker).
  static constexpr uint32_t kCapacity = 256;
  WorkItem ring[kCapacity];
  alignas(64) std::atomic<uint32_t> head{0};
  alignas(64) std::atomic<uint32_t> tail{0};

  std::atomic<bool> stopFlag{false};
  std::thread thread;

  // Per-WorkItem completion events.  Pool must be strictly larger than
  // kCapacity: allocEvent() is called BEFORE push(), so the producer can
  // re-record an event whose old item was just popped but is still being
  // queried by the worker.  2× capacity guarantees no collision.
  static constexpr uint32_t kEventPoolSize = kCapacity * 2;
  cudaEvent_t completionEvents[kEventPoolSize] = {};
  uint32_t nextEventIdx = 0;

  int allocEvent() { return static_cast<int>(nextEventIdx++ % kEventPoolSize); }

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

  // Block until all enqueued items have been consumed by the worker.
  void drain() {
    uint32_t h = head.load(std::memory_order_relaxed);
    while (tail.load(std::memory_order_acquire) < h) {
      std::this_thread::yield();
    }
  }

  ~SendRecvWorkerState() {
    stopFlag.store(true, std::memory_order_release);
    if (thread.joinable()) thread.join();
    for (uint32_t i = 0; i < kEventPoolSize; ++i) {
      if (completionEvents[i]) cudaEventDestroy(completionEvents[i]);
    }
  }
};

struct NcclSendRecvPeerContext {
  int localDevice = -1;
  size_t stagingBytes = 0;  // data capacity (excludes progress counter pad)
  mscclpp::Transport transport = mscclpp::Transport::Unknown;
  bool initialized = false;
  bool isCudaIpc = false;

  // ---------- Common fields ----------
  std::shared_ptr<mscclpp::Host2HostSemaphore> h2hSemaphore;

  // ---------- IB path fields (inter-node) ----------
  std::shared_ptr<char> sendStagingBuffer;  // host pinned
  std::shared_ptr<char> recvStagingBuffer;  // host pinned
  mscclpp::RegisteredMemory sendStagingMemory;
  mscclpp::RegisteredMemory recvStagingMemory;
  mscclpp::RegisteredMemory remoteRecvStagingMemory;
  mscclpp::Connection connection;

  // Monotonic progress counters for RDMA-write-based batch signaling.
  uint64_t sendProgress = 0;
  uint64_t recvExpectedProgress = 0;

  // Fast-path IB fields (cached at init, bypass connection.write() overhead).
  std::shared_ptr<mscclpp::IbQp> ibQp;
  mscclpp::IbMr const* sendStagingMr = nullptr;
  mscclpp::IbMrInfo remoteRecvMrInfo{};
  int ibWrCount = 0;

  // Dedicated CUDA stream for recv H2D copies (PCIe full-duplex overlap).
  cudaStream_t h2dStream = nullptr;
  cudaEvent_t h2dDoneEvent = nullptr;

  // Dedicated CUDA stream for IB send D2H copies.
  cudaStream_t d2hStream = nullptr;

  // Event for cross-stream dependency: record on user stream before send copy
  // so the dedicated stream (d2hStream / ipcStream) waits for prior user work.
  // A pool is used because nccl-tests pipelines iterations without sync —
  // re-recording a single event before the GPU processes the previous WaitEvent
  // would corrupt the dependency.  Pool size = SPSC queue capacity so no event
  // is reused while still being consumed by the dedicated stream.
  static constexpr int kSyncEventPoolSize = SendRecvWorkerState::kCapacity;
  cudaEvent_t syncEventPool[kSyncEventPoolSize] = {};
  int syncEventIdx = 0;
  bool syncEventsCreated = false;

  cudaEvent_t nextSyncEvent() {
    return syncEventPool[syncEventIdx++ % kSyncEventPoolSize];
  }

  // Async worker — destroyed first (declared last) so the thread is joined
  // before the connection / semaphore members are destroyed.
  std::unique_ptr<SendRecvWorkerState> worker;

  // ---------- CudaIpc path fields (intra-node) ----------
  // POSIX-shm based CPU-CPU semaphore for SM-free signaling.
  std::unique_ptr<ShmSemaphore> ipcShmSemaphore;
  // Dedicated stream for D2D copies (fire-and-forget from caller perspective).
  cudaStream_t ipcStream = nullptr;

  // Zero-copy IPC: the peer's recvbuff is directly mapped via CudaIpc so the
  // sender can write to it without staging. The IPC handle covers the entire
  // cudaMalloc allocation. It is exchanged once via bootstrap TCP and cached;
  // per-iteration recvbuff address changes (offset within the same allocation)
  // are communicated through the shm control block (~10ns, no TCP).
  mscclpp::RegisteredMemory remoteRecvbuffRM;
  uintptr_t peerAllocBase = 0;       // peer's allocation base (original addr)
  uintptr_t mappedAllocBase = 0;     // our IPC-mapped base for that allocation
  uint64_t peerAllocGeneration = 0;  // tracks when to re-exchange IPC handle
  // Local recv allocation tracking (to detect when we need to re-register).
  uintptr_t localAllocBase = 0;
  size_t localAllocSize = 0;

  // Compute the IPC-mapped pointer for any address within the peer's
  // allocation.
  void* mapPeerPtr(uintptr_t peerAddr) const {
    return reinterpret_cast<void*>(mappedAllocBase +
                                   (peerAddr - peerAllocBase));
  }

  ~NcclSendRecvPeerContext() {
    worker.reset();
    if (h2dDoneEvent) cudaEventDestroy(h2dDoneEvent);
    if (syncEventsCreated) {
      for (int i = 0; i < kSyncEventPoolSize; ++i) {
        if (syncEventPool[i]) cudaEventDestroy(syncEventPool[i]);
      }
    }
    if (h2dStream) cudaStreamDestroy(h2dStream);
    if (d2hStream) cudaStreamDestroy(d2hStream);
    if (ipcStream) cudaStreamDestroy(ipcStream);
  }
};

inline bool hasIBDevices() { return mscclpp::getIBDeviceCount() > 0; }

inline int sendRecvInitTag(int rank, int worldSize, int peer, int slot) {
  int lo = std::min(rank, peer);
  int hi = std::max(rank, peer);
  int pairIndex = lo * worldSize + hi;
  return kNcclSendRecvInitTagBase + pairIndex * kNcclSendRecvInitTagStride +
         slot;
}

// Tag for runtime recvbuff exchange. The recv side (who knows recvbuff) sends
// a RegisteredMemory to the send side. Direction is always recv→send, so we
// use the ordered pair {sender, receiver} to form a unique tag.
inline int recvbuffExchangeTag(int senderRank, int receiverRank,
                               int worldSize) {
  return kNcclRecvbuffExchangeTagBase + senderRank * worldSize + receiverRank;
}

inline size_t sendRecvStagingBytes() {
  int bytes = mscclpp::env()->ncclSendRecvStagingBytes;
  if (bytes <= 0) {
    throw mscclpp::Error("MSCCLPP_NCCL_SENDRECV_STAGING_BYTES must be positive",
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
    WARN(MSCCLPP_NCCL, std::string(opName),
         " failed: ", std::string(ex.what()));
    return mapMscclppException(ex);
  } catch (...) {
    WARN(MSCCLPP_NCCL, std::string(opName),
         " failed with an unknown exception");
    return ncclInternalError;
  }
}

// Worker thread: spins on the SPSC queue, issues IB operations on the CPU.
// Uses fast-path IB calls (cached MR info, direct QP access) to bypass
// Connection::write() overhead (validation, transport lookup, weak_ptr lock).
static void sendRecvWorkerLoop(NcclSendRecvPeerContext* ctx) {
  auto* ws = ctx->worker.get();
  if (ctx->localDevice >= 0) {
    cudaSetDevice(ctx->localDevice);
  }

  auto* qp = ctx->ibQp.get();
  auto* srcMr = ctx->sendStagingMr;
  auto& dstMrInfo = ctx->remoteRecvMrInfo;

  // Helper: stage a write WR, controlling signaled flag.
  auto stageWrite = [&](uint64_t srcOff, uint64_t dstOff,
                        uint32_t size) -> bool {
    ctx->ibWrCount++;
    bool signaled = (ctx->ibWrCount % kSignalEveryN == 0);
    qp->stageSendWrite(srcMr, dstMrInfo, size, /*wrId=*/0, srcOff, dstOff,
                       signaled);
    return signaled;
  };

  // Helper: flush all outstanding signaled CQ entries.
  auto flushCq = [&]() {
    while (qp->getNumSendCqItems() > 0) {
      int wcNum = qp->pollSendCq();
      if (wcNum < 0) {
        WARN(MSCCLPP_NCCL, "sendRecvWorker: pollSendCq error");
        return;
      }
      for (int i = 0; i < wcNum; ++i) {
        int status = qp->getSendWcStatus(i);
        if (status != 0) {
          WARN(MSCCLPP_NCCL,
               "sendRecvWorker: IB WR failed: ", qp->getSendWcStatusString(i));
        }
      }
    }
  };

  SendRecvWorkerState::WorkItem item;
  while (!ws->stopFlag.load(std::memory_order_acquire)) {
    if (!ws->pop(item)) continue;  // spin

    try {
      if (item.type == SendRecvWorkerState::WorkItem::SEND) {
        while (cudaEventQuery(ws->completionEvents[item.eventIndex]) ==
               cudaErrorNotReady) {
          std::this_thread::yield();
        }
        ctx->ibWrCount = kSignalEveryN - 1;
        stageWrite(0, 0, static_cast<uint32_t>(item.bytes));
        qp->postSend();
        ctx->h2hSemaphore->signal();
        flushCq();
      } else {
        // SEND_BATCH: wait for the batch D2H event, then post multiple RDMA
        // writes (one per kSendChunkBytes sub-chunk) for better NIC pipelining.
        while (cudaEventQuery(ws->completionEvents[item.eventIndex]) ==
               cudaErrorNotReady) {
          std::this_thread::yield();
        }
        bool signaled = false;
        size_t off = item.batchOffset;
        size_t remaining = item.batchBytes;
        for (int c = 0; c < item.numRdmaChunks; ++c) {
          uint32_t chunkSz =
              static_cast<uint32_t>(std::min(remaining, kSendChunkBytes));
          signaled = stageWrite(off, off, chunkSz) || signaled;
          off += chunkSz;
          remaining -= chunkSz;
        }
        if (item.progressVal > 0) {
          size_t cOff = ctx->stagingBytes;
          *reinterpret_cast<uint64_t volatile*>(ctx->sendStagingBuffer.get() +
                                                cOff) = item.progressVal;
          signaled = stageWrite(cOff, cOff, sizeof(uint64_t)) || signaled;
        }
        qp->postSend();

        if (item.lastBatch) {
          if (!signaled) {
            qp->stageSendWrite(srcMr, dstMrInfo, 0, /*wrId=*/0, 0, 0,
                               /*signaled=*/true);
            qp->postSend();
          }
          flushCq();
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
  if (peersShareNode(comm, peer)) {
    // Intra-node: use CudaIpc for data, IB for signaling.
    // Falls back to Unknown if IB is unavailable (→ dlopen fallback).
    return hasIBDevices() ? mscclpp::Transport::CudaIpc
                          : mscclpp::Transport::Unknown;
  }
  if (!hasIBDevices()) {
    return mscclpp::Transport::Unknown;
  }
  // Topology-aware: pick the IB HCA on the same NUMA node as this GPU.
  mscclpp::Transport t = selectIBTransportForGpu(comm->cudaDevice);
  if (t == mscclpp::Transport::Unknown) {
    throw mscclpp::Error(
        "No usable IB transport for GPU " + std::to_string(comm->cudaDevice),
        mscclpp::ErrorCode::InvalidUsage);
  }
  return t;
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
        "transport or intra-node CudaIpc",
        mscclpp::ErrorCode::InvalidUsage);
  }

  ctx.isCudaIpc = (ctx.transport == mscclpp::Transport::CudaIpc);

  if (ctx.isCudaIpc) {
    // --- CudaIpc intra-node path (SM-free, zero-copy) ---
    // Data: cudaMemcpyAsync D2D (DMA engine, no SM).
    // Signaling: POSIX shm semaphore (CPU-CPU, no SM).
    // recvbuff IPC handles are exchanged in GroupEnd on first call.
    ctx.ipcShmSemaphore =
        createShmSemaphore(comm->comm->bootstrap().get(), rank, peer,
                           sendRecvInitTag(rank, comm->worldSize, peer, 3));

    MSCCLPP_CUDATHROW(
        cudaStreamCreateWithFlags(&ctx.ipcStream, cudaStreamNonBlocking));
    // Reuse d2hStream for event-based worker sync (same pattern as IB path).
    MSCCLPP_CUDATHROW(
        cudaStreamCreateWithFlags(&ctx.d2hStream, cudaStreamNonBlocking));
    for (int i = 0; i < NcclSendRecvPeerContext::kSyncEventPoolSize; ++i) {
      MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.syncEventPool[i],
                                                 cudaEventDisableTiming));
    }
    ctx.syncEventsCreated = true;

    // Worker thread: polls cudaEventQuery after D2D copy, then signals shm sem.
    ctx.worker = std::make_unique<SendRecvWorkerState>();
    for (uint32_t i = 0; i < SendRecvWorkerState::kEventPoolSize; ++i) {
      MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(
          &ctx.worker->completionEvents[i], cudaEventDisableTiming));
    }
    ctx.worker->thread = std::thread([&ctx]() {
      if (ctx.localDevice >= 0) cudaSetDevice(ctx.localDevice);
      SendRecvWorkerState::WorkItem item;
      auto* ws = ctx.worker.get();
      while (!ws->stopFlag.load(std::memory_order_acquire)) {
        if (!ws->pop(item)) continue;
        // Wait for the D2D copy event to complete.
        while (cudaEventQuery(ws->completionEvents[item.eventIndex]) ==
               cudaErrorNotReady) {
          std::this_thread::yield();
        }
        // Signal the peer's shm semaphore — data is in peer's recvbuff.
        ctx.ipcShmSemaphore->signal();
      }
    });
  } else {
    // --- IB inter-node path (existing) ---
    size_t allocBytes = ctx.stagingBytes + kProgressCounterPad;
    ctx.sendStagingBuffer =
        mscclpp::detail::gpuCallocHostShared<char>(allocBytes, 0);
    ctx.recvStagingBuffer =
        mscclpp::detail::gpuCallocHostShared<char>(allocBytes, 0);

    mscclpp::TransportFlags transportFlags(ctx.transport);
    ctx.sendStagingMemory = comm->comm->registerMemory(
        ctx.sendStagingBuffer.get(), allocBytes, transportFlags);
    ctx.recvStagingMemory = comm->comm->registerMemory(
        ctx.recvStagingBuffer.get(), allocBytes, transportFlags);

    mscclpp::EndpointConfig::Ib ibCfg;
    ibCfg.maxCqPollNum = 32;
    mscclpp::EndpointConfig endpointConfig(
        ctx.transport, mscclpp::Device(mscclpp::DeviceType::CPU),
        /*maxWriteQueueSize=*/-1, ibCfg);
    auto connectionFuture = comm->comm->connect(
        endpointConfig, peer, sendRecvInitTag(rank, comm->worldSize, peer, 0));
    comm->comm->sendMemory(ctx.recvStagingMemory, peer,
                           sendRecvInitTag(rank, comm->worldSize, peer, 1));
    auto remoteRecvStagingFuture = comm->comm->recvMemory(
        peer, sendRecvInitTag(rank, comm->worldSize, peer, 1));

    ctx.connection = connectionFuture.get();
    auto semaphore =
        comm->comm
            ->buildSemaphore(ctx.connection, peer,
                             sendRecvInitTag(rank, comm->worldSize, peer, 2))
            .get();
    ctx.remoteRecvStagingMemory = remoteRecvStagingFuture.get();
    ctx.h2hSemaphore = std::make_shared<mscclpp::Host2HostSemaphore>(semaphore);

    ctx.ibQp = ctx.connection.getIbQp();
    if (!ctx.ibQp) {
      throw mscclpp::Error("Failed to get IB QP from connection",
                           mscclpp::ErrorCode::InternalError);
    }
    ctx.sendStagingMemory.getIbMrInfo(ctx.transport, &ctx.sendStagingMr,
                                      nullptr);
    // Remote memory is registered with the PEER's IB transport, which may
    // differ from ours (e.g., local=IB1, peer=IB0). Query with their transport.
    // Assumes symmetric NUMA topology across nodes.
    int peerCudaDevice = peer % comm->nRanksPerNode;
    mscclpp::Transport peerIbTransport =
        selectIBTransportForGpu(peerCudaDevice);
    ctx.remoteRecvStagingMemory.getIbMrInfo(peerIbTransport, nullptr,
                                            &ctx.remoteRecvMrInfo);

    MSCCLPP_CUDATHROW(
        cudaStreamCreateWithFlags(&ctx.h2dStream, cudaStreamNonBlocking));
    MSCCLPP_CUDATHROW(
        cudaEventCreateWithFlags(&ctx.h2dDoneEvent, cudaEventDisableTiming));

    // Separate D2H stream for IB sends.
    MSCCLPP_CUDATHROW(
        cudaStreamCreateWithFlags(&ctx.d2hStream, cudaStreamNonBlocking));
    for (int i = 0; i < NcclSendRecvPeerContext::kSyncEventPoolSize; ++i) {
      MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&ctx.syncEventPool[i],
                                                 cudaEventDisableTiming));
    }
    ctx.syncEventsCreated = true;

    ctx.worker = std::make_unique<SendRecvWorkerState>();
    for (uint32_t i = 0; i < SendRecvWorkerState::kEventPoolSize; ++i) {
      MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(
          &ctx.worker->completionEvents[i], cudaEventDisableTiming));
    }
    ctx.worker->thread = std::thread(sendRecvWorkerLoop, &ctx);
  }

  ctx.initialized = true;
}

static NcclSendRecvPeerContext& getSendRecvPeerContext(ncclComm_t comm,
                                                       int peer,
                                                       int localDevice) {
  std::lock_guard<std::mutex> lock(comm->sendRecvMutex);
  auto [it, inserted] = comm->sendRecvPeerContexts.try_emplace(peer);
  if (!it->second.initialized) {
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

// Dispatch one round of the IB batched D2H→RDMA pipeline.
// Enqueues D2H copies on d2hStream and pushes SEND_BATCH items to the worker
// SPSC queue.  This function is non-blocking — the worker processes RDMA
// asynchronously.
//   src       — device pointer to the source region for this round
//   staging   — host staging buffer base (always offset 0 for each round)
//   roundBytes — bytes to transfer in this round (≤ stagingBytes)
//   isLastRound — controls whether the worker issues a CQ flush
static void dispatchIBSendRound(NcclSendRecvPeerContext& peerCtx,
                                char const* src, size_t roundBytes,
                                bool isLastRound) {
  auto* ws = peerCtx.worker.get();
  cudaStream_t d2h = peerCtx.d2hStream;
  char* staging = peerCtx.sendStagingBuffer.get();

  if (roundBytes <= kSendChunkBytes) {
    int eIdx = ws->allocEvent();
    MSCCLPP_CUDATHROW(
        cudaMemcpyAsync(staging, src, roundBytes, cudaMemcpyDeviceToHost, d2h));
    MSCCLPP_CUDATHROW(cudaEventRecord(ws->completionEvents[eIdx], d2h));
    cudaStreamQuery(d2h);
    ws->push({SendRecvWorkerState::WorkItem::SEND, roundBytes, 0, 0, 0, eIdx,
              false, 0});
  } else {
    int totalChunks =
        static_cast<int>((roundBytes + kSendChunkBytes - 1) / kSendChunkBytes);
    int adaptiveBatchChunks =
        std::min(kD2HBatchChunks, std::max(1, totalChunks / 4));
    size_t batchSize =
        static_cast<size_t>(adaptiveBatchChunks) * kSendChunkBytes;
    int numBatches = static_cast<int>((roundBytes + batchSize - 1) / batchSize);
    for (int b = 0; b < numBatches; ++b) {
      size_t off = static_cast<size_t>(b) * batchSize;
      size_t batchBytes = std::min(batchSize, roundBytes - off);
      int rdmaChunks = static_cast<int>((batchBytes + kSendChunkBytes - 1) /
                                        kSendChunkBytes);
      int eIdx = ws->allocEvent();
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(staging + off, src + off, batchBytes,
                                        cudaMemcpyDeviceToHost, d2h));
      MSCCLPP_CUDATHROW(cudaEventRecord(ws->completionEvents[eIdx], d2h));
      cudaStreamQuery(d2h);
      bool isLast = (b == numBatches - 1);
      uint64_t pv = ++peerCtx.sendProgress;
      ws->push({SendRecvWorkerState::WorkItem::SEND_BATCH, 0, off, batchBytes,
                rdmaChunks, eIdx, isLast, pv});
    }
  }
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

  // Check if the custom path can handle this request.
  bool canHandle = comm->hasIB;
  if (!canHandle) {
    if (mscclppNcclDlopenSharedLib == true) {
      return mscclppNcclOps.Send(
          sendbuff, count, datatype, peer,
          *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
    }
    WARN(MSCCLPP_NCCL,
         "CPU-driven ncclSend requires IB and message size "
         "<= ",
         stagingBytes, " bytes");
    return ncclInvalidUsage;
  }

  return runNcclGuarded("ncclSend", [&]() {
    mscclpp::CudaDeviceGuard deviceGuard(localDevice);
    auto& peerCtx = getSendRecvPeerContext(comm, peer, localDevice);

    if (peerCtx.isCudaIpc) {
      // --- CudaIpc intra-node send (SM-free, zero-copy) ---
      // Ensure prior work on user stream completes before reading sendbuff.
      cudaEvent_t syncEvt = peerCtx.nextSyncEvent();
      MSCCLPP_CUDATHROW(cudaEventRecord(syncEvt, stream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(peerCtx.ipcStream, syncEvt, 0));
      // The destination is computed from the peer's recvbuff address (published
      // to shm in GroupEnd Phase 2.5) and our cached allocation mapping.
      uintptr_t peerAddr = peerCtx.ipcShmSemaphore->remote->recvbuffAddr.load(
          std::memory_order_relaxed);
      void* remoteDst = peerCtx.mapPeerPtr(peerAddr);
      auto* ws = peerCtx.worker.get();
      int eIdx = ws->allocEvent();
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(remoteDst, sendbuff, bytes,
                                        cudaMemcpyDeviceToDevice,
                                        peerCtx.ipcStream));
      MSCCLPP_CUDATHROW(
          cudaEventRecord(ws->completionEvents[eIdx], peerCtx.ipcStream));
      cudaStreamQuery(peerCtx.ipcStream);  // flush command buffer
      ws->push({SendRecvWorkerState::WorkItem::SEND, bytes, 0, 0, 0, eIdx,
                false, 0});
    } else {
      // --- IB inter-node send ---
      // Ensure prior work on user stream completes before reading sendbuff.
      cudaStream_t d2h = peerCtx.d2hStream;
      cudaEvent_t syncEvt = peerCtx.nextSyncEvent();
      MSCCLPP_CUDATHROW(cudaEventRecord(syncEvt, stream));
      MSCCLPP_CUDATHROW(cudaStreamWaitEvent(d2h, syncEvt, 0));
      size_t stagingCap = peerCtx.stagingBytes;
      int numRounds = static_cast<int>((bytes + stagingCap - 1) / stagingCap);
      char const* src = static_cast<char const*>(sendbuff);
      for (int r = 0; r < numRounds; ++r) {
        if (r > 0) {
          // Wait for the worker to finish all RDMA of the previous round,
          // then wait for the receiver to signal that it has consumed the
          // staging buffer (H2D complete).
          peerCtx.worker->drain();
          peerCtx.h2hSemaphore->wait(/*maxSpinCount=*/-1);
        }
        size_t roundOff = static_cast<size_t>(r) * stagingCap;
        size_t roundBytes = std::min(stagingCap, bytes - roundOff);
        bool isLastRound = (r == numRounds - 1);
        dispatchIBSendRound(peerCtx, src + roundOff, roundBytes, isLastRound);
      }
    }
  });
}

// Host callback for cudaLaunchHostFunc: spin-waits on the shm semaphore.
// Pure host memory polling — no CUDA API calls (compliant with CUDA callback
// restrictions).
static void CUDART_CB shmSemaphoreWaitCallback(void* arg) {
  static_cast<ShmSemaphore*>(arg)->wait();
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

  bool canHandle = comm->hasIB;
  if (!canHandle) {
    if (mscclppNcclDlopenSharedLib == true) {
      return mscclppNcclOps.Recv(
          recvbuff, count, datatype, peer,
          *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
    }
    WARN(MSCCLPP_NCCL,
         "CPU-driven ncclRecv requires IB and message size "
         "<= ",
         stagingBytes, " bytes");
    return ncclInvalidUsage;
  }

  return runNcclGuarded("ncclRecv", [&]() {
    mscclpp::CudaDeviceGuard deviceGuard(localDevice);
    auto& peerCtx = getSendRecvPeerContext(comm, peer, localDevice);

    if (peerCtx.isCudaIpc) {
      // --- CudaIpc intra-node recv (SM-free, zero-copy) ---
      // Data is written directly to recvbuff by the sender. We enqueue a host
      // callback on the user stream that spin-waits for the shm semaphore.
      // cudaStreamSynchronize(stream) will block until the callback returns,
      // i.e., until the sender has finished writing.
      MSCCLPP_CUDATHROW(cudaLaunchHostFunc(stream, shmSemaphoreWaitCallback,
                                           peerCtx.ipcShmSemaphore.get()));
    } else {
      // --- IB inter-node recv ---
      if (bytes <= kSendChunkBytes) {
        peerCtx.h2hSemaphore->wait(/*maxSpinCount=*/-1);
        MSCCLPP_CUDATHROW(
            cudaMemcpyAsync(recvbuff, peerCtx.recvStagingBuffer.get(), bytes,
                            cudaMemcpyHostToDevice, stream));
      } else {
        size_t stagingCap = peerCtx.stagingBytes;
        int numRounds = static_cast<int>((bytes + stagingCap - 1) / stagingCap);
        char* dst = static_cast<char*>(recvbuff);
        char* staging = peerCtx.recvStagingBuffer.get();
        volatile uint64_t* counterPtr =
            reinterpret_cast<volatile uint64_t*>(staging + stagingCap);

        for (int r = 0; r < numRounds; ++r) {
          size_t roundOff = static_cast<size_t>(r) * stagingCap;
          size_t roundBytes = std::min(stagingCap, bytes - roundOff);

          // Compute batching parameters for this round.
          int totalChunks = static_cast<int>(
              (roundBytes + kSendChunkBytes - 1) / kSendChunkBytes);
          int adaptiveBatchChunks =
              std::min(kD2HBatchChunks, std::max(1, totalChunks / 4));
          size_t batchSize =
              static_cast<size_t>(adaptiveBatchChunks) * kSendChunkBytes;
          int numBatches =
              static_cast<int>((roundBytes + batchSize - 1) / batchSize);

          for (int b = 0; b < numBatches; ++b) {
            uint64_t expected = ++peerCtx.recvExpectedProgress;
            while (*counterPtr < expected) {
            }
            // Staging offset is relative to buffer start (reused each round).
            size_t stagingOff = static_cast<size_t>(b) * batchSize;
            size_t batchBytes = std::min(batchSize, roundBytes - stagingOff);
            MSCCLPP_CUDATHROW(cudaMemcpyAsync(
                dst + roundOff + stagingOff, staging + stagingOff, batchBytes,
                cudaMemcpyHostToDevice, peerCtx.h2dStream));
          }

          if (r < numRounds - 1) {
            // Ensure all H2D copies for this round complete before telling the
            // sender it may reuse the staging buffer.
            MSCCLPP_CUDATHROW(cudaStreamSynchronize(peerCtx.h2dStream));
            peerCtx.h2hSemaphore->signal();
          }
        }
        MSCCLPP_CUDATHROW(
            cudaEventRecord(peerCtx.h2dDoneEvent, peerCtx.h2dStream));
        MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, peerCtx.h2dDoneEvent, 0));
      }
    }
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
  if (gNcclGroupDepth > 0) {
    // Always queue when in a group to ensure peer contexts are initialized
    // in a coordinated order (avoiding bootstrap deadlocks).
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
  if (gNcclGroupDepth > 0) {
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
  // Always use our own group tracking.  Operations are queued and
  // peer contexts initialized in sorted order at GroupEnd to avoid
  // bootstrap handshake deadlocks in multi-peer scenarios.
  gNcclGroupDepth++;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGroupEnd() {
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

  // Phase 1: Pre-initialize all needed peer contexts (sorted by peer rank
  // to ensure deterministic ordering across all ranks, preventing bootstrap
  // handshake deadlocks).
  {
    std::set<int> neededPeers;
    ncclComm_t firstComm = nullptr;
    int localDevice = -1;
    for (auto const& op : ops) {
      neededPeers.insert(op.peer);
      if (!firstComm) {
        firstComm = op.comm;
        localDevice = op.comm->cudaDevice;
      }
    }
    if (firstComm) {
      for (int peer : neededPeers) {
        try {
          getSendRecvPeerContext(firstComm, peer, localDevice);
        } catch (...) {
          // Peer may not support custom path — will use dlopen fallback.
        }
      }
    }
  }

  // Phase 2: Separate operations into custom (handled by us) vs dlopen.
  std::vector<GroupedP2POp> dlopenOps;
  std::vector<GroupedP2POp> customOps;
  for (auto const& op : ops) {
    size_t bytes = op.count * ncclTypeSize(op.datatype);
    bool canHandle = op.comm->hasIB && bytes > 0;
    if (canHandle) {
      // Check if the peer context was actually initialized (transport !=
      // Unknown)
      try {
        auto& ctx =
            getSendRecvPeerContext(op.comm, op.peer, op.comm->cudaDevice);
        (void)ctx;
        customOps.push_back(op);
      } catch (...) {
        dlopenOps.push_back(op);
      }
    } else if (bytes == 0) {
      // Skip zero-byte ops
    } else {
      dlopenOps.push_back(op);
    }
  }

  // Phase 2.5: CudaIpc recvbuff address exchange (zero-copy send support).
  //
  // The IPC handle covers the entire cudaMalloc allocation and is exchanged
  // only ONCE per allocation via bootstrap TCP (~35μs). Per-iteration
  // recvbuff address changes (offsets within the same allocation, e.g. from
  // nccl-tests) are communicated through the shm control block (~10ns).
  //
  // Protocol:
  //   Recv side → publishes recvbuff addr to shm; registers allocation on
  //               first call or when the underlying cudaMalloc changes.
  //   Send side → reads peer's shm; receives IPC handle only when the peer's
  //               allocation generation has advanced; computes mapped ptr.
  {
    // (a) Recv ops: publish recvbuff address and (if needed) IPC handle.
    for (auto const& op : customOps) {
      auto& ctx = getSendRecvPeerContext(op.comm, op.peer, op.comm->cudaDevice);
      if (!ctx.isCudaIpc) continue;
      if (op.kind != GroupedP2POpKind::Recv) continue;

      uintptr_t addr = reinterpret_cast<uintptr_t>(op.recvbuff);

      // Check if the allocation changed (or if this is the first call).
      bool needExchange = false;
      if (ctx.localAllocBase == 0 || addr < ctx.localAllocBase ||
          addr >= ctx.localAllocBase + ctx.localAllocSize) {
        // New allocation — get base + size via CUDA driver API.
        CUdeviceptr base = 0;
        size_t allocSize = 0;
        CUresult res = cuMemGetAddressRange(&base, &allocSize,
                                            static_cast<CUdeviceptr>(addr));
        if (res != CUDA_SUCCESS) {
          throw mscclpp::Error("cuMemGetAddressRange failed",
                               mscclpp::ErrorCode::SystemError);
        }
        ctx.localAllocBase = static_cast<uintptr_t>(base);
        ctx.localAllocSize = allocSize;
        needExchange = true;
      }

      if (needExchange) {
        // Register the FULL allocation (not just the recvbuff slice) so the
        // peer can map any offset within it.
        int rank = op.comm->comm->bootstrap()->getRank();
        int tag = recvbuffExchangeTag(rank, op.peer, op.comm->worldSize);
        mscclpp::TransportFlags ipcFlags(mscclpp::Transport::CudaIpc);
        auto rm = op.comm->comm->registerMemory(
            reinterpret_cast<void*>(ctx.localAllocBase), ctx.localAllocSize,
            ipcFlags);
        op.comm->comm->sendMemory(rm, op.peer, tag);

        // Publish alloc base to shm (sender reads it for offset computation).
        ctx.ipcShmSemaphore->publishAllocBase(ctx.localAllocBase);
      }

      // Always publish the current recvbuff address to shm.
      ctx.ipcShmSemaphore->publishRecvbuff(addr);
    }

    // (b) Send ops: read peer's recvbuff address from shm; receive IPC handle
    //     if the peer's allocation generation advanced.
    for (auto const& op : customOps) {
      auto& ctx = getSendRecvPeerContext(op.comm, op.peer, op.comm->cudaDevice);
      if (!ctx.isCudaIpc) continue;
      if (op.kind != GroupedP2POpKind::Send) continue;

      auto* sem = ctx.ipcShmSemaphore.get();

      // Wait for the peer to publish its recvbuff address (spin is typically
      // < 1μs since both sides enter GroupEnd ~simultaneously).
      uintptr_t peerAddr = sem->readPeerRecvbuff();

      // Check if the peer's allocation changed.
      uint64_t peerAllocGen = sem->readPeerAllocGeneration();
      if (peerAllocGen != ctx.peerAllocGeneration) {
        ctx.peerAllocGeneration = peerAllocGen;
        // Peer registered a new allocation — receive IPC handle.
        int rank = op.comm->comm->bootstrap()->getRank();
        int tag = recvbuffExchangeTag(op.peer, rank, op.comm->worldSize);
        auto rmFuture = op.comm->comm->recvMemory(op.peer, tag);
        ctx.remoteRecvbuffRM = rmFuture.get();
        ctx.mappedAllocBase =
            reinterpret_cast<uintptr_t>(ctx.remoteRecvbuffRM.data());
        ctx.peerAllocBase = sem->readPeerAllocBase();
      }
    }
  }

  // Phase 3: Execute custom operations.
  //
  // Multi-round sends (bytes > stagingBytes) block between rounds waiting for
  // receiver acks.  If dispatched sequentially on the main thread, a send
  // blocking on an ack could prevent the local recv dispatch from starting,
  // while the remote recv (which produces the ack) hasn't started either →
  // deadlock.  To avoid this, multi-round sends are dispatched on dedicated
  // threads so the main thread can proceed to recv dispatch concurrently.
  {
    std::vector<std::thread> sendThreads;
    std::vector<ncclResult_t> sendResults;
    std::mutex sendResultsMu;
    std::vector<GroupedP2POp> recvOps;

    for (auto const& op : customOps) {
      if (op.kind == GroupedP2POpKind::Send) {
        size_t opBytes = op.count * ncclTypeSize(op.datatype);
        size_t stagingCap = op.comm->sendRecvStagingBytesCached;
        if (opBytes > stagingCap) {
          // Large send — run on a separate thread to avoid deadlock.
          sendResults.push_back(ncclSuccess);
          size_t rIdx = sendResults.size() - 1;
          sendThreads.emplace_back(
              [&sendResults, &sendResultsMu, rIdx](
                  void const* buf, size_t cnt, ncclDataType_t dt, int p,
                  ncclComm_t c, cudaStream_t s) {
                ncclResult_t r = executeNcclSendImpl(buf, cnt, dt, p, c, s);
                if (r != ncclSuccess) {
                  std::lock_guard<std::mutex> lk(sendResultsMu);
                  sendResults[rIdx] = r;
                }
              },
              op.sendbuff, op.count, op.datatype, op.peer, op.comm, op.stream);
        } else {
          // Small send — non-blocking, dispatch inline.
          ncclResult_t result = executeNcclSendImpl(
              op.sendbuff, op.count, op.datatype, op.peer, op.comm, op.stream);
          if (result != ncclSuccess) {
            WARN(MSCCLPP_NCCL, "ncclGroupEnd custom send failed for peer ",
                 op.peer);
            return result;
          }
        }
      } else {
        recvOps.push_back(op);
      }
    }

    // Dispatch all recvs on the main thread (blocking polls are normal).
    for (auto const& op : recvOps) {
      ncclResult_t result = executeNcclRecvImpl(
          op.recvbuff, op.count, op.datatype, op.peer, op.comm, op.stream);
      if (result != ncclSuccess) {
        WARN(MSCCLPP_NCCL, "ncclGroupEnd custom recv failed for peer ",
             op.peer);
        for (auto& t : sendThreads) t.join();
        return result;
      }
    }

    // Wait for all large-send threads to complete.
    for (auto& t : sendThreads) t.join();
    for (auto r : sendResults) {
      if (r != ncclSuccess) return r;
    }
  }

  // Phase 4: Execute dlopen fallback operations, wrapped in real NCCL group.
  if (!dlopenOps.empty()) {
    if (!mscclppNcclDlopenSharedLib) {
      WARN(MSCCLPP_NCCL, "ncclGroupEnd has ", dlopenOps.size(),
           " operations requiring NCCL fallback but dlopen not loaded");
      return ncclInvalidUsage;
    }
    mscclppNcclOps.GroupStart();
    for (auto const& op : dlopenOps) {
      ncclResult_t result;
      if (op.kind == GroupedP2POpKind::Send) {
        result = mscclppNcclOps.Send(
            op.sendbuff, op.count, op.datatype, op.peer,
            *reinterpret_cast<ncclComm_t*>(op.comm->mscclppNcclComm),
            op.stream);
      } else {
        result = mscclppNcclOps.Recv(
            op.recvbuff, op.count, op.datatype, op.peer,
            *reinterpret_cast<ncclComm_t*>(op.comm->mscclppNcclComm),
            op.stream);
      }
      if (result != ncclSuccess) {
        WARN(MSCCLPP_NCCL, "ncclGroupEnd dlopen ",
             (op.kind == GroupedP2POpKind::Send ? "send" : "recv"),
             " failed for peer ", op.peer);
        mscclppNcclOps.GroupEnd();
        return result;
      }
    }
    ncclResult_t result = mscclppNcclOps.GroupEnd();
    if (result != ncclSuccess) return result;
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
