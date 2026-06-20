// NodeExchangeBuffer: send/recv slab management for inter-node AllGather.
//
// Encapsulates the two host-memory staging buffers used by multi-node AllGather:
//   - sendSlab: each local rank D2H-stages its shard here; leader RDMA-writes to
//               remote recvSlab.
//   - recvSlab: receives RDMA data from remote nodes; H2D-scattered to recvbuff.
//
// Unlike HostStagingBuffer (GPU-driven intra-node), NodeExchangeBuffer uses a
// CPU-visible completion model because RDMA (IB verbs) is CPU-driven:
//
//   push(stream, src, slabOffset, size, rank, tag)
//     Async D2H of src[0..size) into sendSlab[slabOffset..+size], then writes
//     d2hCtrl[rank] = tag via cuStreamWriteValue64 on the same stream.
//     Non-blocking for the caller — GPU writes the flag after D2H.
//
//   waitCpu(rank, tag)
//     CPU spin until d2hCtrl[rank] >= tag, with acquire fence on return.
//     Called by the leader to confirm all local D2H completions before RDMA.
//
// The design replaces the CPU-blocking waitForCudaStream + atomic.store pattern
// with a GPU-async flag, eliminating CPU stalls on non-leader ranks entirely and
// reducing leader latency from "GPU stream drain + atomic" to "GPU flag write".
//
// RDMA coordination (rdmaReady, rdmaSignal, ackReady, ackSignal, pipeReady) is
// separate (HostControl / RdmaControl struct) and not part of this abstraction.

#pragma once

#include "lite_common.h"
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <fcntl.h>
#include <numa.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>

// ── Layout constants ─────────────────────────────────────────────────────────
static constexpr int kNebMaxRanks = 8;  // max local ranks per node

// ── D2H completion flags (POSIX shm, device-mapped) ─────────────────────────
// One volatile flag per local rank; GPU writes via cuStreamWriteValue64,
// CPU reads via waitCpu() spin loop.
struct NebCtrl {
  alignas(64) volatile uint64_t d2hReady[kNebMaxRanks];
};
static_assert(sizeof(NebCtrl) <= 4096, "NebCtrl fits in one page");

// ── NodeExchangeBuffer ───────────────────────────────────────────────────────
class NodeExchangeBuffer {
 public:
  // Create the buffer collectively (all nRanks must call simultaneously).
  // isOwner: true on the node-group leader rank that creates the POSIX shm.
  // localLeader: the rank (in [0, nRanks)) whose POSIX shm names this rank
  //   should map.  For a 2nx4g setup, l40 ranks use localLeader=0 and l41
  //   ranks use localLeader=4, so each node only maps its own node's shm.
  // slabBytes: size for BOTH sendSlab and recvSlab.
  // nameTag: unique string per context (used to form POSIX shm names).
  static NodeExchangeBuffer create(
      std::shared_ptr<mscclpp::Communicator> bootstrapComm,
      int rank, int nRanks,
      bool isOwner, int localLeader,
      size_t slabBytes,
      int numaNode, int cudaDevice,
      std::string const& nameTag);

  // Non-copyable, moveable.
  NodeExchangeBuffer(NodeExchangeBuffer const&) = delete;
  NodeExchangeBuffer& operator=(NodeExchangeBuffer const&) = delete;

  // Move: swap all fields so moved-from object ends up in default (empty) state.
  NodeExchangeBuffer(NodeExchangeBuffer&& o) noexcept { swap_(o); }
  NodeExchangeBuffer& operator=(NodeExchangeBuffer&& o) noexcept {
    if (this != &o) {
      this->~NodeExchangeBuffer();
      new (this) NodeExchangeBuffer{};
      swap_(o);
    }
    return *this;
  }
  ~NodeExchangeBuffer();

  // ── Core API ──────────────────────────────────────────────────────────────

  // Async D2H: src[0..size) → sendSlab[slabOffset..+size], then GPU writes
  // d2hCtrl[rank] = tag on stream (after D2H completes, by stream ordering).
  void push(cudaStream_t stream,
            void const* src, size_t slabOffset, size_t size,
            int rank, uint64_t tag) const;

  // CPU spin until d2hCtrl[rank] >= tag, then issue acquire fence.
  // Call on the leader after all local ranks have called push().
  void waitCpu(int rank, uint64_t tag) const;

  // ── Accessors ─────────────────────────────────────────────────────────────

  // Host pointer into sendSlab at byte offset.
  char* sendPtr(size_t offset = 0) const { return sendSlab_ + offset; }

  // Device-mapped pointer into sendSlab (nullptr if mapping failed).
  char const* sendDevicePtr() const { return sendDevice_; }

  // Host pointer into recvSlab at byte offset.
  char* recvPtr(size_t offset = 0) const { return recvSlab_ + offset; }

  size_t slabBytes() const { return slabBytes_; }

 private:
  NodeExchangeBuffer() = default;

  void swap_(NodeExchangeBuffer& o) noexcept {
    std::swap(sendMapping_,    o.sendMapping_);
    std::swap(sendSlab_,       o.sendSlab_);
    std::swap(sendDevice_,     o.sendDevice_);
    std::swap(sendRegistered_, o.sendRegistered_);
    std::swap(recvMapping_,    o.recvMapping_);
    std::swap(recvSlab_,       o.recvSlab_);
    std::swap(recvRegistered_, o.recvRegistered_);
    std::swap(ctrlMapping_,    o.ctrlMapping_);
    std::swap(ctrl_,           o.ctrl_);
    std::swap(ctrlDevice_,     o.ctrlDevice_);
    std::swap(ctrlRegistered_, o.ctrlRegistered_);
    std::swap(slabBytes_,      o.slabBytes_);
    std::swap(isOwner_,        o.isOwner_);
    std::swap(sendName_,       o.sendName_);
    std::swap(recvName_,       o.recvName_);
    std::swap(ctrlName_,       o.ctrlName_);
  }

  // sendSlab
  void*  sendMapping_   = nullptr;
  char*  sendSlab_      = nullptr;
  char*  sendDevice_    = nullptr;
  bool   sendRegistered_ = false;

  // recvSlab
  void*  recvMapping_   = nullptr;
  char*  recvSlab_      = nullptr;
  bool   recvRegistered_ = false;

  // D2H ctrl flags (device-mapped)
  void*    ctrlMapping_    = nullptr;
  NebCtrl* ctrl_           = nullptr;
  char*    ctrlDevice_     = nullptr;
  bool     ctrlRegistered_ = false;

  size_t slabBytes_ = 0;
  bool   isOwner_   = false;
  std::string sendName_;
  std::string recvName_;
  std::string ctrlName_;

  static void streamWrite64_(cudaStream_t s, CUdeviceptr addr, uint64_t val) {
    CUresult r = cuStreamWriteValue64(
        reinterpret_cast<CUstream>(s), addr,
        static_cast<cuuint64_t>(val),
        CU_STREAM_WRITE_VALUE_DEFAULT);
    if (r != CUDA_SUCCESS)
      throw mscclpp::Error(
          "cuStreamWriteValue64 failed in NodeExchangeBuffer",
          mscclpp::ErrorCode::SystemError);
  }
};

// ── Inline method implementations ────────────────────────────────────────────

inline void NodeExchangeBuffer::push(
    cudaStream_t stream,
    void const* src, size_t slabOffset, size_t size,
    int rank, uint64_t tag) const {
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(
      sendSlab_ + slabOffset, src, size, cudaMemcpyDeviceToHost, stream));
  if (ctrlDevice_ != nullptr) {
    CUdeviceptr flagAddr = reinterpret_cast<CUdeviceptr>(ctrlDevice_)
                          + static_cast<size_t>(rank) * sizeof(uint64_t);
    streamWrite64_(stream, flagAddr, tag);
  } else {
    // No device-mapped ctrl: fall back to blocking sync + CPU write.
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    ctrl_->d2hReady[rank] = tag;
  }
}

inline void NodeExchangeBuffer::waitCpu(int rank, uint64_t tag) const {
  constexpr int kYieldAfter = 65536;
  int spins = 0;
  while (ctrl_->d2hReady[rank] < tag) {
    if (spins++ < kYieldAfter) {
#if defined(__x86_64__) || defined(__i386__)
      asm volatile("pause" ::: "memory");
#else
      asm volatile("" ::: "memory");
#endif
    } else {
      std::this_thread::yield();
    }
  }
  // Acquire fence: ensure subsequent reads of sendSlab see the D2H data
  // written by the GPU before it set the flag.
  std::atomic_thread_fence(std::memory_order_acquire);
}

// ── NodeExchangeBuffer::create() ─────────────────────────────────────────────

namespace neb_detail {

inline void createShm(std::string const& name, size_t size) {
  shm_unlink(name.c_str());
  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0)
    throw mscclpp::Error("shm_open(create) failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
    close(fd);
    shm_unlink(name.c_str());
    throw mscclpp::Error("ftruncate failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  }
  close(fd);
}

inline void* mapShm(std::string const& name, size_t size) {
  int fd = shm_open(name.c_str(), O_RDWR, 0600);
  if (fd < 0)
    throw mscclpp::Error("shm_open(map) failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED)
    throw mscclpp::Error("mmap failed for " + name,
                         mscclpp::ErrorCode::SystemError);
#ifdef MADV_HUGEPAGE
  (void)madvise(p, size, MADV_HUGEPAGE);
#endif
  return p;
}

struct InitStatus {
  int  result = static_cast<int>(ncclSuccess);
  char message[160] = {};
};

inline void publishStatus(std::shared_ptr<mscclpp::Communicator> boot,
                           int rank, int nRanks, ncclResult_t result,
                           std::string const& msg, char const* stage) {
  std::vector<InitStatus> s(static_cast<size_t>(nRanks));
  s[rank].result = static_cast<int>(result);
  if (!msg.empty()) std::snprintf(s[rank].message, 160, "%s", msg.c_str());
  boot->bootstrap()->allGather(s.data(), sizeof(InitStatus));
  for (int r = 0; r < nRanks; r++) {
    if (s[r].result != static_cast<int>(ncclSuccess)) {
      auto code = (s[r].result == static_cast<int>(ncclInvalidUsage) ||
                   s[r].result == static_cast<int>(ncclInvalidArgument))
                      ? mscclpp::ErrorCode::InvalidUsage
                      : mscclpp::ErrorCode::InternalError;
      throw mscclpp::Error(std::string(stage) + " failed on rank " +
                               std::to_string(r) + ": " +
                               std::string(s[r].message),
                           code);
    }
  }
}

} // namespace neb_detail

inline NodeExchangeBuffer NodeExchangeBuffer::create(
    std::shared_ptr<mscclpp::Communicator> bootstrapComm,
    int rank, int nRanks,
    bool isOwner, int localLeader,
    size_t slabBytes,
    int numaNode, int cudaDevice,
    std::string const& nameTag) {

  NodeExchangeBuffer buf;
  buf.slabBytes_ = slabBytes;
  buf.isOwner_   = isOwner;

  auto boot = bootstrapComm->bootstrap();

  // ── Step 1: owner creates POSIX shm, all ranks share names ───────────────
  struct Names { char send[128]; char recv[128]; char ctrl[128]; };
  Names localNames{};
  ncclResult_t createResult = ncclSuccess;
  std::string  createMsg;
  try {
    if (isOwner) {
      std::snprintf(localNames.send, 128, "/mint_neb_%s_s", nameTag.c_str());
      std::snprintf(localNames.recv, 128, "/mint_neb_%s_r", nameTag.c_str());
      std::snprintf(localNames.ctrl, 128, "/mint_neb_%s_c", nameTag.c_str());
      neb_detail::createShm(localNames.send, slabBytes);
      neb_detail::createShm(localNames.recv, slabBytes);
      neb_detail::createShm(localNames.ctrl, sizeof(NebCtrl));
    }
  } catch (std::exception const& ex) {
    createResult = ncclSystemError;
    createMsg    = ex.what();
  }
  neb_detail::publishStatus(bootstrapComm, rank, nRanks,
                             createResult, createMsg,
                             "NodeExchangeBuffer create");

  std::vector<Names> allNames(static_cast<size_t>(nRanks));
  allNames[rank] = localNames;
  boot->allGather(allNames.data(), sizeof(Names));

  // Each rank maps the shm created by its LOCAL leader (localLeader).
  // In multi-node setups each node has its own leader; use localLeader, not 0.
  buf.sendName_ = allNames[localLeader].send;
  buf.recvName_ = allNames[localLeader].recv;
  buf.ctrlName_ = allNames[localLeader].ctrl;

  // ── Step 2: all ranks map and register ───────────────────────────────────
  ncclResult_t setupResult = ncclSuccess;
  std::string  setupMsg;
  try {
    mscclpp::CudaDeviceGuard devGuard(cudaDevice);

    buf.sendMapping_ = neb_detail::mapShm(buf.sendName_, slabBytes);
    buf.sendSlab_    = static_cast<char*>(buf.sendMapping_);
    buf.recvMapping_ = neb_detail::mapShm(buf.recvName_, slabBytes);
    buf.recvSlab_    = static_cast<char*>(buf.recvMapping_);
    buf.ctrlMapping_ = neb_detail::mapShm(buf.ctrlName_, sizeof(NebCtrl));
    buf.ctrl_        = static_cast<NebCtrl*>(buf.ctrlMapping_);

    // Owner zeros ctrl and NUMA-places its own region.
    if (isOwner) {
      std::memset(buf.ctrlMapping_, 0, sizeof(NebCtrl));
      new (buf.ctrl_) NebCtrl{};
      if (numaNode >= 0 && numa_available() >= 0) {
        std::memset(buf.sendSlab_, 0, slabBytes);
        std::memset(buf.recvSlab_, 0, slabBytes);
        numa_tonode_memory(buf.sendSlab_, slabBytes, numaNode);
        numa_tonode_memory(buf.recvSlab_, slabBytes, numaNode);
      }
    }
    boot->barrier();

    // Register sendSlab: try device-mapped first (for GPU-kernel direct access),
    // fall back to portable-only.
    cudaError_t sendReg = cudaHostRegister(
        buf.sendMapping_, slabBytes,
        cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (sendReg == cudaSuccess) {
      void* dp = nullptr;
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, buf.sendMapping_, 0));
      buf.sendDevice_ = static_cast<char*>(dp);
    } else {
      cudaGetLastError();
      MSCCLPP_CUDATHROW(cudaHostRegister(buf.sendMapping_, slabBytes,
                                         cudaHostRegisterPortable));
      buf.sendDevice_ = nullptr;
    }
    buf.sendRegistered_ = true;

    // Register recvSlab: portable-only (RDMA writes here, no GPU-kernel access).
    MSCCLPP_CUDATHROW(cudaHostRegister(buf.recvMapping_, slabBytes,
                                       cudaHostRegisterPortable));
    buf.recvRegistered_ = true;

    // Register ctrl: device-mapped for cuStreamWriteValue64.
    cudaError_t ctrlReg = cudaHostRegister(
        buf.ctrlMapping_, sizeof(NebCtrl),
        cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (ctrlReg == cudaSuccess) {
      void* dp = nullptr;
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, buf.ctrlMapping_, 0));
      buf.ctrlDevice_ = static_cast<char*>(dp);
    } else {
      cudaGetLastError();
      MSCCLPP_CUDATHROW(cudaHostRegister(buf.ctrlMapping_, sizeof(NebCtrl),
                                         cudaHostRegisterPortable));
      buf.ctrlDevice_ = nullptr;
    }
    buf.ctrlRegistered_ = true;

  } catch (std::exception const& ex) {
    setupResult = ncclSystemError;
    setupMsg    = ex.what();
  }
  neb_detail::publishStatus(bootstrapComm, rank, nRanks,
                             setupResult, setupMsg,
                             "NodeExchangeBuffer setup");
  boot->barrier();
  return buf;
}

inline NodeExchangeBuffer::~NodeExchangeBuffer() {
  if (ctrlRegistered_)  cudaHostUnregister(ctrlMapping_);
  if (recvRegistered_)  cudaHostUnregister(recvMapping_);
  if (sendRegistered_)  cudaHostUnregister(sendMapping_);
  if (ctrlMapping_)     munmap(ctrlMapping_, sizeof(NebCtrl));
  if (recvMapping_)     munmap(recvMapping_, slabBytes_);
  if (sendMapping_)     munmap(sendMapping_, slabBytes_);
  if (isOwner_) {
    if (!ctrlName_.empty()) shm_unlink(ctrlName_.c_str());
    if (!recvName_.empty()) shm_unlink(recvName_.c_str());
    if (!sendName_.empty()) shm_unlink(sendName_.c_str());
  }
}
