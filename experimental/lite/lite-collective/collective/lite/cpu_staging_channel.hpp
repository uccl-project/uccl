// CpuStagingChannel: CPU-initiated shared host-memory collective staging channel.
//
// Encapsulates host-memory staging for intra-node collectives (AllGather etc.):
//   - ONE shared POSIX shm slab; all ranks map the same region.
//   - Each rank r owns slab[slot][r * bytesPerRank .. (r+1)*bytesPerRank).
//   - Rank r NUMA-places its own region on its GPU's NUMA node.
//   - Shared ctrl struct (POSIX shm, device-mapped) holds per-chunk flags.
//
// CPU-initiated stream API (enqueue work into CUDA streams, no CPU blocking):
//   put(stream, slot, chunkId, src, offset, size, tag)
//     D2H src → slab[slot][rank][offset..+size], then write ready flag via
//     cuStreamWriteValue64 on the same stream.
//   wait(stream, slot, chunkId, peer, tag)
//     GPU-side cuStreamWaitValue64 for peer's ready flag.
//   get(stream, slot, firstRank, lastRank, offset, size, dst)
//     H2D slab[slot][firstRank..lastRank][offset..+size] → dst in ONE DMA.
//     (1D contiguous if size==bytesPerRank; 2D strided if chunked.)
//   signalDone(stream, slot, tag)  — slot reuse guard.
//   waitDone(slot, peer, tag)      — CPU spin for slot reuse.
//
// CscDeviceHandle: raw device pointers for GPU-kernel SM-copy paths.
//
// Pair with GpuStagingChannel (gpu_staging_channel.hpp) for the GPU-initiated
// variant where the GPU posts D2H commands to a ring buffer and a CPU service
// thread executes them.  Both channels share the same CscCtrl layout so that
// wait()/get()/waitDone() are interchangeable across both variants.

#pragma once

#include "lite_common.h"
// Note: debug.h and WARN/INFO macros provided by the including TU (nccl.cu).
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
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
static constexpr int    kCscMaxRanks  = 8;
static constexpr int    kCscMaxSlots  = 2;    // double-buffering depth
static constexpr int    kCscMaxChunks = 1024; // max chunks per slot

// ── Ctrl struct (POSIX shm, device-mapped) ───────────────────────────────────
struct CscCounter { alignas(64) std::atomic<uint64_t> value{0}; };

struct CscCtrl {
  // d2hReady[slot][chunk][rank]: rank r sets after D2H of chunk c in slot s.
  // Polled by peers via cuStreamWaitValue64 before H2D.
  CscCounter d2hReady[kCscMaxSlots][kCscMaxChunks][kCscMaxRanks];
  // slotDone[slot][rank]: rank r sets after all H2D for this slot is done.
  // CPU-polled at call entry for slot reuse guard (not hot path).
  CscCounter slotDone[kCscMaxSlots][kCscMaxRanks];
};

// ── Device handle (for GPU kernel paths) ─────────────────────────────────────
struct CscDeviceHandle {
  char*  slabDev;       // device ptr to full slab (null if !mapSlab)
  char*  ctrlDev;       // device ptr to CscCtrl
  int    rank;
  int    nRanks;
  size_t bytesPerRank;
  size_t slotStride;    // bytes between slots: = bytesPerRank * nRanks (padded)
  size_t counterStride; // = sizeof(CscCounter)

  // Byte offset from slab base to slot s.
  size_t slotOffset(int slot) const {
    return static_cast<size_t>(slot) * slotStride;
  }
  // Byte offset from ctrlDev to d2hReady[slot][chunk=0][rank=0].value.
  size_t readyFlagOffset(int slot) const {
    static_assert(std::is_standard_layout<CscCtrl>::value, "");
    static_assert(std::is_standard_layout<CscCounter>::value, "");
    return sizeof(CscCounter) * (
        static_cast<size_t>(slot) * (kCscMaxChunks * kCscMaxRanks));
  }
  // Byte offset from ctrlDev to slotDone[slot][rank=0].value.
  size_t doneFlagOffset(int slot) const {
    return sizeof(CscCounter) * (kCscMaxSlots * kCscMaxChunks * kCscMaxRanks
        + static_cast<size_t>(slot) * kCscMaxRanks);
  }
};

// ── CpuStagingChannel ─────────────────────────────────────────────────────────
class CpuStagingChannel {
 public:
  // Create and initialize the buffer. Called collectively — all ranks must call.
  // Throws on failure (propagated via bootstrap allGather).
  static CpuStagingChannel create(
      size_t bytesPerRank,
      int nSlots,
      std::shared_ptr<mscclpp::Communicator> bootstrapComm,
      int rank, int nRanks, int cudaDevice,
      bool mapSlab, bool numaPlace,
      std::string const& nameTag);  // unique per comm (e.g. comm ptr hex)

  // Non-copyable, moveable.
  CpuStagingChannel(CpuStagingChannel const&) = delete;
  CpuStagingChannel& operator=(CpuStagingChannel const&) = delete;
  CpuStagingChannel(CpuStagingChannel&&) = default;
  CpuStagingChannel& operator=(CpuStagingChannel&&) = default;
  ~CpuStagingChannel();

  // ── Stream API ─────────────────────────────────────────────────────────────

  // D2H devSrc[offset..+size] → slab[slot][rank][offset..+size].
  // Then write d2hReady[slot][chunkId][rank] = tag on stream.
  void put(cudaStream_t stream,
           int slot, int chunkId,
           void const* devSrc, size_t offset, size_t size,
           uint64_t tag) const;

  // GPU-side wait: cuStreamWaitValue64(d2hReady[slot][chunkId][peer], tag).
  void wait(cudaStream_t stream,
            int slot, int chunkId, int peer,
            uint64_t tag) const;

  // H2D slab[slot][firstRank..lastRank][offset..+size] → devDst.
  // One DMA: 1D if size==bytesPerRank (contiguous), 2D if size<bytesPerRank.
  // devDst must point to a buffer of at least nRanks * bytesPerRank bytes;
  // data is written at devDst[firstRank*bytesPerRank+offset..].
  // Caller must have called wait() for each peer in [firstRank..lastRank] on
  // the same stream before calling get().
  void get(cudaStream_t stream,
           int slot, int firstRank, int lastRank,
           size_t offset, size_t size,
           void* devDst) const;

  // Write slotDone[slot][rank] = tag on stream.
  void signalDone(cudaStream_t stream, int slot, uint64_t tag) const;

  // CPU spin: wait until slotDone[slot][peer].value >= tag.
  // Called at call entry for slot reuse guard, not in hot path.
  void waitDone(int slot, int peer, uint64_t tag) const;

  // ── Device handle ──────────────────────────────────────────────────────────
  CscDeviceHandle deviceHandle() const;

  // ── Accessors ──────────────────────────────────────────────────────────────
  int    rank()        const { return rank_; }
  int    nRanks()      const { return nRanks_; }
  size_t bytesPerRank() const { return bytesPerRank_; }
  size_t slotStride()  const { return slotStride_; }
  bool   hasSlabDev()  const { return slabDevice_ != nullptr; }

  // Host ptr to rank r's region in slot s.
  char* rankSlabHost(int slot, int r) const {
    return slab_ + static_cast<size_t>(slot) * slotStride_
                 + static_cast<size_t>(r) * bytesPerRank_;
  }
  // Device ptr to rank r's region in slot s (requires mapSlab).
  char* rankSlabDev(int slot, int r) const {
    return slabDevice_ + static_cast<size_t>(slot) * slotStride_
                       + static_cast<size_t>(r) * bytesPerRank_;
  }
  char* ctrlDev() const { return ctrlDevice_; }
  CscCtrl* ctrl()  const { return ctrl_; }

 private:
  CpuStagingChannel() = default;

  // Slab
  size_t slabBytes_   = 0;
  void*  slabMapping_ = nullptr;
  char*  slab_        = nullptr;
  char*  slabDevice_  = nullptr;
  bool   slabRegistered_ = false;
  std::string slabName_;

  // Ctrl
  void*    ctrlMapping_   = nullptr;
  CscCtrl* ctrl_          = nullptr;
  char*    ctrlDevice_    = nullptr;
  bool     ctrlRegistered_ = false;
  std::string ctrlName_;

  int    rank_        = -1;
  int    nRanks_      = 0;
  size_t bytesPerRank_ = 0;
  size_t slotStride_   = 0;
  int    nSlots_       = 0;
  bool   isLeader_     = false;

  // Internal helpers
  static CUdeviceptr readyFlagCuAddr_(char const* ctrlDev, int slot, int chunk, int rank) {
    size_t off = sizeof(CscCounter) * (
        (static_cast<size_t>(slot) * kCscMaxChunks + static_cast<size_t>(chunk))
        * kCscMaxRanks + static_cast<size_t>(rank));
    return reinterpret_cast<CUdeviceptr>(ctrlDev + off);
  }
  static CUdeviceptr doneFlagCuAddr_(char const* ctrlDev, int slot, int rank) {
    size_t off = sizeof(CscCounter) * (kCscMaxSlots * kCscMaxChunks * kCscMaxRanks
        + static_cast<size_t>(slot) * kCscMaxRanks + static_cast<size_t>(rank));
    return reinterpret_cast<CUdeviceptr>(ctrlDev + off);
  }

  static void streamWrite64_(cudaStream_t s, CUdeviceptr addr, uint64_t val) {
    CUresult r = cuStreamWriteValue64(reinterpret_cast<CUstream>(s), addr,
                                      static_cast<cuuint64_t>(val),
                                      CU_STREAM_WRITE_VALUE_DEFAULT);
    if (r != CUDA_SUCCESS)
      throw mscclpp::Error("cuStreamWriteValue64 failed in CpuStagingChannel",
                           mscclpp::ErrorCode::SystemError);
  }
  static void streamWait64_(cudaStream_t s, CUdeviceptr addr, uint64_t val) {
    CUresult r = cuStreamWaitValue64(reinterpret_cast<CUstream>(s), addr,
                                     static_cast<cuuint64_t>(val),
                                     CU_STREAM_WAIT_VALUE_GEQ);
    if (r != CUDA_SUCCESS)
      throw mscclpp::Error("cuStreamWaitValue64 failed in CpuStagingChannel",
                           mscclpp::ErrorCode::SystemError);
  }
};

// ── Inline method implementations ────────────────────────────────────────────

inline void CpuStagingChannel::put(cudaStream_t stream,
                                   int slot, int chunkId,
                                   void const* devSrc, size_t offset, size_t size,
                                   uint64_t tag) const {
  char* dst = slab_ + static_cast<size_t>(slot) * slotStride_
                    + static_cast<size_t>(rank_) * bytesPerRank_ + offset;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst,
                                    static_cast<char const*>(devSrc) + offset,
                                    size, cudaMemcpyDeviceToHost, stream));
  streamWrite64_(stream, readyFlagCuAddr_(ctrlDevice_, slot, chunkId, rank_), tag);
}

inline void CpuStagingChannel::wait(cudaStream_t stream,
                                    int slot, int chunkId, int peer,
                                    uint64_t tag) const {
  streamWait64_(stream, readyFlagCuAddr_(ctrlDevice_, slot, chunkId, peer), tag);
}

inline void CpuStagingChannel::get(cudaStream_t stream,
                                   int slot, int firstRank, int lastRank,
                                   size_t offset, size_t size,
                                   void* devDst) const {
  if (firstRank > lastRank) return;
  size_t nCopy = static_cast<size_t>(lastRank - firstRank + 1);
  char* src = slab_ + static_cast<size_t>(slot) * slotStride_
                    + static_cast<size_t>(firstRank) * bytesPerRank_;
  char* dst = static_cast<char*>(devDst)
                    + static_cast<size_t>(firstRank) * bytesPerRank_;
  if (size == bytesPerRank_) {
    // Full-rank contiguous copy (no chunk offset, all nCopy ranks consecutive).
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, nCopy * bytesPerRank_,
                                      cudaMemcpyHostToDevice, stream));
  } else {
    // Partial-rank (chunked) strided copy: height=nCopy ranks, width=size bytes.
    MSCCLPP_CUDATHROW(cudaMemcpy2DAsync(
        dst + offset, bytesPerRank_,      // dst: stride=bytesPerRank between rows
        src + offset, bytesPerRank_,      // src: stride=bytesPerRank between rows
        size, nCopy,                       // width × height
        cudaMemcpyHostToDevice, stream));
  }
}

inline void CpuStagingChannel::signalDone(cudaStream_t stream,
                                          int slot, uint64_t tag) const {
  streamWrite64_(stream, doneFlagCuAddr_(ctrlDevice_, slot, rank_), tag);
}

inline void CpuStagingChannel::waitDone(int slot, int peer, uint64_t tag) const {
  auto const& counter = ctrl_->slotDone[slot][peer];
  int spins = 0;
  constexpr int kYieldAfter = 65536;
  while (counter.value.load(std::memory_order_acquire) < tag) {
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
}

inline CscDeviceHandle CpuStagingChannel::deviceHandle() const {
  CscDeviceHandle h;
  h.slabDev      = slabDevice_;
  h.ctrlDev      = ctrlDevice_;
  h.rank         = rank_;
  h.nRanks       = nRanks_;
  h.bytesPerRank = bytesPerRank_;
  h.slotStride   = slotStride_;
  h.counterStride = sizeof(CscCounter);
  return h;
}

// ── CpuStagingChannel::create() ──────────────────────────────────────────────
// (Full implementation; included here as it's a header-only class.)

namespace csc_detail {

inline void createShm(std::string const& name, size_t size) {
  shm_unlink(name.c_str());
  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0)
    throw mscclpp::Error("shm_open(create) failed for " + name,
                         mscclpp::ErrorCode::SystemError);
  if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
    close(fd); shm_unlink(name.c_str());
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
                           std::string(s[r].message), code);
    }
  }
}

} // namespace csc_detail

inline CpuStagingChannel CpuStagingChannel::create(
    size_t bytesPerRank,
    int nSlots,
    std::shared_ptr<mscclpp::Communicator> bootstrapComm,
    int rank, int nRanks, int cudaDevice,
    bool mapSlab, bool numaPlace,
    std::string const& nameTag) {

  CpuStagingChannel buf;
  buf.rank_        = rank;
  buf.nRanks_      = nRanks;
  buf.bytesPerRank_ = bytesPerRank;
  buf.nSlots_      = nSlots;
  buf.isLeader_    = (rank == 0);
  // Slot stride = all ranks' data per slot (no padding needed, already rank-aligned).
  buf.slotStride_  = bytesPerRank * static_cast<size_t>(nRanks);
  buf.slabBytes_   = static_cast<size_t>(nSlots) * buf.slotStride_;

  auto boot = bootstrapComm->bootstrap();

  // ── Step 1: leader creates POSIX shm, all share names ────────────────────
  struct Names { char slab[128]; char ctrl[128]; };
  Names localNames{};
  ncclResult_t createResult = ncclSuccess;
  std::string createMsg;
  try {
    if (buf.isLeader_) {
      std::snprintf(localNames.slab, 128, "/mint_hsb_%s_slab", nameTag.c_str());
      std::snprintf(localNames.ctrl, 128, "/mint_hsb_%s_ctrl", nameTag.c_str());
      csc_detail::createShm(localNames.slab, buf.slabBytes_);
      csc_detail::createShm(localNames.ctrl, sizeof(CscCtrl));
    }
  } catch (std::exception const& ex) {
    createResult = ncclSystemError;
    createMsg = ex.what();
  }
  csc_detail::publishStatus(bootstrapComm, rank, nRanks,
                            createResult, createMsg, "CpuStagingChannel slab create");

  std::vector<Names> allNames(static_cast<size_t>(nRanks));
  allNames[rank] = localNames;
  boot->allGather(allNames.data(), sizeof(Names));
  buf.slabName_ = allNames[0].slab;
  buf.ctrlName_ = allNames[0].ctrl;

  // ── Step 2: all ranks map the slab and ctrl ───────────────────────────────
  ncclResult_t setupResult = ncclSuccess;
  std::string setupMsg;
  try {
    mscclpp::CudaDeviceGuard devGuard(cudaDevice);

    buf.slabMapping_ = csc_detail::mapShm(buf.slabName_, buf.slabBytes_);
    buf.slab_        = static_cast<char*>(buf.slabMapping_);
    buf.ctrlMapping_ = csc_detail::mapShm(buf.ctrlName_, sizeof(CscCtrl));
    buf.ctrl_        = static_cast<CscCtrl*>(buf.ctrlMapping_);

    // Leader zeros ctrl.
    if (buf.isLeader_) {
      std::memset(buf.ctrlMapping_, 0, sizeof(CscCtrl));
      new (buf.ctrl_) CscCtrl{};
    }
    boot->barrier();

    // NUMA-place own rank's region in each slot.
    if (numaPlace) {
      int numaNode = -1;
      try { numaNode = mscclpp::getDeviceNumaNode(cudaDevice); } catch (...) {}
      if (numaNode >= 0 && numa_available() >= 0) {
        for (int s = 0; s < nSlots; s++) {
          char* region = buf.slab_ + static_cast<size_t>(s) * buf.slotStride_
                                   + static_cast<size_t>(rank) * bytesPerRank;
          std::memset(region, 0, bytesPerRank);
          numa_tonode_memory(region, bytesPerRank, numaNode);
        }
      }
    }

    // Register slab with CUDA.
    unsigned int regFlags = mapSlab
        ? (cudaHostRegisterPortable | cudaHostRegisterMapped)
        : cudaHostRegisterPortable;
    MSCCLPP_CUDATHROW(cudaHostRegister(buf.slabMapping_, buf.slabBytes_, regFlags));
    buf.slabRegistered_ = true;
    if (mapSlab) {
      void* p = nullptr;
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&p, buf.slabMapping_, 0));
      buf.slabDevice_ = static_cast<char*>(p);
    }

    // Register ctrl with CUDA (always device-mapped for streamWriteValue64).
    MSCCLPP_CUDATHROW(cudaHostRegister(buf.ctrlMapping_, sizeof(CscCtrl),
                                       cudaHostRegisterPortable | cudaHostRegisterMapped));
    buf.ctrlRegistered_ = true;
    {
      void* p = nullptr;
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&p, buf.ctrlMapping_, 0));
      buf.ctrlDevice_ = static_cast<char*>(p);
    }
  } catch (std::exception const& ex) {
    setupResult = ncclSystemError;
    setupMsg = ex.what();
  }
  csc_detail::publishStatus(bootstrapComm, rank, nRanks,
                            setupResult, setupMsg, "CpuStagingChannel setup");
  boot->barrier();
  return buf;
}

inline CpuStagingChannel::~CpuStagingChannel() {
  if (ctrlRegistered_)  cudaHostUnregister(ctrlMapping_);
  if (slabRegistered_)  cudaHostUnregister(slabMapping_);
  if (ctrlMapping_)     munmap(ctrlMapping_, sizeof(CscCtrl));
  if (slabMapping_)     munmap(slabMapping_, slabBytes_);
  if (isLeader_) {
    if (!ctrlName_.empty()) shm_unlink(ctrlName_.c_str());
    if (!slabName_.empty()) shm_unlink(slabName_.c_str());
  }
}
