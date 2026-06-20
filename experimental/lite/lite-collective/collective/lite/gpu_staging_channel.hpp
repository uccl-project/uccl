// GpuStagingChannel: GPU-initiated shared host-memory collective staging channel.
//
// Variant of CpuStagingChannel where the GPU kernel (not the CPU) decides when
// to stage data.  The GPU posts D2H commands to a device-mapped ring buffer; a
// CPU service thread drains the ring and issues the CUDA DMA + flag writes.
//
// GPU-initiated ring API (__device__, in GscDeviceHandle):
//   put(slot, chunkId, src, offset, size, tag)
//     Posts a D2H command to the ring.  The CPU service thread will issue
//     cudaMemcpyAsync(D2H) + cuStreamWriteValue64(ready flag) asynchronously.
//   signalDone(slot, tag)
//     Posts a slotDone-write command to the ring.
//
// CPU service API:
//   service(stream, maxCmds=0)
//     Drains up to maxCmds (0 = all pending) ring entries, enqueuing
//     cudaMemcpyAsync + streamWriteValue64 on the given stream.
//     Returns the number of commands processed.
//   serviceLoop(stream)
//     Blocking service loop — call from a dedicated background thread.
//     Stops when stop() is called.
//   stop()    — signals the service loop to exit.
//
// Shared ctrl layout (CscCtrl from cpu_staging_channel.hpp):
//   GpuStagingChannel uses the SAME CscCtrl as CpuStagingChannel.  This means
//   wait() and get() from a CpuStagingChannel with the same slab/ctrl can
//   consume data staged by GpuStagingChannel without any changes.
//
// Pair with CpuStagingChannel (cpu_staging_channel.hpp) for the CPU-initiated
// variant.  Both share the same slab layout and CscCtrl structure.

#pragma once

#include "cpu_staging_channel.hpp"
#include <atomic>
#include <cstring>
#include <thread>
#include <cuda_runtime.h>
#include <cuda.h>

// ── Ring constants ────────────────────────────────────────────────────────────
static constexpr uint32_t kGscRingSize  = 1024;   // must be power-of-2
static constexpr uint32_t kGscRingMask  = kGscRingSize - 1;

// ── Ring entry (GPU writes this, CPU reads) ───────────────────────────────────
// Written by GPU via SM stores to device-mapped host memory.  Sized to fit in
// one or two cache lines to avoid partial-write visibility issues.
struct GscRingEntry {
  uint32_t type;      // 0 = D2H put, 1 = slotDone signal
  int32_t  slot;
  int32_t  chunkId;
  int32_t  rank;
  uint64_t srcDevPtr; // source device pointer encoded as uint64
  uint64_t offset;    // byte offset within rank's slot region
  uint64_t size;      // bytes to copy
  uint64_t tag;       // value to write to ready/done flag
};
static_assert(sizeof(GscRingEntry) == 48, "GscRingEntry size");

// Ring control (head written by GPU, tail written by CPU; both device-mapped).
struct GscRingCtrl {
  alignas(64) uint64_t head;  // GPU increments (post command)
  alignas(64) uint64_t tail;  // CPU increments (consume command)
};

// ── Device handle (GPU-callable, __device__ methods) ─────────────────────────
struct GscDeviceHandle {
  GscRingEntry*      ring;      // device ptr to ring buffer
  GscRingCtrl*       ringCtrl;  // device ptr to GscRingCtrl
  uint32_t           ringMask;  // kGscRingSize - 1

  // Rank-specific staging geometry (same meaning as in CscDeviceHandle).
  char*  slabDev;       // device ptr to full slab (null if !mapSlab)
  char*  ctrlDev;       // device ptr to CscCtrl
  int    rank;
  int    nRanks;
  size_t bytesPerRank;
  size_t slotStride;
  size_t counterStride;

  // Post a D2H command.  The CPU service thread will issue the DMA.
  __device__ void put(int slot, int chunkId,
                      void const* src, size_t offset, size_t size,
                      uint64_t tag) {
    uint64_t idx = atomicAdd(
        reinterpret_cast<unsigned long long*>(&ringCtrl->head), 1ULL)
        & ringMask;
    GscRingEntry e;
    e.type      = 0;
    e.slot      = slot;
    e.chunkId   = chunkId;
    e.rank      = rank;
    e.srcDevPtr = reinterpret_cast<uint64_t>(src);
    e.offset    = static_cast<uint64_t>(offset);
    e.size      = static_cast<uint64_t>(size);
    e.tag       = tag;
    // Write entry fields then make head visible to CPU.
    ring[idx] = e;
    __threadfence_system();
  }

  // Post a slotDone signal command.
  __device__ void signalDone(int slot, uint64_t tag) {
    uint64_t idx = atomicAdd(
        reinterpret_cast<unsigned long long*>(&ringCtrl->head), 1ULL)
        & ringMask;
    GscRingEntry e{};
    e.type = 1;
    e.slot = slot;
    e.tag  = tag;
    ring[idx] = e;
    __threadfence_system();
  }

  // Byte offset helpers (same as CscDeviceHandle).
  __device__ __host__ size_t slotOffset(int s) const {
    return static_cast<size_t>(s) * slotStride;
  }
  __device__ __host__ size_t readyFlagOffset(int s) const {
    return counterStride * (static_cast<size_t>(s)
           * (kCscMaxChunks * kCscMaxRanks));
  }
  __device__ __host__ size_t doneFlagOffset(int s) const {
    return counterStride * (kCscMaxSlots * kCscMaxChunks * kCscMaxRanks
           + static_cast<size_t>(s) * kCscMaxRanks);
  }
};

// ── GpuStagingChannel ─────────────────────────────────────────────────────────
class GpuStagingChannel {
 public:
  // Create collectively — all nRanks must call simultaneously.
  // The slab and ctrl are shared with a CpuStagingChannel that uses the same
  // nameTag, allowing wait()/get() from a CpuStagingChannel on the same slab.
  static GpuStagingChannel create(
      size_t bytesPerRank,
      int nSlots,
      std::shared_ptr<mscclpp::Communicator> bootstrapComm,
      int rank, int nRanks, int cudaDevice,
      bool mapSlab, bool numaPlace,
      std::string const& nameTag);

  GpuStagingChannel(GpuStagingChannel const&) = delete;
  GpuStagingChannel& operator=(GpuStagingChannel const&) = delete;
  GpuStagingChannel(GpuStagingChannel&&)            = default;
  GpuStagingChannel& operator=(GpuStagingChannel&&) = default;
  ~GpuStagingChannel();

  // ── CPU service API ────────────────────────────────────────────────────────

  // Process up to maxCmds pending ring entries (0 = all available).
  // Enqueues D2H cudaMemcpyAsync + cuStreamWriteValue64 on stream.
  // Returns the number of commands processed.
  int service(cudaStream_t stream, int maxCmds = 0);

  // Blocking service loop (intended for a dedicated background thread).
  // Calls service(stream) in a tight loop until stop() is called.
  void serviceLoop(cudaStream_t stream);

  // Signal the service loop to stop after the next service() call.
  void stop() { stopFlag_ = true; }

  // ── GPU handle ─────────────────────────────────────────────────────────────
  GscDeviceHandle deviceHandle() const;

  // ── Accessors (mirrors CpuStagingChannel) ─────────────────────────────────
  int    rank()        const { return csc_.rank(); }
  int    nRanks()      const { return csc_.nRanks(); }
  size_t bytesPerRank() const { return csc_.bytesPerRank(); }

 private:
  // Private constructor: takes an already-created CpuStagingChannel.
  // Used by create() to avoid exposing default construction.
  explicit GpuStagingChannel(CpuStagingChannel&& csc) : csc_(std::move(csc)) {}

  // The underlying CpuStagingChannel owns the slab/ctrl POSIX shm.
  // GpuStagingChannel wraps it and adds the ring buffer.
  CpuStagingChannel  csc_;

  // Ring buffer (device-mapped host memory).
  void*        ringMapping_     = nullptr;
  GscRingEntry* ring_           = nullptr;
  GscRingEntry* ringDev_        = nullptr;  // device ptr
  bool         ringRegistered_  = false;

  void*       ctrlMapping_      = nullptr;
  GscRingCtrl* ringCtrl_        = nullptr;
  GscRingCtrl* ringCtrlDev_     = nullptr;  // device ptr
  bool        ctrlRegistered_   = false;

  volatile bool stopFlag_ = false;

  static void streamWrite64_(cudaStream_t s, CUdeviceptr addr, uint64_t val) {
    CUresult r = cuStreamWriteValue64(
        reinterpret_cast<CUstream>(s), addr,
        static_cast<cuuint64_t>(val),
        CU_STREAM_WRITE_VALUE_DEFAULT);
    if (r != CUDA_SUCCESS)
      throw mscclpp::Error("cuStreamWriteValue64 failed in GpuStagingChannel",
                           mscclpp::ErrorCode::SystemError);
  }

  // Compute the device address of CscCtrl ready flag for (slot, chunkId, rank).
  CUdeviceptr readyFlagAddr_(int slot, int chunkId, int rank) const {
    char* base = csc_.ctrlDev();
    size_t off = sizeof(CscCounter) * (
        (static_cast<size_t>(slot) * kCscMaxChunks
         + static_cast<size_t>(chunkId)) * kCscMaxRanks
        + static_cast<size_t>(rank));
    return reinterpret_cast<CUdeviceptr>(base + off);
  }
  CUdeviceptr doneFlagAddr_(int slot, int rank) const {
    char* base = csc_.ctrlDev();
    size_t off = sizeof(CscCounter) * (
        kCscMaxSlots * kCscMaxChunks * kCscMaxRanks
        + static_cast<size_t>(slot) * kCscMaxRanks
        + static_cast<size_t>(rank));
    return reinterpret_cast<CUdeviceptr>(base + off);
  }
};

// ── Inline implementations ────────────────────────────────────────────────────

inline int GpuStagingChannel::service(cudaStream_t stream, int maxCmds) {
  uint64_t head = ringCtrl_->head;
  uint64_t tail = ringCtrl_->tail;
  int processed = 0;
  while (tail < head && (maxCmds == 0 || processed < maxCmds)) {
    GscRingEntry const& e = ring_[tail & kGscRingMask];
    if (e.type == 0) {
      // D2H put command.
      char* dst = csc_.rankSlabHost(e.slot, e.rank) + e.offset;
      auto* src = reinterpret_cast<void const*>(e.srcDevPtr);
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, static_cast<char const*>(src),
                                        e.size, cudaMemcpyDeviceToHost,
                                        stream));
      streamWrite64_(stream, readyFlagAddr_(e.slot, e.chunkId, e.rank), e.tag);
    } else {
      // slotDone signal.
      streamWrite64_(stream, doneFlagAddr_(e.slot, e.rank), e.tag);
    }
    ++tail;
    ++processed;
  }
  ringCtrl_->tail = tail;
  return processed;
}

inline void GpuStagingChannel::serviceLoop(cudaStream_t stream) {
  while (!stopFlag_) {
    if (service(stream, 64) == 0) {
#if defined(__x86_64__) || defined(__i386__)
      asm volatile("pause" ::: "memory");
#else
      asm volatile("" ::: "memory");
#endif
    }
  }
}

inline GscDeviceHandle GpuStagingChannel::deviceHandle() const {
  GscDeviceHandle h;
  h.ring         = ringDev_;
  h.ringCtrl     = ringCtrlDev_;
  h.ringMask     = kGscRingMask;
  auto csc       = csc_.deviceHandle();
  h.slabDev      = csc.slabDev;
  h.ctrlDev      = csc.ctrlDev;
  h.rank         = csc.rank;
  h.nRanks       = csc.nRanks;
  h.bytesPerRank = csc.bytesPerRank;
  h.slotStride   = csc.slotStride;
  h.counterStride = csc.counterStride;
  return h;
}

inline GpuStagingChannel::~GpuStagingChannel() {
  if (ctrlRegistered_)   cudaHostUnregister(ctrlMapping_);
  if (ringRegistered_)   cudaHostUnregister(ringMapping_);
  if (ctrlMapping_)      ::free(ctrlMapping_);
  if (ringMapping_)      ::free(ringMapping_);
}

inline GpuStagingChannel GpuStagingChannel::create(
    size_t bytesPerRank, int nSlots,
    std::shared_ptr<mscclpp::Communicator> bootstrapComm,
    int rank, int nRanks, int cudaDevice,
    bool mapSlab, bool numaPlace,
    std::string const& nameTag) {

  // Build the underlying CpuStagingChannel first.
  GpuStagingChannel ch(
      CpuStagingChannel::create(bytesPerRank, nSlots, bootstrapComm,
                                rank, nRanks, cudaDevice,
                                mapSlab, numaPlace, nameTag));

  mscclpp::CudaDeviceGuard devGuard(cudaDevice);

  // Allocate ring buffer in pinned host memory (device-mapped).
  size_t ringBytes = kGscRingSize * sizeof(GscRingEntry);
  void* rmem = nullptr;
  MSCCLPP_CUDATHROW(cudaMallocHost(&rmem, ringBytes));
  std::memset(rmem, 0, ringBytes);
  ch.ringMapping_    = rmem;
  ch.ring_           = static_cast<GscRingEntry*>(rmem);
  MSCCLPP_CUDATHROW(cudaHostRegister(rmem, ringBytes,
                                     cudaHostRegisterPortable |
                                     cudaHostRegisterMapped));
  ch.ringRegistered_ = true;
  {
    void* dp = nullptr;
    MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, rmem, 0));
    ch.ringDev_ = static_cast<GscRingEntry*>(dp);
  }

  // Allocate ring control struct (head/tail) in pinned host memory.
  void* cmem = nullptr;
  MSCCLPP_CUDATHROW(cudaMallocHost(&cmem, sizeof(GscRingCtrl)));
  std::memset(cmem, 0, sizeof(GscRingCtrl));
  ch.ctrlMapping_    = cmem;
  ch.ringCtrl_       = static_cast<GscRingCtrl*>(cmem);
  MSCCLPP_CUDATHROW(cudaHostRegister(cmem, sizeof(GscRingCtrl),
                                     cudaHostRegisterPortable |
                                     cudaHostRegisterMapped));
  ch.ctrlRegistered_ = true;
  {
    void* dp = nullptr;
    MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, cmem, 0));
    ch.ringCtrlDev_ = static_cast<GscRingCtrl*>(dp);
  }
  return ch;
}
