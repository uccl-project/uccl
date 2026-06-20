// GpuPortChannel: GPU-initiated point-to-point inter-node RDMA channel.
//
// Variant of CpuPortChannel where a GPU kernel (not the CPU) decides when to
// send data.  The GPU posts RDMA commands (write/signal) to a device-mapped
// ring buffer; a CPU proxy thread (serviceLoop) drains the ring, calls
// Connection.write(), and posts semaphore signals via RDMA.
//
// GPU-initiated ring API (__device__, in GpcDeviceHandle):
//   write(localOff, remoteOff, bytes)   — posts an RDMA-write command to ring
//   signal(epoch)                       — posts a semaphore-signal command
//   wait(epoch)                         — spins on local GpcCtrl.semaphore
//                                         (written by remote via CpuPortChannel
//                                          or GpuPortChannel::signal())
//
// CPU proxy API:
//   service(maxCmds=0)   — drain up to maxCmds ring entries; returns # done
//   serviceLoop()        — blocking loop for a dedicated background thread
//   stop()               — signal the service loop to exit
//
// Protocol (semaphore-based, same as CpuPortChannel):
//   GPU:  write(localOff, remoteOff, bytes) + signal(epoch)
//   CPU:  wait(epoch)   or   GPU: wait(epoch) via GpcDeviceHandle
//
// GpcCtrl (from cpu_port_channel.hpp) is shared with CpuPortChannel:
//   CpuPortChannel::signal() and GpuPortChannel::service() both write to the
//   remote's GpcCtrl.semaphore, and both channel types can wait() on the local
//   GpcCtrl.semaphore — from CPU or from GPU kernel.

#pragma once

#include "cpu_port_channel.hpp"
#include <atomic>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>

// ── Ring constants ────────────────────────────────────────────────────────────
static constexpr uint32_t kGpcRingSize = 512;   // must be power-of-2
static constexpr uint32_t kGpcRingMask = kGpcRingSize - 1;

// ── Ring entry (GPU writes, CPU reads) ────────────────────────────────────────
struct GpcRingEntry {
  uint32_t type;        // 0 = RDMA write, 1 = signal
  uint32_t _pad;
  uint64_t localOff;    // byte offset in local data buffer
  uint64_t remoteOff;   // byte offset in remote data buffer
  uint64_t bytes;       // transfer size
  uint64_t epoch;       // semaphore epoch (type==1 only)
};
static_assert(sizeof(GpcRingEntry) == 40, "GpcRingEntry size");

// Ring control (head written by GPU, tail by CPU; both device-mapped).
struct GpcRingCtrl {
  alignas(64) uint64_t head;
  alignas(64) uint64_t tail;
};

// ── Device handle (GPU-callable) ──────────────────────────────────────────────
struct GpcDeviceHandle {
  GpcRingEntry*      ring;       // device ptr to ring entries
  GpcRingCtrl*       ringCtrl;   // device ptr to GpcRingCtrl
  uint32_t           ringMask;

  // Device-mapped semaphore written by the CPU service loop after processing
  // a signal command (type==1).  Avoids needing a device ptr to mscclpp's
  // IB-registered memory, which may not be device-mappable.
  volatile uint64_t* gpuSem;

  // Post an RDMA write command to the ring (single-writer: entry before head).
  __device__ void write(uint64_t localOff, uint64_t remoteOff, uint64_t bytes) {
    uint64_t myHead = ringCtrl->head;
    uint64_t idx    = myHead & ringMask;
    GpcRingEntry e{};
    e.type      = 0;
    e.localOff  = localOff;
    e.remoteOff = remoteOff;
    e.bytes     = bytes;
    ring[idx] = e;
    __threadfence_system();
    ringCtrl->head = myHead + 1;
    __threadfence_system();
  }

  // Post a semaphore signal command to the ring.
  __device__ void signal(uint64_t epoch) {
    uint64_t myHead = ringCtrl->head;
    uint64_t idx    = myHead & ringMask;
    GpcRingEntry e{};
    e.type  = 1;
    e.epoch = epoch;
    ring[idx] = e;
    __threadfence_system();
    ringCtrl->head = myHead + 1;
    __threadfence_system();
  }

  // Spin until gpuSem >= epoch.
  // The proxy service() loop sets gpuSem after calling cpu_.wait(epoch),
  // so this fires once the remote's RDMA signal has been received and serviced.
  __device__ void wait(uint64_t epoch) const {
    while (*gpuSem < epoch) {
#if defined(__CUDA_ARCH__)
      // __nanosleep(100); // sm_70+ only
#endif
    }
    __threadfence_system();
  }
};

// ── GpuPortChannel ────────────────────────────────────────────────────────────
class GpuPortChannel {
 public:
  // Build from a pre-created CpuPortChannel.  The GpuPortChannel wraps it,
  // adding a GPU-writable ring and a proxy service loop.
  explicit GpuPortChannel(CpuPortChannel&& cpu, int cudaDevice);

  GpuPortChannel() = default;
  GpuPortChannel(GpuPortChannel const&) = delete;
  GpuPortChannel& operator=(GpuPortChannel const&) = delete;
  GpuPortChannel(GpuPortChannel&&)            = default;
  GpuPortChannel& operator=(GpuPortChannel&&) = default;
  ~GpuPortChannel();

  // ── CPU proxy API ──────────────────────────────────────────────────────────

  // Drain up to maxCmds (0 = all) ring entries.
  //   type 0 (write):  cpu_.write(localOff, remoteOff, bytes) + cpu_.flush()
  //   type 1 (signal): cpu_.signal(epoch), then gpuSem_ = epoch
  // Returns the number of commands processed.
  int service(int maxCmds = 0);

  // Blocking service loop for a dedicated background thread.
  void serviceLoop();

  // Signal the service loop to exit.
  void stop() { stopFlag_ = true; }

  // ── GPU device handle ──────────────────────────────────────────────────────
  GpcDeviceHandle deviceHandle() const;

  // ── CPU passthrough (for mixed CPU/GPU use) ───────────────────────────────
  CpuPortChannel& cpu() { return cpu_; }

 private:
  CpuPortChannel  cpu_;

  // Ring buffer (pinned, device-mapped).
  void*         ringMapping_     = nullptr;
  GpcRingEntry* ring_            = nullptr;
  GpcRingEntry* ringDev_         = nullptr;
  bool          ringRegistered_  = false;

  void*        ctrlMapping_      = nullptr;
  GpcRingCtrl* ringCtrl_         = nullptr;
  GpcRingCtrl* ringCtrlDev_      = nullptr;
  bool         ctrlRegistered_   = false;

  // Device-mapped semaphore written by service() after executing a signal cmd.
  // GPU kernels spin on this via GpcDeviceHandle::wait().
  void*              semMapping_     = nullptr;
  uint64_t*          gpuSem_         = nullptr;   // host ptr (CPU writes)
  volatile uint64_t* gpuSemDev_      = nullptr;   // device ptr (GPU reads)
  bool               semRegistered_  = false;

  volatile bool stopFlag_ = false;
};

// ── Inline implementations ────────────────────────────────────────────────────

inline GpuPortChannel::GpuPortChannel(CpuPortChannel&& cpu, int cudaDevice)
    : cpu_(std::move(cpu)) {
  mscclpp::CudaDeviceGuard devGuard(cudaDevice);

  auto alloc_mapped = [](size_t sz) -> void* {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, sz) != 0)
      throw mscclpp::Error("posix_memalign failed", mscclpp::ErrorCode::SystemError);
    std::memset(p, 0, sz);
    MSCCLPP_CUDATHROW(cudaHostRegister(p, sz,
                                       cudaHostRegisterPortable |
                                       cudaHostRegisterMapped));
    return p;
  };

  // Ring entries.
  size_t ringBytes = kGpcRingSize * sizeof(GpcRingEntry);
  ringMapping_ = alloc_mapped(ringBytes);
  ring_ = static_cast<GpcRingEntry*>(ringMapping_);
  { void* dp = nullptr;
    MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, ringMapping_, 0));
    ringDev_ = static_cast<GpcRingEntry*>(dp); }

  // Ring control.
  ctrlMapping_ = alloc_mapped(sizeof(GpcRingCtrl));
  ringCtrl_ = static_cast<GpcRingCtrl*>(ctrlMapping_);
  { void* dp = nullptr;
    MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, ctrlMapping_, 0));
    ringCtrlDev_ = static_cast<GpcRingCtrl*>(dp); }

  // GPU-wait semaphore.
  semMapping_ = alloc_mapped(sizeof(uint64_t));
  gpuSem_ = static_cast<uint64_t*>(semMapping_);
  { void* dp = nullptr;
    MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dp, semMapping_, 0));
    gpuSemDev_ = static_cast<volatile uint64_t*>(dp); }
}

inline GpuPortChannel::~GpuPortChannel() {
  if (semMapping_)  { cudaHostUnregister(semMapping_);  ::free(semMapping_); }
  if (ctrlMapping_) { cudaHostUnregister(ctrlMapping_); ::free(ctrlMapping_); }
  if (ringMapping_) { cudaHostUnregister(ringMapping_); ::free(ringMapping_); }
}

inline int GpuPortChannel::service(int maxCmds) {
  uint64_t head = ringCtrl_->head;
  uint64_t tail = ringCtrl_->tail;
  int processed = 0;
  while (tail < head && (maxCmds == 0 || processed < maxCmds)) {
    GpcRingEntry const& e = ring_[tail & kGpcRingMask];
    if (e.type == 0) {
      cpu_.write(e.localOff, e.remoteOff, e.bytes);
      cpu_.flush();
    } else {
      // Execute the RDMA signal, then update the device-mapped GPU semaphore.
      cpu_.signal(e.epoch);
      // Also wait for the INBOUND signal from remote before notifying GPU.
      cpu_.wait(e.epoch);
      *gpuSem_ = e.epoch;  // GPU kernels polling gpuSemDev_ will unblock.
    }
    ++tail;
    ++processed;
  }
  ringCtrl_->tail = tail;
  return processed;
}

inline void GpuPortChannel::serviceLoop() {
  while (!stopFlag_) {
    if (service(64) == 0) {
#if defined(__x86_64__) || defined(__i386__)
      asm volatile("pause" ::: "memory");
#else
      asm volatile("" ::: "memory");
#endif
    }
  }
}

inline GpcDeviceHandle GpuPortChannel::deviceHandle() const {
  GpcDeviceHandle h;
  h.ring     = ringDev_;
  h.ringCtrl = ringCtrlDev_;
  h.ringMask = kGpcRingMask;
  h.gpuSem   = gpuSemDev_;
  return h;
}
