// CpuPortChannel: CPU-initiated point-to-point inter-node RDMA channel.
//
// Encapsulates one directional RDMA connection between two nodes.  The CPU
// application thread calls write()/flush()/signal()/wait() directly — there
// is no proxy or background thread.  This matches the existing pattern in
// allgather_multinode.cu where the leader CPU calls Connection.write() after
// collecting local D2H completions.
//
// Protocol (semaphore-based, compatible with GpuPortChannel):
//   sender: write(localOff, remoteOff, bytes)  — IB RDMA write to remote data
//           flush()                            — push pending writes
//           signal(epoch)                      — RDMA-write epoch to remote's
//                                               GpcCtrl.semaphore + local ack
//   receiver: wait(epoch)                      — CPU spins on local semaphore
//
// GpcCtrl is a small struct (one cache-line) in POSIX-shm that is device-mapped
// so that GpuPortChannel's GPU wait() can spin on the same flag.
//
// Pair with GpuPortChannel (gpu_port_channel.hpp) for the GPU-initiated variant
// where the GPU posts RDMA commands to a ring and a proxy CPU thread executes
// them.  Both channels use the same GpcCtrl layout.

#pragma once

#include "lite_common.h"
#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>

// ── Shared semaphore ctrl (POSIX shm, device-mapped) ─────────────────────────
// The remote side writes its epoch into our GpcCtrl.semaphore via RDMA.
// The local side spins on it (CPU: volatile load; GPU: __device__ volatile load
// via device-mapped pointer).
struct GpcCtrl {
  alignas(64) volatile uint64_t semaphore;  // written by remote via RDMA
};
static_assert(sizeof(GpcCtrl) <= 4096, "GpcCtrl must fit in one page");

// ── CpuPortChannel ────────────────────────────────────────────────────────────
class CpuPortChannel {
 public:
  // Lightweight constructor from pre-created mscclpp objects.
  // Callers should use the bootstrapped factory (see createFromBootstrap()) or
  // build the mscclpp connection themselves and call this directly.
  CpuPortChannel(
      mscclpp::Connection            connection,
      mscclpp::RegisteredMemory      localData,
      mscclpp::RegisteredMemory      remoteData,
      mscclpp::RegisteredMemory      localCtrl,
      mscclpp::RegisteredMemory      remoteCtrl,
      GpcCtrl*                       localSemaphore)  // host ptr for CPU wait()
      : conn_(std::move(connection))
      , localData_(std::move(localData))
      , remoteData_(std::move(remoteData))
      , localCtrl_(std::move(localCtrl))
      , remoteCtrl_(std::move(remoteCtrl))
      , localSem_(localSemaphore) {}

  CpuPortChannel() = default;
  CpuPortChannel(CpuPortChannel const&) = delete;
  CpuPortChannel& operator=(CpuPortChannel const&) = delete;
  CpuPortChannel(CpuPortChannel&&)            = default;
  CpuPortChannel& operator=(CpuPortChannel&&) = default;

  // ── Data transport ─────────────────────────────────────────────────────────

  // IB RDMA-write bytes from localData[localOff..] to remoteData[remoteOff..].
  void write(size_t localOff, size_t remoteOff, size_t bytes) {
    conn_.write(remoteData_, remoteOff, localData_, localOff, bytes);
  }

  // Flush all pending writes (posts accumulated WRs to the HCA).
  void flush() { conn_.flush(); }

  // Write data + flush as one call.
  void writeAndFlush(size_t localOff, size_t remoteOff, size_t bytes) {
    write(localOff, remoteOff, bytes);
    flush();
  }

  // Signal remote: RDMA-write epoch into remote's GpcCtrl.semaphore.
  // The remote side observes this via wait(epoch).
  void signal(uint64_t epoch) {
    // Write the epoch value into our local signalValue and RDMA-write it to the
    // remote's semaphore field.
    signalValue_ = epoch;
    conn_.write(remoteCtrl_, offsetof(GpcCtrl, semaphore),
                localCtrl_, offsetof(GpcCtrl, semaphore), sizeof(uint64_t));
    conn_.flush();
  }

  // CPU spin until local GpcCtrl.semaphore >= epoch (written by remote's signal).
  void wait(uint64_t epoch) const {
    constexpr int kYieldAfter = 65536;
    int spins = 0;
    while (localSem_->semaphore < epoch) {
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
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  // ── Accessors ──────────────────────────────────────────────────────────────
  mscclpp::Connection&       connection()    { return conn_; }
  mscclpp::RegisteredMemory& localDataMem()  { return localData_; }
  mscclpp::RegisteredMemory& remoteDataMem() { return remoteData_; }
  mscclpp::RegisteredMemory& localCtrlMem()  { return localCtrl_; }
  mscclpp::RegisteredMemory& remoteCtrlMem() { return remoteCtrl_; }

 private:
  mscclpp::Connection       conn_;
  mscclpp::RegisteredMemory localData_;
  mscclpp::RegisteredMemory remoteData_;
  mscclpp::RegisteredMemory localCtrl_;
  mscclpp::RegisteredMemory remoteCtrl_;
  GpcCtrl*                  localSem_    = nullptr;
  uint64_t                  signalValue_ = 0;  // staging area for signal RDMA src
};
