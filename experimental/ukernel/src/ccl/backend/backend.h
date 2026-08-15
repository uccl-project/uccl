#pragma once

#include "../coll_types.h"
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace Transport {
class Communicator;
}
namespace CCL {

// Put op routing: which backend handles the data transfer.
// Values match the uint8_t stored in Cmd::put_path.
// PutPath is defined in coll_types.h (plans carry per-op hints).

// Command descriptor

struct Cmd {
  ExecOpKind kind;      // 4
  uint32_t src_buf;     // 4
  uint32_t src2_buf;    // 4 — fused-reduce local contribution (Input)
  uint32_t dst_buf;     // 4
  uint32_t bytes;       // 4
  uint32_t src_peer;    // 4
  uint32_t dst_peer;    // 4
  uint32_t copy_dst_buf;  // 4 — fused reduce+copy: peer accum target
  uint32_t copy_dst_peer; // 4
  uint32_t flag_slot;     // 4 — device-flag slot (WaitSignal poll /
                          //     fused reduce write), ~0u = unused
  uint32_t flag_count;    // 4 — WaitSignal: consecutive slots to poll
  ReductionKind redop;  // 4
  ScalarType dtype;     // 4 — element type for device-kernel reduce/copy
  PutPath put_path;     // 1 — Device/IPC/RDMA for ops
  // kCmdFlagPutSignal: this Put carries its partner Signal's tag (in
  // Cmd::tag); the transport emits the signal once the data lands.
  uint8_t flags;
  // WaitSignal: expected tag arrivals (0/1 = 1). A fused signal group
  // delivers one arrival per tile, so the wait counts group_size.
  uint16_t wait_count;
  uint64_t tag;      // 8 — for Signal/SignalWait/PutSignal
  uint64_t src_off;  // 8 — byte offset within src_buf's allocation
  uint64_t dst_off;  // 8 — byte offset within dst_buf's allocation
  uint64_t copy_dst_off;  // 8 — fused reduce+copy target offset
};
// Total: 4*12 + 1 + 1 + 2 + 8*4 = 84 bytes

static_assert(sizeof(Cmd) <= 96, "Cmd too large");

// Cmd::flags bits
inline constexpr uint8_t kCmdFlagPutSignal = 1u << 0;
// kCmdFlagImmWait: this WaitSignal expects RDMA write-with-imm arrivals
// from fused PutSignal puts. Immediates carry only the tag's low 32 bits
// (the run epoch lives in the high bits), which collide across runs, so
// matching is per-peer FIFO in arrival order — Cmd::tag then carries the
// UNSALTED tag and wait_count counts the group's fused puts (one imm
// each).
inline constexpr uint8_t kCmdFlagImmWait = 1u << 1;
// kCmdFlagReduce3Way: the Reduce writes dst = src op src2 (fresh, no dst
// read) instead of dst = dst op src. src = the peer's buffer (src_peer),
// src2 = this rank's local Input at src_off (fused out-of-place reduce).
inline constexpr uint8_t kCmdFlagReduce3Way = 1u << 2;
// kCmdFlagReduceCopy: the Reduce task also copies dst to the peer
// (copy_dst_*). The data-ready signal is a separate host-written Signal
// op (B300 has no GPU-mapped signal ring, so the kernel cannot write it).
inline constexpr uint8_t kCmdFlagReduceCopy = 1u << 3;
// kCmdFlagCopySignal: the Put is a fused AG copy — a device task that
// copies to the peer (dst_peer) and device-writes the completion flag
// (flag_slot, tag) when done. No CE, no host signal op.
inline constexpr uint8_t kCmdFlagCopySignal = 1u << 4;

struct CmdWithId {
  Cmd cmd;
  uint32_t caller_id;
};

struct BufSpec {
  void* ptr;
  size_t bytes;
};

class BatchBackend {
 public:
  virtual ~BatchBackend() = default;
  virtual char const* name() const = 0;
  virtual bool supports(ExecOpKind kind) const = 0;
  void set_comm(UKernel::Transport::Communicator* comm) { comm_ = comm; }

  // Backend API (called directly by SprayExecutor)
  virtual size_t do_enqueue(Cmd const* cmds, size_t n,
                            uint32_t* out_indices = nullptr) = 0;
  virtual size_t do_drain(uint32_t* completed, size_t max) = 0;
  virtual size_t capacity() const = 0;
  virtual void release(uint32_t cmd_idx) { (void)cmd_idx; }

  // Whether this backend can carry a Put's partner signal tag itself
  // (Cmd::kCmdFlagPutSignal): DeviceBackend's kernels write the tag
  // into the peer's shm signal ring after the copy. Default: no.
  virtual bool can_fuse_put_signal(int peer) const {
    (void)peer;
    return false;
  }

  // Reserve-then-enqueue API for ops whose completion may arrive
  // synchronously during enqueue (e.g. same-host IPC signals). The
  // executor publishes its slot-table entry between reserve_slot() and
  // do_enqueue_reserved(), so the drain side never observes a
  // completion for an unpublished slot. Default: unsupported
  // (kInvalidBeIdx) — the executor then falls back to plain do_enqueue.
  static constexpr uint32_t kInvalidBeIdx = ~0u;
  virtual uint32_t reserve_slot() { return kInvalidBeIdx; }
  virtual bool do_enqueue_reserved(Cmd const& cmd, uint32_t be_idx) {
    (void)cmd;
    (void)be_idx;
    return false;
  }

  // Batch variants. reserve_slots() fills out[0..k) and returns k;
  // do_enqueue_reserved_batch() submits previously reserved ops and
  // returns the accepted prefix length — slots at or beyond the return
  // value were NOT submitted and must be released by the caller.
  virtual size_t reserve_slots(uint32_t* out, size_t n) {
    size_t k = 0;
    while (k < n && (out[k] = reserve_slot()) != kInvalidBeIdx) ++k;
    return k;
  }
  virtual size_t do_enqueue_reserved_batch(Cmd const* cmds,
                                           uint32_t const* be_idx, size_t n) {
    size_t k = 0;
    while (k < n && do_enqueue_reserved(cmds[k], be_idx[k])) ++k;
    return k;
  }

 protected:
  UKernel::Transport::Communicator* comm_ = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
