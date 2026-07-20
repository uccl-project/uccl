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
enum class PutPath : uint8_t { Device = 0, Ipc = 1, Rdma = 2, None = 3 };

// Command descriptor

struct Cmd {
  ExecOpKind kind;      // 4
  uint32_t src_buf;     // 4
  uint32_t dst_buf;     // 4
  uint32_t src_off;     // 4
  uint32_t dst_off;     // 4
  uint32_t bytes;       // 4
  uint32_t src_peer;    // 4
  uint32_t dst_peer;    // 4
  ReductionKind redop;  // 4
  PutPath put_path;     // 1 — Device/IPC/RDMA for ops
  // kCmdFlagPutSignal: this Put carries its partner Signal's tag (in
  // Cmd::tag); the transport emits the signal once the data lands.
  uint8_t flags;
  // WaitSignal: expected tag arrivals (0/1 = 1). A fused signal group
  // delivers one arrival per tile, so the wait counts group_size.
  uint16_t wait_count;
  uint64_t tag;         // 8 — for Signal/SignalWait/PutSignal
};
// Total: 4*9 + 1 + 1 + 2 + 8 = 48 bytes

static_assert(sizeof(Cmd) <= 64, "Cmd too large");

// Cmd::flags bits
inline constexpr uint8_t kCmdFlagPutSignal = 1u << 0;

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
