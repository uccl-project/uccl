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
  LogicalOpKind kind;   // 4 — logical op; execution details via flags
  uint32_t src_buf;     // 4
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
  // Execution-side orthogonals (channel/mechanism), chosen at enqueue
  // time; the logical kind already expresses the op's behavior.
  uint8_t flags;
  // Wait: expected tag arrivals (0/1 = 1).
  uint16_t wait_count;
  uint64_t tag;      // 8 — for Signal/SignalWait/PutSignal
  uint64_t src_off;  // 8 — byte offset within src_buf's allocation
  uint64_t dst_off;  // 8 — byte offset within dst_buf's allocation
  uint64_t copy_dst_off;  // 8 — fused reduce+copy target offset
  // RDMA proxy: the device writes rdma_fused_cmd_index into the D2H ring
  // after reducing; the host posts the linked put from the CmdPool.
  void* rdma_fused_ring = nullptr;
  uint64_t rdma_fused_cmd_index = UINT64_MAX;
};
// Total: 4*11 + 1 + 1 + 2 + 8*6 = 104 bytes

static_assert(sizeof(Cmd) == 104, "Cmd size changed");

// Wait matches RDMA write-with-imm values (epoch-encoded tag).
inline constexpr uint8_t kCmdFlagImmWait = 1u << 0;
// Device put also writes the peer completion flag (fused AG copy).
inline constexpr uint8_t kCmdFlagCopySignal = 1u << 1;
// Device reduce notifies the CCL proxy via the D2H ring; the proxy
// posts the RDMA put.
inline constexpr uint8_t kCmdFlagRdmaFusedProxy = 1u << 2;

struct CmdWithId {
  Cmd cmd;
  uint32_t caller_id;
};

struct BufSpec {
  void* ptr;
  size_t bytes;
};

// Pure abstract backend interface: each backend owns its submission and
// completion queues; threading and queue management live in
// SprayExecutor (do_enqueue / do_drain / capacity / supports).
class BatchBackend {
 public:
  virtual ~BatchBackend() = default;
  virtual char const* name() const = 0;
  virtual bool supports(LogicalOpKind kind) const = 0;
  void set_comm(UKernel::Transport::Communicator* comm) { comm_ = comm; }

  // Backend API (called directly by SprayExecutor)
  virtual size_t do_enqueue(Cmd const* cmds, size_t n,
                            uint32_t* out_indices = nullptr) = 0;
  virtual size_t do_drain(uint32_t* completed, size_t max) = 0;
  virtual size_t capacity() const = 0;
  virtual void release(uint32_t cmd_idx) { (void)cmd_idx; }

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
