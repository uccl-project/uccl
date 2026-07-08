#pragma once

#include "../coll_types.h"
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace Transport {
class Communicator;
}
namespace CCL {

// ── Command descriptor ──────────────────────────────────────────────────

struct Cmd {
  OpKind kind;          // 4
  uint32_t src_buf;     // 4
  uint32_t dst_buf;     // 4
  uint32_t src_off;     // 4
  uint32_t dst_off;     // 4
  uint32_t bytes;       // 4
  uint32_t src_peer;    // 4
  uint32_t dst_peer;    // 4
  ReductionKind redop;  // 4
  uint8_t transport;    // 1 — 0=auto/device default, 1=IPC, 2=RDMA
  uint8_t _pad[3];      // 3 — alignment
  uint64_t tag;         // 8 — for Signal/SignalWait
};
// Total: 4*9 + 1 + 3 + 8 = 48 bytes

static_assert(sizeof(Cmd) <= 64, "Cmd too large");

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
  virtual bool supports(OpKind kind) const = 0;

  virtual void init(BufSpec bufs[3]) = 0;

  void set_comm(UKernel::Transport::Communicator* comm) { comm_ = comm; }

  virtual size_t enqueue(Cmd const* cmds, size_t n,
                         uint32_t* out_indices = nullptr) = 0;
  virtual size_t drain(uint32_t* completed, size_t max) = 0;
  virtual size_t capacity() const = 0;
  virtual void release(uint32_t cmd_idx) { (void)cmd_idx; }

 protected:
  UKernel::Transport::Communicator* comm_ = nullptr;
};

struct GpuSignalPeer {
  constexpr static uint32_t kPending = 0;
  constexpr static uint32_t kDone = 1;
  uint32_t* local = nullptr;
  uint32_t* remote = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
