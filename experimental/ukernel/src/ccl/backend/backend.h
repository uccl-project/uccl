#pragma once

#include "../coll_types.h"
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace CCL {

// ── Command descriptor ──────────────────────────────────────────────────

struct Cmd {
  OpKind kind;
  uint32_t src_buf;   // 1=input, 2=output, 3=scratch
  uint32_t dst_buf;
  uint32_t src_off;
  uint32_t dst_off;
  uint32_t bytes;
  uint32_t src_peer;  // ~0u = local
  uint32_t dst_peer;  // ~0u = local
  ReductionKind redop; // Reduce / RecvReduce
};

static_assert(sizeof(Cmd) <= 64, "Cmd too large");

// ── Command with caller-assigned global index (for async rings) ────────

struct CmdWithId {
  Cmd cmd;
  uint32_t caller_id;
};

// ── Buffer descriptor ───────────────────────────────────────────────────

struct BufSpec {
  void* ptr;
  size_t bytes;
};

// ── Batch-capable backend interface ─────────────────────────────────────
//
//   init()     — register input/output/scratch buffers once
//   enqueue()  — batch-submit N commands; returns # accepted (0 = full)
//   drain()    — harvest completed commands; returns their enqueue indices
//   capacity() — max concurrent commands the backend can absorb

class BatchBackend {
 public:
  virtual ~BatchBackend() = default;

  virtual char const* name() const = 0;
  virtual bool supports(OpKind kind) const = 0;

  virtual void init(BufSpec bufs[3]) = 0;

  // Returns the number of commands actually enqueued (<= n).
  // Returns 0 if the backend is completely full (backpressure).
  // If out_indices is non-null, fills backend-assigned cmd_idx per accepted Cmd.
  virtual size_t enqueue(Cmd const* cmds, size_t n,
                         uint32_t* out_indices = nullptr) = 0;

  // Returns the number of completed commands harvested.
  // out[i] = the index of the completed command in its original
  //          enqueue batch (0-based, monotonically increasing).
  virtual size_t drain(uint32_t* completed, size_t max) = 0;

  // Maximum number of commands that can be in-flight concurrently.
  virtual size_t capacity() const = 0;

  // Cleanup a specific command index (release resources).
  virtual void release(uint32_t cmd_idx) { (void)cmd_idx; }
};

struct GpuSignalPeer {
  constexpr static uint32_t kPending = 0;
  constexpr static uint32_t kDone = 1;
  uint32_t* local = nullptr;
  uint32_t* remote = nullptr;
};

struct BatchExecutorBackends {
  BatchBackend* transport = nullptr;
  BatchBackend* device = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
