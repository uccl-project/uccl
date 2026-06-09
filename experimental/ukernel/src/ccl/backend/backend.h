#pragma once

#include "../coll_types.h"
#include "../lower.h"
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

struct BackendToken {
  uint64_t value = 0;
  bool failed = false;
};

struct GpuSignalPeer {
  void* local = nullptr;
  void* remote = nullptr;
};

struct OpBindings {
  void const* resolved_src = nullptr;
  void* resolved_dst = nullptr;
  int src_device = -1;
  int dst_device = -1;
  uint64_t signal_seq = 0;
  uint32_t stream_index = 0;
};

inline CollectiveBufferRole buf_role(OpKind kind, bool is_src,
                                     bool copy_from_staging) {
  switch (kind) {
    case OpKind::Send:
    case OpKind::Recv:
    case OpKind::RecvReduce:
    case OpKind::Reduce:
      return CollectiveBufferRole::Input;
    case OpKind::Copy:
      return is_src ? (copy_from_staging ? CollectiveBufferRole::Scratch
                                         : CollectiveBufferRole::Input)
                    : CollectiveBufferRole::Output;
  }
  return CollectiveBufferRole::Input;
}

// Minimal backend interface:
//   validate  — one-time init (memory bindings, peer paths, worker warmup)
//   submit    — enqueue one op, returns opaque token (value==0 = backpressure)
//   drain     — harvest all completed ops; backend cleans its own resources
//   stop      — end-of-stream cleanup (optional)
class Backend {
 public:
  virtual ~Backend() = default;

  virtual char const* name() const = 0;
  virtual void validate(TiledResult const& tiled,
                         void* input_ptr, void* output_ptr,
                         void* scratch_ptr) = 0;
  virtual bool supports(OpKind kind) const = 0;
  virtual BackendToken submit(Op const& op, OpBindings const& bind,
                               void* input_ptr, void* output_ptr,
                               void* scratch_ptr) = 0;
  virtual size_t drain(BackendToken* out, size_t max_count) = 0;
  virtual void stop(uint32_t stream_id) { (void)stream_id; }
};

struct ExecutorBackends {
  Backend* transport = nullptr;
  Backend* device = nullptr;
  Backend* rdma_copy = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
