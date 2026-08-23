/*
 * RDMA fused proxy (CCL layer).
 *
 * This is NOT a new transport. It is an alternate producer that feeds the
 * existing TransportBackend:
 *
 *   GPU kernel writes cmd_index into D2H ring
 *   -> RdmaFusedProxy::progress() pops index
 *   -> looks up Cmd in RdmaFusedCmdPool
 *   -> calls TransportBackend::do_enqueue() (or Communicator send API)
 *   -> TransportBackend / Communicator / RdmaTransportAdapter handle the
 *      actual RDMA put and completion as usual.
 *
 * The transport layer stays generic and unchanged.
 */
#pragma once

#include "backend.h"
#include "../../include/transport.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace CCL {

// Host-visible ring of command indices shared with the persistent kernel.
class RdmaFusedRing {
 public:
  explicit RdmaFusedRing(size_t capacity = 4096);
  ~RdmaFusedRing();

  RdmaFusedRing(RdmaFusedRing const&) = delete;
  RdmaFusedRing& operator=(RdmaFusedRing const&) = delete;

  // Device-side handle (to be passed to the kernel later).
  void* device_handle();

  // Host-side pop. Returns false when empty.
  bool pop(uint64_t& index);

  // Host-side push for bring-up/testing.
  void push_from_host(uint64_t index);

 private:
  struct DeviceHandle;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Stable command pool: DeviceBackend writes Cmd entries here and the kernel
// only references them by index. Slots are not freed until the consumer has
// posted the RDMA operation.
//
// Each slot also carries the executor completion context (opaque run
// pointer + op_idx) so that after the RDMA put is accepted, the caller can
// publish a BeSlot entry and let the existing drain_tpt_loop process the
// completion normally.
struct FusedCmdSlot {
  Cmd cmd;
  void* run = nullptr;
  uint32_t op_idx = 0;
  PutPath put_path = PutPath::Rdma;
};

class RdmaFusedCmdPool {
 public:
  explicit RdmaFusedCmdPool(size_t capacity = 4096);
  ~RdmaFusedCmdPool();

  // Allocate a slot and copy cmd + completion context into it.
  // Returns index or UINT64_MAX.
  uint64_t alloc(Cmd const& cmd, void* run, uint32_t op_idx,
                 PutPath put_path = PutPath::Rdma);

  // Get a slot by index. The caller must ensure the index is valid.
  FusedCmdSlot const& get(uint64_t index) const;

  // Release a slot after the RDMA put has been posted.
  void release(uint64_t index);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class RdmaFusedProxy {
 public:
  // first_attempt == true only for the initial ring pop: the executor
  // mirrors its normal acceptance accounting exactly once per command.
  // Retries (post rejected earlier) must skip that accounting.
  using PostFn = std::function<bool(uint64_t cmd_index, bool first_attempt)>;

  RdmaFusedProxy(PostFn post_fn, size_t ring_capacity = 4096,
                 size_t pool_capacity = 4096);
  ~RdmaFusedProxy();

  RdmaFusedRing& ring() { return ring_; }
  RdmaFusedCmdPool& pool() { return pool_; }

  // Drain pending fused commands. Returns number of successfully posted puts.
  size_t progress();

 private:
  RdmaFusedRing ring_;
  RdmaFusedCmdPool pool_;
  PostFn post_fn_;
  // Per-peer FIFO of commands whose post was rejected. The head MUST
  // succeed before any later command to the same peer is posted:
  // RDMA write-with-imm arrivals are matched per-peer in issue order, so
  // posting a later imm first would strand the receiver's FIFO wait.
  // progress() is only ever called from the drain_tpt_loop thread, so no
  // locking is needed.
  std::unordered_map<int, std::deque<uint64_t>> pending_;
};

}  // namespace CCL
}  // namespace UKernel
