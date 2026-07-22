#pragma once

#include "backend.h"
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace UKernel {
namespace Device {
class WorkerPool;
struct TaskArgs;
enum class TaskType : uint64_t;
}  // namespace Device
namespace CCL {

struct DeviceBackendConfig {
  uint32_t task_capacity = 4096;
  uint32_t max_fifos = 2;
  uint32_t threads_per_block = 256;
  uint32_t blocks_per_worker = 1;
  uint32_t fifo_capacity = 64;
  uint32_t smem_size = 0;
  uint32_t bytes_per_block = 0;  // 0=auto, >0=override
  // Grace period (µs) of continuous fifo emptiness after which a
  // persistent worker kernel exits (relaunched on next enqueue).
  // 0 = always resident. Enable for torch coexistence — see
  // WorkerPool::Config::idleExitAfterUs.
  uint32_t idle_exit_after_us = 0;
};

class DeviceBackend final : public BatchBackend {
 public:
  explicit DeviceBackend(DeviceBackendConfig const& cfg = {});
  ~DeviceBackend() override;

  char const* name() const override { return "device"; }
  bool supports(ExecOpKind kind) const override;

  size_t do_enqueue(Cmd const* cmds, size_t n,
                    uint32_t* out_indices = nullptr) override;
  uint32_t reserve_slot() override;
  bool do_enqueue_reserved(Cmd const& cmd, uint32_t be_idx) override;
  size_t do_enqueue_reserved_batch(Cmd const* cmds, uint32_t const* be_idx,
                                   size_t n) override;
  size_t do_drain(uint32_t* completed, size_t max) override;
  size_t capacity() const override;
  // True when the peer's IPC signal ring is GPU-mapped, so a fused
  // PutSignal (CollPut task) can write the tag from the kernel.
  bool can_fuse_put_signal(int peer) const override;

 private:
  void ensure_runtime();
  // Fill TaskArgs/TaskType for a device op; returns false for op kinds
  // this backend does not handle (caller skips them, matching the
  // historical behavior). Throws on unresolvable buffer pointers.
  bool build_task(Cmd const& c, Device::TaskArgs& args, Device::TaskType& tt);

  DeviceBackendConfig cfg_;
  int sm_count_ = 1;
  int device_idx_ = 0;

  bool owns_task_manager_ = false;

  std::unique_ptr<UKernel::Device::WorkerPool> worker_pool_;

  // FIFO management
  uint32_t next_fifo_ = 0;
  struct CmdRec {
    uint32_t fifo_id;
    uint64_t task_id;
    uint32_t args_id;
    uint32_t cmd_idx;
  };
  // Per-FIFO submission-ordered queues. Each FIFO completes tasks in
  // order (monotonic tail counter), so do_drain only pops done prefixes
  // instead of scanning every pending record.
  std::vector<std::deque<CmdRec>> pending_by_fifo_;
  size_t pending_total_ = 0;
  std::mutex pending_mu_;

  // Resolved remote IPC pointer cache — written once, read without lock
  struct ResolvedRemote {
    int remote_rank = -1;
    uint32_t buffer_id = 0;
    void* ptr = nullptr;
    int device_idx = -1;
  };
  std::vector<ResolvedRemote> resolved_remote_cache_;

  // Local buffer base-pointer cache — populated once per collective, read
  // lock-free
  static constexpr size_t kMaxLocalBufs = 8;
  void* local_ptr_cache_[kMaxLocalBufs] = {};

  // Global command sequence counter; atomic so reserve_slot() is
  // lock-free. Failed submissions leave harmless gaps.
  std::atomic<uint32_t> cmd_next_{0};
  uint32_t cmd_done_ = 0;  // completed up to this point
};

}  // namespace CCL
}  // namespace UKernel
