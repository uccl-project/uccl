#pragma once

#include "backend.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace UKernel {
namespace Device {
class WorkerPool;
}
namespace CCL {

struct DeviceBackendConfig {
  uint32_t task_capacity = 4096;
  uint32_t max_fifos = 8;
  uint32_t threads_per_block = 256;
  uint32_t fifo_capacity = 64;
  uint32_t smem_size = 0;
};

class DeviceBackend final : public Backend {
 public:
  explicit DeviceBackend(DeviceBackendConfig const& config = {});
  ~DeviceBackend() override;

  char const* name() const override;
  void validate(ExecutionPlan const& plan,
                CollectiveBinding& binding) const override;
  bool supports(ExecOpKind kind) const override;
  BackendToken submit(ExecOp const& op, CollectiveBinding& binding) override;
  bool poll(BackendToken token) override;
  bool try_pop_completed(BackendToken& token) override;
  void release(BackendToken token) override;
  void stop(uint32_t flow_id) override;

 private:
  struct SubmittedTask {
    uint32_t fifo_id;
    uint64_t task_id;
    uint32_t flow_id;
    uint32_t args_id;
    bool args_released = false;
    bool completion_queued = false;
  };

  struct ActiveFlow {
    uint32_t fifo_id = 0;
    uint32_t inflight = 0;
  };

  void* byte_offset(void* base, size_t offset) const;
  void const* byte_offset(void const* base, size_t offset) const;
  void* resolve_mutable(CollectiveBinding const& binding, BufferRef const& ref,
                        size_t bytes) const;
  void const* resolve_const(CollectiveBinding const& binding,
                            BufferRef const& ref,
                            size_t bytes) const;
  void ensure_device_context() const;
  void ensure_runtime();
  uint32_t acquire_fifo(uint32_t flow_id, uint32_t num_blocks);
  void release_task_args(SubmittedTask& task);
  void stop_flow(uint32_t flow_id);
  uint32_t suggested_num_blocks(ExecOp const& op) const;

  DeviceBackendConfig config_{};
  bool owns_task_manager_ = false;
  int local_device_idx_ = 0;
  int sm_count_ = 1;
  std::unique_ptr<UKernel::Device::WorkerPool> worker_pool_;
  uint64_t next_token_ = 1;
  std::unordered_map<uint32_t, ActiveFlow> active_flows_;
  std::deque<uint32_t> free_fifos_;
  std::unordered_map<uint64_t, SubmittedTask> submitted_;
  std::deque<uint64_t> completed_tokens_;
};

}  // namespace CCL
}  // namespace UKernel
