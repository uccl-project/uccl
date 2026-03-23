#pragma once

#include "backend.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <unordered_map>

namespace UKernel {
namespace Device {
class WorkerPool;
}
namespace CCL {

class DeviceBackend final : public Backend {
 public:
  DeviceBackend(UKernel::Device::WorkerPool* worker_pool,
                CollectiveMemory memory, int dtype, int reduce_type);

  char const* name() const override;
  void validate(ExecutionPlan const& plan) const override;
  bool supports(ExecOpKind kind) const override;
  BackendToken submit(ExecOp const& op) override;
  bool poll(BackendToken token) override;
  bool try_pop_completed(BackendToken& token) override;
  void release(BackendToken token) override;

 private:
  struct SubmittedTask {
    uint32_t fifo_id;
    uint64_t task_id;
    bool completion_queued = false;
  };

  void* byte_offset(void* base, size_t offset) const;
  void const* byte_offset(void const* base, size_t offset) const;
  void* resolve_mutable(BufferRef const& ref, size_t bytes) const;
  void const* resolve_const(BufferRef const& ref, size_t bytes) const;
  uint32_t fifo_id_for(ExecOp const& op);

  UKernel::Device::WorkerPool* worker_pool_;
  CollectiveMemory memory_{};
  int dtype_;
  int reduce_type_;
  uint64_t next_token_ = 1;
  uint32_t next_fifo_cursor_ = 0;
  std::unordered_map<uint32_t, uint32_t> lane_fifo_assignments_;
  std::unordered_map<uint64_t, SubmittedTask> submitted_;
  std::deque<uint64_t> completed_tokens_;
};

}  // namespace CCL
}  // namespace UKernel
