#pragma once

#include "../backend.h"
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
  DeviceBackend(
      UKernel::Device::WorkerPool* worker_pool,
      CollectiveMemory memory,
      int dtype,
      int reduce_type,
      uint32_t num_blocks = 1);

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
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
  void* resolve_mutable(MemoryRef const& ref, size_t bytes) const;
  void const* resolve_const(MemoryRef const& ref, size_t bytes) const;
  uint32_t fifo_id_for(ExecutionOp const& op) const;

  UKernel::Device::WorkerPool* worker_pool_;
  CollectiveMemory memory_{};
  int dtype_;
  int reduce_type_;
  uint32_t num_blocks_ = 1;
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, SubmittedTask> submitted_;
  std::deque<uint64_t> completed_tokens_;
};

}  // namespace CCL
}  // namespace UKernel
