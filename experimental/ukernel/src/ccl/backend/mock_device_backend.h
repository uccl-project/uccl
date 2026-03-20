#pragma once

#include "../backend.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace UKernel {
namespace CCL {

// Simplified mock device backend for testing that doesn't depend on external libraries
class MockDeviceBackend final : public Backend {
 public:
  explicit MockDeviceBackend(
      void* workerPool,  // Mock parameter - won't actually use real worker pool
      CollectiveBuffers buffers, 
      int dtype,      // Mock parameter - will use our own mock type
      int reduce_type, // Mock parameter - will use our own mock type
      uint32_t num_blocks = 1);

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
  bool poll(BackendToken token) override;
  void release(BackendToken token) override;

 private:
  struct SubmittedTask {
    uint32_t block_id;
    uint64_t task_id;
  };

  void* byte_offset(void* base, size_t offset) const;
  void const* byte_offset(void const* base, size_t offset) const;
  void* resolve_dst(BufferRole role, size_t offset) const;
  void const* resolve_src(BufferRole role, size_t offset) const;

  void* workerPool_;
  CollectiveBuffers buffers_{};
  int dtype_;
  int reduce_type_;
  uint32_t num_blocks_ = 1;
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, SubmittedTask> submitted_;
};

}  // namespace CCL
}  // namespace UKernel