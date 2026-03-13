#pragma once

#include "../ccl/backend.h"
#include "persistent.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace UKernel {
namespace Compute {

struct CollectiveBuffers {
  void const* local_input = nullptr;
  void const* remote_input = nullptr;
  void const* remote_reduced = nullptr;
  void* final_output = nullptr;
  void* recv_staging = nullptr;
};

class ComputePersistentKernelBackend final : public UKernel::CCL::Backend {
 public:
  ComputePersistentKernelBackend(
      PersistentKernel<Task>& kernel, CollectiveBuffers buffers,
      DataType dtype, ReduceType reduce_type = ReduceType::Sum,
      TransferPath transfer_path = TransferPath::Auto, uint32_t num_blocks = 1);

  char const* name() const override;
  bool supports(UKernel::CCL::ExecutionOpKind kind) const override;
  UKernel::CCL::BackendToken submit(UKernel::CCL::ExecutionOp const& op) override;
  bool poll(UKernel::CCL::BackendToken token) override;
  void release(UKernel::CCL::BackendToken token) override;

 private:
  struct SubmittedTask {
    uint32_t block_id = 0;
    uint64_t task_id = 0;
  };

  void* byte_offset(void* base, size_t offset) const;
  void const* byte_offset(void const* base, size_t offset) const;
  void* resolve_dst(UKernel::CCL::BufferRole role, size_t offset) const;
  void const* resolve_src(UKernel::CCL::BufferRole role, size_t offset) const;

  PersistentKernel<Task>& kernel_;
  CollectiveBuffers buffers_{};
  DataType dtype_ = DataType::Fp32;
  ReduceType reduce_type_ = ReduceType::Sum;
  TransferPath transfer_path_ = TransferPath::Auto;
  uint32_t num_blocks_ = 1;
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, SubmittedTask> submitted_;
};

class ComputeCopyEngineBackend final : public UKernel::CCL::Backend {
 public:
  ComputeCopyEngineBackend(CollectiveBuffers buffers, int dst_device = -1,
                           int src_device = -1, gpuStream_t stream = nullptr);
  ~ComputeCopyEngineBackend() override;

  char const* name() const override;
  bool supports(UKernel::CCL::ExecutionOpKind kind) const override;
  UKernel::CCL::BackendToken submit(UKernel::CCL::ExecutionOp const& op) override;
  bool poll(UKernel::CCL::BackendToken token) override;
  void release(UKernel::CCL::BackendToken token) override;

  uint64_t submissions() const { return submissions_; }

 private:
  struct SubmittedCopy {
    gpuEvent_t event = nullptr;
  };

  void* byte_offset(void* base, size_t offset) const;
  void const* byte_offset(void const* base, size_t offset) const;
  void* resolve_dst(UKernel::CCL::BufferRole role, size_t offset) const;
  void const* resolve_src(UKernel::CCL::BufferRole role, size_t offset) const;
  void set_device(int device) const;

  CollectiveBuffers buffers_{};
  int dst_device_ = 0;
  int src_device_ = 0;
  gpuStream_t stream_ = nullptr;
  bool owns_stream_ = false;
  uint64_t next_token_ = 1;
  uint64_t submissions_ = 0;
  std::unordered_map<uint64_t, SubmittedCopy> submitted_;
};

}  // namespace Compute
}  // namespace UKernel
