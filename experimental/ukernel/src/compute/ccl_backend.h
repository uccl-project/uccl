#pragma once

#include "../ccl/backend.h"
#include "persistent.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace UKernel {
namespace Compute {

class ComputePersistentKernelBackend final : public UKernel::CCL::Backend {
 public:
  ComputePersistentKernelBackend(
      PersistentKernel<Task>& kernel, void* dst_base, void const* src_base,
      DataType dtype, ReduceType reduce_type = ReduceType::Sum,
      TransferPath transfer_path = TransferPath::Auto, uint32_t num_blocks = 1,
      void* staging_base = nullptr);

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

  PersistentKernel<Task>& kernel_;
  void* dst_base_ = nullptr;
  void const* src_base_ = nullptr;
  void* staging_base_ = nullptr;
  DataType dtype_ = DataType::Fp32;
  ReduceType reduce_type_ = ReduceType::Sum;
  TransferPath transfer_path_ = TransferPath::Auto;
  uint32_t num_blocks_ = 1;
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, SubmittedTask> submitted_;
};

class ComputeCopyEngineBackend final : public UKernel::CCL::Backend {
 public:
  ComputeCopyEngineBackend(void* dst_base, void const* src_base, int dst_device = -1,
                           int src_device = -1, void* staging_base = nullptr,
                           gpuStream_t stream = nullptr);
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
  void set_device(int device) const;

  void* dst_base_ = nullptr;
  void const* src_base_ = nullptr;
  void* staging_base_ = nullptr;
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
