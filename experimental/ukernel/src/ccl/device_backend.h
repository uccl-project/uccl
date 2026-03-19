#pragma once

#include "backend.h"
#include "../device/worker.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace UKernel {
namespace CCL {

// Adapts persistent-kernel execution into the CCL backend interface. It stays
// in ccl/ because it is specific to collective execution, not a pure device
// primitive.
class PersistentKernelBackend final : public Backend {
 public:
  PersistentKernelBackend(
      UKernel::Device::WorkerPool& workerPool,
      CollectiveBuffers buffers, UKernel::Device::DataType dtype,
      UKernel::Device::ReduceType reduce_type =
          UKernel::Device::ReduceType::Sum,
      uint32_t num_blocks = 1);

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
  bool poll(BackendToken token) override;
  void release(BackendToken token) override;

 private:
  struct SubmittedTask {
    uint32_t block_id = 0;
    uint64_t task_id = 0;
  };

  void* byte_offset(void* base, size_t offset) const;
  void const* byte_offset(void const* base, size_t offset) const;
  void* resolve_dst(BufferRole role, size_t offset) const;
  void const* resolve_src(BufferRole role, size_t offset) const;

  UKernel::Device::WorkerPool& workerPool_;
  CollectiveBuffers buffers_{};
  UKernel::Device::DataType dtype_ = UKernel::Device::DataType::Fp32;
  UKernel::Device::ReduceType reduce_type_ =
      UKernel::Device::ReduceType::Sum;
  uint32_t num_blocks_ = 1;
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, SubmittedTask> submitted_;
};

// Adapts copy-engine copies into the CCL backend interface.
class CopyEngineBackend final : public Backend {
 public:
  CopyEngineBackend(CollectiveBuffers buffers, int dst_device = -1,
                    int src_device = -1, gpuStream_t stream = nullptr);
  ~CopyEngineBackend() override;

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
  bool poll(BackendToken token) override;
  void release(BackendToken token) override;

  uint64_t submissions() const { return submissions_; }

 private:
  struct SubmittedCopy {
    gpuEvent_t event = nullptr;
  };

  void* byte_offset(void* base, size_t offset) const;
  void const* byte_offset(void const* base, size_t offset) const;
  void* resolve_dst(BufferRole role, size_t offset) const;
  void const* resolve_src(BufferRole role, size_t offset) const;
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

}  // namespace CCL
}  // namespace UKernel
