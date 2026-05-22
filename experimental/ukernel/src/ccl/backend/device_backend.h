#pragma once

#include "backend.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

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
  void validate(CollectivePlan const& plan,
                CollectiveBinding& binding) override;
  bool supports(OpKind kind) const override;
  BackendToken submit(Op const& op, CollectiveBinding& binding) override;
  size_t drain(BackendToken* out, size_t max_count) override;
  void stop(uint32_t flow_id) override;

 private:
  struct TaskRec {
    uint64_t task_id;
    uint32_t flow_id;
    uint32_t args_id;
  };

  struct FlowRec {
    uint32_t fifo_id = 0;
    uint32_t inflight = 0;
  };

  void ensure_runtime();
  uint32_t acquire_fifo(uint32_t flow_id, uint32_t num_blocks);
  void release_task_args(uint32_t args_id);
  void stop_flow(uint32_t flow_id);
  uint32_t suggested_num_blocks(Op const& op) const;

  DeviceBackendConfig config_{};
  bool owns_task_manager_ = false;
  int local_device_idx_ = 0;
  int sm_count_ = 1;
  std::unique_ptr<UKernel::Device::WorkerPool> worker_pool_;
  uint64_t next_token_ = 1;
  std::unordered_map<uint32_t, FlowRec> active_flows_;
  std::vector<uint32_t> free_fifos_;
  // Per-FIFO submitted task map.  Indexed by fifo_id; empty when no worker.
  std::vector<std::unordered_map<uint64_t, TaskRec>> submitted_per_fifo_;
};

}  // namespace CCL
}  // namespace UKernel
