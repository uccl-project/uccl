#pragma once

#include "backend.h"
#include <cstdint>
#include <memory>
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
  uint32_t bytes_per_block = 0;   // 0=auto, >0=override
};

class DeviceBackend final : public BatchBackend {
 public:
  explicit DeviceBackend(DeviceBackendConfig const& cfg = {});
  ~DeviceBackend() override;

  char const* name() const override { return "device"; }
  bool supports(OpKind kind) const override;

  void init(BufSpec bufs[3]) override;
  size_t enqueue(Cmd const* cmds, size_t n) override;
  size_t drain(uint32_t* completed, size_t max) override;
  size_t capacity() const override;

  void set_signal_buffers(std::vector<GpuSignalPeer> const& peers);

 private:
  void ensure_runtime();

  DeviceBackendConfig cfg_;
  int sm_count_ = 1;
  int device_idx_ = 0;

  BufSpec bufs_[3] = {};
  bool inited_ = false;
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
  std::vector<CmdRec> pending_;  // indexed by internal seq

  std::vector<GpuSignalPeer> gpu_signal_bufs_;

  uint32_t cmd_next_ = 0;        // global command sequence counter
  uint32_t cmd_done_ = 0;        // completed up to this point
};

}  // namespace CCL
}  // namespace UKernel
