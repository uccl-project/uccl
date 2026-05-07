#pragma once

#include "backend/backend.h"
#include "plan.h"
#include "selector.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace UKernel {
namespace Transport {
struct CommunicatorConfig;
class Communicator;
}  // namespace Transport
namespace CCL {

enum class CollectiveOpStatus : uint32_t {
  Queued,
  Running,
  Completed,
  Failed,
};

struct CollectiveConfig {
  int nranks = 1;
  int rank = 0;
  uint32_t num_flows = 1;
  size_t tensor_bytes = 0;
  // Optional alltoall-specific byte sizes. When unset, tensor_bytes is used.
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  size_t staging_bytes = 0;
  // Optional alltoall-specific per-peer split sizes in bytes.
  std::vector<size_t> input_split_bytes;
  std::vector<size_t> output_split_bytes;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  ScalarType dtype = ScalarType::Float32;
  ReductionKind reduction = ReductionKind::Sum;
};

struct ExecutorConfig {
  int gpu_id = 0;
  int rank = 0;
  int world_size = 1;
  std::shared_ptr<UKernel::Transport::CommunicatorConfig> communicator_config;
  uint32_t device_task_capacity = 4096;
  uint32_t max_device_fifos = 8;
  uint32_t threads_per_block = 256;
  uint32_t fifo_capacity = 64;
  uint32_t smem_size = 0;
};

PlanRequest make_plan_request(CollectiveKind kind,
                              CollectiveConfig const& config,
                              CollectiveBufferRoles const& roles);

struct CollectiveOpHandle {
  uint64_t value = 0;
};

class Executor {
 public:
  // Executor owns runtime scheduling for the primitive DAG emitted by the
  // planner. submit() enqueues a collective, and a dedicated progress thread
  // drives one queued collective at a time until completion. Ops become ready
  // when all dependency counts reach zero, and backend completions unlock
  // successor ops.
  explicit Executor(
      ExecutorBackends backends,
      // Callback contract: second argument is remote `buffer_id` (not MR id).
      std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
          resolve_ipc_buffer_pointer = {});
  Executor(ExecutorConfig const& config = {});
  ~Executor();

  Executor(Executor const&) = delete;
  Executor& operator=(Executor const&) = delete;

  // Each submit binds one concrete registered-buffer snapshot to the logical
  // collective plan. Executor itself no longer owns mutable global collective
  // memory state.
  CollectiveOpHandle submit(CollectivePlan plan,
                            std::shared_ptr<CollectiveBinding> runtime_binding);
  CollectiveOpHandle submit_allreduce(
      CollectiveConfig const& config,
      std::shared_ptr<CollectiveBinding> runtime_binding);
  CollectiveOpHandle submit_alltoall(
      CollectiveConfig const& config,
      std::shared_ptr<CollectiveBinding> runtime_binding);
  // poll() is now a non-blocking terminal-state query. Execution progresses on
  // the internal progress thread rather than on the caller thread.
  bool poll(CollectiveOpHandle handle);
  void wait(CollectiveOpHandle handle);
  void release(CollectiveOpHandle handle);

  CollectiveOpStatus status(CollectiveOpHandle handle) const;
  std::string error_message(CollectiveOpHandle handle) const;
  size_t inflight_steps(CollectiveOpHandle handle) const;
  UKernel::Transport::Communicator* communicator();
  UKernel::Transport::Communicator const* communicator() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace CCL
}  // namespace UKernel
