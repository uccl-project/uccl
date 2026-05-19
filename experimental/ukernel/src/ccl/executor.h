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

struct CollectiveConfig {
  int nranks = 1;
  int rank = 0;
  uint32_t num_flows = 1;
  size_t tensor_bytes = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  size_t staging_bytes = 0;
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
                              CollectiveConfig const& config, bool inplace);

class Executor {
 public:
  explicit Executor(
      ExecutorBackends backends,
      std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
          resolve_ipc_buffer_pointer = {});
  Executor(ExecutorConfig const& config = {});
  ~Executor();

  Executor(Executor const&) = delete;
  Executor& operator=(Executor const&) = delete;

  void run(CollectivePlan plan, CollectiveBinding& binding);
  void allreduce(CollectiveConfig const& config, CollectiveBinding& binding);
  void alltoall(CollectiveConfig const& config, CollectiveBinding& binding);

  UKernel::Transport::Communicator* communicator();
  UKernel::Transport::Communicator const* communicator() const;

 private:
  ExecutorBackends backends_{};
  std::vector<Backend*> completion_sources_;
  std::unique_ptr<Backend> owned_transport_backend_;
  std::unique_ptr<Backend> owned_device_backend_;
  std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
      resolve_ipc_buffer_pointer_;
};

}  // namespace CCL
}  // namespace UKernel
