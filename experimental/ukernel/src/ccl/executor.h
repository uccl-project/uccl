#pragma once

#include "backend/backend.h"
#include "plan.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
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

using CollectiveOpHandle = uint64_t;
inline constexpr CollectiveOpHandle kInvalidHandle = 0;

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

// Per-invocation mutable execution state.
// The CollectivePlan is immutable and can be shared across runs.
struct CollectiveRun {
  CollectiveOpHandle handle = kInvalidHandle;
  CollectiveOpStatus status = CollectiveOpStatus::Queued;
  CollectivePlan plan;
  CollectiveBinding* binding = nullptr;
  std::string error_message;

  // Per-op tracking
  std::vector<bool> completed;
  std::vector<BackendToken> tokens;
  std::vector<Backend*> op_backend;

  struct TokenKey {
    Backend* backend;
    uint64_t value;
    bool operator==(TokenKey const& o) const {
      return backend == o.backend && value == o.value;
    }
  };
  struct TokenKeyHash {
    size_t operator()(TokenKey const& k) const {
      return std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(k.backend)) ^
             std::hash<uint64_t>()(k.value);
    }
  };
  std::unordered_map<TokenKey, size_t, TokenKeyHash> token_to_op_idx;
  std::vector<size_t> inflight_op_indices;
  std::vector<std::vector<uint32_t>> flow_ops;
  std::vector<size_t> flow_head;
  size_t completed_count = 0;
  size_t total_ops = 0;
};

class Executor {
 public:
  explicit Executor(
      ExecutorBackends backends,
      std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
          resolve_ipc_buffer_pointer = {});

  explicit Executor(ExecutorConfig const& config = {});
  ~Executor();

  Executor(Executor const&) = delete;
  Executor& operator=(Executor const&) = delete;

  // === Asynchronous submission ===
  CollectiveOpHandle submit_allreduce(CollectiveConfig const& config,
                                      CollectiveBinding& binding);
  CollectiveOpHandle submit_alltoall(CollectiveConfig const& config,
                                     CollectiveBinding& binding);

  // === Progress inquiry ===
  CollectiveOpStatus status(CollectiveOpHandle handle) const;
  bool poll(CollectiveOpHandle handle);
  void progress();
  bool wait(CollectiveOpHandle handle,
            std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
  void release(CollectiveOpHandle handle);
  std::string error_message(CollectiveOpHandle handle) const;

  // Synchronous execution of a pre-built plan (convenience for bindings).
  void run_plan(CollectivePlan const& plan, CollectiveBinding& binding);

  // === Utilities ===
  size_t active_count() const;
  UKernel::Transport::Communicator* communicator();
  UKernel::Transport::Communicator const* communicator() const;

 private:
  CollectiveRun* get_run(CollectiveOpHandle handle);
  CollectiveRun const* get_run(CollectiveOpHandle handle) const;
  CollectiveOpHandle submit_collective(CollectiveKind kind,
                                       CollectiveConfig const& config,
                                       CollectiveBinding& binding);
  void advance_run(CollectiveRun& run);

  ExecutorBackends backends_{};
  std::vector<Backend*> completion_sources_;
  std::unique_ptr<Backend> owned_transport_backend_;
  std::unique_ptr<Backend> owned_device_backend_;
  std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
      resolve_ipc_buffer_pointer_;
  std::unordered_map<CollectiveOpHandle, CollectiveRun> runs_;
  uint64_t next_handle_ = 1;

  // Plan cache keyed by a hash of the config shape.
  struct PlanCacheKey {
    CollectiveKind kind;
    bool inplace;
    CollectiveConfig config;
    bool operator==(PlanCacheKey const& o) const;
  };
  struct PlanCacheKeyHash {
    size_t operator()(PlanCacheKey const& key) const;
  };
  std::unordered_map<PlanCacheKey, CollectivePlan, PlanCacheKeyHash>
      plan_cache_;
};

}  // namespace CCL
}  // namespace UKernel
