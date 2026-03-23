#pragma once

#include "backend/backend.h"
#include "plan.h"
#include "selector.h"
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace CCL {

enum class CollectiveOpStatus : uint32_t {
  Pending,
  Running,
  Completed,
  Failed,
};

struct CollectiveConfig {
  int nranks = 1;
  int rank = 0;
  uint32_t num_flows = 1;
  size_t tensor_bytes = 0;
  size_t tile_bytes = 0;
  size_t staging_bytes = 0;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
};

PlanRequest make_plan_request(CollectiveKind kind, CollectiveConfig const& config);

struct CollectiveOpHandle {
  uint64_t value = 0;
};

class Executor {
 public:
  // Executor owns runtime scheduling for the primitive DAG emitted by the
  // planner. Ops become ready when all dependency counts reach zero, and
  // backend completions unlock successor ops.
  // CCL v1 keeps one collective active per executor instance.
  explicit Executor(ExecutorBackends backends);
  ~Executor();

  Executor(Executor const&) = delete;
  Executor& operator=(Executor const&) = delete;

  CollectiveOpHandle submit(CollectivePlan plan);
  CollectiveOpHandle submit_allreduce(CollectiveConfig const& config);
  CollectiveOpHandle submit_alltoall(CollectiveConfig const& config);
  bool poll(CollectiveOpHandle handle);
  void wait(CollectiveOpHandle handle);
  void release(CollectiveOpHandle handle);

  CollectiveOpStatus status(CollectiveOpHandle handle) const;
  size_t inflight_steps(CollectiveOpHandle handle) const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace CCL
}  // namespace UKernel
