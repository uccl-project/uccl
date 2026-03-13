#include "executor.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

PlanRequest make_plan_request(CollectiveKind kind, CollectiveConfig const& config) {
  PlanRequest request;
  request.collective = kind;
  request.algorithm = config.algorithm;
  request.nranks = config.nranks;
  request.rank = config.rank;
  request.channels = config.channels;
  request.bytes_per_rank = config.bytes_per_rank;
  request.chunk_bytes = config.chunk_bytes;
  return request;
}

void assign_copy_backends(CollectivePlan& plan, CollectiveConfig const& config,
                          ExecutorBackends const& backends) {
  for (auto& step : plan.steps) {
    for (auto& op : step.ops) {
      if (op.kind != ExecutionOpKind::PkCopy) continue;

      auto selected = UKernel::Compute::resolve_cpu_backend_kind(
          config.requested_cpu_backend, true, op.chunk.size_bytes,
          config.device_caps, config.cpu_selector);
      if (selected == UKernel::Compute::CpuBackendKind::Ce) {
        if (backends.ce != nullptr &&
            backends.ce->supports(ExecutionOpKind::CeCopy)) {
          op.kind = ExecutionOpKind::CeCopy;
        } else if (config.requested_cpu_backend ==
                   UKernel::Compute::CpuBackendKind::Ce) {
          throw std::invalid_argument("CE backend requested but unavailable");
        }
      }
    }
  }
}

enum class StepState : uint32_t { Pending, Running, Completed };

struct InflightOp {
  size_t step_index = 0;
  Backend* backend = nullptr;
  BackendToken token{};
};

struct OpState {
  CollectivePlan plan;
  CollectiveOpStatus status = CollectiveOpStatus::Pending;
  std::vector<uint32_t> remaining_predecessors;
  std::vector<std::vector<size_t>> successors;
  std::vector<StepState> step_states;
  std::vector<size_t> step_completed_ops;
  std::vector<InflightOp> inflight;
  size_t completed_steps = 0;
  std::string error;
};

Backend* select_backend(ExecutorBackends const& backends, ExecutionOpKind kind) {
  Backend* preferred = nullptr;
  switch (kind) {
    case ExecutionOpKind::RdmaSend:
    case ExecutionOpKind::RdmaRecv:
      preferred = backends.rdma;
      break;
    case ExecutionOpKind::CeCopy:
      preferred = backends.ce;
      break;
    case ExecutionOpKind::PkCopy:
    case ExecutionOpKind::PkReduce:
      preferred = backends.persistent;
      break;
    case ExecutionOpKind::EventWait:
    case ExecutionOpKind::Barrier:
      preferred = backends.fallback;
      break;
  }

  if (preferred != nullptr && preferred->supports(kind)) return preferred;
  if (backends.fallback != nullptr && backends.fallback->supports(kind)) {
    return backends.fallback;
  }
  return nullptr;
}

}  // namespace

struct Executor::Impl {
  explicit Impl(ExecutorBackends backends_in) : backends(backends_in) {}

  CollectiveOpHandle submit(CollectivePlan plan) {
    if (plan.steps.empty()) {
      throw std::invalid_argument("collective plan must contain at least one step");
    }

    CollectiveOpHandle handle{next_handle++};
    OpState state;
    state.plan = std::move(plan);
    state.status = CollectiveOpStatus::Running;
    state.remaining_predecessors.resize(state.plan.steps.size(), 0);
    state.successors.resize(state.plan.steps.size());
    state.step_states.resize(state.plan.steps.size(), StepState::Pending);
    state.step_completed_ops.resize(state.plan.steps.size(), 0);

    for (size_t step_index = 0; step_index < state.plan.steps.size(); ++step_index) {
      auto const& step = state.plan.steps[step_index];
      state.remaining_predecessors[step_index] =
          static_cast<uint32_t>(step.predecessors.size());
      for (uint32_t pred : step.predecessors) {
        if (pred >= state.plan.steps.size()) {
          throw std::invalid_argument("step predecessor out of range");
        }
        state.successors[pred].push_back(step_index);
      }
    }

    auto [it, inserted] = ops.emplace(handle.value, std::move(state));
    if (!inserted) {
      throw std::runtime_error("duplicate collective handle");
    }
    submit_ready_steps(it->second);
    return handle;
  }

  bool poll(CollectiveOpHandle handle) {
    OpState& state = get(handle);
    if (state.status == CollectiveOpStatus::Completed ||
        state.status == CollectiveOpStatus::Failed) {
      return true;
    }

    std::vector<size_t> ready_indices;
    ready_indices.reserve(state.inflight.size());
    for (size_t i = 0; i < state.inflight.size(); ++i) {
      if (state.inflight[i].backend->poll(state.inflight[i].token)) {
        ready_indices.push_back(i);
      }
    }

    for (size_t idx = ready_indices.size(); idx > 0; --idx) {
      complete_inflight(state, ready_indices[idx - 1]);
      if (state.status == CollectiveOpStatus::Failed) return true;
    }

    if (state.completed_steps == state.plan.steps.size()) {
      state.status = CollectiveOpStatus::Completed;
      return true;
    }
    return false;
  }

  void wait(CollectiveOpHandle handle) {
    while (!poll(handle)) {
    }
  }

  void release(CollectiveOpHandle handle) {
    auto it = ops.find(handle.value);
    if (it == ops.end()) return;

    for (auto& inflight : it->second.inflight) {
      inflight.backend->release(inflight.token);
    }
    ops.erase(it);
  }

  CollectiveOpStatus status(CollectiveOpHandle handle) const {
    return get_const(handle).status;
  }

  size_t inflight_steps(CollectiveOpHandle handle) const {
    return get_const(handle).inflight.size();
  }

  void submit_ready_steps(OpState& state) {
    bool progress = true;
    while (progress && state.status == CollectiveOpStatus::Running) {
      progress = false;
      for (size_t step_index = 0; step_index < state.plan.steps.size(); ++step_index) {
        if (state.step_states[step_index] != StepState::Pending) continue;
        if (state.remaining_predecessors[step_index] != 0) continue;
        submit_step(state, step_index);
        progress = true;
      }
    }
  }

  void submit_step(OpState& state, size_t step_index) {
    auto const& step = state.plan.steps[step_index];
    state.step_states[step_index] = StepState::Running;

    for (auto const& op : step.ops) {
      Backend* backend = select_backend(backends, op.kind);
      if (backend == nullptr) {
        state.status = CollectiveOpStatus::Failed;
        state.error = "no backend available for op";
        return;
      }
      BackendToken token = backend->submit(op);
      state.inflight.push_back(InflightOp{step_index, backend, token});
    }

    if (step.ops.empty()) {
      complete_step(state, step_index);
    }
  }

  void complete_inflight(OpState& state, size_t inflight_index) {
    InflightOp inflight = state.inflight[inflight_index];
    inflight.backend->release(inflight.token);
    state.inflight.erase(state.inflight.begin() + static_cast<long>(inflight_index));

    size_t step_index = inflight.step_index;
    ++state.step_completed_ops[step_index];
    if (state.step_completed_ops[step_index] ==
        state.plan.steps[step_index].ops.size()) {
      complete_step(state, step_index);
    }
  }

  void complete_step(OpState& state, size_t step_index) {
    if (state.step_states[step_index] == StepState::Completed) return;
    state.step_states[step_index] = StepState::Completed;
    ++state.completed_steps;

    for (size_t succ : state.successors[step_index]) {
      if (state.remaining_predecessors[succ] > 0) {
        --state.remaining_predecessors[succ];
      }
    }
    submit_ready_steps(state);
  }

  OpState& get(CollectiveOpHandle handle) {
    auto it = ops.find(handle.value);
    if (it == ops.end()) {
      throw std::invalid_argument("unknown collective handle");
    }
    return it->second;
  }

  OpState const& get_const(CollectiveOpHandle handle) const {
    auto it = ops.find(handle.value);
    if (it == ops.end()) {
      throw std::invalid_argument("unknown collective handle");
    }
    return it->second;
  }

  ExecutorBackends backends{};
  uint64_t next_handle = 1;
  std::unordered_map<uint64_t, OpState> ops;
};

Executor::Executor(ExecutorBackends backends) : impl_(new Impl(backends)) {}

Executor::~Executor() { delete impl_; }

CollectiveOpHandle Executor::submit(CollectivePlan plan) {
  return impl_->submit(std::move(plan));
}

CollectiveOpHandle Executor::submit_allgather(CollectiveConfig const& config) {
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllGather, config));
  assign_copy_backends(plan, config, impl_->backends);
  return submit(std::move(plan));
}

CollectiveOpHandle Executor::submit_allreduce(CollectiveConfig const& config) {
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, config));
  assign_copy_backends(plan, config, impl_->backends);
  return submit(std::move(plan));
}

bool Executor::poll(CollectiveOpHandle handle) { return impl_->poll(handle); }

void Executor::wait(CollectiveOpHandle handle) { impl_->wait(handle); }

void Executor::release(CollectiveOpHandle handle) { impl_->release(handle); }

CollectiveOpStatus Executor::status(CollectiveOpHandle handle) const {
  return impl_->status(handle);
}

size_t Executor::inflight_steps(CollectiveOpHandle handle) const {
  return impl_->inflight_steps(handle);
}

}  // namespace CCL
}  // namespace UKernel
