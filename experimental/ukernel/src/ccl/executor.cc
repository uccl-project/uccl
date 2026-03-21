#include "executor.h"
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

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

namespace {

enum class StepState : uint32_t { Pending, Ready, Running, Completed };
enum class OpRunState : uint32_t { Pending, Queued, Running, Completed };

struct InflightOp {
  size_t step_index = 0;
  size_t op_index = 0;
  Backend* backend = nullptr;
  BackendToken token{};
};

struct OpLocation {
  size_t step_index = 0;
  size_t op_offset = 0;
};

struct OpState {
  CollectivePlan plan;
  CollectiveConfig config;
  CollectiveOpStatus status = CollectiveOpStatus::Pending;
  std::vector<StepState> step_states;
  std::vector<uint32_t> step_remaining_predecessors;
  std::vector<std::vector<size_t>> step_successors;
  std::vector<size_t> step_remaining_ops;
  std::vector<std::vector<size_t>> step_op_indices;
  std::vector<OpLocation> op_locations;
  std::vector<OpRunState> op_states;
  std::vector<uint32_t> op_remaining_predecessors;
  std::vector<std::vector<size_t>> op_successors;
  std::vector<InflightOp> inflight;
  std::deque<size_t> ready_ops;
  size_t completed_steps = 0;
  std::string error;
};

Backend* select_backend(ExecutorBackends const& backends,
                        CollectiveConfig const& config,
                        ExecutionOp const& op) {
  BackendKind kind =
      resolve_backend_kind(config.requested_backend, op,
                           config.runtime_caps, config.backend_selector);

  Backend* preferred = nullptr;
  switch (kind) {
    case BackendKind::Device:
      preferred = backends.device;
      break;
    case BackendKind::Transport:
      preferred = backends.transport;
      break;
    case BackendKind::Auto:
      break;
  }

  if (preferred != nullptr && preferred->supports(op.kind)) return preferred;
  if (backends.fallback != nullptr && backends.fallback->supports(op.kind)) {
    return backends.fallback;
  }
  return nullptr;
}

std::vector<Backend*> backend_sources(ExecutorBackends const& backends) {
  std::vector<Backend*> out;
  if (backends.transport != nullptr) out.push_back(backends.transport);
  if (backends.device != nullptr && backends.device != backends.transport) {
    out.push_back(backends.device);
  }
  if (backends.fallback != nullptr &&
      backends.fallback != backends.transport &&
      backends.fallback != backends.device) {
    out.push_back(backends.fallback);
  }
  return out;
}

}  // namespace

struct Executor::Impl {
  explicit Impl(ExecutorBackends backends_in)
      : backends(backends_in), completion_sources(backend_sources(backends_in)) {}

  CollectiveOpHandle submit(CollectivePlan plan, CollectiveConfig config) {
    if (!ops.empty()) {
      throw std::runtime_error(
          "executor supports only one active collective at a time; release the "
          "previous handle before submitting another");
    }
    if (plan.steps.empty()) {
      throw std::invalid_argument("collective plan must contain at least one step");
    }

    CollectiveOpHandle handle{next_handle++};
    OpState state;
    state.plan = std::move(plan);
    state.config = std::move(config);
    state.status = CollectiveOpStatus::Running;
    state.step_states.resize(state.plan.steps.size(), StepState::Pending);
    state.step_remaining_predecessors.resize(state.plan.steps.size(), 0);
    state.step_successors.resize(state.plan.steps.size());
    state.step_remaining_ops.resize(state.plan.steps.size(), 0);
    state.step_op_indices.resize(state.plan.steps.size());

    std::unordered_map<uint32_t, size_t> op_id_to_index;

    for (size_t step_index = 0; step_index < state.plan.steps.size(); ++step_index) {
      auto const& step = state.plan.steps[step_index];
      state.step_remaining_predecessors[step_index] =
          static_cast<uint32_t>(step.predecessors.size());
      state.step_remaining_ops[step_index] = step.ops.size();
      for (uint32_t pred : step.predecessors) {
        if (pred >= state.plan.steps.size()) {
          throw std::invalid_argument("step predecessor out of range");
        }
        state.step_successors[pred].push_back(step_index);
      }
      for (size_t op_offset = 0; op_offset < step.ops.size(); ++op_offset) {
        auto const& op = step.ops[op_offset];
        size_t op_index = state.op_locations.size();
        auto [_, inserted] = op_id_to_index.emplace(op.op_id, op_index);
        if (!inserted) {
          throw std::invalid_argument("duplicate op id in collective plan");
        }
        state.step_op_indices[step_index].push_back(op_index);
        state.op_locations.push_back(OpLocation{step_index, op_offset});
        state.op_states.push_back(OpRunState::Pending);
        state.op_remaining_predecessors.push_back(0);
        state.op_successors.emplace_back();
      }
    }

    for (size_t op_index = 0; op_index < state.op_locations.size(); ++op_index) {
      auto const& location = state.op_locations[op_index];
      auto const& op =
          state.plan.steps[location.step_index].ops[location.op_offset];
      for (uint32_t dep : op.deps) {
        auto dep_it = op_id_to_index.find(dep);
        if (dep_it == op_id_to_index.end()) {
          throw std::invalid_argument("op dependency out of range");
        }
        ++state.op_remaining_predecessors[op_index];
        state.op_successors[dep_it->second].push_back(op_index);
      }
    }

    auto [it, inserted] = ops.emplace(handle.value, std::move(state));
    if (!inserted) {
      throw std::runtime_error("duplicate collective handle");
    }

    initialize_ready_work(it->second);
    drain_ready_ops(it->second);
    return handle;
  }

  bool poll(CollectiveOpHandle handle) {
    OpState& state = get(handle);
    if (state.status == CollectiveOpStatus::Completed ||
        state.status == CollectiveOpStatus::Failed) {
      return true;
    }

    bool progress = false;
    progress |= drain_backend_completions(state);
    if (!progress) {
      progress |= poll_inflight(state);
    }
    if (progress) {
      drain_ready_ops(state);
    }

    if (state.completed_steps == state.plan.steps.size()) {
      state.status = CollectiveOpStatus::Completed;
      return true;
    }
    return false;
  }

  void wait(CollectiveOpHandle handle) {
    while (!poll(handle)) {
      std::this_thread::yield();
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

  void initialize_ready_work(OpState& state) {
    for (size_t step_index = 0; step_index < state.plan.steps.size(); ++step_index) {
      if (state.step_remaining_predecessors[step_index] == 0) {
        mark_step_ready(state, step_index);
      }
    }
  }

  void mark_step_ready(OpState& state, size_t step_index) {
    if (state.step_states[step_index] != StepState::Pending) return;
    if (state.step_remaining_predecessors[step_index] != 0) return;
    if (state.step_remaining_ops[step_index] == 0) {
      complete_step(state, step_index);
      return;
    }

    state.step_states[step_index] = StepState::Ready;
    for (size_t op_index : state.step_op_indices[step_index]) {
      enqueue_op_if_ready(state, op_index);
    }
  }

  void enqueue_op_if_ready(OpState& state, size_t op_index) {
    if (state.op_states[op_index] != OpRunState::Pending) return;
    if (state.op_remaining_predecessors[op_index] != 0) return;

    size_t step_index = state.op_locations[op_index].step_index;
    StepState step_state = state.step_states[step_index];
    if (step_state != StepState::Ready && step_state != StepState::Running) return;

    state.op_states[op_index] = OpRunState::Queued;
    state.ready_ops.push_back(op_index);
  }

  void drain_ready_ops(OpState& state) {
    while (!state.ready_ops.empty() && state.status == CollectiveOpStatus::Running) {
      size_t op_index = state.ready_ops.front();
      state.ready_ops.pop_front();
      if (state.op_states[op_index] != OpRunState::Queued) continue;
      submit_op(state, op_index);
    }
  }

  void submit_op(OpState& state, size_t op_index) {
    auto const& location = state.op_locations[op_index];
    auto const& op =
        state.plan.steps[location.step_index].ops[location.op_offset];
    state.op_states[op_index] = OpRunState::Running;
    if (state.step_states[location.step_index] == StepState::Ready) {
      state.step_states[location.step_index] = StepState::Running;
    }

    Backend* backend = select_backend(backends, state.config, op);
    if (backend == nullptr) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "no backend available for op";
      return;
    }
    BackendToken token = backend->submit(op);
    state.inflight.push_back(
        InflightOp{location.step_index, op_index, backend, token});
  }

  bool drain_backend_completions(OpState& state) {
    bool progress = false;
    for (Backend* backend : completion_sources) {
      if (backend == nullptr) continue;
      BackendToken token{};
      while (backend->try_pop_completed(token)) {
        if (complete_inflight_by_token(state, backend, token)) {
          progress = true;
        }
      }
    }
    return progress;
  }

  bool poll_inflight(OpState& state) {
    std::vector<size_t> ready_indices;
    ready_indices.reserve(state.inflight.size());
    for (size_t i = 0; i < state.inflight.size(); ++i) {
      if (state.inflight[i].backend->poll(state.inflight[i].token)) {
        ready_indices.push_back(i);
      }
    }

    bool progress = false;
    for (size_t idx = ready_indices.size(); idx > 0; --idx) {
      complete_inflight(state, ready_indices[idx - 1]);
      progress = true;
      if (state.status == CollectiveOpStatus::Failed) return true;
    }
    return progress;
  }

  bool complete_inflight_by_token(OpState& state, Backend* backend,
                                  BackendToken token) {
    for (size_t i = 0; i < state.inflight.size(); ++i) {
      if (state.inflight[i].backend != backend) continue;
      if (state.inflight[i].token.value != token.value) continue;
      complete_inflight(state, i);
      return true;
    }
    return false;
  }

  void complete_inflight(OpState& state, size_t inflight_index) {
    InflightOp inflight = state.inflight[inflight_index];
    inflight.backend->release(inflight.token);
    state.inflight.erase(state.inflight.begin() + static_cast<long>(inflight_index));

    state.op_states[inflight.op_index] = OpRunState::Completed;
    for (size_t succ : state.op_successors[inflight.op_index]) {
      if (state.op_remaining_predecessors[succ] > 0) {
        --state.op_remaining_predecessors[succ];
      }
      enqueue_op_if_ready(state, succ);
    }

    size_t step_index = inflight.step_index;
    if (state.step_remaining_ops[step_index] > 0) {
      --state.step_remaining_ops[step_index];
    }
    if (state.step_remaining_ops[step_index] == 0) {
      complete_step(state, step_index);
    }
  }

  void complete_step(OpState& state, size_t step_index) {
    if (state.step_states[step_index] == StepState::Completed) return;
    state.step_states[step_index] = StepState::Completed;
    ++state.completed_steps;

    for (size_t succ : state.step_successors[step_index]) {
      if (state.step_remaining_predecessors[succ] > 0) {
        --state.step_remaining_predecessors[succ];
      }
      if (state.step_remaining_predecessors[succ] == 0) {
        mark_step_ready(state, succ);
      }
    }
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
  std::vector<Backend*> completion_sources;
  uint64_t next_handle = 1;
  std::unordered_map<uint64_t, OpState> ops;
};

Executor::Executor(ExecutorBackends backends) : impl_(new Impl(backends)) {}

Executor::~Executor() { delete impl_; }

CollectiveOpHandle Executor::submit(CollectivePlan plan) {
  CollectiveConfig config;
  config.nranks = plan.nranks;
  config.rank = plan.rank;
  config.channels = plan.channels;
  config.bytes_per_rank = plan.bytes_per_rank;
  config.chunk_bytes = plan.chunk_bytes;
  config.algorithm = plan.algorithm;
  return impl_->submit(std::move(plan), std::move(config));
}

CollectiveOpHandle Executor::submit_allreduce(CollectiveConfig const& config) {
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, config));
  return impl_->submit(std::move(plan), config);
}

CollectiveOpHandle Executor::submit_alltoall(CollectiveConfig const& config) {
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllToAll, config));
  return impl_->submit(std::move(plan), config);
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
