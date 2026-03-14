#include "executor.h"
#include <cstddef>
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
  auto rewrite_allgather_rdma_step = [&](CollectiveStep& step) {
    if (!step.has_forward_chunk) {
      throw std::invalid_argument("RDMA allgather step missing forward chunk");
    }
    if (step.ops.size() != 1 || step.ops.front().kind != ExecutionOpKind::PkCopy) {
      throw std::invalid_argument(
          "RDMA allgather rewrite expects a single copy op per step");
    }

    ExecutionOp send_op;
    send_op.op_id = step.ops.front().op_id;
    send_op.kind = ExecutionOpKind::RdmaSend;
    send_op.src_rank = step.forward_src_rank;
    send_op.dst_rank = step.forward_dst_rank;
    send_op.chunk = step.forward_chunk;
    send_op.flags = step.ops.front().flags;
    send_op.src_role = step.forward_src_role;
    send_op.dst_role = BufferRole::FinalOutput;

    ExecutionOp recv_op = step.ops.front();
    recv_op.kind = ExecutionOpKind::RdmaRecv;
    recv_op.src_role = BufferRole::None;

    step.ops.clear();
    step.ops.push_back(std::move(send_op));
    step.ops.push_back(std::move(recv_op));
  };

  for (auto& step : plan.steps) {
    for (auto& op : step.ops) {
      if (op.kind != ExecutionOpKind::PkCopy) continue;

      auto selected = resolve_backend_kind(config.requested_backend, true,
                                           op.chunk.size_bytes,
                                           config.runtime_caps,
                                           config.backend_selector);
      if (selected == BackendKind::Rdma) {
        if (plan.collective != CollectiveKind::AllGather) {
          throw std::invalid_argument(
              "RDMA path is only wired for AllGather in this phase");
        }
        if (backends.transport == nullptr ||
            !backends.transport->supports(ExecutionOpKind::RdmaSend) ||
            !backends.transport->supports(ExecutionOpKind::RdmaRecv)) {
          throw std::invalid_argument(
              "transport backend requested but unavailable");
        }
        rewrite_allgather_rdma_step(step);
        break;
      }
      if (selected == BackendKind::Ce) {
        if (backends.copy_engine != nullptr &&
            backends.copy_engine->supports(ExecutionOpKind::CeCopy)) {
          op.kind = ExecutionOpKind::CeCopy;
        } else if (config.requested_backend == BackendKind::Ce) {
          throw std::invalid_argument("CE backend requested but unavailable");
        }
      }
    }
  }
}

enum class StepState : uint32_t { Pending, Running, Completed };
enum class OpRunState : uint32_t { Pending, Running, Completed };

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
  CollectiveOpStatus status = CollectiveOpStatus::Pending;
  std::vector<StepState> step_states;
  std::vector<uint32_t> step_remaining_predecessors;
  std::vector<std::vector<size_t>> step_successors;
  std::vector<size_t> step_remaining_ops;
  std::vector<OpLocation> op_locations;
  std::vector<OpRunState> op_states;
  std::vector<uint32_t> op_remaining_predecessors;
  std::vector<std::vector<size_t>> op_successors;
  std::vector<InflightOp> inflight;
  size_t completed_steps = 0;
  std::string error;
};

Backend* select_backend(ExecutorBackends const& backends, ExecutionOpKind kind) {
  Backend* preferred = nullptr;
  switch (kind) {
    case ExecutionOpKind::RdmaSend:
    case ExecutionOpKind::RdmaRecv:
      preferred = backends.transport;
      break;
    case ExecutionOpKind::CeCopy:
      preferred = backends.copy_engine;
      break;
    case ExecutionOpKind::PkCopy:
    case ExecutionOpKind::PkReduce:
      preferred = backends.persistent_kernel;
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
    state.step_states.resize(state.plan.steps.size(), StepState::Pending);
    state.step_remaining_predecessors.resize(state.plan.steps.size(), 0);
    state.step_successors.resize(state.plan.steps.size());
    state.step_remaining_ops.resize(state.plan.steps.size(), 0);

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
    submit_ready_work(it->second);
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

  void submit_ready_work(OpState& state) {
    bool progress = true;
    while (progress && state.status == CollectiveOpStatus::Running) {
      progress = false;

      for (size_t step_index = 0; step_index < state.plan.steps.size(); ++step_index) {
        if (state.step_states[step_index] != StepState::Pending) continue;
        if (state.step_remaining_predecessors[step_index] != 0) continue;
        if (state.step_remaining_ops[step_index] != 0) continue;
        complete_step(state, step_index);
        progress = true;
      }

      for (size_t op_index = 0; op_index < state.op_locations.size(); ++op_index) {
        if (state.op_states[op_index] != OpRunState::Pending) continue;
        size_t step_index = state.op_locations[op_index].step_index;
        if (state.step_remaining_predecessors[step_index] != 0) continue;
        if (state.op_remaining_predecessors[op_index] != 0) continue;
        submit_op(state, op_index);
        if (state.status != CollectiveOpStatus::Running) return;
        progress = true;
      }
    }
  }

  void submit_op(OpState& state, size_t op_index) {
    auto const& location = state.op_locations[op_index];
    auto const& op =
        state.plan.steps[location.step_index].ops[location.op_offset];
    state.op_states[op_index] = OpRunState::Running;
    if (state.step_states[location.step_index] == StepState::Pending) {
      state.step_states[location.step_index] = StepState::Running;
    }

    Backend* backend = select_backend(backends, op.kind);
    if (backend == nullptr) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "no backend available for op";
      return;
    }
    BackendToken token = backend->submit(op);
    state.inflight.push_back(
        InflightOp{location.step_index, op_index, backend, token});
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
    }

    size_t step_index = inflight.step_index;
    if (state.step_remaining_ops[step_index] > 0) {
      --state.step_remaining_ops[step_index];
    }
    if (state.step_remaining_ops[step_index] == 0) {
      complete_step(state, step_index);
    } else {
      submit_ready_work(state);
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
    }
    submit_ready_work(state);
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
