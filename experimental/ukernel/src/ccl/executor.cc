#include "executor.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

PlanRequest make_plan_request(CollectiveKind kind,
                              CollectiveConfig const& config) {
  PlanRequest request;
  request.collective = kind;
  request.algorithm = config.algorithm;
  request.nranks = config.nranks;
  request.rank = config.rank;
  request.num_flows = config.num_flows;
  request.tensor_bytes = config.tensor_bytes;
  request.tile_bytes = config.tile_bytes;
  request.staging_bytes = config.staging_bytes;
  return request;
}

namespace {

struct InflightOp {
  Backend* backend = nullptr;
  BackendToken token{};
};

struct OpState {
  uint32_t remaining_deps = 0;
  std::vector<uint32_t> successors;
  bool submitted = false;
  bool completed = false;
  InflightOp inflight{};
};

struct HandleState {
  ExecutionPlan exec_plan;
  CollectiveOpStatus status = CollectiveOpStatus::Pending;
  std::vector<OpState> op_states;
  std::unordered_map<uint64_t, uint32_t> inflight_lookup;
  std::deque<uint32_t> ready_ops;
  size_t completed_ops = 0;
  std::string error;
};

uint64_t inflight_key(Backend const* backend, BackendToken token) {
  uint64_t backend_bits =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(backend));
  return backend_bits ^ (token.value + 0x9e3779b97f4a7c15ULL +
                         (backend_bits << 6) + (backend_bits >> 2));
}

std::vector<Backend*> backend_sources(ExecutorBackends const& backends) {
  std::vector<Backend*> out;
  if (backends.transport != nullptr) out.push_back(backends.transport);
  if (backends.device != nullptr && backends.device != backends.transport) {
    out.push_back(backends.device);
  }
  if (backends.fallback != nullptr && backends.fallback != backends.transport &&
      backends.fallback != backends.device) {
    out.push_back(backends.fallback);
  }
  return out;
}

void validate_backends(std::vector<Backend*> const& backends,
                       ExecutionPlan const& exec_plan) {
  for (Backend* backend : backends) {
    if (backend != nullptr) {
      backend->validate(exec_plan);
    }
  }
}

bool is_transport_op(ExecOpKind kind) {
  return kind == ExecOpKind::TransportSend || kind == ExecOpKind::TransportRecv;
}

Backend* pick_backend(ExecutorBackends const& backends, ExecOpKind kind) {
  if (is_transport_op(kind)) {
    return backends.transport != nullptr ? backends.transport
                                         : backends.fallback;
  }
  return backends.device != nullptr ? backends.device : backends.fallback;
}

}  // namespace

struct Executor::Impl {
  explicit Impl(ExecutorBackends backends_in)
      : backends(backends_in),
        completion_sources(backend_sources(backends_in)) {}

  CollectiveOpHandle submit(CollectivePlan plan) {
    if (!handles.empty()) {
      throw std::runtime_error(
          "executor supports only one active collective at a time; release the "
          "previous handle before submitting another");
    }
    if (plan.ops.empty()) {
      throw std::invalid_argument(
          "collective plan must contain at least one op");
    }

    ExecutionPlan exec_plan = lower_plan(plan);
    if (exec_plan.ops.empty()) {
      throw std::invalid_argument(
          "lowered execution plan must contain at least one op");
    }
    validate_backends(completion_sources, exec_plan);

    HandleState state;
    state.exec_plan = std::move(exec_plan);
    state.status = CollectiveOpStatus::Running;
    state.op_states.resize(state.exec_plan.ops.size());

    for (size_t index = 0; index < state.exec_plan.ops.size(); ++index) {
      ExecOp const& op = state.exec_plan.ops[index];
      if (op.op_id != index) {
        throw std::invalid_argument(
            "execution plan op ids must be dense and ordered");
      }

      Backend* backend = pick_backend(backends, op.kind);
      if (backend == nullptr || !backend->supports(op.kind)) {
        throw std::invalid_argument("no backend available for execution op");
      }

      OpState& op_state = state.op_states[index];
      op_state.remaining_deps = static_cast<uint32_t>(op.deps.size());
      for (uint32_t dep : op.deps) {
        if (dep >= state.exec_plan.ops.size()) {
          throw std::invalid_argument("execution plan dependency out of range");
        }
        if (dep >= op.op_id) {
          throw std::invalid_argument(
              "execution plan dependencies must reference earlier ops");
        }
        state.op_states[dep].successors.push_back(op.op_id);
      }
    }

    for (size_t index = 0; index < state.op_states.size(); ++index) {
      if (state.op_states[index].remaining_deps == 0) {
        state.ready_ops.push_back(static_cast<uint32_t>(index));
      }
    }

    CollectiveOpHandle handle{next_handle++};
    auto [it, inserted] = handles.emplace(handle.value, std::move(state));
    if (!inserted) {
      throw std::runtime_error("duplicate collective handle");
    }

    drive_ready_ops(it->second);
    return handle;
  }

  bool poll(CollectiveOpHandle handle) {
    HandleState& state = get(handle);
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
      drive_ready_ops(state);
    }

    if (state.completed_ops == state.exec_plan.ops.size()) {
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
    auto it = handles.find(handle.value);
    if (it == handles.end()) return;

    for (auto& op_state : it->second.op_states) {
      if (op_state.inflight.backend != nullptr) {
        it->second.inflight_lookup.erase(
            inflight_key(op_state.inflight.backend, op_state.inflight.token));
        op_state.inflight.backend->release(op_state.inflight.token);
        op_state.inflight = {};
      }
    }
    handles.erase(it);
  }

  CollectiveOpStatus status(CollectiveOpHandle handle) const {
    return get_const(handle).status;
  }

  size_t inflight_steps(CollectiveOpHandle handle) const {
    size_t inflight = 0;
    for (auto const& op_state : get_const(handle).op_states) {
      if (op_state.inflight.backend != nullptr) {
        ++inflight;
      }
    }
    return inflight;
  }

  void drive_ready_ops(HandleState& state) {
    while (!state.ready_ops.empty() &&
           state.status == CollectiveOpStatus::Running) {
      uint32_t op_id = state.ready_ops.front();
      state.ready_ops.pop_front();
      submit_ready_op(state, op_id);
    }
  }

  void submit_ready_op(HandleState& state, uint32_t op_id) {
    if (op_id >= state.op_states.size()) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "ready op id out of range";
      return;
    }

    OpState& op_state = state.op_states[op_id];
    if (op_state.completed || op_state.submitted ||
        op_state.remaining_deps != 0) {
      return;
    }

    ExecOp const& op = state.exec_plan.ops[op_id];
    Backend* backend = pick_backend(backends, op.kind);
    if (backend == nullptr || !backend->supports(op.kind)) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "no backend available for ready execution op";
      return;
    }

    BackendToken token = backend->submit(op);
    op_state.submitted = true;
    op_state.inflight.backend = backend;
    op_state.inflight.token = token;
    state.inflight_lookup[inflight_key(backend, token)] = op_id;
  }

  bool drain_backend_completions(HandleState& state) {
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

  bool poll_inflight(HandleState& state) {
    bool progress = false;
    for (uint32_t op_id = 0; op_id < state.op_states.size(); ++op_id) {
      OpState& op_state = state.op_states[op_id];
      if (op_state.inflight.backend == nullptr) continue;
      if (!op_state.inflight.backend->poll(op_state.inflight.token)) continue;
      complete_inflight(state, op_id);
      progress = true;
      if (state.status == CollectiveOpStatus::Failed) return true;
    }
    return progress;
  }

  bool complete_inflight_by_token(HandleState& state, Backend* backend,
                                  BackendToken token) {
    auto it = state.inflight_lookup.find(inflight_key(backend, token));
    if (it == state.inflight_lookup.end()) return false;
    complete_inflight(state, it->second);
    return true;
  }

  void complete_inflight(HandleState& state, uint32_t op_id) {
    if (op_id >= state.op_states.size()) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "completion op id out of range";
      return;
    }

    OpState& op_state = state.op_states[op_id];
    if (op_state.inflight.backend == nullptr) return;

    state.inflight_lookup.erase(
        inflight_key(op_state.inflight.backend, op_state.inflight.token));
    op_state.inflight.backend->release(op_state.inflight.token);
    op_state.inflight = {};
    op_state.completed = true;
    ++state.completed_ops;

    for (uint32_t successor : op_state.successors) {
      if (successor >= state.op_states.size()) {
        state.status = CollectiveOpStatus::Failed;
        state.error = "successor op id out of range";
        return;
      }
      OpState& successor_state = state.op_states[successor];
      if (successor_state.remaining_deps == 0) {
        state.status = CollectiveOpStatus::Failed;
        state.error = "dependency underflow while completing op";
        return;
      }
      --successor_state.remaining_deps;
      if (successor_state.remaining_deps == 0 && !successor_state.submitted &&
          !successor_state.completed) {
        state.ready_ops.push_back(successor);
      }
    }
  }

  HandleState& get(CollectiveOpHandle handle) {
    auto it = handles.find(handle.value);
    if (it == handles.end()) {
      throw std::invalid_argument("unknown collective handle");
    }
    return it->second;
  }

  HandleState const& get_const(CollectiveOpHandle handle) const {
    auto it = handles.find(handle.value);
    if (it == handles.end()) {
      throw std::invalid_argument("unknown collective handle");
    }
    return it->second;
  }

  ExecutorBackends backends{};
  std::vector<Backend*> completion_sources;
  uint64_t next_handle = 1;
  std::unordered_map<uint64_t, HandleState> handles;
};

Executor::Executor(ExecutorBackends backends) : impl_(new Impl(backends)) {}

Executor::~Executor() { delete impl_; }

CollectiveOpHandle Executor::submit(CollectivePlan plan) {
  return impl_->submit(std::move(plan));
}

CollectiveOpHandle Executor::submit_allreduce(CollectiveConfig const& config) {
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, config));
  return impl_->submit(std::move(plan));
}

CollectiveOpHandle Executor::submit_alltoall(CollectiveConfig const& config) {
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllToAll, config));
  return impl_->submit(std::move(plan));
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
