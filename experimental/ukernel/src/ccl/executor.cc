#include "executor.h"
#include "executor_impl.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace UKernel {
namespace CCL {

namespace {

bool is_peer_ref(BufferRef const& ref) {
  return ref.kind == BufferKind::PeerTensor ||
         ref.kind == BufferKind::PeerStaging;
}

void validate_local_span(char const* what, size_t offset, size_t bytes,
                         size_t capacity) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::invalid_argument(std::string(what) + " out of range");
  }
}

void* local_mutable_ptr(CollectiveMemory const& memory, BufferRef const& ref,
                        size_t bytes) {
  switch (ref.kind) {
    case BufferKind::Tensor:
      if (memory.tensor.local_ptr == nullptr) {
        throw std::invalid_argument("local tensor buffer is missing");
      }
      validate_local_span("local tensor", ref.offset_bytes, bytes,
                          memory.tensor.bytes);
      return static_cast<char*>(memory.tensor.local_ptr) + ref.offset_bytes;
    case BufferKind::Staging:
      if (memory.staging.local_ptr == nullptr) {
        throw std::invalid_argument("local staging buffer is missing");
      }
      validate_local_span("local staging", ref.offset_bytes, bytes,
                          memory.staging.bytes);
      return static_cast<char*>(memory.staging.local_ptr) + ref.offset_bytes;
    case BufferKind::PeerTensor:
    case BufferKind::PeerStaging:
      break;
  }
  throw std::invalid_argument("local binding cannot target peer buffer");
}

void const* local_const_ptr(CollectiveMemory const& memory, BufferRef const& ref,
                            size_t bytes) {
  return local_mutable_ptr(memory, ref, bytes);
}

uint32_t remote_mr_id(CollectiveMemory const& memory, BufferRef const& ref,
                      int local_rank) {
  (void)local_rank;
  if (!is_peer_ref(ref) || ref.rank < 0) {
    throw std::invalid_argument("remote MR lookup requires a remote buffer ref");
  }
  size_t peer = static_cast<size_t>(ref.rank);
  switch (ref.kind) {
    case BufferKind::PeerTensor:
      if (peer >= memory.tensor.peer_views.size()) {
        throw std::invalid_argument("remote tensor peer rank out of range");
      }
      return memory.tensor.peer_views[peer].mr_id;
    case BufferKind::PeerStaging:
      if (peer >= memory.staging.peer_views.size()) {
        throw std::invalid_argument("remote staging peer rank out of range");
      }
      return memory.staging.peer_views[peer].mr_id;
    case BufferKind::Tensor:
    case BufferKind::Staging:
      break;
  }
  throw std::invalid_argument("unknown remote buffer kind");
}

ExecOp bind_device_exec_op(ExecOp const& op, CollectiveMemory const& memory,
                           std::function<bool(int, uint32_t, size_t, size_t,
                                              void**, int*)> const&
                               resolve_remote_ptr) {
  ExecOp bound = op;

  if (is_peer_ref(op.src)) {
    uint32_t mr_id = remote_mr_id(memory, op.src, memory.tensor.local_rank);
    void* ptr = nullptr;
    int device_idx = -1;
    if (!resolve_remote_ptr ||
        !resolve_remote_ptr(op.src.rank, mr_id, op.src.offset_bytes,
                            op.tile.size_bytes, &ptr, &device_idx)) {
      throw std::runtime_error("failed to resolve remote source pointer");
    }
    bound.resolved_src = ptr;
    bound.src_device = device_idx;
  } else {
    bound.resolved_src = local_const_ptr(memory, op.src, op.tile.size_bytes);
  }

  if (is_peer_ref(op.dst)) {
    uint32_t mr_id = remote_mr_id(memory, op.dst, memory.tensor.local_rank);
    void* ptr = nullptr;
    int device_idx = -1;
    if (!resolve_remote_ptr ||
        !resolve_remote_ptr(op.dst.rank, mr_id, op.dst.offset_bytes,
                            op.tile.size_bytes, &ptr, &device_idx)) {
      throw std::runtime_error("failed to resolve remote destination pointer");
    }
    bound.resolved_dst = ptr;
    bound.dst_device = device_idx;
  } else {
    bound.resolved_dst = local_mutable_ptr(memory, op.dst, op.tile.size_bytes);
  }

  return bound;
}

}  // namespace

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
  request.dtype = config.dtype;
  request.reduction = config.reduction;
  return request;
}

Executor::Impl::~Impl() {
  {
    std::lock_guard<std::mutex> lock(mu);
    stop_requested = true;
    cv.notify_all();
  }
  if (progress_thread.joinable()) {
    progress_thread.join();
  }

  std::lock_guard<std::mutex> lock(mu);
  for (auto& [handle_value, state] : handles) {
    cleanup_inflight_locked(state);
  }
}

CollectiveOpHandle Executor::Impl::submit(CollectivePlan plan) {
    if (plan.ops.empty()) {
      throw std::invalid_argument(
          "collective plan must contain at least one op");
    }

    ExecutionPlan exec_plan = lower_plan(plan);
    if (exec_plan.ops.empty()) {
      throw std::invalid_argument(
          "lowered execution plan must contain at least one op");
    }
    detail::validate_backends(completion_sources, exec_plan);

    detail::HandleState state;
    state.exec_plan = std::move(exec_plan);
    state.status = CollectiveOpStatus::Queued;
    state.op_states.resize(state.exec_plan.ops.size());

    for (size_t index = 0; index < state.exec_plan.ops.size(); ++index) {
      ExecOp const& op = state.exec_plan.ops[index];
      if (op.op_id != index) {
        throw std::invalid_argument(
            "execution plan op ids must be dense and ordered");
      }

      Backend* backend = detail::pick_backend(backends, op.kind);
      if (backend == nullptr || !backend->supports(op.kind)) {
        throw std::invalid_argument("no backend available for execution op");
      }

      detail::OpState& op_state = state.op_states[index];
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

    std::lock_guard<std::mutex> lock(mu);
    CollectiveOpHandle handle{next_handle++};
    auto [it, inserted] = handles.emplace(handle.value, std::move(state));
    if (!inserted) {
      throw std::runtime_error("duplicate collective handle");
    }
    pending_handles.push_back(handle.value);
    cv.notify_all();
    return handle;
}

bool Executor::Impl::poll(CollectiveOpHandle handle) {
    std::lock_guard<std::mutex> lock(mu);
    return detail::is_terminal(get_locked(handle).status);
}

void Executor::Impl::wait(CollectiveOpHandle handle) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] {
      return detail::is_terminal(get_locked(handle).status);
    });
}

void Executor::Impl::release(CollectiveOpHandle handle) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = handles.find(handle.value);
    if (it == handles.end()) return;
    if (active_handle == handle.value || it->second.status == CollectiveOpStatus::Queued ||
        it->second.status == CollectiveOpStatus::Running) {
      throw std::runtime_error("cannot release a collective that is still queued or running");
    }
    handles.erase(it);
}

CollectiveOpStatus Executor::Impl::status(CollectiveOpHandle handle) const {
    std::lock_guard<std::mutex> lock(mu);
    return get_locked_const(handle).status;
}

std::string Executor::Impl::error_message(CollectiveOpHandle handle) const {
    std::lock_guard<std::mutex> lock(mu);
    return get_locked_const(handle).error;
}

size_t Executor::Impl::inflight_steps(CollectiveOpHandle handle) const {
    std::lock_guard<std::mutex> lock(mu);
    size_t inflight = 0;
    for (auto const& op_state : get_locked_const(handle).op_states) {
      if (op_state.inflight.backend != nullptr) {
        ++inflight;
      }
    }
    return inflight;
}

void Executor::Impl::progress_loop() {
    while (true) {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [&] {
        return stop_requested || active_handle != 0 || !pending_handles.empty();
      });
      if (stop_requested) {
        return;
      }

      if (active_handle == 0) {
        start_next_collective_locked();
      }

      bool progress = false;
      try {
        progress = progress_active_collective_locked();
      } catch (std::exception const& ex) {
        fail_active_collective_locked(ex.what());
        progress = true;
      } catch (...) {
        fail_active_collective_locked("executor progress loop hit unknown exception");
        progress = true;
      }

      if (active_handle != 0) {
        detail::HandleState& state = handles.at(active_handle);
        if (detail::is_terminal(state.status)) {
          if (state.status == CollectiveOpStatus::Failed &&
              !state.error.empty()) {
            std::fprintf(stderr,
                         "[ccl executor] collective %" PRIu64
                         " failed: %s\n",
                         active_handle, state.error.c_str());
          }
          active_handle = 0;
          cv.notify_all();
          continue;
        }
      }

      if (!progress) {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
}

void Executor::Impl::start_next_collective_locked() {
    while (!pending_handles.empty()) {
      uint64_t handle_value = pending_handles.front();
      pending_handles.pop_front();
      auto it = handles.find(handle_value);
      if (it == handles.end()) {
        continue;
      }
      it->second.status = CollectiveOpStatus::Running;
      active_handle = handle_value;
      cv.notify_all();
      return;
    }
}

bool Executor::Impl::progress_active_collective_locked() {
    if (active_handle == 0) {
      return false;
    }

    detail::HandleState& state = handles.at(active_handle);
    if (detail::is_terminal(state.status)) {
      return true;
    }

    bool progress = false;
    progress |= drive_ready_ops_locked(state);

    bool completion_progress = drain_backend_completions_locked(state);
    if (!completion_progress) {
      completion_progress = poll_inflight_locked(state);
    }
    progress |= completion_progress;

    if (completion_progress && state.status == CollectiveOpStatus::Running) {
      progress |= drive_ready_ops_locked(state);
    }

    maybe_quiesce_backends_locked(state);

    if (state.completed_ops == state.exec_plan.ops.size()) {
      state.status = CollectiveOpStatus::Completed;
      cv.notify_all();
      return true;
    }

    if (state.status == CollectiveOpStatus::Running && state.ready_ops.empty() &&
        state.inflight_lookup.empty()) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "collective stalled with no ready or inflight ops";
      cv.notify_all();
      return true;
    }

    return progress;
}

bool Executor::Impl::drive_ready_ops_locked(detail::HandleState& state) {
    bool progress = false;
    while (!state.ready_ops.empty() &&
           state.status == CollectiveOpStatus::Running) {
      uint32_t op_id = state.ready_ops.front();
      state.ready_ops.pop_front();
      submit_ready_op_locked(state, op_id);
      progress = true;
    }
    return progress;
}

void Executor::Impl::maybe_quiesce_backends_locked(
    detail::HandleState const& state) {
    Backend* device_backend = backends.device != nullptr ? backends.device
                                                         : backends.fallback;
    if (device_backend == nullptr) return;
    for (uint32_t flow_id = 0; flow_id < state.exec_plan.num_flows; ++flow_id) {
      bool flow_has_unfinished_device_work = false;
      for (size_t op_id = 0; op_id < state.op_states.size(); ++op_id) {
        ExecOp const& op = state.exec_plan.ops[op_id];
        if (op.kind != ExecOpKind::DeviceCopy &&
            op.kind != ExecOpKind::DeviceReduce) {
          continue;
        }
        if (op.tile.flow_index != flow_id) continue;
        if (!state.op_states[op_id].completed) {
          flow_has_unfinished_device_work = true;
          break;
        }
      }
      if (!flow_has_unfinished_device_work) {
        device_backend->stop(flow_id);
      }
    }
}

void Executor::Impl::submit_ready_op_locked(detail::HandleState& state,
                                            uint32_t op_id) {
    if (op_id >= state.op_states.size()) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "ready op id out of range";
      cv.notify_all();
      return;
    }

    detail::OpState& op_state = state.op_states[op_id];
    if (op_state.completed || op_state.submitted ||
        op_state.remaining_deps != 0) {
      return;
    }

    ExecOp const& op = state.exec_plan.ops[op_id];
    Backend* backend = detail::pick_backend(backends, op.kind);
    if (backend == nullptr || !backend->supports(op.kind)) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "no backend available for ready execution op";
      cv.notify_all();
      return;
    }

    BackendToken token{};
    if ((op.kind == ExecOpKind::DeviceCopy || op.kind == ExecOpKind::DeviceReduce) &&
        runtime_memory != nullptr) {
      ExecOp bound =
          bind_device_exec_op(op, *runtime_memory, resolve_remote_buffer_ptr);
      token = backend->submit(bound);
    } else {
      token = backend->submit(op);
    }
    op_state.submitted = true;
    op_state.inflight.backend = backend;
    op_state.inflight.token = token;
    state.inflight_lookup[detail::inflight_key(backend, token)] = op_id;
}

bool Executor::Impl::drain_backend_completions_locked(
    detail::HandleState& state) {
    bool progress = false;
    for (Backend* backend : completion_sources) {
      if (backend == nullptr) continue;
      BackendToken token{};
      while (backend->try_pop_completed(token)) {
        if (complete_inflight_by_token_locked(state, backend, token)) {
          progress = true;
        }
      }
    }
    return progress;
}

bool Executor::Impl::poll_inflight_locked(detail::HandleState& state) {
    bool progress = false;
    for (uint32_t op_id = 0; op_id < state.op_states.size(); ++op_id) {
      detail::OpState& op_state = state.op_states[op_id];
      if (op_state.inflight.backend == nullptr) continue;
      if (!op_state.inflight.backend->poll(op_state.inflight.token)) continue;
      complete_inflight_locked(state, op_id);
      progress = true;
      if (state.status == CollectiveOpStatus::Failed) return true;
    }
    return progress;
}

bool Executor::Impl::complete_inflight_by_token_locked(
    detail::HandleState& state, Backend* backend, BackendToken token) {
    auto it = state.inflight_lookup.find(detail::inflight_key(backend, token));
    if (it == state.inflight_lookup.end()) return false;
    complete_inflight_locked(state, it->second);
    return true;
}

void Executor::Impl::complete_inflight_locked(detail::HandleState& state,
                                              uint32_t op_id) {
    if (op_id >= state.op_states.size()) {
      state.status = CollectiveOpStatus::Failed;
      state.error = "completion op id out of range";
      cv.notify_all();
      return;
    }

    detail::OpState& op_state = state.op_states[op_id];
    if (op_state.inflight.backend == nullptr) return;

    state.inflight_lookup.erase(detail::inflight_key(op_state.inflight.backend,
                                                     op_state.inflight.token));
    op_state.inflight.backend->release(op_state.inflight.token);
    op_state.inflight = {};
    op_state.completed = true;
    ++state.completed_ops;

    for (uint32_t successor : op_state.successors) {
      if (successor >= state.op_states.size()) {
        state.status = CollectiveOpStatus::Failed;
        state.error = "successor op id out of range";
        cv.notify_all();
        return;
      }
      detail::OpState& successor_state = state.op_states[successor];
      if (successor_state.remaining_deps == 0) {
        state.status = CollectiveOpStatus::Failed;
        state.error = "dependency underflow while completing op";
        cv.notify_all();
        return;
      }
      --successor_state.remaining_deps;
      if (successor_state.remaining_deps == 0 && !successor_state.submitted &&
          !successor_state.completed) {
        state.ready_ops.push_back(successor);
      }
    }
}

void Executor::Impl::cleanup_inflight_locked(detail::HandleState& state) {
    for (auto& op_state : state.op_states) {
      if (op_state.inflight.backend == nullptr) {
        continue;
      }
      state.inflight_lookup.erase(detail::inflight_key(
          op_state.inflight.backend, op_state.inflight.token));
      op_state.inflight.backend->release(op_state.inflight.token);
      op_state.inflight = {};
    }
}

void Executor::Impl::fail_active_collective_locked(std::string message) {
    if (active_handle == 0) {
      return;
    }
    detail::HandleState& state = handles.at(active_handle);
    cleanup_inflight_locked(state);
    state.status = CollectiveOpStatus::Failed;
    state.error = std::move(message);
    cv.notify_all();
}

detail::HandleState& Executor::Impl::get_locked(CollectiveOpHandle handle) {
    auto it = handles.find(handle.value);
    if (it == handles.end()) {
      throw std::invalid_argument("unknown collective handle");
    }
    return it->second;
}

detail::HandleState const& Executor::Impl::get_locked_const(
    CollectiveOpHandle handle) const {
    auto it = handles.find(handle.value);
    if (it == handles.end()) {
      throw std::invalid_argument("unknown collective handle");
    }
    return it->second;
}

Executor::Executor(ExecutorBackends backends)
    : impl_(std::make_unique<Impl>(backends)) {}

Executor::~Executor() = default;

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

std::string Executor::error_message(CollectiveOpHandle handle) const {
  return impl_->error_message(handle);
}

size_t Executor::inflight_steps(CollectiveOpHandle handle) const {
  return impl_->inflight_steps(handle);
}

}  // namespace CCL
}  // namespace UKernel
