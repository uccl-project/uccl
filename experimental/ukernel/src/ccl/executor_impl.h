#pragma once

#include "executor.h"
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cinttypes>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

namespace detail {

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
  CollectiveOpStatus status = CollectiveOpStatus::Queued;
  std::vector<OpState> op_states;
  std::unordered_map<uint64_t, uint32_t> inflight_lookup;
  std::deque<uint32_t> ready_ops;
  size_t completed_ops = 0;
  std::string error;
};

inline bool is_terminal(CollectiveOpStatus status) {
  return status == CollectiveOpStatus::Completed ||
         status == CollectiveOpStatus::Failed;
}

inline uint64_t inflight_key(Backend const* backend, BackendToken token) {
  uint64_t backend_bits =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(backend));
  return backend_bits ^ (token.value + 0x9e3779b97f4a7c15ULL +
                         (backend_bits << 6) + (backend_bits >> 2));
}

inline std::vector<Backend*> backend_sources(ExecutorBackends const& backends) {
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

inline void validate_backends(std::vector<Backend*> const& backends,
                              ExecutionPlan const& exec_plan) {
  for (Backend* backend : backends) {
    if (backend != nullptr) {
      backend->validate(exec_plan);
    }
  }
}

inline bool is_transport_op(ExecOpKind kind) {
  return kind == ExecOpKind::TransportSend || kind == ExecOpKind::TransportRecv;
}

inline Backend* pick_backend(ExecutorBackends const& backends, ExecOpKind kind) {
  if (is_transport_op(kind)) {
    return backends.transport != nullptr ? backends.transport
                                         : backends.fallback;
  }
  return backends.device != nullptr ? backends.device : backends.fallback;
}

inline ExecutorBackends make_owned_backends(Backend& transport,
                                            Backend& device) {
  ExecutorBackends out{};
  out.transport = &transport;
  out.device = &device;
  return out;
}

}  // namespace detail

struct Executor::Impl {
  explicit Impl(ExecutorBackends backends_in)
      : backends(backends_in),
        completion_sources(detail::backend_sources(backends_in)),
        progress_thread(&Impl::progress_loop, this) {}

  Impl(std::unique_ptr<Backend> transport_backend_in,
       std::unique_ptr<Backend> device_backend_in)
      : owned_transport_backend(std::move(transport_backend_in)),
        owned_device_backend(std::move(device_backend_in)),
        backends(detail::make_owned_backends(*owned_transport_backend,
                                             *owned_device_backend)),
        completion_sources(detail::backend_sources(backends)),
        progress_thread(&Impl::progress_loop, this) {}

  ~Impl();

  CollectiveOpHandle submit(CollectivePlan plan);
  bool poll(CollectiveOpHandle handle);
  void wait(CollectiveOpHandle handle);
  void release(CollectiveOpHandle handle);
  CollectiveOpStatus status(CollectiveOpHandle handle) const;
  std::string error_message(CollectiveOpHandle handle) const;
  size_t inflight_steps(CollectiveOpHandle handle) const;

  void progress_loop();
  void start_next_collective_locked();
  bool progress_active_collective_locked();
  bool drive_ready_ops_locked(detail::HandleState& state);
  void maybe_quiesce_backends_locked(detail::HandleState const& state);
  void submit_ready_op_locked(detail::HandleState& state, uint32_t op_id);
  bool drain_backend_completions_locked(detail::HandleState& state);
  bool poll_inflight_locked(detail::HandleState& state);
  bool complete_inflight_by_token_locked(detail::HandleState& state,
                                         Backend* backend, BackendToken token);
  void complete_inflight_locked(detail::HandleState& state, uint32_t op_id);
  void cleanup_inflight_locked(detail::HandleState& state);
  void fail_active_collective_locked(std::string message);
  detail::HandleState& get_locked(CollectiveOpHandle handle);
  detail::HandleState const& get_locked_const(CollectiveOpHandle handle) const;

  std::unique_ptr<Backend> owned_transport_backend;
  std::unique_ptr<Backend> owned_device_backend;
  std::shared_ptr<CollectiveMemory> runtime_memory;
  std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
      resolve_ipc_buffer_pointer;
  ExecutorBackends backends{};
  std::vector<Backend*> completion_sources;
  uint64_t next_handle = 1;
  uint64_t active_handle = 0;
  bool stop_requested = false;
  std::unordered_map<uint64_t, detail::HandleState> handles;
  std::deque<uint64_t> pending_handles;
  mutable std::mutex mu;
  std::condition_variable cv;
  std::thread progress_thread;
};

}  // namespace CCL
}  // namespace UKernel
