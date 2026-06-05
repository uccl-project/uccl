#include "executor.h"
#include "scheduler.h"
#include "utils.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

bool is_device_op(OpKind kind) {
  return kind == OpKind::DeviceCopy || kind == OpKind::DeviceReduce ||
         kind == OpKind::DeviceSend || kind == OpKind::DeviceRecvReduce ||
         kind == OpKind::DeviceRecv;
}

bool is_transport_op(OpKind kind) {
  return kind == OpKind::TransportSend || kind == OpKind::TransportRecv;
}

void* local_ptr(CollectiveBinding const& binding, CollectiveBufferRole role,
                size_t offset) {
  return byte_offset(binding.role_buffer(role).local_ptr, offset);
}

uint32_t remote_buffer_id_for_role(CollectiveBinding const& binding,
                                   CollectiveBufferRole role, int peer_rank) {
  RegisteredBuffer const& buf = binding.role_buffer(role);
  if (static_cast<size_t>(peer_rank) >= buf.peer_views.size())
    throw std::invalid_argument(
        "peer rank out of range for remote buffer lookup");
  return buf.peer_views[static_cast<size_t>(peer_rank)].buffer_id;
}

void resolve_remote_ptrs(
    Op const& op, OpBindings& bind, CollectiveBinding const& binding,
    std::function<bool(int, uint32_t, size_t, size_t, void**, int*)> const&
        resolve_remote) {
  if (op.src_peer != ~0u) {
    auto role = buf_role(op.kind, true, op.copy_from_staging);
    uint32_t remote_id = remote_buffer_id_for_role(binding, role, op.src_peer);
    void* ptr = nullptr;
    int dev = -1;
    if (!resolve_remote || !resolve_remote(op.src_peer, remote_id, op.src_off,
                                           op.bytes, &ptr, &dev))
      throw std::runtime_error("failed to resolve remote source pointer");
    bind.resolved_src = ptr;
    bind.src_device = dev;
  } else {
    bind.resolved_src = local_ptr(
        binding, buf_role(op.kind, true, op.copy_from_staging), op.src_off);
  }
  if (op.dst_peer != ~0u) {
    auto role = buf_role(op.kind, false, op.copy_from_staging);
    uint32_t remote_id = remote_buffer_id_for_role(binding, role, op.dst_peer);
    void* ptr = nullptr;
    int dev = -1;
    if (!resolve_remote || !resolve_remote(op.dst_peer, remote_id, op.dst_off,
                                           op.bytes, &ptr, &dev))
      throw std::runtime_error("failed to resolve remote destination pointer");
    bind.resolved_dst = ptr;
    bind.dst_device = dev;
  } else {
    bind.resolved_dst = local_ptr(
        binding, buf_role(op.kind, false, op.copy_from_staging), op.dst_off);
  }
}

Backend* pick_backend(ExecutorBackends const& backends, OpKind kind) {
  if (is_transport_op(kind)) return backends.transport;
  return backends.device;
}

}  // namespace

static std::atomic<uint64_t> g_seq{1};

static void init_run_state(CollectiveRun& run) {
  run.completed.assign(run.total_ops, false);
  run.tokens.resize(run.total_ops);
  run.stream_head.assign(run.num_streams, 0);
}

Executor::Executor(
    ExecutorBackends backends,
    std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
        resolve_ipc_buffer_pointer)
    : backends_(backends),
      resolve_ipc_buffer_pointer_(std::move(resolve_ipc_buffer_pointer)) {}

Executor::~Executor() = default;

CollectiveRun* Executor::get_run(CollectiveOpHandle handle) {
  auto it = runs_.find(handle);
  return it != runs_.end() ? &it->second : nullptr;
}

CollectiveRun const* Executor::get_run(CollectiveOpHandle handle) const {
  auto it = runs_.find(handle);
  return it != runs_.end() ? &it->second : nullptr;
}

CollectiveOpHandle Executor::submit_collective(CollectiveKind kind,
                                               CollectiveConfig const& config,
                                               CollectiveBinding& binding) {
  bool inplace =
      binding.roles.input_buffer_id == binding.roles.output_buffer_id;

  CollectiveConfig cfg = config;
  cfg.collective = kind;
  TiledResult tiled = build_plan(cfg, inplace);

  uint64_t seq_base =
      tiled.ops.empty() ? 0
                       : g_seq.fetch_add(static_cast<uint64_t>(tiled.ops.size()),
                                         std::memory_order_relaxed);

  uint64_t sig =
      reinterpret_cast<uintptr_t>(&binding) ^
      (static_cast<uint64_t>(tiled.ops.size()) << 32) ^
      static_cast<uint64_t>(std::max(tiled.input_bytes, tiled.output_bytes));
  if (sig != validated_sig_) {
    if (backends_.transport) backends_.transport->validate(tiled, binding);
    if (backends_.device && backends_.device != backends_.transport)
      backends_.device->validate(tiled, binding);
    validated_sig_ = sig;
  }

  CollectiveOpHandle handle = next_handle_++;
  CollectiveRun run;
  run.status = CollectiveOpStatus::Running;
  run.tiled = std::move(tiled);
  run.binding = &binding;
  run.total_ops = run.tiled.ops.size();
  run.signal_seq_base = seq_base;
  Schedule schedule = schedule_ops(run.tiled.ops);
  run.num_streams = schedule.num_streams;
  run.stream_ops = std::move(schedule.stream_ops);

  if (run.total_ops == 0) {
    run.status = CollectiveOpStatus::Completed;
    runs_.emplace(handle, std::move(run));
    return handle;
  }

  init_run_state(run);
  run.done_buf.resize(run.total_ops);
  runs_.emplace(handle, std::move(run));
  return handle;
}

CollectiveOpHandle Executor::submit_allreduce(CollectiveConfig const& config,
                                              CollectiveBinding& binding) {
  return submit_collective(CollectiveKind::AllReduce, config, binding);
}

CollectiveOpHandle Executor::submit_alltoall(CollectiveConfig const& config,
                                             CollectiveBinding& binding) {
  return submit_collective(CollectiveKind::AllToAll, config, binding);
}

CollectiveOpStatus Executor::status(CollectiveOpHandle handle) const {
  auto* run = get_run(handle);
  return run ? run->status : CollectiveOpStatus::Completed;
}

void Executor::advance_run(CollectiveRun& run) {
  if (run.status != CollectiveOpStatus::Running) return;
  if (run.completed_count == run.total_ops) {
    run.status = CollectiveOpStatus::Completed;
    return;
  }

  run.ready.clear();
  run.ready.reserve(run.num_streams * 2);

  for (uint32_t fid = 0; fid < run.num_streams; ++fid) {
    while (run.stream_head[fid] < run.stream_ops[fid].size()) {
      uint32_t op_idx = run.stream_ops[fid][run.stream_head[fid]];
      if (run.completed[op_idx]) {
        ++run.stream_head[fid];
        continue;
      }
      bool deps_ok = true;
      for (uint32_t dep : run.tiled.ops[op_idx].deps) {
        if (!run.completed[dep]) {
          deps_ok = false;
          break;
        }
      }
      if (!deps_ok) break;
      run.ready.push_back({fid, op_idx});
      ++run.stream_head[fid];
    }
  }

  for (auto& r : run.ready) {
    uint32_t fid = r.first;
    uint32_t op_idx = r.second;
    Op const& plan_op = run.tiled.ops[op_idx];
    Backend* backend = pick_backend(backends_, plan_op.kind);
    if (backend == nullptr) continue;

    OpBindings bind;
    bind.stream_index = fid;
    if (is_device_op(plan_op.kind) && resolve_ipc_buffer_pointer_) {
      try {
        resolve_remote_ptrs(plan_op, bind, *run.binding,
                            resolve_ipc_buffer_pointer_);
      } catch (std::exception const& e) {
        run.status = CollectiveOpStatus::Failed;
        run.error_message = std::string("IPC resolution failed for op ") +
                            std::to_string(op_idx) + ": " + e.what();
        return;
      }
    } else {
      if (plan_op.src_peer == ~0u)
        bind.resolved_src =
            local_ptr(*run.binding,
                      buf_role(plan_op.kind, true, plan_op.copy_from_staging),
                      plan_op.src_off);
      if (plan_op.dst_peer == ~0u)
        bind.resolved_dst =
            local_ptr(*run.binding,
                      buf_role(plan_op.kind, false, plan_op.copy_from_staging),
                      plan_op.dst_off);
    }
    bind.signal_seq = op_idx + run.signal_seq_base;

    BackendToken token;
    try {
      token = backend->submit(plan_op, bind, *run.binding);
    } catch (std::exception const& e) {
      run.status = CollectiveOpStatus::Failed;
      run.error_message = e.what();
      return;
    }

    if (token.value == 0) {
      --run.stream_head[fid];
      continue;
    }
    run.tokens[op_idx] = token;
    run.token_to_op_idx[{backend, token.value}] = op_idx;
  }

  std::string failure_msg;
  for (Backend* backend : {backends_.transport, backends_.device}) {
    if (backend == nullptr) continue;
    size_t n = backend->drain(run.done_buf.data(), run.total_ops);
    for (size_t i = 0; i < n; ++i) {
      auto it = run.token_to_op_idx.find({backend, run.done_buf[i].value});
      if (it == run.token_to_op_idx.end()) continue;
      size_t op_idx = it->second;
      if (run.completed[op_idx]) {
        run.token_to_op_idx.erase(it);
        continue;
      }

      if (run.done_buf[i].failed && failure_msg.empty())
        failure_msg = std::string("backend '") + backend->name() +
                      "' reported failure for op " + std::to_string(op_idx);

      run.completed[op_idx] = true;
      ++run.completed_count;
      run.token_to_op_idx.erase(it);
    }
  }
  if (!failure_msg.empty()) {
    run.status = CollectiveOpStatus::Failed;
    run.error_message = std::move(failure_msg);
    return;
  }

  if (run.completed_count == run.total_ops)
    run.status = CollectiveOpStatus::Completed;
}

bool Executor::poll(CollectiveOpHandle handle) {
  auto* run = get_run(handle);
  if (run == nullptr) return true;
  if (run->status != CollectiveOpStatus::Running) return true;
  advance_run(*run);
  return run->status == CollectiveOpStatus::Completed ||
         run->status == CollectiveOpStatus::Failed;
}

void Executor::progress() {
  for (auto& [handle, run] : runs_) advance_run(run);
}

bool Executor::wait(CollectiveOpHandle handle,
                    std::chrono::milliseconds timeout) {
  constexpr int kSpinIters = 1000;
  constexpr int kBackoffBaseUs = 10;
  constexpr int kBackoffMaxUs = 200;
  int spin_count = 0;
  int backoff_us = kBackoffBaseUs;

  if (timeout.count() == 0) {
    while (!poll(handle)) {
      if (spin_count < kSpinIters) {
        ++spin_count;
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
        backoff_us = std::min(backoff_us * 2, kBackoffMaxUs);
      }
    }
    return true;
  }
  auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (poll(handle)) return true;
    if (spin_count < kSpinIters) {
      ++spin_count;
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
      backoff_us = std::min(backoff_us * 2, kBackoffMaxUs);
    }
  } while (std::chrono::steady_clock::now() < deadline);
  return poll(handle);
}

void Executor::release(CollectiveOpHandle handle) {
  auto it = runs_.find(handle);
  if (it == runs_.end()) return;
  if (it->second.status == CollectiveOpStatus::Queued ||
      it->second.status == CollectiveOpStatus::Running) {
    throw std::logic_error("cannot release a queued or running collective");
  }
  runs_.erase(it);
}

std::string Executor::error_message(CollectiveOpHandle handle) const {
  auto* run = get_run(handle);
  return run ? run->error_message : std::string{};
}

size_t Executor::active_count() const {
  size_t count = 0;
  for (auto const& [h, run] : runs_)
    if (run.status == CollectiveOpStatus::Running) ++count;
  return count;
}

void Executor::run_plan(TiledResult const& tiled,
                        CollectiveBinding& binding) {
  uint64_t sig =
      reinterpret_cast<uintptr_t>(&binding) ^
      (static_cast<uint64_t>(tiled.ops.size()) << 32) ^
      static_cast<uint64_t>(std::max(tiled.input_bytes, tiled.output_bytes));
  if (sig != validated_sig_) {
    if (backends_.transport) backends_.transport->validate(tiled, binding);
    if (backends_.device && backends_.device != backends_.transport)
      backends_.device->validate(tiled, binding);
    validated_sig_ = sig;
  }

  CollectiveRun run;
  run.status = CollectiveOpStatus::Running;
  run.tiled = tiled;
  run.binding = &binding;
  run.total_ops = tiled.ops.size();

  Schedule schedule = schedule_ops(tiled.ops);
  run.num_streams = schedule.num_streams;
  run.stream_ops = std::move(schedule.stream_ops);

  uint64_t seq_base =
      tiled.ops.empty() ? 0
                       : g_seq.fetch_add(static_cast<uint64_t>(tiled.ops.size()),
                                         std::memory_order_relaxed);
  run.signal_seq_base = seq_base;

  if (run.total_ops == 0) return;

  init_run_state(run);
  run.done_buf.resize(run.total_ops);

  while (run.status == CollectiveOpStatus::Running) {
    advance_run(run);
    if (run.status == CollectiveOpStatus::Running) std::this_thread::yield();
  }
  if (run.status == CollectiveOpStatus::Failed)
    throw std::runtime_error(run.error_message);
}

}  // namespace CCL
}  // namespace UKernel
