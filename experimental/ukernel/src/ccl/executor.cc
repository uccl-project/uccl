#include "executor.h"
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

void* local_ptr(CollectiveBufferRole role, size_t offset,
                void* input_ptr, void* output_ptr, void* scratch_ptr) {
  switch (role) {
    case CollectiveBufferRole::Input:
      return byte_offset(input_ptr, offset);
    case CollectiveBufferRole::Output:
      return byte_offset(output_ptr, offset);
    case CollectiveBufferRole::Scratch:
      return byte_offset(scratch_ptr, offset);
  }
  return nullptr;
}

uint32_t remote_buffer_id_for_role(CollectiveBufferRole role) {
  switch (role) {
    case CollectiveBufferRole::Input:
      return 1;
    case CollectiveBufferRole::Output:
      return 2;
    case CollectiveBufferRole::Scratch:
      return 3;
  }
  return 0;
}

Backend* pick_backend(ExecutorBackends const& backends, OpKind kind) {
  if (kind == OpKind::Copy && backends.rdma_copy)
    return backends.rdma_copy;
  if (backends.transport && backends.transport->supports(kind))
    return backends.transport;
  return backends.device;
}

}  // namespace

static std::atomic<uint64_t> g_seq{1};

// ── Executor ────────────────────────────────────────────────────────────

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

// ── validate guard ─────────────────────────────────────────────────────

void Executor::ensure_validated(TiledResult const& tiled,
                                void* input_ptr, void* output_ptr,
                                void* scratch_ptr) {
  uint64_t sig =
      reinterpret_cast<uintptr_t>(input_ptr) ^
      (static_cast<uint64_t>(tiled.ops.size()) << 32) ^
      static_cast<uint64_t>(std::max(tiled.input_bytes, tiled.output_bytes));
  if (sig == validated_sig_) return;
  if (backends_.transport)
    backends_.transport->validate(tiled, input_ptr, output_ptr, scratch_ptr);
  if (backends_.device && backends_.device != backends_.transport)
    backends_.device->validate(tiled, input_ptr, output_ptr, scratch_ptr);
  if (backends_.rdma_copy && backends_.rdma_copy != backends_.transport &&
      backends_.rdma_copy != backends_.device)
    backends_.rdma_copy->validate(tiled, input_ptr, output_ptr, scratch_ptr);
  validated_sig_ = sig;
}

// ── IPC resolution ─────────────────────────────────────────────────────

void Executor::bind_op(Op const& op, OpBindings& bind,
                       void* input_ptr, void* output_ptr, void* scratch_ptr) {
  auto role_src = buf_role(op.kind, true, op.copy_from_staging);
  auto role_dst = buf_role(op.kind, false, op.copy_from_staging);

  if (op.src_peer != ~0u) {
    uint32_t remote_id = remote_buffer_id_for_role(role_src);
    void* ptr = nullptr;
    int dev = -1;
    if (!resolve_ipc_buffer_pointer_ ||
        !resolve_ipc_buffer_pointer_(op.src_peer, remote_id, op.src_off,
                                     op.bytes, &ptr, &dev))
      throw std::runtime_error("failed to resolve remote source pointer");
    bind.resolved_src = ptr;
    bind.src_device = dev;
  } else {
    bind.resolved_src =
        local_ptr(role_src, op.src_off, input_ptr, output_ptr, scratch_ptr);
  }

  if (op.dst_peer != ~0u) {
    uint32_t remote_id = remote_buffer_id_for_role(role_dst);
    void* ptr = nullptr;
    int dev = -1;
    if (!resolve_ipc_buffer_pointer_ ||
        !resolve_ipc_buffer_pointer_(op.dst_peer, remote_id, op.dst_off,
                                     op.bytes, &ptr, &dev))
      throw std::runtime_error("failed to resolve remote destination pointer");
    bind.resolved_dst = ptr;
    bind.dst_device = dev;
  } else {
    bind.resolved_dst =
        local_ptr(role_dst, op.dst_off, input_ptr, output_ptr, scratch_ptr);
  }
}

// ── submit_collective ──────────────────────────────────────────────────

CollectiveOpHandle Executor::submit_collective(
    CollKind kind, CollectiveConfig const& config, void* input_ptr,
    void* output_ptr, void* scratch_ptr) {
  bool inplace = (input_ptr == output_ptr);

  CollectiveConfig cfg = config;
  cfg.kind = kind;
  TiledResult tiled = build_tiled(cfg, inplace);

  uint64_t seq_base =
      tiled.ops.empty() ? 0
                        : g_seq.fetch_add(static_cast<uint64_t>(tiled.ops.size()),
                                          std::memory_order_relaxed);

  ensure_validated(tiled, input_ptr, output_ptr, scratch_ptr);

  CollectiveOpHandle handle = next_handle_++;
  CollectiveRun run;
  run.status = CollectiveOpStatus::Running;
  run.tiled = std::move(tiled);
  run.input_ptr = input_ptr;
  run.output_ptr = output_ptr;
  run.scratch_ptr = scratch_ptr;
  run.total_ops = run.tiled.ops.size();
  run.signal_seq_base = seq_base;
  run.layers = run.tiled.layers;

  if (run.total_ops == 0) {
    run.status = CollectiveOpStatus::Completed;
    runs_.emplace(handle, std::move(run));
    return handle;
  }

  run.completed.assign(run.total_ops, false);
  run.done_buf.resize(run.total_ops);
  runs_.emplace(handle, std::move(run));
  return handle;
}

CollectiveOpHandle Executor::submit_allreduce(CollectiveConfig const& config,
                                              void* input_ptr,
                                              void* output_ptr,
                                              void* scratch_ptr) {
  return submit_collective(CollKind::AllReduceRing, config, input_ptr,
                           output_ptr, scratch_ptr);
}

CollectiveOpHandle Executor::submit_alltoall(CollectiveConfig const& config,
                                             void* input_ptr,
                                             void* output_ptr,
                                             void* scratch_ptr) {
  return submit_collective(CollKind::AllToAllPairwise, config, input_ptr,
                           output_ptr, scratch_ptr);
}

CollectiveOpStatus Executor::status(CollectiveOpHandle handle) const {
  auto* run = get_run(handle);
  return run ? run->status : CollectiveOpStatus::Completed;
}

// ── advance_run: 3-phase pipeline ──────────────────────────────────────

void Executor::advance_run(CollectiveRun& run) {
  if (run.status != CollectiveOpStatus::Running) return;
  if (run.completed_count == run.total_ops) {
    run.status = CollectiveOpStatus::Completed;
    return;
  }

  collect_ready(run);
  submit_ready(run);
  drain_completed(run);

  if (run.completed_count == run.total_ops)
    run.status = CollectiveOpStatus::Completed;
}

// Phase 1: scan every stream, collect ops whose deps are satisfied.
void Executor::collect_ready(CollectiveRun& run) {
  run.ready.clear();
  for (uint32_t l = run.first_unfinished; l < run.layers.size(); ++l) {
    bool layer_done = true;
    for (uint32_t op_idx : run.layers[l]) {
      if (run.completed[op_idx]) continue;
      layer_done = false;
      bool deps_ok = true;
      for (uint32_t dep : run.tiled.ops[op_idx].deps) {
        if (!run.completed[dep]) { deps_ok = false; break; }
      }
      if (deps_ok) run.ready.push_back({l, op_idx});
    }
    if (layer_done) run.first_unfinished = l + 1;
  }
}

// Phase 2: resolve pointers and submit every ready op to its backend.
void Executor::submit_ready(CollectiveRun& run) {
  for (auto& r : run.ready) {
    uint32_t layer_idx = r.first;
    uint32_t op_idx = r.second;
    Op const& op = run.tiled.ops[op_idx];
    Backend* backend = pick_backend(backends_, op.kind);
    if (backend == nullptr) continue;

    OpBindings bind;
    bind.stream_index = op_idx;

    if (resolve_ipc_buffer_pointer_) {
      try {
        bind_op(op, bind, run.input_ptr, run.output_ptr, run.scratch_ptr);
      } catch (std::exception const& e) {
        run.status = CollectiveOpStatus::Failed;
        run.error_message =
            std::string("IPC resolution failed for op ") +
            std::to_string(op_idx) + ": " + e.what();
        return;
      }
    } else {
      if (op.src_peer == ~0u)
        bind.resolved_src = local_ptr(
            buf_role(op.kind, true, op.copy_from_staging), op.src_off,
            run.input_ptr, run.output_ptr, run.scratch_ptr);
      if (op.dst_peer == ~0u)
        bind.resolved_dst = local_ptr(
            buf_role(op.kind, false, op.copy_from_staging), op.dst_off,
            run.input_ptr, run.output_ptr, run.scratch_ptr);
    }
    bind.signal_seq = op_idx + run.signal_seq_base;

    BackendToken token;
    try {
      token = backend->submit(op, bind, run.input_ptr, run.output_ptr,
                              run.scratch_ptr);
    } catch (std::exception const& e) {
      run.status = CollectiveOpStatus::Failed;
      run.error_message = e.what();
      return;
    }

    if (token.value == 0) {
      // backpressure handled by layer rescan;  // backpressure — retry next cycle
      continue;
    }
    run.token_to_op_idx[{backend, token.value}] = op_idx;
  }
}

// Phase 3: drain completed ops from all backends.
void Executor::drain_completed(CollectiveRun& run) {
  std::string failure_msg;
  for (Backend* backend : {backends_.transport, backends_.device, backends_.rdma_copy}) {
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
  }
}

// ── public API ──────────────────────────────────────────────────────────

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
      it->second.status == CollectiveOpStatus::Running)
    throw std::logic_error("cannot release a queued or running collective");
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

// ── run_tiled (synchronous, for debugging / tests) ─────────────────────

void Executor::run_tiled(TiledResult const& tiled, void* input_ptr,
                         void* output_ptr, void* scratch_ptr) {
  ensure_validated(tiled, input_ptr, output_ptr, scratch_ptr);

  CollectiveRun run;
  run.status = CollectiveOpStatus::Running;
  run.tiled = tiled;
  run.input_ptr = input_ptr;
  run.output_ptr = output_ptr;
  run.scratch_ptr = scratch_ptr;
  run.total_ops = tiled.ops.size();
  run.layers = tiled.layers;

  uint64_t seq_base =
      tiled.ops.empty() ? 0
                        : g_seq.fetch_add(static_cast<uint64_t>(tiled.ops.size()),
                                          std::memory_order_relaxed);
  run.signal_seq_base = seq_base;

  if (run.total_ops == 0) return;

  run.completed.assign(run.total_ops, false);
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
