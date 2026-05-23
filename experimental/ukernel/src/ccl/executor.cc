#include "executor.h"
#include "utils.h"
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

bool is_remote_ref(BufferRef const& ref) {
  return ref.kind == BufferKind::Remote;
}

bool is_device_op(OpKind kind) {
  return kind == OpKind::DeviceCopy || kind == OpKind::DeviceReduce;
}

bool is_transport_op(OpKind kind) {
  return kind == OpKind::TransportSend || kind == OpKind::TransportRecv;
}

void* local_mutable_ptr(CollectiveBinding const& binding, BufferRef const& ref,
                        size_t bytes) {
  RegisteredBuffer const& buffer = binding.buffer_for_ref(ref);
  if (ref.kind != BufferKind::Local)
    throw std::invalid_argument("local binding cannot target remote buffer");
  if (buffer.local_ptr == nullptr)
    throw std::invalid_argument("local registered buffer is missing");
  validate_span("local registered buffer", ref.offset_bytes, bytes, buffer.bytes);
  return byte_offset(buffer.local_ptr, ref.offset_bytes);
}

void const* local_const_ptr(CollectiveBinding const& binding,
                            BufferRef const& ref, size_t bytes) {
  return local_mutable_ptr(binding, ref, bytes);
}

uint32_t remote_buffer_id_for_ipc(CollectiveBinding const& binding,
                                  BufferRef const& ref) {
  if (!is_remote_ref(ref) || ref.rank < 0)
    throw std::invalid_argument("remote IPC lookup requires a remote buffer ref");
  RegisteredBuffer const& buffer = binding.buffer_for_ref(ref);
  if (static_cast<size_t>(ref.rank) >= buffer.peer_views.size())
    throw std::invalid_argument("remote buffer peer rank out of range");
  return buffer.peer_views[static_cast<size_t>(ref.rank)].buffer_id;
}

void resolve_remote_ptrs(
    Op& bound, CollectiveBinding const& binding,
    std::function<bool(int, uint32_t, size_t, size_t, void**, int*)> const&
        resolve_remote) {
  if (is_remote_ref(bound.src)) {
    uint32_t remote_id = remote_buffer_id_for_ipc(binding, bound.src);
    void* ptr = nullptr;
    int dev = -1;
    if (!resolve_remote ||
        !resolve_remote(bound.src.rank, remote_id, bound.src.offset_bytes,
                        bound.tile.size_bytes, &ptr, &dev))
      throw std::runtime_error("failed to resolve remote source pointer");
    bound.resolved_src = ptr;
    bound.src_device = dev;
  } else {
    bound.resolved_src = local_const_ptr(binding, bound.src, bound.tile.size_bytes);
  }
  if (is_remote_ref(bound.dst)) {
    uint32_t remote_id = remote_buffer_id_for_ipc(binding, bound.dst);
    void* ptr = nullptr;
    int dev = -1;
    if (!resolve_remote ||
        !resolve_remote(bound.dst.rank, remote_id, bound.dst.offset_bytes,
                        bound.tile.size_bytes, &ptr, &dev))
      throw std::runtime_error("failed to resolve remote destination pointer");
    bound.resolved_dst = ptr;
    bound.dst_device = dev;
  } else {
    bound.resolved_dst = local_mutable_ptr(binding, bound.dst, bound.tile.size_bytes);
  }
}

Backend* pick_backend(ExecutorBackends const& backends, OpKind kind) {
  if (is_transport_op(kind)) return backends.transport;
  return backends.device;
}

}  // namespace

// PlanCacheKey helpers
bool Executor::PlanCacheKey::operator==(PlanCacheKey const& o) const {
  return kind == o.kind && inplace == o.inplace &&
         config.nranks == o.config.nranks &&
         config.rank == o.config.rank &&
         config.num_flows == o.config.num_flows &&
         config.tensor_bytes == o.config.tensor_bytes &&
         config.input_bytes == o.config.input_bytes &&
         config.output_bytes == o.config.output_bytes &&
         config.tile_bytes == o.config.tile_bytes &&
         config.input_split_bytes == o.config.input_split_bytes &&
         config.output_split_bytes == o.config.output_split_bytes &&
         config.algorithm == o.config.algorithm &&
         config.dtype == o.config.dtype &&
         config.reduction == o.config.reduction;
}

size_t Executor::PlanCacheKeyHash::operator()(PlanCacheKey const& key) const {
  auto h = [](size_t seed, size_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
  };
  size_t s = static_cast<size_t>(key.kind);
  s = h(s, key.inplace ? 1 : 0);
  s = h(s, static_cast<size_t>(key.config.nranks));
  s = h(s, static_cast<size_t>(key.config.rank));
  s = h(s, key.config.num_flows);
  s = h(s, key.config.tensor_bytes);
  s = h(s, key.config.input_bytes);
  s = h(s, key.config.output_bytes);
  s = h(s, key.config.tile_bytes);
  s = h(s, static_cast<size_t>(key.config.algorithm));
  s = h(s, static_cast<size_t>(key.config.dtype));
  s = h(s, static_cast<size_t>(key.config.reduction));
  for (auto v : key.config.input_split_bytes) s = h(s, v);
  for (auto v : key.config.output_split_bytes) s = h(s, v);
  return s;
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

CollectiveOpHandle Executor::submit_collective(
    CollectiveKind kind, CollectiveConfig const& config,
    CollectiveBinding& binding) {
  bool inplace =
      binding.roles.input_buffer_id == binding.roles.output_buffer_id;

  CollectiveConfig cfg = config;
  cfg.collective = kind;
  PlanCacheKey key{kind, inplace, cfg};
  auto cache_it = plan_cache_.find(key);
  CollectivePlan plan;
  if (cache_it != plan_cache_.end()) {
    plan = cache_it->second;
  } else {
    plan = build_plan(cfg, inplace);
    plan_cache_.emplace(key, plan);
  }

  // One-time backend init.
  if (backends_.transport) backends_.transport->validate(plan, binding);
  if (backends_.device && backends_.device != backends_.transport)
    backends_.device->validate(plan, binding);

  CollectiveOpHandle handle = next_handle_++;
  CollectiveRun run;
  run.handle = handle;
  run.status = CollectiveOpStatus::Running;
  run.plan = std::move(plan);
  run.binding = &binding;
  run.total_ops = run.plan.ops.size();

  if (run.total_ops == 0) {
    run.status = CollectiveOpStatus::Completed;
    runs_.emplace(handle, std::move(run));
    return handle;
  }

  run.completed.assign(run.total_ops, false);
  run.tokens.resize(run.total_ops);
  run.op_backend.resize(run.total_ops, nullptr);
  run.flow_ops.resize(run.plan.num_flows);
  run.flow_head.assign(run.plan.num_flows, 0);

  for (size_t i = 0; i < run.total_ops; i++) {
    uint32_t f = run.plan.ops[i].tile.flow_index;
    if (f < run.plan.num_flows)
      run.flow_ops[f].push_back(static_cast<uint32_t>(i));
  }

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

// ── advance_run: two-phase pipeline ────────────────────────────────────

void Executor::advance_run(CollectiveRun& run) {
  if (run.status != CollectiveOpStatus::Running) return;
  if (run.completed_count == run.total_ops) {
    run.status = CollectiveOpStatus::Completed;
    return;
  }

  // Phase 1: Collect all ops whose dependencies are satisfied.
  struct ReadyOp { uint32_t fid; uint32_t op_idx; };
  std::vector<ReadyOp> ready;
  ready.reserve(run.plan.num_flows * 2);  // typical inflight per cycle

  for (uint32_t fid = 0; fid < run.plan.num_flows; ++fid) {
    while (run.flow_head[fid] < run.flow_ops[fid].size()) {
      uint32_t op_idx = run.flow_ops[fid][run.flow_head[fid]];
      if (run.completed[op_idx]) {
        ++run.flow_head[fid];
        continue;
      }
      bool deps_ok = true;
      for (uint32_t dep : run.plan.ops[op_idx].deps) {
        if (!run.completed[dep]) { deps_ok = false; break; }
      }
      if (!deps_ok) break;
      ready.push_back({fid, op_idx});
      ++run.flow_head[fid];
    }
  }

  // Phase 2: Submit ready ops.
  for (auto& r : ready) {
    Op const& plan_op = run.plan.ops[r.op_idx];
    Backend* backend = pick_backend(backends_, plan_op.kind);
    if (backend == nullptr) continue;

    Op submit_op = plan_op;
    if (is_device_op(plan_op.kind) && resolve_ipc_buffer_pointer_) {
      try {
        resolve_remote_ptrs(submit_op, *run.binding, resolve_ipc_buffer_pointer_);
      } catch (std::exception const& e) {
        run.status = CollectiveOpStatus::Failed;
        run.error_message = std::string("IPC resolution failed for op ") +
                            std::to_string(r.op_idx) + ": " + e.what();
        return;
      }
    } else {
      if (!is_remote_ref(submit_op.src))
        submit_op.resolved_src =
            local_const_ptr(*run.binding, submit_op.src, submit_op.tile.size_bytes);
      if (!is_remote_ref(submit_op.dst))
        submit_op.resolved_dst =
            local_mutable_ptr(*run.binding, submit_op.dst, submit_op.tile.size_bytes);
    }

    BackendToken token;
    try {
      token = backend->submit(submit_op, *run.binding);
    } catch (std::exception const& e) {
      run.status = CollectiveOpStatus::Failed;
      run.error_message = e.what();
      return;
    }

    if (token.value == 0) {
      --run.flow_head[r.fid];  // backpressure — retry next cycle
      continue;
    }
    run.tokens[r.op_idx] = token;
    run.op_backend[r.op_idx] = backend;
    run.token_to_op_idx[{backend, token.value}] = r.op_idx;
  }

  // Phase 3: Drain completions from each backend.
  // Backend cleans its own resources inside drain(); we update op bookkeeping.
  // Process ALL drain results even if a failure is detected, because drain
  // has already released backend resources for every returned token.
  std::vector<BackendToken> done_buf(run.total_ops);
  std::string failure_msg;
  for (Backend* backend : {backends_.transport, backends_.device}) {
    if (backend == nullptr) continue;
    size_t n = backend->drain(done_buf.data(), run.total_ops);
    for (size_t i = 0; i < n; ++i) {
      auto it = run.token_to_op_idx.find({backend, done_buf[i].value});
      if (it == run.token_to_op_idx.end()) continue;
      size_t op_idx = it->second;
      if (run.completed[op_idx]) { run.token_to_op_idx.erase(it); continue; }

      if (done_buf[i].failed && failure_msg.empty())
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

// ── public API ─────────────────────────────────────────────────────────

bool Executor::poll(CollectiveOpHandle handle) {
  auto* run = get_run(handle);
  if (run == nullptr) return true;
  if (run->status != CollectiveOpStatus::Running) return true;
  advance_run(*run);
  return run->status == CollectiveOpStatus::Completed ||
         run->status == CollectiveOpStatus::Failed;
}

void Executor::progress() {
  for (auto& [handle, run] : runs_)
    advance_run(run);
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
      if (spin_count < kSpinIters) { ++spin_count; std::this_thread::yield(); }
      else {
        std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
        backoff_us = std::min(backoff_us * 2, kBackoffMaxUs);
      }
    }
    return true;
  }
  auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (poll(handle)) return true;
    if (spin_count < kSpinIters) { ++spin_count; std::this_thread::yield(); }
    else {
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

void Executor::run_plan(CollectivePlan const& plan,
                        CollectiveBinding& binding) {
  if (backends_.transport) backends_.transport->validate(plan, binding);
  if (backends_.device && backends_.device != backends_.transport)
    backends_.device->validate(plan, binding);

  CollectiveOpHandle handle = next_handle_++;
  CollectiveRun run;
  run.handle = handle;
  run.status = CollectiveOpStatus::Running;
  run.plan = plan;
  run.binding = &binding;
  run.total_ops = plan.ops.size();

  if (run.total_ops == 0) return;

  run.completed.assign(run.total_ops, false);
  run.tokens.resize(run.total_ops);
  run.op_backend.resize(run.total_ops, nullptr);
  run.flow_ops.resize(run.plan.num_flows);
  run.flow_head.assign(run.plan.num_flows, 0);

  for (size_t i = 0; i < run.total_ops; i++) {
    uint32_t f = run.plan.ops[i].tile.flow_index;
    if (f < run.plan.num_flows)
      run.flow_ops[f].push_back(static_cast<uint32_t>(i));
  }

  while (run.status == CollectiveOpStatus::Running) {
    advance_run(run);
    if (run.status == CollectiveOpStatus::Running)
      std::this_thread::yield();
  }
  if (run.status == CollectiveOpStatus::Failed)
    throw std::runtime_error(run.error_message);
}

}  // namespace CCL
}  // namespace UKernel
