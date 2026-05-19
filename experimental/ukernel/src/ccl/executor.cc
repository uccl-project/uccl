#include "executor.h"
#include "backend/device_backend.h"
#include "backend/transport_backend.h"
#include "../include/transport.h"
#include <cstddef>
#include <cstdint>
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

bool is_device_op(ExecOpKind kind) {
  return kind == ExecOpKind::DeviceCopy || kind == ExecOpKind::DeviceReduce;
}

bool is_transport_op(ExecOpKind kind) {
  return kind == ExecOpKind::TransportSend || kind == ExecOpKind::TransportRecv;
}

void validate_local_span(char const* what, size_t offset, size_t bytes,
                         size_t capacity) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::invalid_argument(std::string(what) + " out of range");
  }
}

void* local_mutable_ptr(CollectiveBinding const& binding, BufferRef const& ref,
                        size_t bytes) {
  if (ref.kind != BufferKind::Local) {
    throw std::invalid_argument("local binding cannot target remote buffer");
  }
  RegisteredBuffer const& buffer = binding.plan_buffer(ref);
  if (buffer.local_ptr == nullptr) {
    throw std::invalid_argument("local registered buffer is missing");
  }
  validate_local_span("local registered buffer", ref.offset_bytes, bytes,
                      buffer.bytes);
  return static_cast<char*>(buffer.local_ptr) + ref.offset_bytes;
}

void const* local_const_ptr(CollectiveBinding const& binding,
                            BufferRef const& ref, size_t bytes) {
  return local_mutable_ptr(binding, ref, bytes);
}

uint32_t remote_buffer_id_for_ipc(CollectiveBinding const& binding,
                                  BufferRef const& ref) {
  if (!is_remote_ref(ref) || ref.rank < 0) {
    throw std::invalid_argument(
        "remote IPC lookup requires a remote buffer ref");
  }
  size_t peer = static_cast<size_t>(ref.rank);
  RegisteredBuffer const& buffer = binding.plan_buffer(ref);
  if (peer >= buffer.peer_views.size()) {
    throw std::invalid_argument("remote buffer peer rank out of range");
  }
  return buffer.peer_views[peer].buffer_id;
}

ExecOp bind_device_exec_op(
    ExecOp const& op, CollectiveBinding const& binding,
    std::function<bool(int, uint32_t, size_t, size_t, void**, int*)> const&
        resolve_remote_ptr) {
  ExecOp bound = op;
  if (is_remote_ref(op.src)) {
    uint32_t remote_buffer_id = remote_buffer_id_for_ipc(binding, op.src);
    void* ptr = nullptr;
    int device_idx = -1;
    if (!resolve_remote_ptr ||
        !resolve_remote_ptr(op.src.rank, remote_buffer_id, op.src.offset_bytes,
                            op.tile.size_bytes, &ptr, &device_idx)) {
      throw std::runtime_error("failed to resolve remote source pointer");
    }
    bound.resolved_src = ptr;
    bound.src_device = device_idx;
  } else {
    bound.resolved_src = local_const_ptr(binding, op.src, op.tile.size_bytes);
  }
  if (is_remote_ref(op.dst)) {
    uint32_t remote_buffer_id = remote_buffer_id_for_ipc(binding, op.dst);
    void* ptr = nullptr;
    int device_idx = -1;
    if (!resolve_remote_ptr ||
        !resolve_remote_ptr(op.dst.rank, remote_buffer_id, op.dst.offset_bytes,
                            op.tile.size_bytes, &ptr, &device_idx)) {
      throw std::runtime_error("failed to resolve remote destination pointer");
    }
    bound.resolved_dst = ptr;
    bound.dst_device = device_idx;
  } else {
    bound.resolved_dst = local_mutable_ptr(binding, op.dst, op.tile.size_bytes);
  }
  return bound;
}

inline Backend* pick_backend(ExecutorBackends const& backends,
                             ExecOpKind kind) {
  if (is_transport_op(kind)) {
    return backends.transport != nullptr ? backends.transport
                                         : backends.fallback;
  }
  return backends.device != nullptr ? backends.device : backends.fallback;
}

inline std::vector<Backend*> backend_sources(ExecutorBackends const& backends) {
  std::vector<Backend*> out;
  if (backends.transport != nullptr) out.push_back(backends.transport);
  if (backends.device != nullptr && backends.device != backends.transport)
    out.push_back(backends.device);
  if (backends.fallback != nullptr && backends.fallback != backends.transport &&
      backends.fallback != backends.device)
    out.push_back(backends.fallback);
  return out;
}

inline void validate_backends(std::vector<Backend*> const& sources,
                              ExecutionPlan const& exec_plan,
                              CollectiveBinding& binding) {
  for (Backend* backend : sources) {
    if (backend != nullptr) backend->validate(exec_plan, binding);
  }
}

inline uint64_t inflight_key(Backend const* backend, BackendToken token) {
  uint64_t backend_bits =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(backend));
  return backend_bits ^ (token.value + 0x9e3779b97f4a7c15ULL +
                         (backend_bits << 6) + (backend_bits >> 2));
}

}  // namespace

PlanRequest make_plan_request(CollectiveKind kind,
                              CollectiveConfig const& config, bool inplace) {
  PlanRequest request;
  request.collective = kind;
  request.algorithm = config.algorithm;
  request.nranks = config.nranks;
  request.rank = config.rank;
  request.num_flows = config.num_flows;
  request.tensor_bytes = config.tensor_bytes;
  request.input_bytes = config.input_bytes;
  request.output_bytes = config.output_bytes;
  request.tile_bytes = config.tile_bytes;
  request.staging_bytes = config.staging_bytes;
  request.input_split_bytes = config.input_split_bytes;
  request.output_split_bytes = config.output_split_bytes;
  request.inplace = inplace;
  request.dtype = config.dtype;
  request.reduction = config.reduction;
  return request;
}

Executor::Executor(
    ExecutorBackends backends,
    std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
        resolve_ipc_buffer_pointer)
    : backends_(backends),
      completion_sources_(backend_sources(backends)),
      resolve_ipc_buffer_pointer_(std::move(resolve_ipc_buffer_pointer)) {}

Executor::Executor(ExecutorConfig const& config) {
  owned_transport_backend_ =
      std::make_unique<CommunicatorTransportBackend>(TransportBackendConfig{
          config.gpu_id, config.rank, config.world_size,
          config.communicator_config});
  auto* communicator =
      &static_cast<CommunicatorTransportBackend*>(
           owned_transport_backend_.get())
           ->communicator();
  owned_device_backend_ = std::make_unique<DeviceBackend>(DeviceBackendConfig{
      config.device_task_capacity, config.max_device_fifos,
      config.threads_per_block, config.fifo_capacity, config.smem_size});
  backends_.transport = owned_transport_backend_.get();
  backends_.device = owned_device_backend_.get();
  completion_sources_ = backend_sources(backends_);
  resolve_ipc_buffer_pointer_ =
      [communicator](int remote_rank, uint32_t remote_buffer_id, size_t offset,
                     size_t bytes, void** out_ptr, int* out_device_idx) {
        return communicator->try_resolve_remote_ipc_pointer(
            remote_rank, remote_buffer_id, offset, bytes, out_ptr,
            out_device_idx);
      };
}

Executor::~Executor() = default;

void Executor::run(CollectivePlan plan, CollectiveBinding& binding) {
  ExecutionPlan exec = lower_plan(plan);
  validate_backends(completion_sources_, exec, binding);

  size_t total = exec.ops.size();
  if (total == 0) return;

  // Build per-flow op lists.  Ops in the plan are already topologically
  // sorted, so the per-flow vectors inherit the dependency order.
  std::vector<std::vector<uint32_t>> flow_ops(exec.num_flows);
  for (size_t i = 0; i < exec.ops.size(); i++) {
    uint32_t f = exec.ops[i].tile.flow_index;
    if (f < exec.num_flows) flow_ops[f].push_back(static_cast<uint32_t>(i));
  }

  std::vector<bool> completed(total, false);
  std::vector<BackendToken> tokens(total);
  std::vector<Backend*> op_backend(total, nullptr);
  std::unordered_map<uint64_t, size_t> inflight;
  std::vector<size_t> flow_head(exec.num_flows, 0);
  size_t completed_count = 0;

  while (completed_count < total) {
    // Phase 1: Advance every flow — submit its next ready op if all of
    // that op's dependencies (which may cross flows) are completed.
    bool any_submitted = false;
    for (uint32_t fid = 0; fid < exec.num_flows; fid++) {
      size_t head = flow_head[fid];
      if (head >= flow_ops[fid].size()) continue;

      uint32_t op_id = flow_ops[fid][head];
      if (completed[op_id]) {
        flow_head[fid] = head + 1;
        continue;
      }

      bool deps_ready = true;
      for (uint32_t dep : exec.ops[op_id].deps) {
        if (dep >= total || !completed[dep]) {
          deps_ready = false;
          break;
        }
      }
      if (!deps_ready) continue;

      Backend* backend = pick_backend(backends_, exec.ops[op_id].kind);
      BackendToken token;
      if (is_device_op(exec.ops[op_id].kind) &&
          resolve_ipc_buffer_pointer_ != nullptr) {
        ExecOp bound = bind_device_exec_op(
            exec.ops[op_id], binding, resolve_ipc_buffer_pointer_);
        token = backend->submit(bound, binding);
      } else {
        token = backend->submit(exec.ops[op_id], binding);
      }

      if (token.value == 0) continue;  // backpressure – retry next cycle

      tokens[op_id] = token;
      op_backend[op_id] = backend;
      inflight[inflight_key(backend, token)] = op_id;
      flow_head[fid] = head + 1;
      any_submitted = true;
    }

    // Phase 2: Drain batched completions, then immediately rescan flow
    // heads so that completion→successor-submit has zero-yield latency.
    {
      bool any_completed = false;

      for (Backend* backend : completion_sources_) {
        if (!backend) continue;
        BackendToken token{};
        while (backend->try_pop_completed(token)) {
          auto it = inflight.find(inflight_key(backend, token));
          if (it != inflight.end()) {
            size_t op_id = it->second;
            backend->release(token);
            inflight.erase(it);
            completed[op_id] = true;
            completed_count++;
            any_completed = true;
          }
        }
      }

      // Phase 3: Fallback poll every inflight op individually.
      for (size_t i = 0; i < total; i++) {
        if (completed[i] || !op_backend[i]) continue;
        if (op_backend[i]->poll(tokens[i])) {
          op_backend[i]->release(tokens[i]);
          inflight.erase(inflight_key(op_backend[i], tokens[i]));
          completed[i] = true;
          completed_count++;
          any_completed = true;
        }
      }

    if (completed_count == total) break;

    // Stalled: nothing submitted, nothing completed, nothing inflight.
    if (!any_submitted && !any_completed && inflight.empty()) {
      throw std::runtime_error(
          "collective stalled: all flow heads blocked with nothing in flight");
    }

    // If completions happened, immediately re-scan flow heads to
    // submit newly-unlocked successors without a yield() in between.
    if (any_completed) continue;

    std::this_thread::yield();
  }
}

void Executor::allreduce(CollectiveConfig const& config,
                         CollectiveBinding& binding) {
  bool inplace = binding.roles.input_buffer_id ==
                 binding.roles.output_buffer_id;
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, config, inplace));
  run(std::move(plan), binding);
}

void Executor::alltoall(CollectiveConfig const& config,
                        CollectiveBinding& binding) {
  bool inplace = binding.roles.input_buffer_id ==
                 binding.roles.output_buffer_id;
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllToAll, config, inplace));
  run(std::move(plan), binding);
}

UKernel::Transport::Communicator* Executor::communicator() {
  if (backends_.transport == nullptr) return nullptr;
  return &static_cast<CommunicatorTransportBackend*>(backends_.transport)
              ->communicator();
}

UKernel::Transport::Communicator const* Executor::communicator() const {
  if (backends_.transport == nullptr) return nullptr;
  return &static_cast<CommunicatorTransportBackend const*>(backends_.transport)
              ->communicator();
}

}  // namespace CCL
}  // namespace UKernel
