#include "transport_backend.h"
#include "../../include/transport.h"
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace UKernel {
namespace CCL {

namespace {

void validate_span(char const* what, size_t offset, size_t bytes,
                   size_t capacity) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::invalid_argument(std::string(what) + " out of range");
  }
}

}  // namespace

CommunicatorTransportBackend::CommunicatorTransportBackend(
    TransportBackendConfig const& config,
    std::shared_ptr<CollectiveMemory> memory)
    : communicator_(std::make_unique<UKernel::Transport::Communicator>(
          config.gpu_id, config.rank, config.world_size,
          config.communicator_config != nullptr
              ? config.communicator_config
              : std::make_shared<UKernel::Transport::CommunicatorConfig>())),
      memory_(std::move(memory)),
      peer_paths_(static_cast<size_t>(communicator_->world_size())) {
  if (memory_ == nullptr) {
    throw std::invalid_argument(
        "transport backend requires collective memory");
  }
  completion_notifier_ = communicator_->register_completion_notifier(
      [this](unsigned request_id, std::chrono::steady_clock::time_point) {
        on_transport_completion(request_id);
      });
}

char const* CommunicatorTransportBackend::name() const {
  return "communicator-transport";
}

UKernel::Transport::Communicator& CommunicatorTransportBackend::communicator() {
  return *communicator_;
}

UKernel::Transport::Communicator const&
CommunicatorTransportBackend::communicator() const {
  return *communicator_;
}

CollectiveMemory& CommunicatorTransportBackend::memory() { return *memory_; }

CollectiveMemory const& CommunicatorTransportBackend::memory() const {
  return *memory_;
}

void CommunicatorTransportBackend::validate(ExecutionPlan const& plan) const {
  ensure_memory_bindings_initialized();
  ensure_plan_paths(plan);
  if (plan.staging_bytes_required != 0 &&
      memory_->staging.local_ptr == nullptr) {
    throw std::invalid_argument("transport backend staging buffer is missing");
  }
  if (plan.staging_bytes_required > memory_->staging.bytes) {
    throw std::invalid_argument(
        "transport backend staging capacity is insufficient");
  }
}

void CommunicatorTransportBackend::ensure_plan_paths(
    ExecutionPlan const& plan) const {
  if (plan.rank != communicator_->rank()) {
    throw std::invalid_argument(
        "execution plan rank does not match transport communicator");
  }
  if (plan.nranks != communicator_->world_size()) {
    throw std::invalid_argument(
        "execution plan world size does not match transport communicator");
  }

  std::vector<bool> need_send(static_cast<size_t>(plan.nranks), false);
  std::vector<bool> need_recv(static_cast<size_t>(plan.nranks), false);
  for (ExecOp const& op : plan.ops) {
    if (op.peer_rank < 0 || op.peer_rank >= plan.nranks) continue;
    if (op.kind == ExecOpKind::TransportSend) {
      need_send[static_cast<size_t>(op.peer_rank)] = true;
    } else if (op.kind == ExecOpKind::TransportRecv) {
      need_recv[static_cast<size_t>(op.peer_rank)] = true;
    }
  }

  // Phase 1 establishes lower-rank -> higher-rank paths only.
  for (int peer = 0; peer < plan.nranks; ++peer) {
    if (peer == plan.rank) continue;
    bool need_send_to_peer = need_send[static_cast<size_t>(peer)];
    bool need_recv_from_peer = need_recv[static_cast<size_t>(peer)];
    if (plan.rank < peer) {
      ensure_peer_paths(peer, need_send_to_peer, false);
    } else {
      ensure_peer_paths(peer, false, need_recv_from_peer);
    }
  }

  // Phase 2 establishes higher-rank -> lower-rank paths only.
  for (int peer = 0; peer < plan.nranks; ++peer) {
    if (peer == plan.rank) continue;
    bool need_send_to_peer = need_send[static_cast<size_t>(peer)];
    bool need_recv_from_peer = need_recv[static_cast<size_t>(peer)];
    if (plan.rank < peer) {
      ensure_peer_paths(peer, false, need_recv_from_peer);
    } else {
      ensure_peer_paths(peer, need_send_to_peer, false);
    }
  }
}

void CommunicatorTransportBackend::ensure_peer_paths(int peer_rank,
                                                     bool need_send,
                                                     bool need_recv) const {
  if (!need_send && !need_recv) return;

  std::lock_guard<std::mutex> lock(path_mu_);
  PeerPathState& state = peer_paths_.at(static_cast<size_t>(peer_rank));
  bool ok = true;
  if (need_send) {
    if (!state.send_ready) {
      ok = communicator_->connect_to(peer_rank);
      state.send_ready = ok;
    }
  }
  if (ok && need_recv) {
    if (!state.recv_ready) {
      ok = communicator_->accept_from(peer_rank);
      state.recv_ready = ok;
    }
  }

  if (!ok) {
    throw std::runtime_error("failed to lazily establish transport path with "
                             "peer " +
                             std::to_string(peer_rank));
  }
}

void CommunicatorTransportBackend::ensure_memory_bindings_initialized() const {
  if (bindings_initialized_) return;
  std::lock_guard<std::mutex> lock(init_mu_);
  if (bindings_initialized_) return;
  initialize_memory_bindings();
  bindings_initialized_ = true;
}

void CommunicatorTransportBackend::initialize_memory_bindings() const {
  if (memory_->tensor.local_ptr == nullptr || memory_->tensor.bytes == 0) {
    throw std::invalid_argument(
        "transport backend requires a valid local tensor buffer");
  }

  memory_->tensor.local_rank = communicator_->rank();
  memory_->tensor.peer_views.resize(
      static_cast<size_t>(communicator_->world_size()));
  memory_->staging.peer_views.resize(
      static_cast<size_t>(communicator_->world_size()));

  Transport::MR tensor_mr =
      communicator_->reg_mr(memory_->tensor.local_ptr, memory_->tensor.bytes);
  memory_->tensor.local_mr_id = tensor_mr.id;
  Transport::MR staging_mr{};
  bool has_staging = memory_->staging.local_ptr != nullptr &&
                     memory_->staging.bytes != 0;

  if (has_staging) {
    staging_mr =
        communicator_->reg_mr(memory_->staging.local_ptr, memory_->staging.bytes);
    memory_->staging.local_mr_id = staging_mr.id;
  }

  for (int peer = 0; peer < communicator_->world_size(); ++peer) {
    bool same_node =
        (peer != communicator_->rank()) && communicator_->same_host(peer);
    auto& tensor_view = memory_->tensor.peer_views[static_cast<size_t>(peer)];
    auto& staging_view = memory_->staging.peer_views[static_cast<size_t>(peer)];
    tensor_view.same_node = same_node;
    staging_view.same_node = same_node;
    if (peer == communicator_->rank()) continue;
    if (!communicator_->notify_mr(peer, tensor_mr)) {
      throw std::runtime_error("transport backend notify_mr failed for peer " +
                               std::to_string(peer));
    }
    if (!communicator_->notify_mr(peer, staging_mr)) {
      throw std::runtime_error(
          "transport backend notify staging_mr failed for peer " +
          std::to_string(peer));
    }
    if (same_node) {
      if (!communicator_->notify_ipc_buffer(peer, tensor_mr.id,
                                            memory_->tensor.local_ptr,
                                            memory_->tensor.bytes)) {
        throw std::runtime_error(
            "transport backend notify tensor ipc buffer failed for peer " +
            std::to_string(peer));
      }
      if (has_staging &&
          !communicator_->notify_ipc_buffer(peer, staging_mr.id,
                                            memory_->staging.local_ptr,
                                            memory_->staging.bytes)) {
        throw std::runtime_error(
            "transport backend notify staging ipc buffer failed for peer " +
            std::to_string(peer));
      }
    }
  }

  for (int peer = 0; peer < communicator_->world_size(); ++peer) {
    if (peer == communicator_->rank()) continue;
    Transport::MR remote_tensor_mr{};
    if (!communicator_->wait_mr_notify(peer, remote_tensor_mr)) {
      throw std::runtime_error(
          "transport backend wait tensor_mr_notify failed for peer " +
          std::to_string(peer));
    }
    memory_->tensor.peer_views[static_cast<size_t>(peer)].mr_id =
        remote_tensor_mr.id;
    if (memory_->tensor.peer_views[static_cast<size_t>(peer)].same_node &&
        remote_tensor_mr.id != 0 &&
        !communicator_->wait_ipc_buffer(peer, remote_tensor_mr.id)) {
      throw std::runtime_error(
          "transport backend wait tensor ipc buffer failed for peer " +
          std::to_string(peer));
    }

    Transport::MR remote_staging_mr{};
    if (!communicator_->wait_mr_notify(peer, remote_staging_mr)) {
      throw std::runtime_error(
          "transport backend wait staging_mr_notify failed for peer " +
          std::to_string(peer));
    }
    memory_->staging.peer_views[static_cast<size_t>(peer)].mr_id =
        remote_staging_mr.id;
    if (memory_->staging.peer_views[static_cast<size_t>(peer)].same_node &&
        remote_staging_mr.id != 0 &&
        !communicator_->wait_ipc_buffer(peer, remote_staging_mr.id)) {
      throw std::runtime_error(
          "transport backend wait staging ipc buffer failed for peer " +
          std::to_string(peer));
    }
  }
}

bool CommunicatorTransportBackend::supports(ExecOpKind kind) const {
  return kind == ExecOpKind::TransportSend || kind == ExecOpKind::TransportRecv;
}

BackendToken CommunicatorTransportBackend::submit(ExecOp const& op) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for transport backend");
  }
  if (op.peer_rank < 0) {
    throw std::invalid_argument("transport op requires a valid peer rank");
  }
  ensure_memory_bindings_initialized();

  BackendToken token{next_token_++};
  unsigned request_id = 0;
  if (op.kind == ExecOpKind::TransportSend) {
    void const* src = resolve_const(op.src, op.tile.size_bytes);
    request_id =
        communicator_->isend(op.peer_rank, const_cast<void*>(src), 0,
                             op.tile.size_bytes,
                             resolve_local_mr_id(op.src, op.tile.size_bytes),
                             resolve_remote_mr_id(op.peer_rank), true);
  } else {
    void* dst = resolve_mutable(op.dst, op.tile.size_bytes);
    request_id = communicator_->irecv(op.peer_rank, dst, 0, op.tile.size_bytes,
                                      true);
  }

  if (request_id == 0) {
    throw std::runtime_error(
        "communicator transport request submission failed");
  }

  std::lock_guard<std::mutex> lk(mu_);
  pending_[token.value] = PendingRequest{request_id, false, false};
  request_to_token_[request_id] = token.value;
  return token;
}

bool CommunicatorTransportBackend::poll(BackendToken token) {
  unsigned request_id = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(token.value);
    if (it == pending_.end()) return true;
    if (it->second.completed) return true;
    request_id = it->second.request_id;
  }

  if (!communicator_->poll(request_id)) return false;
  on_transport_completion(request_id);
  return true;
}

bool CommunicatorTransportBackend::try_pop_completed(BackendToken& token) {
  std::lock_guard<std::mutex> lk(mu_);
  if (completed_tokens_.empty()) return false;
  token.value = completed_tokens_.front();
  completed_tokens_.pop_front();
  return true;
}

void CommunicatorTransportBackend::release(BackendToken token) {
  unsigned request_id = 0;
  bool should_release = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(token.value);
    if (it == pending_.end()) return;
    request_id = it->second.request_id;
    it->second.released = true;
    if (it->second.completed) {
      should_release = true;
      pending_.erase(it);
      request_to_token_.erase(request_id);
    }
  }
  if (should_release) {
    communicator_->release(request_id);
  }
}

void* CommunicatorTransportBackend::resolve_mutable(BufferRef const& ref,
                                                    size_t bytes) const {
  switch (ref.kind) {
    case BufferKind::Staging:
      if (memory_->staging.local_ptr == nullptr) {
        throw std::invalid_argument("transport staging buffer is missing");
      }
      validate_span("transport staging", ref.offset_bytes, bytes,
                    memory_->staging.bytes);
      return byte_offset(memory_->staging.local_ptr, ref.offset_bytes);
    case BufferKind::Tensor:
      if (memory_->tensor.local_ptr == nullptr) {
        throw std::invalid_argument("transport local tensor is missing");
      }
      validate_span("transport local tensor", ref.offset_bytes, bytes,
                    memory_->tensor.bytes);
      return byte_offset(memory_->tensor.local_ptr, ref.offset_bytes);
    case BufferKind::PeerTensor:
    case BufferKind::PeerStaging:
      break;
  }
  throw std::invalid_argument("transport backend cannot bind remote destination buffer");
}

void const* CommunicatorTransportBackend::resolve_const(BufferRef const& ref,
                                                        size_t bytes) const {
  switch (ref.kind) {
    case BufferKind::Staging:
      if (memory_->staging.local_ptr == nullptr) {
        throw std::invalid_argument("transport staging buffer is missing");
      }
      validate_span("transport staging", ref.offset_bytes, bytes,
                    memory_->staging.bytes);
      return byte_offset(memory_->staging.local_ptr, ref.offset_bytes);
    case BufferKind::Tensor:
      if (memory_->tensor.local_ptr == nullptr) {
        throw std::invalid_argument("transport local tensor is missing");
      }
      validate_span("transport local tensor", ref.offset_bytes, bytes,
                    memory_->tensor.bytes);
      return byte_offset(memory_->tensor.local_ptr, ref.offset_bytes);
    case BufferKind::PeerTensor:
    case BufferKind::PeerStaging:
      break;
  }
  throw std::invalid_argument("transport backend cannot bind remote source buffer");
}

uint32_t CommunicatorTransportBackend::resolve_local_mr_id(BufferRef const& ref,
                                                           size_t bytes) const {
  if (ref.kind == BufferKind::Tensor) {
    if (memory_->tensor.local_mr_id == 0) {
      throw std::invalid_argument("transport local tensor MR id is missing");
    }
    return memory_->tensor.local_mr_id;
  }
  if (ref.kind == BufferKind::PeerTensor || ref.kind == BufferKind::PeerStaging) {
    throw std::invalid_argument("transport backend local MR requires local buffer ref");
  }

  void* ptr = resolve_mutable(ref, bytes);
  return communicator_->get_local_mr(ptr).id;
}

uint32_t CommunicatorTransportBackend::resolve_remote_mr_id(
    int peer_rank) const {
  if (peer_rank < 0 ||
      static_cast<size_t>(peer_rank) >= memory_->tensor.peer_views.size()) {
    throw std::invalid_argument("transport peer rank out of range");
  }
  uint32_t mr_id =
      memory_->tensor.peer_views[static_cast<size_t>(peer_rank)].mr_id;
  if (mr_id == 0) {
    throw std::invalid_argument("transport remote MR id is missing");
  }
  return mr_id;
}

void CommunicatorTransportBackend::on_transport_completion(
    unsigned request_id) {
  bool should_release = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = request_to_token_.find(request_id);
    if (it == request_to_token_.end()) return;

    auto pending_it = pending_.find(it->second);
    if (pending_it == pending_.end() || pending_it->second.completed) return;

    pending_it->second.completed = true;
    if (pending_it->second.released) {
      should_release = true;
      pending_.erase(pending_it);
      request_to_token_.erase(it);
    } else {
      completed_tokens_.push_back(it->second);
    }
  }
  if (should_release) {
    communicator_->release(request_id);
  }
}

void* CommunicatorTransportBackend::byte_offset(void* base, size_t offset) {
  return static_cast<void*>(static_cast<char*>(base) + offset);
}

void const* CommunicatorTransportBackend::byte_offset(void const* base,
                                                      size_t offset) {
  return static_cast<void const*>(static_cast<char const*>(base) + offset);
}

}  // namespace CCL
}  // namespace UKernel
