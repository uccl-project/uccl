#include "transport_backend.h"
#include "../../include/transport.h"
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace UKernel {
namespace CCL {

namespace {

std::atomic<uint64_t> g_next_transport_backend_cache_key{1};

void validate_span(char const* what, size_t offset, size_t bytes,
                   size_t capacity) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::invalid_argument(std::string(what) + " out of range");
  }
}

bool is_peer_ref(BufferRef const& ref) {
  return ref.kind == BufferKind::Remote;
}

int transport_peer_rank(ExecOp const& op) {
  if (op.kind == ExecOpKind::TransportSend) {
    return is_peer_ref(op.dst) ? op.dst.rank : -1;
  }
  if (op.kind == ExecOpKind::TransportRecv) {
    return is_peer_ref(op.src) ? op.src.rank : -1;
  }
  return -1;
}

}  // namespace

CommunicatorTransportBackend::CommunicatorTransportBackend(
    TransportBackendConfig const& config)
    : communicator_(std::make_unique<UKernel::Transport::Communicator>(
          config.gpu_id, config.rank, config.world_size,
          config.communicator_config != nullptr
              ? config.communicator_config
              : std::make_shared<UKernel::Transport::CommunicatorConfig>())),
      backend_cache_key_(g_next_transport_backend_cache_key.fetch_add(
          1, std::memory_order_relaxed)),
      peer_paths_ready_(static_cast<size_t>(communicator_->world_size()),
                        false) {
  completion_notifier_ = communicator_->register_completion_notifier(
      [this](unsigned request_id, std::chrono::steady_clock::time_point) {
        on_transport_completion(request_id);
      });
}

CommunicatorTransportBackend::~CommunicatorTransportBackend() = default;

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

void CommunicatorTransportBackend::validate(ExecutionPlan const& plan,
                                            CollectiveBinding& binding) const {
  ensure_memory_bindings_initialized(binding);
  ensure_plan_paths(plan);
  if (plan.staging_bytes_required != 0 &&
      binding.role_buffer(CollectiveBufferRole::Scratch).local_ptr == nullptr) {
    throw std::invalid_argument("transport backend staging buffer is missing");
  }
  if (plan.staging_bytes_required >
      binding.role_buffer(CollectiveBufferRole::Scratch).bytes) {
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

  std::vector<bool> need_peer(static_cast<size_t>(plan.nranks), false);
  for (ExecOp const& op : plan.ops) {
    int peer_rank = transport_peer_rank(op);
    if (peer_rank < 0 || peer_rank >= plan.nranks) continue;
    if (op.kind == ExecOpKind::TransportSend ||
        op.kind == ExecOpKind::TransportRecv) {
      need_peer[static_cast<size_t>(peer_rank)] = true;
    }
  }

  for (int peer = 0; peer < plan.nranks; ++peer) {
    if (peer == plan.rank) continue;
    if (!need_peer[static_cast<size_t>(peer)]) continue;
    ensure_peer_paths(peer);
  }
}

void CommunicatorTransportBackend::ensure_peer_paths(int peer_rank) const {
  std::lock_guard<std::mutex> lock(path_mu_);
  size_t idx = static_cast<size_t>(peer_rank);
  if (peer_paths_ready_.at(idx)) return;

  // Communicator exposes a duplex-ready semantic at peer granularity.
  // Keep connect/accept API surface unchanged; choose one side
  // deterministically.
  bool ok = (communicator_->rank() < peer_rank)
                ? communicator_->connect(peer_rank)
                : communicator_->accept(peer_rank);
  peer_paths_ready_.at(idx) = ok;
  if (!ok) {
    throw std::runtime_error(
        "failed to lazily establish transport path with "
        "peer " +
        std::to_string(peer_rank));
  }
}

void CommunicatorTransportBackend::ensure_memory_bindings_initialized(
    CollectiveBinding& binding) const {
  auto fully_bound = [&]() {
    if (binding.registry == nullptr) return false;
    if (binding.transport_initialized_backend_key != backend_cache_key_) {
      return false;
    }
    if (binding.transport_initialized_signature !=
        binding.transport_signature()) {
      return false;
    }
    for (BufferId id : binding.registry->registered_buffer_ids()) {
      RegisteredBuffer const& buffer = binding.buffer(id);
      if (buffer.peer_views.size() !=
          static_cast<size_t>(communicator_->world_size())) {
        return false;
      }
      if (buffer.local_ptr == nullptr || buffer.bytes == 0 ||
          !buffer.remotely_accessible) {
        continue;
      }
      if (buffer.local_buffer_id == 0) {
        return false;
      }
      for (int peer = 0; peer < communicator_->world_size(); ++peer) {
        if (peer == communicator_->rank()) continue;
        if (buffer.peer_views[static_cast<size_t>(peer)].buffer_id == 0) {
          return false;
        }
      }
    }
    return true;
  };
  if (fully_bound()) return;
  std::lock_guard<std::mutex> lock(init_mu_);
  if (fully_bound()) return;
  if (binding.transport_initialized_backend_key != backend_cache_key_ ||
      binding.transport_initialized_signature !=
          binding.transport_signature()) {
    binding.invalidate_transport_cache();
  }
  initialize_memory_bindings(binding);
}

void CommunicatorTransportBackend::initialize_memory_bindings(
    CollectiveBinding& binding) const {
  if (!binding.has_buffer(binding.buffer_id(CollectiveBufferRole::Input))) {
    throw std::invalid_argument(
        "transport backend requires a registered input buffer");
  }

  std::vector<BufferId> buffer_ids = binding.registry->registered_buffer_ids();
  for (BufferId id : buffer_ids) {
    RegisteredBuffer& buffer = binding.buffer(id);
    buffer.peer_views.resize(static_cast<size_t>(communicator_->world_size()));
    if (buffer.local_ptr != nullptr && buffer.bytes != 0 &&
        buffer.remotely_accessible) {
      if (!communicator_->reg_mr(id, buffer.local_ptr, buffer.bytes, true)) {
        throw std::runtime_error("transport backend reg_mr failed for buffer " +
                                 std::to_string(id));
      }
      buffer.local_buffer_id = id;
    } else {
      buffer.local_buffer_id = 0;
    }
    if (buffer.local_ptr != nullptr && buffer.bytes != 0 &&
        buffer.remotely_accessible) {
      (void)communicator_->reg_ipc(id, buffer.local_ptr, buffer.bytes, true);
    } else {
      (void)communicator_->reg_ipc(id, nullptr, 0, true);
    }
  }

  for (int peer = 0; peer < communicator_->world_size(); ++peer) {
    bool same_node =
        (peer != communicator_->rank()) && communicator_->same_host(peer);
    for (BufferId id : buffer_ids) {
      RegisteredBuffer& buffer = binding.buffer(id);
      buffer.peer_views[static_cast<size_t>(peer)].same_node = same_node;
    }
    if (peer == communicator_->rank()) continue;
  }

  for (int peer = 0; peer < communicator_->world_size(); ++peer) {
    if (peer == communicator_->rank()) continue;
    for (BufferId id : buffer_ids) {
      RegisteredBuffer& buffer = binding.buffer(id);
      if (buffer.local_ptr != nullptr && buffer.bytes != 0 &&
          buffer.remotely_accessible && !communicator_->wait_mr(peer, id)) {
        throw std::runtime_error("transport backend wait_mr failed for peer " +
                                 std::to_string(peer) + ", buffer " +
                                 std::to_string(id));
      }
      if (buffer.local_ptr != nullptr && buffer.bytes != 0 &&
          buffer.remotely_accessible) {
        buffer.peer_views[static_cast<size_t>(peer)].buffer_id = id;
      } else {
        buffer.peer_views[static_cast<size_t>(peer)].buffer_id = 0;
      }
      if (buffer.peer_views[static_cast<size_t>(peer)].same_node &&
          buffer.peer_views[static_cast<size_t>(peer)].buffer_id != 0 &&
          !communicator_->wait_ipc(peer, id)) {
        throw std::runtime_error(
            "transport backend wait ipc buffer failed for peer " +
            std::to_string(peer));
      }
    }
  }
  binding.transport_initialized_backend_key = backend_cache_key_;
  binding.transport_initialized_signature = binding.transport_signature();
}

bool CommunicatorTransportBackend::supports(ExecOpKind kind) const {
  return kind == ExecOpKind::TransportSend || kind == ExecOpKind::TransportRecv;
}

BackendToken CommunicatorTransportBackend::submit(ExecOp const& op,
                                                  CollectiveBinding& binding) {
  if (!supports(op.kind)) {
    throw std::invalid_argument("unsupported op kind for transport backend");
  }
  ensure_memory_bindings_initialized(binding);

  int peer_rank = resolve_peer_rank(op);
  BackendToken token{next_token_++};
  unsigned request_id = 0;
  if (op.kind == ExecOpKind::TransportSend) {
    (void)resolve_const(binding, op.src, op.tile.size_bytes);
    Transport::LocalSlice src_slice{
        resolve_local_buffer_id(binding, op.src, op.tile.size_bytes),
        op.src.offset_bytes,
        op.tile.size_bytes,
    };
    std::optional<Transport::RemoteSlice> dst_hint = std::nullopt;
    if (op.dst.kind == BufferKind::Remote) {
      // Communicator::isend auto-enriches write hints (addr/rkey/capacity)
      // from exchanged remote metadata when only buffer_id/offset are provided.
      dst_hint = Transport::RemoteSlice{
          resolve_remote_buffer_id(binding, op.dst), op.dst.offset_bytes, {}};
    }
    request_id = communicator_->isend(peer_rank, src_slice, dst_hint);
  } else {
    (void)resolve_mutable(binding, op.dst, op.tile.size_bytes);
    Transport::LocalSlice dst_slice{
        resolve_local_buffer_id(binding, op.dst, op.tile.size_bytes),
        op.dst.offset_bytes,
        op.tile.size_bytes,
    };
    request_id = communicator_->irecv(peer_rank, dst_slice);
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

void* CommunicatorTransportBackend::resolve_mutable(
    CollectiveBinding const& binding, BufferRef const& ref,
    size_t bytes) const {
  if (ref.kind != BufferKind::Local) {
    throw std::invalid_argument(
        "transport backend cannot bind remote destination buffer");
  }
  RegisteredBuffer const& buffer = binding.buffer(ref.buffer_id);
  if (buffer.local_ptr == nullptr) {
    throw std::invalid_argument("transport local buffer is missing");
  }
  validate_span("transport local buffer", ref.offset_bytes, bytes,
                buffer.bytes);
  return byte_offset(buffer.local_ptr, ref.offset_bytes);
}

void const* CommunicatorTransportBackend::resolve_const(
    CollectiveBinding const& binding, BufferRef const& ref,
    size_t bytes) const {
  if (ref.kind != BufferKind::Local) {
    throw std::invalid_argument(
        "transport backend cannot bind remote source buffer");
  }
  RegisteredBuffer const& buffer = binding.buffer(ref.buffer_id);
  if (buffer.local_ptr == nullptr) {
    throw std::invalid_argument("transport local buffer is missing");
  }
  validate_span("transport local buffer", ref.offset_bytes, bytes,
                buffer.bytes);
  return byte_offset(buffer.local_ptr, ref.offset_bytes);
}

uint32_t CommunicatorTransportBackend::resolve_local_buffer_id(
    CollectiveBinding const& binding, BufferRef const& ref,
    size_t bytes) const {
  if (ref.kind == BufferKind::Remote) {
    throw std::invalid_argument(
        "transport backend local buffer id requires local buffer ref");
  }
  RegisteredBuffer const& buffer = binding.buffer(ref.buffer_id);
  if (buffer.local_buffer_id != 0) {
    return buffer.local_buffer_id;
  }
  (void)bytes;
  return ref.buffer_id;
}

int CommunicatorTransportBackend::resolve_peer_rank(ExecOp const& op) const {
  if (op.kind == ExecOpKind::TransportSend) {
    if (!is_peer_ref(op.dst) || op.dst.rank < 0) {
      throw std::invalid_argument(
          "transport send requires a peer destination buffer");
    }
    return op.dst.rank;
  }
  if (op.kind == ExecOpKind::TransportRecv) {
    if (!is_peer_ref(op.src) || op.src.rank < 0) {
      throw std::invalid_argument(
          "transport recv requires a peer source buffer");
    }
    return op.src.rank;
  }
  throw std::invalid_argument(
      "transport peer rank requested for non-transport op");
}

uint32_t CommunicatorTransportBackend::resolve_remote_buffer_id(
    CollectiveBinding const& binding, BufferRef const& ref) const {
  if (!is_peer_ref(ref) || ref.rank < 0) {
    throw std::invalid_argument(
        "transport remote buffer id requires peer buffer ref");
  }
  int peer_rank = ref.rank;
  RegisteredBuffer const& buffer = binding.buffer(ref.buffer_id);
  if (peer_rank < 0 ||
      static_cast<size_t>(peer_rank) >= buffer.peer_views.size()) {
    throw std::invalid_argument("transport peer rank out of range");
  }
  uint32_t remote_buffer_id =
      buffer.peer_views[static_cast<size_t>(peer_rank)].buffer_id;
  if (remote_buffer_id == 0) {
    throw std::invalid_argument("transport remote buffer id is missing");
  }
  return remote_buffer_id;
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
