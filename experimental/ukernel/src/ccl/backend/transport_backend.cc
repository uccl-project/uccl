#include "transport_backend.h"
#include "../utils.h"
#include "../../include/transport.h"
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

namespace UKernel {
namespace CCL {

namespace {

std::atomic<uint64_t> g_next_transport_backend_cache_key{1};
constexpr int kMemoryBindingTimeoutMs = 30000;

bool is_peer_ref(BufferRef const& ref) {
  return ref.kind == BufferKind::Remote;
}

int transport_peer_rank(Op const& op) {
  if (op.kind == OpKind::TransportSend)
    return is_peer_ref(op.dst) ? op.dst.rank : -1;
  if (op.kind == OpKind::TransportRecv)
    return is_peer_ref(op.src) ? op.src.rank : -1;
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
      peer_put_ready_(static_cast<size_t>(communicator_->world_size()), 0),
      peer_wait_ready_(static_cast<size_t>(communicator_->world_size()), 0) {
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

// ── validate — one-time init ──────────────────────────────────────────

void CommunicatorTransportBackend::validate(CollectivePlan const& plan,
                                            CollectiveBinding& binding) {
  if (plan.rank != communicator_->rank())
    throw std::invalid_argument("plan rank != communicator rank");
  if (plan.nranks != communicator_->world_size())
    throw std::invalid_argument("plan world_size != communicator world_size");

  if (plan.staging_bytes_required != 0 &&
      binding.role_buffer(CollectiveBufferRole::Scratch).local_ptr == nullptr)
    throw std::invalid_argument("staging buffer is missing");
  if (plan.staging_bytes_required >
      binding.role_buffer(CollectiveBufferRole::Scratch).bytes)
    throw std::invalid_argument("staging buffer too small");

  // Phase 1: Pre-establish peer paths first — they require both ranks
  // to be active simultaneously (OOB exchange).  Doing them before memory
  // bindings minimises the window where one rank is waiting.
  std::vector<bool> need_put(static_cast<size_t>(plan.nranks), false);
  std::vector<bool> need_wait(static_cast<size_t>(plan.nranks), false);
  for (Op const& op : plan.ops) {
    int peer_rank = transport_peer_rank(op);
    if (peer_rank < 0 || peer_rank >= plan.nranks) continue;
    if (op.kind == OpKind::TransportSend)
      need_put[static_cast<size_t>(peer_rank)] = true;
    else if (op.kind == OpKind::TransportRecv)
      need_wait[static_cast<size_t>(peer_rank)] = true;
  }
  for (int peer = 0; peer < plan.nranks; ++peer) {
    if (peer == plan.rank) continue;
    if (!need_put[static_cast<size_t>(peer)] && !need_wait[static_cast<size_t>(peer)])
      continue;
    ensure_peer_path(peer, need_put[static_cast<size_t>(peer)],
                     need_wait[static_cast<size_t>(peer)]);
  }

  // Phase 2: Memory bindings — purely local, no peer synchronisation needed.
  {
    std::lock_guard<std::mutex> lock(init_mu_);
    if (!is_transport_fresh(binding)) return;
    binding.invalidate_transport_cache();
    initialize_memory_bindings(binding);
  }
}

bool CommunicatorTransportBackend::is_transport_fresh(
    CollectiveBinding const& binding) const {
  if (binding.transport_initialized_backend_key != backend_cache_key_)
    return true;
  if (binding.transport_initialized_signature != binding.transport_signature())
    return true;
  return false;
}

void CommunicatorTransportBackend::ensure_peer_path(int peer_rank,
                                                     bool need_put,
                                                     bool need_wait) {
  size_t idx = static_cast<size_t>(peer_rank);

  auto establish_put = [&]() {
    if (peer_put_ready_[idx]) return true;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(connect_timeout_s_);
    while (!peer_put_ready_[idx]) {
      if (communicator_->connect(peer_rank)) {
        peer_put_ready_[idx] = 1;
        return true;
      }
      if (std::chrono::steady_clock::now() >= deadline)
        return false;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(connect_retry_ms_));
    }
    return false;
  };

  auto establish_wait = [&]() {
    if (peer_wait_ready_[idx]) return true;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(connect_timeout_s_);
    while (!peer_wait_ready_[idx]) {
      if (communicator_->accept(peer_rank)) {
        peer_wait_ready_[idx] = 1;
        return true;
      }
      if (std::chrono::steady_clock::now() >= deadline)
        return false;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(connect_retry_ms_));
    }
    return false;
  };

  bool ok = true;
  if (!need_put && !need_wait) return;

  // When both directions are needed for the same peer, order by rank
  // to avoid symmetric deadlock.
  if (need_put && need_wait)
    ok = (communicator_->rank() < peer_rank)
             ? (establish_put() && establish_wait())
             : (establish_wait() && establish_put());
  else if (need_put)
    ok = establish_put();
  else
    ok = establish_wait();

  if (!ok)
    throw std::runtime_error(
        "timed out establishing transport path with peer " +
        std::to_string(peer_rank));
}

// ── memory bindings — MR / IPC registration ───────────────────────────

void CommunicatorTransportBackend::initialize_memory_bindings(
    CollectiveBinding& binding) {
  if (!binding.has_buffer(binding.buffer_id(CollectiveBufferRole::Input)))
    throw std::invalid_argument("transport backend requires a registered input buffer");

  auto buffer_ids = binding.registry->registered_buffer_ids();
  int world = communicator_->world_size();

  for (BufferId id : buffer_ids) {
    RegisteredBuffer& buf = binding.buffer(id);
    buf.peer_views.resize(static_cast<size_t>(world));
    if (buf.local_ptr && buf.bytes && buf.remotely_accessible) {
      if (!communicator_->reg_mr(id, buf.local_ptr, buf.bytes, true))
        throw std::runtime_error("reg_mr failed for buffer " + std::to_string(id));
      buf.local_buffer_id = id;
    } else {
      buf.local_buffer_id = 0;
    }
    communicator_->reg_ipc(id, (buf.local_ptr && buf.bytes && buf.remotely_accessible)
                                   ? buf.local_ptr : nullptr,
                           (buf.local_ptr && buf.bytes && buf.remotely_accessible)
                                   ? buf.bytes : 0,
                           true);
  }

  for (int peer = 0; peer < world; ++peer) {
    bool same_node = (peer != communicator_->rank()) && communicator_->same_host(peer);
    for (BufferId id : buffer_ids)
      binding.buffer(id).peer_views[static_cast<size_t>(peer)].same_node = same_node;
    if (peer == communicator_->rank()) continue;
  }

  for (int peer = 0; peer < world; ++peer) {
    if (peer == communicator_->rank()) continue;
    for (BufferId id : buffer_ids) {
      RegisteredBuffer& buf = binding.buffer(id);
      if (buf.local_ptr && buf.bytes && buf.remotely_accessible) {
        if (!communicator_->wait_mr(peer, id, kMemoryBindingTimeoutMs))
          throw std::runtime_error("wait_mr failed peer=" + std::to_string(peer) +
                                   " buf=" + std::to_string(id));
        buf.peer_views[static_cast<size_t>(peer)].buffer_id = id;
      } else {
        buf.peer_views[static_cast<size_t>(peer)].buffer_id = 0;
      }
      if (buf.peer_views[static_cast<size_t>(peer)].same_node &&
          buf.peer_views[static_cast<size_t>(peer)].buffer_id != 0 &&
          !communicator_->wait_ipc(peer, id, kMemoryBindingTimeoutMs))
        throw std::runtime_error("wait_ipc failed peer=" + std::to_string(peer));
    }
  }
  binding.transport_initialized_backend_key = backend_cache_key_;
  binding.transport_initialized_signature = binding.transport_signature();
}

// ── submit / drain ─────────────────────────────────────────────────────

bool CommunicatorTransportBackend::supports(OpKind kind) const {
  return kind == OpKind::TransportSend || kind == OpKind::TransportRecv;
}

BackendToken CommunicatorTransportBackend::submit(Op const& op,
                                                   CollectiveBinding& binding) {
  (void)binding;
  int peer_rank = resolve_peer_rank(op);
  size_t idx = static_cast<size_t>(peer_rank);

  if (op.kind == OpKind::TransportSend && !peer_put_ready_[idx])
    throw std::runtime_error(
        "transport put path not established — call validate() first");
  if (op.kind == OpKind::TransportRecv && !peer_wait_ready_[idx])
    throw std::runtime_error(
        "transport wait path not established — call validate() first");

  unsigned request_id = 0;
  if (op.kind == OpKind::TransportSend) {
    uint32_t dst_buf_id = 0;
    size_t dst_off = 0;
    if (op.dst.kind == BufferKind::Remote) {
      dst_buf_id = resolve_remote_buffer_id(binding, op.dst);
      dst_off = op.dst.offset_bytes;
    }
    uint32_t src_buf = resolve_local_buffer_id(binding, op.src);
    request_id = communicator_->isend(
        peer_rank, src_buf, op.src.offset_bytes, op.tile.size_bytes,
        dst_buf_id, dst_off);
  } else {
    uint32_t dst_buf = resolve_local_buffer_id(binding, op.dst);
    request_id = communicator_->irecv(
        peer_rank, dst_buf, op.dst.offset_bytes, op.tile.size_bytes);
  }
  if (request_id == 0)
    throw std::runtime_error("transport request submission failed");

  BackendToken token{next_token_++};
  {
    std::lock_guard<std::mutex> lk(mu_);
    pending_[token.value] = request_id;
    req_to_token_[request_id] = token.value;
  }
  return token;
}

size_t CommunicatorTransportBackend::drain(BackendToken* out, size_t max_count) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (pending_.empty()) return 0;
  }
  auto done = communicator_->progress();
  if (done.empty()) return 0;

  size_t count = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [req_id, req_failed] : done) {
      if (count >= max_count) break;
      auto ti = req_to_token_.find(req_id);
      if (ti == req_to_token_.end()) continue;
      auto pi = pending_.find(ti->second);
      if (pi == pending_.end()) continue;  // already drained
      communicator_->release(req_id);
      out[count].value = ti->second;
      out[count].failed = req_failed;
      ++count;
      pending_.erase(pi);
      req_to_token_.erase(ti);
    }
  }
  return count;
}

// ── helpers ────────────────────────────────────────────────────────────

int CommunicatorTransportBackend::resolve_peer_rank(Op const& op) const {
  if (op.kind == OpKind::TransportSend) {
    if (!is_peer_ref(op.dst) || op.dst.rank < 0)
      throw std::invalid_argument("transport send requires a peer destination");
    return op.dst.rank;
  }
  if (op.kind == OpKind::TransportRecv) {
    if (!is_peer_ref(op.src) || op.src.rank < 0)
      throw std::invalid_argument("transport recv requires a peer source");
    return op.src.rank;
  }
  throw std::invalid_argument("non-transport op");
}

uint32_t CommunicatorTransportBackend::resolve_local_buffer_id(
    CollectiveBinding const& binding, BufferRef const& ref) const {
  if (ref.kind == BufferKind::Remote)
    throw std::invalid_argument("local buffer id requires a local ref");
  RegisteredBuffer const& buf = binding.buffer_for_ref(ref);
  return buf.local_buffer_id != 0 ? buf.local_buffer_id : ref.buffer_id;
}

uint32_t CommunicatorTransportBackend::resolve_remote_buffer_id(
    CollectiveBinding const& binding, BufferRef const& ref) const {
  if (!is_peer_ref(ref) || ref.rank < 0)
    throw std::invalid_argument("remote buffer id requires a peer ref");
  RegisteredBuffer const& buf = binding.buffer_for_ref(ref);
  if (static_cast<size_t>(ref.rank) >= buf.peer_views.size())
    throw std::invalid_argument("peer rank out of range");
  uint32_t rid = buf.peer_views[static_cast<size_t>(ref.rank)].buffer_id;
  if (rid == 0)
    throw std::invalid_argument("remote buffer id is missing");
  return rid;
}

}  // namespace CCL
}  // namespace UKernel
