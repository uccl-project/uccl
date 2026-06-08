#include "transport_backend.h"
#include "../../include/transport.h"
#include "../utils.h"
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

int transport_peer_rank(Op const& op) {
  if (op.kind == OpKind::Send) return op.dst_peer;
  if (op.kind == OpKind::Recv) return op.src_peer;
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
      peer_wait_ready_(static_cast<size_t>(communicator_->world_size()), 0) {}

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

void CommunicatorTransportBackend::validate(TiledResult const& tiled,
                                            void* input_ptr, void* output_ptr,
                                            void* scratch_ptr) {
  (void)output_ptr;
  if (tiled.rank != communicator_->rank())
    throw std::invalid_argument("tiled rank != communicator rank");
  if (tiled.nranks != communicator_->world_size())
    throw std::invalid_argument("tiled world_size != communicator world_size");

  if (tiled.staging_bytes_required != 0 && scratch_ptr == nullptr)
    throw std::invalid_argument("staging buffer is missing");

  // Phase 1: Pre-establish peer paths.
  std::vector<bool> need_put(static_cast<size_t>(tiled.nranks), false);
  std::vector<bool> need_wait(static_cast<size_t>(tiled.nranks), false);
  for (Op const& op : tiled.ops) {
    int peer_rank = transport_peer_rank(op);
    if (peer_rank < 0 || peer_rank >= tiled.nranks) continue;
    if (op.kind == OpKind::Send)
      need_put[static_cast<size_t>(peer_rank)] = true;
    else if (op.kind == OpKind::Recv)
      need_wait[static_cast<size_t>(peer_rank)] = true;
  }
  for (int peer = 0; peer < tiled.nranks; ++peer) {
    if (peer == tiled.rank) continue;
    if (!need_put[static_cast<size_t>(peer)] &&
        !need_wait[static_cast<size_t>(peer)])
      continue;
    ensure_peer_path(peer, need_put[static_cast<size_t>(peer)],
                     need_wait[static_cast<size_t>(peer)]);
  }

  // Phase 2: Memory bindings.
  {
    std::lock_guard<std::mutex> lock(init_mu_);
    initialize_memory_bindings(input_ptr, output_ptr, scratch_ptr);
  }
}


void CommunicatorTransportBackend::initialize_memory_bindings(
    void* input_ptr, void* output_ptr, void* scratch_ptr) {
  int world = communicator_->world_size();

  auto reg_buf = [&](uint32_t id, void* ptr, size_t bytes) {
    if (!ptr) return;
    if (!communicator_->reg_mr(id, ptr, bytes, true))
      throw std::runtime_error("reg_mr failed for buffer " + std::to_string(id));
    communicator_->reg_ipc(id, ptr, bytes, true);
  };

  reg_buf(1, input_ptr, 0);
  reg_buf(2, output_ptr, 0);
  reg_buf(3, scratch_ptr, 0);

  for (int peer = 0; peer < world; ++peer) {
    if (peer == communicator_->rank()) continue;
    for (uint32_t id = 1; id <= 3; ++id) {
      void* ptr = (id == 1) ? input_ptr : (id == 2) ? output_ptr : scratch_ptr;
      if (!ptr) continue;
      if (!communicator_->wait_mr(peer, id, kMemoryBindingTimeoutMs))
        throw std::runtime_error("wait_mr failed peer=" + std::to_string(peer) +
                                 " buf=" + std::to_string(id));
      if (communicator_->same_host(peer) &&
          !communicator_->wait_ipc(peer, id, kMemoryBindingTimeoutMs))
        throw std::runtime_error("wait_ipc failed peer=" + std::to_string(peer));
    }
  }
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
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(connect_retry_ms_));
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
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(connect_retry_ms_));
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

// ── submit / drain ─────────────────────────────────────────────────────

bool CommunicatorTransportBackend::supports(OpKind kind) const {
  return kind == OpKind::Send || kind == OpKind::Recv;
}

BackendToken CommunicatorTransportBackend::submit(Op const& op,
                                                   OpBindings const& bind,
                                                   void* input_ptr,
                                                   void* output_ptr,
                                                   void* scratch_ptr) {
  (void)bind;
  (void)input_ptr;
  (void)output_ptr;
  (void)scratch_ptr;
  int peer_rank = resolve_peer_rank(op);
  size_t idx = static_cast<size_t>(peer_rank);

  if (op.kind == OpKind::Send && !peer_put_ready_[idx])
    throw std::runtime_error(
        "transport put path not established — call validate() first");
  if (op.kind == OpKind::Recv && !peer_wait_ready_[idx])
    throw std::runtime_error(
        "transport wait path not established — call validate() first");

  unsigned request_id = 0;
  if (op.kind == OpKind::Send) {
    request_id = communicator_->isend(peer_rank, /*src_buf=*/1, op.src_off,
                                      op.bytes, /*dst_buf=*/3, op.dst_off);
  } else {
    request_id = communicator_->irecv(peer_rank, /*dst_buf=*/3, op.dst_off,
                                      op.bytes);
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

size_t CommunicatorTransportBackend::drain(BackendToken* out,
                                           size_t max_count) {
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
  if (op.kind == OpKind::Send) {
    if (op.dst_peer == ~0u)
      throw std::invalid_argument("transport send requires a peer destination");
    return op.dst_peer;
  }
  if (op.kind == OpKind::Recv) {
    if (op.src_peer == ~0u)
      throw std::invalid_argument("transport recv requires a peer source");
    return op.src_peer;
  }
  throw std::invalid_argument("non-transport op");
}

}  // namespace CCL
}  // namespace UKernel
