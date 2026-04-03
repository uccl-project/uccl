#include "oob.h"
#include "util/util.h"
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

static inline void shm_ring_send(jring_t* ring, ShmCtrlMsg const& msg) {
  alignas(16) ShmCtrlMsg tmp = msg;
  while (jring_mp_enqueue_bulk(ring, &tmp, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
}

static inline bool shm_ring_try_recv(jring_t* ring, ShmCtrlMsg& msg) {
  if (jring_mc_dequeue_bulk(ring, &msg, 1, nullptr) != 1) {
    return false;
  }
  return true;
}

static inline jring_t* attach_shared_ring_quiet(char const* shm_name,
                                                int& shm_fd, size_t shm_size) {
  shm_fd = shm_open(shm_name, O_RDWR, 0666);
  if (shm_fd < 0) return nullptr;

  void* ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    close(shm_fd);
    shm_fd = -1;
    return nullptr;
  }
  return reinterpret_cast<jring_t*>(ptr);
}

static inline bool timeout_expired(
    std::chrono::steady_clock::time_point deadline, int timeout_ms) {
  if (timeout_ms < 0) return false;
  return std::chrono::steady_clock::now() >= deadline;
}

}  // namespace

ShmRingExchanger::ShmRingExchanger(int self_rank, int world_size,
                                   std::string ring_namespace,
                                   int self_local_id)
    : self_rank_(self_rank),
      world_size_(world_size),
      self_local_id_(self_local_id >= 0 ? self_local_id : self_rank),
      ring_namespace_(std::move(ring_namespace)),
      peers_(static_cast<size_t>(world_size)),
      peer_local_ids_(static_cast<size_t>(world_size), -1),
      next_connect_seq_(static_cast<size_t>(world_size), 1) {}

void ShmRingExchanger::set_peer_local_id(int peer_rank, int local_id) {
  if (peer_rank < 0 || peer_rank >= world_size_ || peer_rank == self_rank_) {
    return;
  }
  std::lock_guard<std::mutex> lk(mu_);
  peer_local_ids_[static_cast<size_t>(peer_rank)] = local_id;
}

ShmRingExchanger::~ShmRingExchanger() {
  running_.store(false, std::memory_order_relaxed);

  std::lock_guard<std::mutex> lk(mu_);
  for (int peer_rank = 0; peer_rank < world_size_; ++peer_rank) {
    if (peer_rank == self_rank_) continue;
    auto const& peer = peers_[static_cast<size_t>(peer_rank)];
    if (!peer) continue;

    if (peer->remote_inbox.ring != nullptr && peer->remote_inbox.attached) {
      uccl::detach_shared_ring(peer->remote_inbox.ring,
                               peer->remote_inbox.shm_fd,
                               peer->remote_inbox.shm_size);
    }
    if (peer->local_inbox.ring != nullptr) {
      if (peer->local_inbox.creator) {
        uccl::destroy_shared_ring(
            peer->local_inbox.shm_name.c_str(), peer->local_inbox.ring,
            peer->local_inbox.shm_fd, peer->local_inbox.shm_size);
      } else {
        uccl::detach_shared_ring(peer->local_inbox.ring,
                                 peer->local_inbox.shm_fd,
                                 peer->local_inbox.shm_size);
      }
    }
  }
}

bool ShmRingExchanger::ensure_server_started() {
  if (running_.load(std::memory_order_relaxed)) return true;

  std::lock_guard<std::mutex> lk(mu_);
  if (running_.load(std::memory_order_relaxed)) return true;
  for (int peer_rank = 0; peer_rank < world_size_; ++peer_rank) {
    if (peer_rank == self_rank_) continue;
    if (!ensure_local_ring(peer_rank)) return false;
  }
  running_.store(true, std::memory_order_relaxed);
  return true;
}

bool ShmRingExchanger::connect_to(int peer_rank, int timeout_ms) {
  if (peer_rank == self_rank_) return true;
  if (!ensure_server_started()) return false;
  if (!ensure_remote_ring_attached(peer_rank, timeout_ms)) return false;

  uint64_t connect_seq = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    connect_seq = next_connect_seq_[static_cast<size_t>(peer_rank)]++;
  }

  ShmCtrlMsg msg;
  msg.from_rank = static_cast<uint32_t>(self_rank_);
  msg.to_rank = static_cast<uint32_t>(peer_rank);
  msg.type = ShmCtrlMsgType::Connect;
  msg.seq = connect_seq;
  if (!send_msg(peer_rank, msg)) return false;

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
  while (true) {
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      if (try_take_cached_connect_ack_locked(peer_rank, connect_seq, nullptr)) {
        break;
      }
    }

    ShmCtrlMsg in_msg;
    bool got = false;
    {
      std::shared_ptr<PeerState> peer;
      {
        std::lock_guard<std::mutex> lk(mu_);
        peer = peers_[static_cast<size_t>(peer_rank)];
      }
      if (!peer || peer->local_inbox.ring == nullptr) return false;
      std::lock_guard<std::mutex> recv_lk(peer->recv_mu);
      got = try_recv_one_locked(peer_rank, in_msg);
    }
    if (got) continue;

    if (timeout_expired(deadline, timeout_ms)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto& peer = peers_[static_cast<size_t>(peer_rank)];
  if (peer) peer->connected = true;
  return true;
}

bool ShmRingExchanger::accept_from(int peer_rank, int timeout_ms) {
  if (peer_rank == self_rank_) return true;
  if (!ensure_server_started()) return false;

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
  uint64_t connect_seq = 0;
  while (true) {
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      if (try_take_connect_locked(peer_rank, &connect_seq)) {
        break;
      }
    }

    ShmCtrlMsg msg;
    bool got = false;
    {
      std::shared_ptr<PeerState> peer;
      {
        std::lock_guard<std::mutex> lk(mu_);
        peer = peers_[static_cast<size_t>(peer_rank)];
      }
      if (!peer || peer->local_inbox.ring == nullptr) return false;
      std::lock_guard<std::mutex> recv_lk(peer->recv_mu);
      got = try_recv_one_locked(peer_rank, msg);
    }
    if (got) continue;

    if (timeout_expired(deadline, timeout_ms)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (!ensure_remote_ring_attached(peer_rank, timeout_ms)) return false;
  ShmCtrlMsg ack{};
  ack.from_rank = static_cast<uint32_t>(self_rank_);
  ack.to_rank = static_cast<uint32_t>(peer_rank);
  ack.type = ShmCtrlMsgType::ConnectAck;
  ack.seq = connect_seq;
  if (!send_msg(peer_rank, ack)) return false;

  std::lock_guard<std::mutex> lk(mu_);
  auto& peer = peers_[static_cast<size_t>(peer_rank)];
  if (peer) peer->connected = true;
  return true;
}

bool ShmRingExchanger::send(int peer_rank, uint16_t type, uint64_t seq,
                            void const* payload, uint32_t bytes) {
  if (type == kTypeIpcCache && bytes != sizeof(IpcCacheWire)) return false;
  if (type == kTypeAck && bytes != sizeof(AckWire)) return false;

  ShmCtrlMsg msg;
  msg.from_rank = static_cast<uint32_t>(self_rank_);
  msg.to_rank = static_cast<uint32_t>(peer_rank);
  msg.seq = seq;
  if (type == kTypeIpcCache) {
    msg.type = ShmCtrlMsgType::IpcCache;
    std::memcpy(&msg.cache, payload, sizeof(IpcCacheWire));
  } else if (type == kTypeAck) {
    msg.type = ShmCtrlMsgType::Ack;
    std::memcpy(&msg.ack, payload, sizeof(AckWire));
  } else {
    return false;
  }
  return send_msg(peer_rank, msg);
}

bool ShmRingExchanger::send_ipc_cache(int peer_rank, uint64_t seq,
                                      IpcCacheWire const& cache) {
  return send(peer_rank, kTypeIpcCache, seq, &cache,
              static_cast<uint32_t>(sizeof(IpcCacheWire)));
}

bool ShmRingExchanger::recv_ipc_cache(int peer_rank, IpcCacheWire& out_cache,
                                      uint64_t* out_seq, int timeout_ms,
                                      uint64_t expected_seq) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
  while (true) {
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      if (try_take_cached_ipc_cache_locked(peer_rank, expected_seq, out_cache,
                                           out_seq)) {
        return true;
      }
    }

    ShmCtrlMsg msg;
    bool got = false;
    {
      std::shared_ptr<PeerState> peer;
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (peer_rank < 0 || peer_rank >= world_size_) return false;
        peer = peers_[static_cast<size_t>(peer_rank)];
      }
      if (!peer || peer->local_inbox.ring == nullptr) return false;
      std::lock_guard<std::mutex> recv_lk(peer->recv_mu);
      got = try_recv_one_locked(peer_rank, msg);
    }
    if (got) continue;

    if (timeout_expired(deadline, timeout_ms)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

bool ShmRingExchanger::send_ack(int peer_rank, uint64_t seq, uint32_t status) {
  AckWire ack{};
  ack.status = status;
  ack.reserved = 0;
  return send(peer_rank, kTypeAck, seq, &ack,
              static_cast<uint32_t>(sizeof(ack)));
}

bool ShmRingExchanger::recv_ack(int peer_rank, uint32_t* out_status,
                                uint64_t* out_seq, int timeout_ms,
                                uint64_t expected_seq) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
  while (true) {
    AckWire ack{};
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      if (try_take_cached_ack_locked(peer_rank, expected_seq, ack, out_seq)) {
        if (out_status) *out_status = ack.status;
        return true;
      }
    }

    ShmCtrlMsg msg;
    bool got = false;
    {
      std::shared_ptr<PeerState> peer;
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (peer_rank < 0 || peer_rank >= world_size_) return false;
        peer = peers_[static_cast<size_t>(peer_rank)];
      }
      if (!peer || peer->local_inbox.ring == nullptr) return false;
      std::lock_guard<std::mutex> recv_lk(peer->recv_mu);
      got = try_recv_one_locked(peer_rank, msg);
    }
    if (got) continue;

    if (timeout_expired(deadline, timeout_ms)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void ShmRingExchanger::close_peer(int peer_rank) {
  std::shared_ptr<PeerState> peer;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (peer_rank < 0 || peer_rank >= world_size_) return;
    peer = peers_[static_cast<size_t>(peer_rank)];
    peers_[static_cast<size_t>(peer_rank)].reset();
  }

  if (peer) {
    if (peer->remote_inbox.ring != nullptr && peer->remote_inbox.attached) {
      uccl::detach_shared_ring(peer->remote_inbox.ring,
                               peer->remote_inbox.shm_fd,
                               peer->remote_inbox.shm_size);
    }
    if (peer->local_inbox.ring != nullptr) {
      if (peer->local_inbox.creator) {
        uccl::destroy_shared_ring(
            peer->local_inbox.shm_name.c_str(), peer->local_inbox.ring,
            peer->local_inbox.shm_fd, peer->local_inbox.shm_size);
      } else {
        uccl::detach_shared_ring(peer->local_inbox.ring,
                                 peer->local_inbox.shm_fd,
                                 peer->local_inbox.shm_size);
      }
    }
  }

  std::lock_guard<std::mutex> pending_lk(pending_mu_);
  pending_connect_.erase(peer_rank);
  pending_connect_acks_.erase(peer_rank);
  rank_to_pending_ipc_cache_.erase(peer_rank);
  rank_to_pending_acks_.erase(peer_rank);
}

std::string ShmRingExchanger::ring_name(int from_rank, int to_rank) const {
  auto resolve_local_id = [&](int rank) {
    if (rank == self_rank_) return self_local_id_;
    int local_id = peer_local_ids_[static_cast<size_t>(rank)];
    return local_id >= 0 ? local_id : rank;
  };
  return uccl::Format("/uk_t_oob_%s_l%d_l%d", ring_namespace_.c_str(),
                      resolve_local_id(from_rank), resolve_local_id(to_rank));
}

void ShmRingExchanger::cleanup_stale_ring(int peer_rank) {
  if (peer_rank == self_rank_) return;
  std::string local_ring = ring_name(peer_rank, self_rank_);
  shm_unlink(local_ring.c_str());
}

bool ShmRingExchanger::ensure_local_ring(int peer_rank) {
  auto& peer = peers_[static_cast<size_t>(peer_rank)];
  if (!peer) peer = std::make_shared<PeerState>();
  if (peer->local_inbox.ring != nullptr) return true;

  cleanup_stale_ring(peer_rank);

  peer->local_inbox.shm_name = ring_name(peer_rank, self_rank_);
  shm_unlink(peer->local_inbox.shm_name.c_str());
  peer->local_inbox.ring = uccl::create_shared_ring(
      peer->local_inbox.shm_name.c_str(), sizeof(ShmCtrlMsg), 1024,
      peer->local_inbox.shm_fd, peer->local_inbox.shm_size,
      &peer->local_inbox.creator);
  if (peer->local_inbox.ring == nullptr) return false;
  peer->local_inbox.attached = true;
  return true;
}

bool ShmRingExchanger::ensure_remote_ring_attached(int peer_rank,
                                                   int timeout_ms) {
  std::shared_ptr<PeerState> peer;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (peer_rank < 0 || peer_rank >= world_size_) return false;
    peer = peers_[static_cast<size_t>(peer_rank)];
    if (!peer) {
      peer = std::make_shared<PeerState>();
      peers_[static_cast<size_t>(peer_rank)] = peer;
    }
    if (peer->remote_inbox.ring != nullptr) return true;
    peer->remote_inbox.shm_name = ring_name(self_rank_, peer_rank);
    peer->remote_inbox.shm_size =
        jring_get_buf_ring_size(sizeof(ShmCtrlMsg), 1024);
  }

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
  while (true) {
    jring_t* ring = attach_shared_ring_quiet(
        peer->remote_inbox.shm_name.c_str(), peer->remote_inbox.shm_fd,
        peer->remote_inbox.shm_size);
    if (ring != nullptr) {
      std::lock_guard<std::mutex> lk(mu_);
      peer->remote_inbox.ring = ring;
      peer->remote_inbox.attached = true;
      return true;
    }
    if (timeout_expired(deadline, timeout_ms)) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

bool ShmRingExchanger::try_recv_one_locked(int peer_rank, ShmCtrlMsg& msg) {
  auto& peer = peers_[static_cast<size_t>(peer_rank)];
  if (!peer || peer->local_inbox.ring == nullptr) return false;
  if (!shm_ring_try_recv(peer->local_inbox.ring, msg)) return false;

  if (msg.to_rank != static_cast<uint32_t>(self_rank_) ||
      msg.from_rank != static_cast<uint32_t>(peer_rank)) {
    return true;
  }

  if (msg.type == ShmCtrlMsgType::Connect) {
    cache_connect_message(peer_rank, msg.seq);
    return true;
  }
  if (msg.type == ShmCtrlMsgType::ConnectAck) {
    cache_connect_ack_message(peer_rank, msg.seq);
    return true;
  }
  if (msg.type == ShmCtrlMsgType::IpcCache) {
    cache_ipc_cache_message(peer_rank, msg.seq, msg.cache);
    return true;
  }
  if (msg.type == ShmCtrlMsgType::Ack) {
    cache_ack_message(peer_rank, msg.seq, msg.ack);
    return true;
  }
  return true;
}

bool ShmRingExchanger::try_take_connect_locked(int peer_rank, uint64_t* out_seq) {
  auto it = pending_connect_.find(peer_rank);
  if (it == pending_connect_.end()) return false;
  auto& q = it->second;
  if (q.empty()) return false;
  uint64_t seq = q.front();
  q.pop_front();
  if (q.empty()) pending_connect_.erase(it);
  if (out_seq) *out_seq = seq;
  return true;
}

bool ShmRingExchanger::try_take_cached_connect_ack_locked(
    int peer_rank, uint64_t expected_seq, uint64_t* out_seq) {
  auto it = pending_connect_acks_.find(peer_rank);
  if (it == pending_connect_acks_.end()) return false;
  auto& q = it->second;
  for (auto msg_it = q.begin(); msg_it != q.end(); ++msg_it) {
    if (expected_seq != UINT64_MAX && *msg_it != expected_seq) continue;
    uint64_t seq = *msg_it;
    q.erase(msg_it);
    if (q.empty()) pending_connect_acks_.erase(it);
    if (out_seq) *out_seq = seq;
    return true;
  }
  return false;
}

void ShmRingExchanger::cache_connect_message(int peer_rank, uint64_t seq) {
  std::lock_guard<std::mutex> lk(pending_mu_);
  pending_connect_[peer_rank].push_back(seq);
}

void ShmRingExchanger::cache_connect_ack_message(int peer_rank, uint64_t seq) {
  std::lock_guard<std::mutex> lk(pending_mu_);
  pending_connect_acks_[peer_rank].push_back(seq);
}

bool ShmRingExchanger::try_take_cached_ipc_cache_locked(int peer_rank,
                                                        uint64_t expected_seq,
                                                        IpcCacheWire& out_cache,
                                                        uint64_t* out_seq) {
  auto it = rank_to_pending_ipc_cache_.find(peer_rank);
  if (it == rank_to_pending_ipc_cache_.end()) return false;
  auto& q = it->second;
  for (auto msg_it = q.begin(); msg_it != q.end(); ++msg_it) {
    if (expected_seq != UINT64_MAX && msg_it->seq != expected_seq) continue;
    out_cache = msg_it->cache;
    if (out_seq) *out_seq = msg_it->seq;
    q.erase(msg_it);
    return true;
  }
  return false;
}

bool ShmRingExchanger::try_take_cached_ack_locked(int peer_rank,
                                                  uint64_t expected_seq,
                                                  AckWire& out_ack,
                                                  uint64_t* out_seq) {
  auto it = rank_to_pending_acks_.find(peer_rank);
  if (it == rank_to_pending_acks_.end()) return false;
  auto& q = it->second;
  for (auto msg_it = q.begin(); msg_it != q.end(); ++msg_it) {
    if (expected_seq != UINT64_MAX && msg_it->seq != expected_seq) continue;
    out_ack = msg_it->ack;
    if (out_seq) *out_seq = msg_it->seq;
    q.erase(msg_it);
    return true;
  }
  return false;
}

void ShmRingExchanger::cache_ipc_cache_message(int peer_rank, uint64_t seq,
                                               IpcCacheWire const& cache) {
  std::lock_guard<std::mutex> lk(pending_mu_);
  rank_to_pending_ipc_cache_[peer_rank].push_back(
      PendingIpcCacheMsg{seq, cache});
}

void ShmRingExchanger::cache_ack_message(int peer_rank, uint64_t seq,
                                         AckWire const& ack) {
  std::lock_guard<std::mutex> lk(pending_mu_);
  rank_to_pending_acks_[peer_rank].push_back(PendingAckMsg{seq, ack});
}

bool ShmRingExchanger::send_msg(int peer_rank, ShmCtrlMsg const& msg) {
  std::shared_ptr<PeerState> peer;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (peer_rank < 0 || peer_rank >= world_size_) return false;
    peer = peers_[static_cast<size_t>(peer_rank)];
  }
  if (!peer) return false;
  if (peer->remote_inbox.ring == nullptr) return false;

  std::lock_guard<std::mutex> send_lk(peer->send_mu);
  shm_ring_send(peer->remote_inbox.ring, msg);
  return true;
}

}  // namespace Transport
}  // namespace UKernel
