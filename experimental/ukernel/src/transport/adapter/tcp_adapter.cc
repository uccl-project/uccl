#include "tcp_adapter.h"
#include "../util/utils.h"
#include "gpu_rt.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr auto kConnectRetrySleep = std::chrono::milliseconds(50);
constexpr auto kDefaultConnectTimeout = std::chrono::seconds(30);
constexpr auto kAcceptPollSleep = std::chrono::milliseconds(10);
constexpr size_t kTaskRingSize = 1024;
constexpr size_t kSignalWireBytes = sizeof(uint64_t);

enum class FrameType : uint32_t {
  Data = 1,
  Signal = 2,
};

struct WireHeader {
  uint32_t type = 0;
  uint64_t payload_len = 0;
};

bool is_retryable_errno(int err) {
  return err == EINTR || err == EAGAIN || err == EWOULDBLOCK;
}

bool enqueue_one_request_id(jring_t* ring, unsigned request_id,
                            std::atomic<bool> const& stop) {
  unsigned elem = request_id;
  while (!stop.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(ring, &elem, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  return !stop.load(std::memory_order_acquire);
}

bool recv_discard_all(int fd, uint64_t len) {
  constexpr size_t kDiscardChunk = 4096;
  char discard[kDiscardChunk];
  uint64_t remaining = len;
  while (remaining > 0) {
    size_t chunk = static_cast<size_t>(
        std::min<uint64_t>(remaining, static_cast<uint64_t>(kDiscardChunk)));
    size_t received = 0;
    while (received < chunk) {
      ssize_t rc = ::recv(fd, discard + received, chunk - received, 0);
      if (rc > 0) {
        received += static_cast<size_t>(rc);
        continue;
      }
      if (rc == 0) return false;
      if (is_retryable_errno(errno)) continue;
      return false;
    }
    remaining -= chunk;
  }
  return true;
}

}  // namespace

TcpTransportAdapter::TcpTransportAdapter(std::string local_ip, int local_rank)
    : local_ip_(std::move(local_ip)), local_rank_(local_rank) {
  listen_fd_ = create_listen_socket(listen_port_);
  if (listen_fd_ < 0) {
    throw std::runtime_error("failed to create tcp transport listen socket");
  }
  send_task_ring_ =
      UKernel::Transport::create_ring(sizeof(unsigned), kTaskRingSize);
  recv_task_ring_ =
      UKernel::Transport::create_ring(sizeof(unsigned), kTaskRingSize);
  request_slots_ = std::make_unique<RequestSlot[]>(kRequestSlotCount);
  if (send_task_ring_ == nullptr || recv_task_ring_ == nullptr ||
      !request_slots_) {
    if (send_task_ring_ != nullptr) {
      free(send_task_ring_);
      send_task_ring_ = nullptr;
    }
    if (recv_task_ring_ != nullptr) {
      free(recv_task_ring_);
      recv_task_ring_ = nullptr;
    }
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    throw std::runtime_error("failed to initialize tcp request infrastructure");
  }
  stop_.store(false, std::memory_order_release);
  send_worker_ = std::thread([this] { send_worker_loop(); });
  recv_worker_ = std::thread([this] { recv_worker_loop(); });
}

TcpTransportAdapter::~TcpTransportAdapter() {
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();

  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [peer_rank, ctx] : peer_contexts_) {
      (void)peer_rank;
      if (!ctx) continue;
      int send_fd = ctx->send_fd;
      int recv_fd = ctx->recv_fd;
      if (send_fd >= 0) {
        ::shutdown(send_fd, SHUT_RDWR);
      }
      if (recv_fd >= 0 && recv_fd != send_fd) {
        ::shutdown(recv_fd, SHUT_RDWR);
      }
    }
  }

  if (send_worker_.joinable()) send_worker_.join();
  if (recv_worker_.joinable()) recv_worker_.join();

  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [peer_rank, ctx] : peer_contexts_) {
      (void)peer_rank;
      if (!ctx) continue;
      int send_fd = ctx->send_fd;
      int recv_fd = ctx->recv_fd;
      if (send_fd >= 0) {
        ::close(send_fd);
      }
      if (recv_fd >= 0 && recv_fd != send_fd) {
        ::close(recv_fd);
      }
      ctx->send_fd = -1;
      ctx->recv_fd = -1;
    }
  }

  if (send_task_ring_ != nullptr) {
    free(send_task_ring_);
    send_task_ring_ = nullptr;
  }
  if (recv_task_ring_ != nullptr) {
    free(recv_task_ring_);
    recv_task_ring_ = nullptr;
  }
}

uint16_t TcpTransportAdapter::get_listen_port() const { return listen_port_; }

bool TcpTransportAdapter::connect_to_peer(int peer_rank, std::string remote_ip,
                                          uint16_t remote_port) {
  if (has_put_path(peer_rank)) return true;

  auto deadline = std::chrono::steady_clock::now() + kDefaultConnectTimeout;
  int fd = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    if (!connect_socket(fd, remote_ip, remote_port, std::chrono::seconds(1))) {
      continue;
    }

    Handshake hs;
    hs.src_rank = static_cast<uint32_t>(local_rank_);
    HandshakeAck ack{};
    if (send_handshake(fd, hs) && recv_handshake_ack(fd, ack) &&
        ack.accepted == 1) {
      break;
    }

    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    fd = -1;
    std::this_thread::sleep_for(kConnectRetrySleep);
  }
  if (fd < 0) {
    return false;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto& ctx = peer_contexts_[peer_rank];
  if (!ctx) ctx = std::make_shared<PeerContext>();
  if (ctx->send_fd >= 0) {
    ::close(fd);
    return true;
  }
  ctx->send_fd = fd;
  return true;
}

bool TcpTransportAdapter::accept_from_peer(
    int peer_rank, std::string const& expected_remote_ip) {
  if (has_wait_path(peer_rank)) return true;

  auto deadline = std::chrono::steady_clock::now() + kDefaultConnectTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    sockaddr_in addr{};
    socklen_t addr_len = sizeof(addr);
    int fd =
        ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &addr_len);
    if (fd < 0) {
      if (is_retryable_errno(errno)) {
        std::this_thread::sleep_for(kAcceptPollSleep);
        continue;
      }
      return false;
    }

    char remote_ip_buf[INET_ADDRSTRLEN] = {};
    std::string remote_ip;
    if (::inet_ntop(AF_INET, &addr.sin_addr, remote_ip_buf,
                    sizeof(remote_ip_buf)) != nullptr) {
      remote_ip = remote_ip_buf;
    }

    if (!expected_remote_ip.empty() && remote_ip != expected_remote_ip) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      continue;
    }

    Handshake hs{};
    if (!recv_handshake(fd, hs) || static_cast<int>(hs.src_rank) != peer_rank) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      continue;
    }
    HandshakeAck ack{};
    ack.accepted = 1;
    if (!send_handshake_ack(fd, ack)) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      continue;
    }

    std::lock_guard<std::mutex> lk(mu_);
    auto& ctx = peer_contexts_[peer_rank];
    if (!ctx) ctx = std::make_shared<PeerContext>();
    if (ctx->recv_fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      return true;
    }
    ctx->recv_fd = fd;
    return true;
  }
  return false;
}

bool TcpTransportAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_put_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Connect) return false;
  if (!std::holds_alternative<TcpPeerConnectSpec>(spec.detail)) return false;
  auto const& tcp = std::get<TcpPeerConnectSpec>(spec.detail);
  if (!connect_to_peer(spec.peer_rank, tcp.remote_ip, tcp.remote_port)) {
    return false;
  }
  return has_put_path(spec.peer_rank);
}

bool TcpTransportAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_wait_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Accept) return false;
  if (!std::holds_alternative<TcpPeerConnectSpec>(spec.detail)) return false;
  auto const& tcp = std::get<TcpPeerConnectSpec>(spec.detail);
  if (!accept_from_peer(spec.peer_rank, tcp.remote_ip)) return false;
  return has_wait_path(spec.peer_rank);
}

bool TcpTransportAdapter::has_put_path(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second && it->second->send_fd >= 0;
}

bool TcpTransportAdapter::has_wait_path(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second && it->second->recv_fd >= 0;
}

unsigned TcpTransportAdapter::put_async(
    int peer_rank, void* local_ptr, uint32_t local_buffer_id,
    void* remote_ptr, uint32_t remote_buffer_id, size_t len) {
  (void)local_buffer_id;
  (void)remote_ptr;
  (void)remote_buffer_id;
  if (!has_put_path(peer_rank)) return 0;

  unsigned request_id = 0;
  RequestSlot* slot = try_acquire_request_slot(&request_id);
  if (!slot) return 0;
  slot->peer_rank = peer_rank;
  slot->kind = RequestSlot::Kind::DataPut;
  slot->host_ptr = local_ptr;
  slot->len = len;
  slot->mark_queued();
  if (!enqueue_request(request_id, /*is_send=*/true)) {
    slot->mark_completed(false);
    release_request_slot(request_id);
    return 0;
  }
  return request_id;
}

unsigned TcpTransportAdapter::signal_async(int peer_rank, uint64_t tag) {
  if (!has_put_path(peer_rank)) return 0;
  unsigned request_id = 0;
  RequestSlot* slot = try_acquire_request_slot(&request_id);
  if (!slot) return 0;
  slot->peer_rank = peer_rank;
  slot->kind = RequestSlot::Kind::Signal;
  slot->signal_payload = tag;
  // host_ptr must point to signal_payload so the send worker reads the tag value
  // from the slot itself — this is safe because the slot is exclusively owned.
  slot->host_ptr = &slot->signal_payload;
  slot->len = sizeof(uint64_t);
  slot->mark_queued();
  if (!enqueue_request(request_id, /*is_send=*/true)) {
    slot->mark_completed(false);
    release_request_slot(request_id);
    return 0;
  }
  return request_id;
}

unsigned TcpTransportAdapter::wait_async(int peer_rank, uint64_t expected_tag,
                                         std::optional<WaitTarget> target) {
  if (!has_wait_path(peer_rank)) return 0;
  unsigned request_id = 0;
  RequestSlot* slot = try_acquire_request_slot(&request_id);
  if (!slot) return 0;
  slot->peer_rank = peer_rank;
  if (!target.has_value()) {
    slot->kind = RequestSlot::Kind::SignalWait;
    slot->expected_tag = expected_tag;
    slot->signal_payload = 0;
    slot->host_ptr = &slot->signal_payload;
    slot->len = kSignalWireBytes;
  } else {
    slot->kind = RequestSlot::Kind::DataWait;
    slot->host_ptr = target->local_ptr;
    slot->len = target->len;
  }
  slot->mark_queued();
  if (!enqueue_request(request_id, /*is_send=*/false)) {
    slot->mark_completed(false);
    release_request_slot(request_id);
    return 0;
  }
  return request_id;
}

bool TcpTransportAdapter::request_failed(unsigned id) {
  RequestSlot* slot = resolve_request_slot_const(id);
  if (!slot) return false;
  return slot->is_failed();
}

bool TcpTransportAdapter::poll_completion(unsigned request_id) {
  RequestSlot* slot = resolve_request_slot_const(request_id);
  if (!slot) return true;
  return slot->is_completed();
}

bool TcpTransportAdapter::wait_completion(unsigned request_id) {
  for (;;) {
    RequestSlot* slot = resolve_request_slot_const(request_id);
    if (!slot) return false;
    if (slot->is_completed()) return true;
    std::unique_lock<std::mutex> lk(cv_mu_);
    cv_.wait(lk, [&] {
      RequestSlot* cur = resolve_request_slot_const(request_id);
      return stop_.load(std::memory_order_acquire) || cur == nullptr ||
             cur->is_completed();
    });
  }
}

void TcpTransportAdapter::release_request(unsigned request_id) {
  release_request_slot(request_id);
}

TcpTransportAdapter::RequestSlot* TcpTransportAdapter::try_acquire_request_slot(
    unsigned* out_request_id) {
  if (out_request_id == nullptr || !request_slots_) return nullptr;
  for (uint32_t n = 0; n < kRequestSlotCount; ++n) {
    uint32_t idx =
        request_alloc_cursor_.fetch_add(1, std::memory_order_relaxed) &
        kRequestSlotMask;
    auto& slot = request_slots_[idx];
    RequestState expected = RequestState::Free;
    if (!slot.state.compare_exchange_strong(expected, RequestState::Running,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
      continue;
    }
    uint32_t gen = slot.generation.load(std::memory_order_acquire);
    if (gen == 0) {
      gen = 1;
      slot.generation.store(gen, std::memory_order_release);
    }
    slot.peer_rank = -1;
    slot.kind = RequestSlot::Kind::DataPut;
    slot.host_ptr = nullptr;
    slot.len = 0;
    slot.expected_tag = 0;
    slot.signal_payload = 0;
    slot.completed.store(false, std::memory_order_release);
    slot.failed.store(false, std::memory_order_release);
    *out_request_id = make_request_id(idx, gen);
    return &slot;
  }
  return nullptr;
}

TcpTransportAdapter::RequestSlot* TcpTransportAdapter::resolve_request_slot(
    unsigned request_id) {
  if (request_id == 0 || !request_slots_) return nullptr;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return nullptr;
  uint32_t idx = request_slot_index(request_id);
  auto& slot = request_slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) {
    return nullptr;
  }
  if (slot.state.load(std::memory_order_acquire) == RequestState::Free) {
    return nullptr;
  }
  return &slot;
}

TcpTransportAdapter::RequestSlot*
TcpTransportAdapter::resolve_request_slot_const(unsigned request_id) const {
  if (request_id == 0 || !request_slots_) return nullptr;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return nullptr;
  uint32_t idx = request_slot_index(request_id);
  auto const& slot = request_slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) {
    return nullptr;
  }
  if (slot.state.load(std::memory_order_acquire) == RequestState::Free) {
    return nullptr;
  }
  return const_cast<RequestSlot*>(&slot);
}

void TcpTransportAdapter::release_request_slot(unsigned request_id) {
  if (request_id == 0 || !request_slots_) return;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return;
  uint32_t idx = request_slot_index(request_id);
  auto& slot = request_slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) return;
  auto st = slot.state.load(std::memory_order_acquire);
  if (st == RequestState::Queued || st == RequestState::Running) return;
  // Hold slot in a non-Free state while rolling generation to avoid ABA reuse.
  slot.state.store(RequestState::Running, std::memory_order_release);
  slot.completed.store(false, std::memory_order_release);
  slot.failed.store(false, std::memory_order_release);
  slot.peer_rank = -1;
  slot.kind = RequestSlot::Kind::DataPut;
  slot.host_ptr = nullptr;
  slot.len = 0;
  slot.expected_tag = 0;
  slot.signal_payload = 0;
  uint32_t old_gen = slot.generation.load(std::memory_order_acquire);
  while (true) {
    uint32_t next_gen = old_gen + 1;
    if (next_gen == 0) next_gen = 1;
    if (slot.generation.compare_exchange_weak(old_gen, next_gen,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      break;
    }
  }
  slot.state.store(RequestState::Free, std::memory_order_release);
  cv_.notify_all();
}

bool TcpTransportAdapter::enqueue_request(unsigned request_id, bool is_send) {
  if (stop_.load(std::memory_order_acquire)) return false;
  if (is_send) {
    if (send_task_ring_ == nullptr ||
        !enqueue_one_request_id(send_task_ring_, request_id, stop_)) {
      return false;
    }
    pending_send_.fetch_add(1, std::memory_order_relaxed);
  } else {
    if (recv_task_ring_ == nullptr ||
        !enqueue_one_request_id(recv_task_ring_, request_id, stop_)) {
      return false;
    }
    pending_recv_.fetch_add(1, std::memory_order_relaxed);
  }
  cv_.notify_all();
  return true;
}

void TcpTransportAdapter::send_worker_loop() {
  while (!stop_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_acquire) ||
               pending_send_.load(std::memory_order_relaxed) > 0;
      });
    }
    if (stop_.load(std::memory_order_acquire)) break;

    unsigned request_id = 0;
    if (jring_sc_dequeue_bulk(send_task_ring_, &request_id, 1, nullptr) != 1) {
      continue;
    }
    pending_send_.fetch_sub(1, std::memory_order_relaxed);
    RequestSlot* slot = resolve_request_slot(request_id);
    if (!slot) continue;
    slot->mark_running();
    bool ok = false;
    std::shared_ptr<PeerContext> ctx;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = peer_contexts_.find(slot->peer_rank);
      if (it != peer_contexts_.end() && it->second &&
          it->second->send_fd >= 0) {
        ctx = it->second;
      }
    }
    if (ctx) {
      std::lock_guard<std::mutex> send_lk(ctx->send_mu);
      WireHeader header{};
      header.type =
          static_cast<uint32_t>((slot->kind == RequestSlot::Kind::Signal)
                                    ? FrameType::Signal
                                    : FrameType::Data);
      header.payload_len = static_cast<uint64_t>(slot->len);
      ok = TcpTransportAdapter::send_all(ctx->send_fd, &header, sizeof(header));
      if (ok && slot->len > 0) {
        ok = TcpTransportAdapter::send_all(ctx->send_fd, slot->host_ptr,
                                           slot->len);
      }
    }
    slot->mark_completed(ok);
    cv_.notify_all();
  }

  while (true) {
    unsigned request_id = 0;
    if (jring_mc_dequeue_bulk(send_task_ring_, &request_id, 1, nullptr) != 1) {
      break;
    }
    RequestSlot* slot = resolve_request_slot(request_id);
    if (!slot) continue;
    slot->mark_completed(false);
    cv_.notify_all();
  }
}

void TcpTransportAdapter::recv_worker_loop() {
  while (!stop_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_acquire) ||
               pending_recv_.load(std::memory_order_relaxed) > 0;
      });
    }
    if (stop_.load(std::memory_order_acquire)) break;

    unsigned request_id = 0;
    if (jring_sc_dequeue_bulk(recv_task_ring_, &request_id, 1, nullptr) != 1) {
      continue;
    }
    pending_recv_.fetch_sub(1, std::memory_order_relaxed);
    RequestSlot* slot = resolve_request_slot(request_id);
    if (!slot) continue;
    slot->mark_running();
    bool ok = false;
    std::shared_ptr<PeerContext> ctx;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = peer_contexts_.find(slot->peer_rank);
      if (it != peer_contexts_.end() && it->second &&
          it->second->recv_fd >= 0) {
        ctx = it->second;
      }
    }
    if (ctx) {
      std::lock_guard<std::mutex> recv_lk(ctx->recv_mu);
      WireHeader header{};
      ok = TcpTransportAdapter::recv_all(ctx->recv_fd, &header, sizeof(header));
      if (ok) {
        FrameType expected_type =
            (slot->kind == RequestSlot::Kind::SignalWait) ? FrameType::Signal
                                                           : FrameType::Data;
        FrameType got_type = static_cast<FrameType>(header.type);
        bool frame_ok = (got_type == expected_type) &&
                        (header.payload_len == static_cast<uint64_t>(slot->len));
        if (!frame_ok) {
          if (header.payload_len > 0) {
            recv_discard_all(ctx->recv_fd, header.payload_len);
          }
          ok = false;
        } else if (slot->len > 0) {
          ok = TcpTransportAdapter::recv_all(ctx->recv_fd, slot->host_ptr,
                                             slot->len);
        }
        if (ok && slot->kind == RequestSlot::Kind::SignalWait) {
          ok = (slot->signal_payload == slot->expected_tag);
        }
      }
    }
    slot->mark_completed(ok);
    cv_.notify_all();
  }

  while (true) {
    unsigned request_id = 0;
    if (jring_mc_dequeue_bulk(recv_task_ring_, &request_id, 1, nullptr) != 1) {
      break;
    }
    RequestSlot* slot = resolve_request_slot(request_id);
    if (!slot) continue;
    slot->mark_completed(false);
    cv_.notify_all();
  }
}

int TcpTransportAdapter::create_listen_socket(uint16_t& out_port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  int opt = 1;
  if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
    ::close(fd);
    return -1;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(0);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }
  if (::listen(fd, 128) != 0) {
    ::close(fd);
    return -1;
  }

  sockaddr_in bound_addr{};
  socklen_t bound_len = sizeof(bound_addr);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&bound_addr), &bound_len) !=
      0) {
    ::close(fd);
    return -1;
  }
  out_port = ntohs(bound_addr.sin_port);
  return fd;
}

bool TcpTransportAdapter::connect_socket(int& out_fd,
                                         std::string const& remote_ip,
                                         uint16_t remote_port,
                                         std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(remote_port);
    if (::inet_pton(AF_INET, remote_ip.c_str(), &addr.sin_addr) != 1) {
      ::close(fd);
      return false;
    }
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      out_fd = fd;
      return true;
    }
    ::close(fd);
    std::this_thread::sleep_for(kConnectRetrySleep);
  }
  return false;
}

bool TcpTransportAdapter::send_handshake(int fd, Handshake const& hs) {
  return send_all(fd, &hs, sizeof(hs));
}

bool TcpTransportAdapter::recv_handshake(int fd, Handshake& hs) {
  return recv_all(fd, &hs, sizeof(hs));
}

bool TcpTransportAdapter::send_handshake_ack(int fd, HandshakeAck const& ack) {
  return send_all(fd, &ack, sizeof(ack));
}

bool TcpTransportAdapter::recv_handshake_ack(int fd, HandshakeAck& ack) {
  return recv_all(fd, &ack, sizeof(ack));
}

bool TcpTransportAdapter::send_all(int fd, void const* buf, size_t len) {
  size_t sent = 0;
  auto const* ptr = static_cast<char const*>(buf);
  while (sent < len) {
    ssize_t rc = ::send(fd, ptr + sent, len - sent, 0);
    if (rc > 0) {
      sent += static_cast<size_t>(rc);
      continue;
    }
    if (rc == 0) return false;
    if (is_retryable_errno(errno)) continue;
    return false;
  }
  return true;
}

bool TcpTransportAdapter::recv_all(int fd, void* buf, size_t len) {
  size_t received = 0;
  auto* ptr = static_cast<char*>(buf);
  while (received < len) {
    ssize_t rc = ::recv(fd, ptr + received, len - received, 0);
    if (rc > 0) {
      received += static_cast<size_t>(rc);
      continue;
    }
    if (rc == 0) return false;
    if (is_retryable_errno(errno)) continue;
    return false;
  }
  return true;
}

}  // namespace Transport
}  // namespace UKernel
