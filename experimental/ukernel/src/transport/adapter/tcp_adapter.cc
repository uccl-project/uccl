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

enum class FrameType : uint32_t { Data = 1, Signal = 2 };

struct WireHeader {
  uint32_t type = 0;
  uint64_t payload_len = 0;
};

bool is_retryable(int e) { return e == EINTR || e == EAGAIN || e == EWOULDBLOCK; }

bool enqueue_rid(jring_t* ring, unsigned rid, std::atomic<bool> const& stop) {
  while (!stop.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(ring, &rid, 1, nullptr) != 1)
    std::this_thread::yield();
  return !stop.load(std::memory_order_acquire);
}

bool recv_discard(int fd, uint64_t len) {
  constexpr size_t kChunk = 4096;
  char buf[kChunk];
  uint64_t remain = len;
  while (remain > 0) {
    size_t n = std::min<uint64_t>(remain, kChunk);
    size_t got = 0;
    while (got < n) {
      ssize_t rc = ::recv(fd, buf + got, n - got, 0);
      if (rc > 0) { got += rc; continue; }
      if (rc == 0) return false;
      if (is_retryable(errno)) continue;
      return false;
    }
    remain -= n;
  }
  return true;
}

}  // namespace

TcpTransportAdapter::TcpTransportAdapter(std::string local_ip, int local_rank,
                                         int gpu_id)
    : local_ip_(std::move(local_ip)), local_rank_(local_rank), gpu_id_(gpu_id) {
  if (gpu_id_ >= 0)
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&gpu_stream_, gpuStreamNonBlocking));
  listen_fd_ = create_listen_socket(listen_port_);
  if (listen_fd_ < 0)
    throw std::runtime_error("failed to create tcp listen socket");

  send_task_ring_ = create_ring(sizeof(unsigned), kTaskRingSize);
  recv_task_ring_ = create_ring(sizeof(unsigned), kTaskRingSize);
  slots_ = std::make_unique<RequestSlot[]>(kSlotCount);
  if (!send_task_ring_ || !recv_task_ring_ || !slots_) {
    free(send_task_ring_); send_task_ring_ = nullptr;
    free(recv_task_ring_); recv_task_ring_ = nullptr;
    ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); listen_fd_ = -1;
    throw std::runtime_error("failed to init tcp request infra");
  }
  send_worker_ = std::thread([this] { send_worker_loop(); });
  recv_worker_ = std::thread([this] { recv_worker_loop(); });
}

TcpTransportAdapter::~TcpTransportAdapter() {
  stop_.store(true);
  if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); }
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [_, ctx] : peer_contexts_) {
      if (!ctx) continue;
      if (ctx->send_fd >= 0) ::shutdown(ctx->send_fd, SHUT_RDWR);
      if (ctx->recv_fd >= 0 && ctx->recv_fd != ctx->send_fd) ::shutdown(ctx->recv_fd, SHUT_RDWR);
    }
  }
  if (send_worker_.joinable()) send_worker_.join();
  if (recv_worker_.joinable()) recv_worker_.join();
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [_, ctx] : peer_contexts_)
      if (ctx) { if (ctx->send_fd >= 0) ::close(ctx->send_fd);
                 if (ctx->recv_fd >= 0 && ctx->recv_fd != ctx->send_fd) ::close(ctx->recv_fd); }
  }
  free(send_task_ring_); send_task_ring_ = nullptr;
  free(recv_task_ring_); recv_task_ring_ = nullptr;
  if (gpu_stream_) { gpuStreamDestroy(gpu_stream_); gpu_stream_ = nullptr; }
}

uint16_t TcpTransportAdapter::get_listen_port() const { return listen_port_; }

// ── Path management ─────────────────────────────────────────────────────

bool TcpTransportAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  auto const* tcp = std::get_if<TcpPeerConnectSpec>(&spec.detail);
  if (!tcp) return false;
  if (spec.type == PeerConnectType::Connect)
    return connect_to_peer(spec.peer_rank, tcp->remote_ip, tcp->remote_port);
  return accept_from_peer(spec.peer_rank, "");
}

bool TcpTransportAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  auto const* tcp = std::get_if<TcpPeerConnectSpec>(&spec.detail);
  if (!tcp) return false;
  if (spec.type == PeerConnectType::Connect)
    return connect_to_peer(spec.peer_rank, tcp->remote_ip, tcp->remote_port);
  return accept_from_peer(spec.peer_rank, "");
}

bool TcpTransportAdapter::has_put_path(int rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(rank);
  return it != peer_contexts_.end() && it->second && it->second->send_fd >= 0;
}

bool TcpTransportAdapter::has_wait_path(int rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(rank);
  return it != peer_contexts_.end() && it->second && it->second->recv_fd >= 0;
}

// ── Submission ──────────────────────────────────────────────────────────

unsigned TcpTransportAdapter::put_async(int peer, void* local_ptr, uint32_t,
                                        void*, uint32_t, size_t len,
                                        unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  unsigned rid = 0;
  RequestSlot* s = acquire_slot(&rid);
  if (!s) return 0;
  s->comm_rid = comm_rid;
  s->peer_rank = peer;
  s->kind = RequestSlot::Kind::DataPut;
  s->host_ptr = local_ptr;
  s->len = len;
  s->mark_queued();
  if (!enqueue_send(rid)) { s->mark_completed(false); release_slot(rid); return 0; }
  return rid;
}

unsigned TcpTransportAdapter::signal_async(int peer, uint64_t tag,
                                           unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  unsigned rid = 0;
  RequestSlot* s = acquire_slot(&rid);
  if (!s) return 0;
  s->comm_rid = comm_rid;
  s->peer_rank = peer;
  s->kind = RequestSlot::Kind::Signal;
  s->signal_payload = tag;
  s->mark_queued();
  if (!enqueue_send(rid)) { s->mark_completed(false); release_slot(rid); return 0; }
  return rid;
}

unsigned TcpTransportAdapter::wait_async(int peer, uint64_t tag,
                                         std::optional<WaitTarget> target,
                                         unsigned comm_rid) {
  if (!has_wait_path(peer)) return 0;
  unsigned rid = 0;
  RequestSlot* s = acquire_slot(&rid);
  if (!s) return 0;
  s->comm_rid = comm_rid;
  s->peer_rank = peer;
  s->kind = target ? RequestSlot::Kind::DataWait : RequestSlot::Kind::SignalWait;
  s->host_ptr = target ? target->local_ptr : nullptr;
  s->len = target ? target->len : 0;
  s->expected_tag = tag;
  s->mark_queued();
  if (!enqueue_recv(rid)) { s->mark_completed(false); release_slot(rid); return 0; }
  return rid;
}

void TcpTransportAdapter::release(unsigned id) { release_slot(id); }

// ── Peer connection ─────────────────────────────────────────────────────

bool TcpTransportAdapter::connect_to_peer(int rank, std::string ip, uint16_t port) {
  std::lock_guard<std::mutex> lk(mu_);
  if (peer_contexts_.count(rank) && peer_contexts_[rank] &&
      peer_contexts_[rank]->send_fd >= 0)
    return true;
  auto ctx = std::make_shared<PeerContext>();
  if (!connect_socket(ctx->send_fd, ip, port, kDefaultConnectTimeout)) return false;
  ctx->recv_fd = ctx->send_fd;
  handshake(ctx->send_fd, local_rank_, true);
  peer_contexts_[rank] = ctx;
  return true;
}

bool TcpTransportAdapter::accept_from_peer(int rank, std::string const&) {
  // Handled by Communicator via get_listen_port() — accept happens at a higher level
  (void)rank;
  return true;
}

// ── Slot management ─────────────────────────────────────────────────────

TcpTransportAdapter::RequestSlot* TcpTransportAdapter::acquire_slot(unsigned* out) {
  uint32_t start = alloc_cursor_.fetch_add(1) & (kSlotCount - 1u);
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    uint32_t idx = (start + i) & (kSlotCount - 1u);
    RequestSlot* s = &slots_[idx];
    auto expected = RequestState::Free;
    if (!s->state.compare_exchange_strong(expected, RequestState::Queued,
                                          std::memory_order_acq_rel, std::memory_order_acquire))
      continue;
    uint32_t gen = (s->generation.fetch_add(1) + 1u) & ((1u << (32u - kSlotBits)) - 1u);
    if (gen == 0) { gen = 1; s->generation.store(gen); }
    unsigned rid = make_rid(idx, gen);
    s->comm_rid = 0;
    s->peer_rank = -1;
    s->host_ptr = nullptr;
    s->len = 0;
    *out = rid;
    return s;
  }
  return nullptr;
}

TcpTransportAdapter::RequestSlot* TcpTransportAdapter::resolve_slot(unsigned rid) {
  uint32_t idx = slot_index(rid);
  if (idx >= kSlotCount) return nullptr;
  RequestSlot* s = &slots_[idx];
  uint32_t gen = s->generation.load() & ((1u << (32u - kSlotBits)) - 1u);
  if (gen == 0 || gen != slot_gen(rid)) return nullptr;
  return s;
}

void TcpTransportAdapter::release_slot(unsigned rid) {
  RequestSlot* s = resolve_slot(rid);
  if (!s) return;
  s->state.store(RequestState::Free, std::memory_order_release);
}

bool TcpTransportAdapter::enqueue_send(unsigned rid) {
  return enqueue_rid(send_task_ring_, rid, stop_);
}

bool TcpTransportAdapter::enqueue_recv(unsigned rid) {
  return enqueue_rid(recv_task_ring_, rid, stop_);
}

// ── Workers ─────────────────────────────────────────────────────────────

void TcpTransportAdapter::send_worker_loop() {
  while (!stop_.load(std::memory_order_acquire)) {
    unsigned rid = 0;
    if (jring_sc_dequeue_bulk(send_task_ring_, &rid, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    RequestSlot* s = resolve_slot(rid);
    if (!s) continue;
    bool ok = false;
    std::shared_ptr<PeerContext> ctx;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = peer_contexts_.find(s->peer_rank);
      if (it != peer_contexts_.end() && it->second && it->second->send_fd >= 0)
        ctx = it->second;
    }
    if (ctx) {
      std::lock_guard<std::mutex> lk(ctx->send_mu);
      // GPU→host bounce
      void* ptr = s->host_ptr;
      void* bounce = nullptr;
      if (s->len > 0 && ptr) {
        gpuPointerAttributes attr{};
        if (gpuPointerGetAttributes(&attr, ptr) == gpuSuccess &&
            attr.type == gpuMemoryTypeDevice) {
          GPU_RT_CHECK(gpuSetDevice(gpu_id_));
          GPU_RT_CHECK(gpuMallocHost(&bounce, s->len));
          GPU_RT_CHECK(gpuMemcpyAsync(bounce, ptr, s->len,
                                       gpuMemcpyDeviceToHost, gpu_stream_));
          GPU_RT_CHECK(gpuStreamSynchronize(gpu_stream_));
          ptr = bounce;
        }
      }
      WireHeader hdr{};
      hdr.type = static_cast<uint32_t>(s->kind == RequestSlot::Kind::Signal
                                           ? FrameType::Signal : FrameType::Data);
      hdr.payload_len = s->len;
      ok = send_all(ctx->send_fd, &hdr, sizeof(hdr));
      if (ok && s->len > 0)
        ok = send_all(ctx->send_fd, ptr, s->len);
      if (bounce) GPU_RT_CHECK(gpuFreeHost(bounce));
    }
    s->mark_completed(ok);
    publish_completion(s->comm_rid, !ok);
  }
  // Drain remaining
  unsigned rid;
  while (jring_mc_dequeue_bulk(send_task_ring_, &rid, 1, nullptr) == 1)
    if (RequestSlot* s = resolve_slot(rid))
      s->mark_completed(false);
}

void TcpTransportAdapter::recv_worker_loop() {
  while (!stop_.load(std::memory_order_acquire)) {
    unsigned rid = 0;
    if (jring_sc_dequeue_bulk(recv_task_ring_, &rid, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    RequestSlot* s = resolve_slot(rid);
    if (!s) continue;
    bool ok = false;
    std::shared_ptr<PeerContext> ctx;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = peer_contexts_.find(s->peer_rank);
      if (it != peer_contexts_.end() && it->second && it->second->recv_fd >= 0)
        ctx = it->second;
    }
    if (ctx) {
      std::lock_guard<std::mutex> lk(ctx->recv_mu);
      WireHeader hdr{};
      ok = recv_all(ctx->recv_fd, &hdr, sizeof(hdr));
      if (ok && hdr.payload_len > 0) {
        if (s->kind == RequestSlot::Kind::SignalWait) {
          ok = recv_all(ctx->recv_fd, &s->signal_payload, sizeof(uint64_t));
        } else if (s->kind == RequestSlot::Kind::DataWait && s->host_ptr) {
          ok = recv_all(ctx->recv_fd, s->host_ptr, s->len);
        } else {
          ok = recv_discard(ctx->recv_fd, hdr.payload_len);
        }
      }
    }
    s->mark_completed(ok);
    publish_completion(s->comm_rid, !ok);
  }
  unsigned rid;
  while (jring_mc_dequeue_bulk(recv_task_ring_, &rid, 1, nullptr) == 1)
    if (RequestSlot* s = resolve_slot(rid))
      s->mark_completed(false);
}

// ── Socket helpers ──────────────────────────────────────────────────────

int TcpTransportAdapter::create_listen_socket(uint16_t& out_port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(fd); return -1; }
  if (::listen(fd, 128) < 0) { ::close(fd); return -1; }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  if (::getsockname(fd, (sockaddr*)&bound, &len) == 0)
    out_port = ntohs(bound.sin_port);
  return fd;
}

bool TcpTransportAdapter::connect_socket(int& out_fd, std::string const& ip,
                                         uint16_t port, std::chrono::milliseconds timeout) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return false;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  ::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (::connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    if (errno != EINPROGRESS && errno != EALREADY && errno != EAGAIN &&
        errno != ECONNREFUSED) {
      ::close(fd); return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) { ::close(fd); return false; }
    std::this_thread::sleep_for(kConnectRetrySleep);
  }
  out_fd = fd;
  return true;
}

bool TcpTransportAdapter::handshake(int fd, uint32_t rank, bool) {
  return send_all(fd, &rank, sizeof(rank));
}

bool TcpTransportAdapter::recv_handshake(int fd, uint32_t& rank) {
  return recv_all(fd, &rank, sizeof(rank));
}

bool TcpTransportAdapter::send_all(int fd, void const* buf, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    ssize_t rc = ::send(fd, (char const*)buf + sent, len - sent, MSG_NOSIGNAL);
    if (rc > 0) { sent += rc; continue; }
    if (is_retryable(errno)) continue;
    return false;
  }
  return true;
}

bool TcpTransportAdapter::recv_all(int fd, void* buf, size_t len) {
  size_t got = 0;
  while (got < len) {
    ssize_t rc = ::recv(fd, (char*)buf + got, len - got, 0);
    if (rc > 0) { got += rc; continue; }
    if (rc == 0) return false;
    if (is_retryable(errno)) continue;
    return false;
  }
  return true;
}

}  // namespace Transport
}  // namespace UKernel
