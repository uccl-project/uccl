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

bool is_retryable(int e) {
  return e == EINTR || e == EAGAIN || e == EWOULDBLOCK;
}

template <typename T>
bool enqueue_elem(jring_t* ring, T const& elem, std::atomic<bool> const& stop) {
  while (!stop.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(ring, &elem, 1, nullptr) != 1)
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
      if (rc > 0) {
        got += rc;
        continue;
      }
      if (rc == 0) return false;
      if (is_retryable(errno)) continue;
      return false;
    }
    remain -= n;
  }
  return true;
}

}  // namespace

// ── CpuBouncePool ─────────────────────────────────────────────────────

CpuBouncePool::CpuBouncePool(size_t buffer_size, size_t num_buffers)
    : buffer_size_(buffer_size) {
  if (num_buffers == 0) return;
  free_list_.reserve(num_buffers);
  for (size_t i = 0; i < num_buffers; ++i) {
    void* buf = nullptr;
    GPU_RT_CHECK(gpuMallocHost(&buf, buffer_size_));
    free_list_.push_back(buf);
    all_bufs_.insert(buf);
  }
}

CpuBouncePool::~CpuBouncePool() {
  for (void* buf : all_bufs_) {
    if (buf) GPU_RT_CHECK(gpuFreeHost(buf));
  }
}

void* CpuBouncePool::acquire(size_t size) {
  std::lock_guard<std::mutex> lk(mu_);
  if (size > buffer_size_) {
    // Oversized request - allocate directly, not tracked in pool
    void* buf = nullptr;
    GPU_RT_CHECK(gpuMallocHost(&buf, size));
    return buf;
  }
  if (!free_list_.empty()) {
    void* buf = free_list_.back();
    free_list_.pop_back();
    return buf;
  }
  // Pool exhausted - allocate another and track it
  void* buf = nullptr;
  GPU_RT_CHECK(gpuMallocHost(&buf, buffer_size_));
  all_bufs_.insert(buf);
  return buf;
}

void CpuBouncePool::release(void* buf) {
  if (!buf) return;
  std::lock_guard<std::mutex> lk(mu_);
  if (all_bufs_.count(buf)) {
    free_list_.push_back(buf);
  } else {
    // Oversized one-off allocation - free directly
    GPU_RT_CHECK(gpuFreeHost(buf));
  }
}

TcpTransportAdapter::TcpTransportAdapter(std::string local_ip, int local_rank,
                                         int gpu_id)
    : local_ip_(std::move(local_ip)), local_rank_(local_rank), gpu_id_(gpu_id) {
  if (gpu_id_ >= 0) {
    GPU_RT_CHECK(gpuSetDevice(gpu_id_));
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&gpu_stream_, gpuStreamNonBlocking));
    bounce_pool_ = std::make_unique<CpuBouncePool>();
  }
  listen_fd_ = create_listen_socket(listen_port_);
  if (listen_fd_ < 0)
    throw std::runtime_error("failed to create tcp listen socket");

  send_task_ring_ = create_ring(sizeof(RingElem), kTaskRingSize);
  recv_task_ring_ = create_ring(sizeof(RingElem), kTaskRingSize);
  if (!send_task_ring_ || !recv_task_ring_) {
    free(send_task_ring_);
    send_task_ring_ = nullptr;
    free(recv_task_ring_);
    recv_task_ring_ = nullptr;
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
    throw std::runtime_error("failed to init tcp request infra");
  }
  send_worker_ = std::thread([this] { send_worker_loop(); });
  recv_worker_ = std::thread([this] { recv_worker_loop(); });
}

TcpTransportAdapter::~TcpTransportAdapter() {
  stop_.store(true);
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [_, ctx] : peer_contexts_) {
      if (!ctx) continue;
      if (ctx->send_fd >= 0) ::shutdown(ctx->send_fd, SHUT_RDWR);
      if (ctx->recv_fd >= 0 && ctx->recv_fd != ctx->send_fd)
        ::shutdown(ctx->recv_fd, SHUT_RDWR);
    }
  }
  if (send_worker_.joinable()) send_worker_.join();
  if (recv_worker_.joinable()) recv_worker_.join();
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [_, ctx] : peer_contexts_)
      if (ctx) {
        if (ctx->send_fd >= 0) ::close(ctx->send_fd);
        if (ctx->recv_fd >= 0 && ctx->recv_fd != ctx->send_fd)
          ::close(ctx->recv_fd);
      }
  }
  free(send_task_ring_);
  send_task_ring_ = nullptr;
  free(recv_task_ring_);
  recv_task_ring_ = nullptr;
  if (gpu_stream_) {
    gpuStreamDestroy(gpu_stream_);
    gpu_stream_ = nullptr;
  }
}

uint16_t TcpTransportAdapter::get_listen_port() const { return listen_port_; }

// ── Path management ─────────────────────────────────────────────────────

bool TcpTransportAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  auto const* tcp = std::get_if<TcpPeerConnectSpec>(&spec.detail);
  if (!tcp) return false;
  if (spec.type != PeerConnectType::Connect) return false;
  return connect_to_peer(spec.peer_rank, tcp->remote_ip, tcp->remote_port);
}

bool TcpTransportAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  auto const* tcp = std::get_if<TcpPeerConnectSpec>(&spec.detail);
  if (!tcp) return false;
  if (spec.type != PeerConnectType::Accept) return false;
  return accept_from_peer(spec.peer_rank, tcp->remote_ip);
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

unsigned TcpTransportAdapter::send_put_async(int peer, void* local_ptr,
                                             uint32_t, void*, uint32_t,
                                             size_t len, unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  RingElem e{comm_rid, peer, Kind::DataPut, local_ptr, len, 0};
  if (!enqueue_elem(send_task_ring_, e, stop_)) return 0;
  return 1;
}

unsigned TcpTransportAdapter::send_signal_async(int peer, uint64_t tag,
                                                unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  RingElem e{comm_rid, peer, Kind::Signal, nullptr, 0, tag};
  if (!enqueue_elem(send_task_ring_, e, stop_)) return 0;
  return 1;
}

unsigned TcpTransportAdapter::wait_signal_async(
    int peer, uint64_t tag, std::optional<WaitTarget> target,
    unsigned comm_rid) {
  if (!has_wait_path(peer)) return 0;
  void* ptr = target ? target->local_ptr : nullptr;
  size_t len = target ? target->len : 0;
  Kind k = target ? Kind::DataWait : Kind::SignalWait;
  RingElem e{comm_rid, peer, k, ptr, len, tag};
  if (!enqueue_elem(recv_task_ring_, e, stop_)) return 0;
  return 1;
}

// ── Workers ─────────────────────────────────────────────────────────────

void TcpTransportAdapter::send_worker_loop() {
  RingElem e;
  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(send_task_ring_, &e, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    bool ok = false;
    std::shared_ptr<PeerContext> ctx;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = peer_contexts_.find(e.peer);
      if (it != peer_contexts_.end() && it->second && it->second->send_fd >= 0)
        ctx = it->second;
    }
    if (ctx) {
      std::lock_guard<std::mutex> lk(ctx->send_mu);
      void* ptr = e.ptr;
      void* bounce = nullptr;
      if (e.len > 0 && ptr) {
        gpuPointerAttributes attr{};
        if (gpuPointerGetAttributes(&attr, ptr) == gpuSuccess &&
            attr.type == gpuMemoryTypeDevice) {
          GPU_RT_CHECK(gpuSetDevice(gpu_id_));
          bounce = bounce_pool_->acquire(e.len);
          GPU_RT_CHECK(gpuMemcpyAsync(bounce, ptr, e.len, gpuMemcpyDeviceToHost,
                                      gpu_stream_));
          GPU_RT_CHECK(gpuStreamSynchronize(gpu_stream_));
          ptr = bounce;
        }
      }
      WireHeader hdr{};
      hdr.type = static_cast<uint32_t>(
          e.kind == Kind::Signal ? FrameType::Signal : FrameType::Data);
      hdr.payload_len = e.len;
      ok = send_all(ctx->send_fd, &hdr, sizeof(hdr));
      if (ok && e.len > 0) ok = send_all(ctx->send_fd, ptr, e.len);
      if (bounce) bounce_pool_->release(bounce);
    }
    publish_completion(e.comm_rid, !ok);
  }
  // Drain remaining
  RingElem drain;
  while (jring_mc_dequeue_bulk(send_task_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

void TcpTransportAdapter::recv_worker_loop() {
  RingElem e;
  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(recv_task_ring_, &e, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    bool ok = false;
    std::shared_ptr<PeerContext> ctx;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = peer_contexts_.find(e.peer);
      if (it != peer_contexts_.end() && it->second && it->second->recv_fd >= 0)
        ctx = it->second;
    }
    if (ctx) {
      std::lock_guard<std::mutex> lk(ctx->recv_mu);
      WireHeader hdr{};
      ok = recv_all(ctx->recv_fd, &hdr, sizeof(hdr));
      uint64_t tag = 0;
      if (ok && hdr.payload_len > 0) {
        if (e.kind == Kind::SignalWait) {
          ok = recv_all(ctx->recv_fd, &tag, sizeof(uint64_t));
        } else if (e.kind == Kind::DataWait && e.ptr) {
          gpuPointerAttributes attr{};
          bool is_gpu = (gpuPointerGetAttributes(&attr, e.ptr) == gpuSuccess &&
                         attr.type == gpuMemoryTypeDevice);
          if (is_gpu) {
            GPU_RT_CHECK(gpuSetDevice(gpu_id_));
            void* bounce = bounce_pool_->acquire(e.len);
            ok = recv_all(ctx->recv_fd, bounce, e.len);
            if (ok) {
              GPU_RT_CHECK(gpuMemcpyAsync(e.ptr, bounce, e.len,
                                          gpuMemcpyHostToDevice, gpu_stream_));
              GPU_RT_CHECK(gpuStreamSynchronize(gpu_stream_));
            }
            bounce_pool_->release(bounce);
          } else {
            ok = recv_all(ctx->recv_fd, e.ptr, e.len);
          }
        } else {
          ok = recv_discard(ctx->recv_fd, hdr.payload_len);
        }
      }
    }
    publish_completion(e.comm_rid, !ok);
  }
  RingElem drain;
  while (jring_mc_dequeue_bulk(recv_task_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

// ── Peer connection ─────────────────────────────────────────────────────

bool TcpTransportAdapter::connect_to_peer(int rank, std::string ip,
                                          uint16_t port) {
  std::lock_guard<std::mutex> lk(mu_);
  if (peer_contexts_.count(rank) && peer_contexts_[rank] &&
      peer_contexts_[rank]->send_fd >= 0)
    return true;

  int fd = -1;
  if (!connect_socket(fd, ip, port, kDefaultConnectTimeout)) return false;

  Handshake hs{static_cast<uint32_t>(local_rank_)};
  HandshakeAck ack{};
  if (!send_handshake(fd, hs) || !recv_handshake_ack(fd, ack) ||
      ack.accepted != 1) {
    ::close(fd);
    return false;
  }

  auto& ctx = peer_contexts_[rank];
  if (!ctx) ctx = std::make_shared<PeerContext>();
  ctx->send_fd = fd;
  return true;
}

bool TcpTransportAdapter::accept_from_peer(
    int rank, std::string const& expected_remote_ip) {
  if (has_wait_path(rank)) return true;

  auto deadline = std::chrono::steady_clock::now() + kDefaultConnectTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    sockaddr_in addr{};
    socklen_t addr_len = sizeof(addr);
    int fd =
        ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &addr_len);
    if (fd < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        std::this_thread::sleep_for(kAcceptPollSleep);
        continue;
      }
      return false;
    }

    if (!expected_remote_ip.empty()) {
      char remote_ip_buf[INET_ADDRSTRLEN] = {};
      ::inet_ntop(AF_INET, &addr.sin_addr, remote_ip_buf,
                  sizeof(remote_ip_buf));
      if (expected_remote_ip != remote_ip_buf) {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
        continue;
      }
    }

    Handshake hs{};
    if (!recv_handshake(fd, hs) || static_cast<int>(hs.src_rank) != rank) {
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
    auto& ctx = peer_contexts_[rank];
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

// ── Socket helpers ──────────────────────────────────────────────────────

int TcpTransportAdapter::create_listen_socket(uint16_t& out_port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  if (::listen(fd, 128) < 0) {
    ::close(fd);
    return -1;
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  if (::getsockname(fd, (sockaddr*)&bound, &len) == 0)
    out_port = ntohs(bound.sin_port);
  return fd;
}

bool TcpTransportAdapter::connect_socket(int& out_fd, std::string const& ip,
                                         uint16_t port,
                                         std::chrono::milliseconds timeout) {
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
      ::close(fd);
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      ::close(fd);
      return false;
    }
    std::this_thread::sleep_for(kConnectRetrySleep);
  }
  out_fd = fd;
  return true;
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
  while (sent < len) {
    ssize_t rc = ::send(fd, (char const*)buf + sent, len - sent, MSG_NOSIGNAL);
    if (rc > 0) {
      sent += rc;
      continue;
    }
    if (is_retryable(errno)) continue;
    return false;
  }
  return true;
}

bool TcpTransportAdapter::recv_all(int fd, void* buf, size_t len) {
  size_t got = 0;
  while (got < len) {
    ssize_t rc = ::recv(fd, (char*)buf + got, len - got, 0);
    if (rc > 0) {
      got += rc;
      continue;
    }
    if (rc == 0) return false;
    if (is_retryable(errno)) continue;
    return false;
  }
  return true;
}

}  // namespace Transport
}  // namespace UKernel
