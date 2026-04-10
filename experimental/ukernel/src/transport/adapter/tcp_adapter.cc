#include "tcp_adapter.h"
#include "gpu_rt.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
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

bool is_retryable_errno(int err) {
  return err == EINTR || err == EAGAIN || err == EWOULDBLOCK;
}

}  // namespace

TcpTransportAdapter::TcpTransportAdapter(std::string local_ip, int local_rank)
    : local_ip_(std::move(local_ip)), local_rank_(local_rank) {
  listen_fd_ = create_listen_socket(listen_port_);
  if (listen_fd_ < 0) {
    throw std::runtime_error("failed to create tcp transport listen socket");
  }
}

TcpTransportAdapter::~TcpTransportAdapter() {
  std::unordered_map<unsigned, std::shared_ptr<PendingRequest>> pending_copy;
  {
    std::lock_guard<std::mutex> lk(mu_);
    pending_copy = pending_;
  }
  for (auto& [request_id, pending] : pending_copy) {
    (void)request_id;
    if (pending && pending->worker.joinable()) {
      pending->worker.join();
    }
  }

  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [peer_rank, ctx] : peer_contexts_) {
    (void)peer_rank;
    if (!ctx) continue;
    int send_fd = ctx->send_fd;
    int recv_fd = ctx->recv_fd;
    if (send_fd >= 0) {
      ::shutdown(ctx->send_fd, SHUT_RDWR);
      ::close(ctx->send_fd);
    }
    if (recv_fd >= 0 && recv_fd != send_fd) {
      ::shutdown(ctx->recv_fd, SHUT_RDWR);
      ::close(ctx->recv_fd);
    }
    ctx->send_fd = -1;
    ctx->recv_fd = -1;
  }
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
}

uint16_t TcpTransportAdapter::get_listen_port() const { return listen_port_; }

bool TcpTransportAdapter::connect_to_peer(int peer_rank, std::string remote_ip,
                                          uint16_t remote_port) {
  if (has_send_peer(peer_rank) && has_recv_peer(peer_rank)) return true;

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
  if (ctx->send_fd >= 0 && ctx->recv_fd >= 0) {
    ::close(fd);
    return true;
  }
  if (ctx->send_fd < 0) ctx->send_fd = fd;
  if (ctx->recv_fd < 0) ctx->recv_fd = fd;
  ctx->remote_ip = std::move(remote_ip);
  return true;
}

bool TcpTransportAdapter::accept_from_peer(
    int peer_rank, std::string const& expected_remote_ip,
    AcceptedPeer* accepted_peer) {
  if (has_send_peer(peer_rank) && has_recv_peer(peer_rank)) return true;

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
    if (ctx->send_fd >= 0 && ctx->recv_fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      return true;
    }
    if (ctx->send_fd < 0) ctx->send_fd = fd;
    if (ctx->recv_fd < 0) ctx->recv_fd = fd;
    ctx->remote_ip = remote_ip;
    if (accepted_peer != nullptr) {
      accepted_peer->remote_rank = static_cast<int>(hs.src_rank);
      accepted_peer->remote_ip = remote_ip;
    }
    return true;
  }
  return false;
}

bool TcpTransportAdapter::ensure_peer(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_peer(spec.peer_rank)) return true;
  if (!std::holds_alternative<TcpPeerConnectSpec>(spec.detail)) return false;
  auto const& tcp = std::get<TcpPeerConnectSpec>(spec.detail);
  if (spec.type == PeerConnectType::Connect) {
    return connect_to_peer(spec.peer_rank, tcp.remote_ip, tcp.remote_port);
  }
  return accept_from_peer(spec.peer_rank, tcp.remote_ip, nullptr);
}

bool TcpTransportAdapter::has_send_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second && it->second->send_fd >= 0;
}

bool TcpTransportAdapter::has_recv_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second && it->second->recv_fd >= 0;
}

bool TcpTransportAdapter::has_peer(int peer_rank) const {
  return has_send_peer(peer_rank) && has_recv_peer(peer_rank);
}

unsigned TcpTransportAdapter::send_async(int peer_rank, void* local_ptr,
                                         size_t len, uint64_t local_mr_id,
                                         std::optional<RemoteSlice> remote_hint,
                                         BounceBufferProvider bounce_provider) {
  (void)local_mr_id;
  (void)remote_hint;

  void* send_ptr = local_ptr;
  if (bounce_provider) {
    BounceBufferInfo info = bounce_provider(len);
    if (info.ptr != nullptr) {
      GPU_RT_CHECK(gpuMemcpy(info.ptr, local_ptr, len, gpuMemcpyDeviceToHost));
      send_ptr = info.ptr;
    }
  }

  unsigned request_id =
      next_request_id_.fetch_add(1, std::memory_order_relaxed);
  return send_async_tcp(peer_rank, send_ptr, len, request_id) == 0 ? request_id
                                                                   : 0;
}

unsigned TcpTransportAdapter::recv_async(int peer_rank, void* local_ptr,
                                         size_t len, uint64_t local_mr_id,
                                         BounceBufferProvider bounce_provider) {
  (void)local_mr_id;
  void* recv_ptr = local_ptr;
  if (bounce_provider) {
    BounceBufferInfo info = bounce_provider(len);
    if (info.ptr != nullptr) {
      recv_ptr = info.ptr;
    }
  }
  unsigned request_id =
      next_request_id_.fetch_add(1, std::memory_order_relaxed);
  return recv_async_tcp(peer_rank, recv_ptr, len, request_id) == 0 ? request_id
                                                                   : 0;
}

bool TcpTransportAdapter::request_failed(unsigned id) {
  std::shared_ptr<PendingRequest> pending;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(id);
    if (it == pending_.end()) return false;
    pending = it->second;
  }
  return pending->failed.load(std::memory_order_acquire);
}

int TcpTransportAdapter::send_async_tcp(int peer_rank, void const* host_ptr,
                                        size_t len, unsigned request_id) {
  std::shared_ptr<PendingRequest> pending = std::make_shared<PendingRequest>();
  std::shared_ptr<PeerContext> ctx;

  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peer_contexts_.find(peer_rank);
    if (it == peer_contexts_.end() || !it->second || it->second->send_fd < 0) {
      return -1;
    }
    ctx = it->second;
    pending_[request_id] = pending;
  }

  pending->worker = std::thread([pending, ctx, host_ptr, len]() {
    bool ok = false;
    {
      std::lock_guard<std::mutex> send_lk(ctx->send_mu);
      ok = TcpTransportAdapter::send_all(ctx->send_fd, host_ptr, len);
    }
    pending->failed.store(!ok, std::memory_order_release);
    pending->completed.store(true, std::memory_order_release);
  });

  return 0;
}

int TcpTransportAdapter::recv_async_tcp(int peer_rank, void* host_ptr,
                                        size_t len, unsigned request_id) {
  std::shared_ptr<PendingRequest> pending = std::make_shared<PendingRequest>();
  std::shared_ptr<PeerContext> ctx;

  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peer_contexts_.find(peer_rank);
    if (it == peer_contexts_.end() || !it->second || it->second->recv_fd < 0) {
      return -1;
    }
    ctx = it->second;
    pending_[request_id] = pending;
  }

  pending->worker = std::thread([pending, ctx, host_ptr, len]() {
    bool ok = false;
    {
      std::lock_guard<std::mutex> recv_lk(ctx->recv_mu);
      ok = TcpTransportAdapter::recv_all(ctx->recv_fd, host_ptr, len);
    }
    pending->failed.store(!ok, std::memory_order_release);
    pending->completed.store(true, std::memory_order_release);
  });

  return 0;
}

bool TcpTransportAdapter::poll_completion(unsigned request_id) {
  std::shared_ptr<PendingRequest> pending;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(request_id);
    if (it == pending_.end()) return true;
    pending = it->second;
  }
  return pending->completed.load(std::memory_order_acquire);
}

bool TcpTransportAdapter::wait_completion(unsigned request_id) {
  std::shared_ptr<PendingRequest> pending;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(request_id);
    if (it == pending_.end()) return false;
    pending = it->second;
  }
  while (!pending->completed.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return true;
}

void TcpTransportAdapter::release_request(unsigned request_id) {
  join_and_erase_request(request_id);
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

void TcpTransportAdapter::join_and_erase_request(unsigned request_id) {
  std::shared_ptr<PendingRequest> pending;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(request_id);
    if (it == pending_.end()) return;
    pending = it->second;
    pending_.erase(it);
  }
  if (pending && pending->worker.joinable()) {
    pending->worker.join();
  }
}

}  // namespace Transport
}  // namespace UKernel
