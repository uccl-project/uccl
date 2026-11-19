#pragma once

// CPU-only control traffic for the ABI integration. The opaque NCCL ID is a
// session key, never interpreted as an address or another private NCCL field.
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace uccl_lite::nccl_detail {
inline uint16_t hostControlInitialPort(void const* session, size_t bytes) {
  uint64_t salt = 14695981039346656037ULL;
  auto key = static_cast<unsigned char const*>(session);
  for (size_t i = 0; i < bytes; ++i) salt = (salt ^ key[i]) * 1099511628211ULL;
  // Stay below the usual Linux ephemeral range used by NCCL/bootstrap TCP.
  return uint16_t(20000 + salt % 10000);
}
class HostControl {
  using Clock = std::chrono::steady_clock;
  struct Fd {
    int value = -1;
    Fd() = default;
    explicit Fd(int fd) : value(fd) {}
    Fd(Fd const&) = delete;
    Fd& operator=(Fd const&) = delete;
    Fd(Fd&& other) noexcept : value(other.value) { other.value = -1; }
    Fd& operator=(Fd&& other) noexcept {
      if (value >= 0) close(value);
      value = other.value;
      other.value = -1;
      return *this;
    }
    ~Fd() {
      if (value >= 0) close(value);
    }
  };
  static constexpr uint64_t magic = 0x5543434c4354524cULL;
  static constexpr uint64_t acknowledgement = ~magic;
  static constexpr unsigned portAttempts = 64;
  struct Hello {
    uint64_t magic;
    uint32_t version, ranks, rank, reserved;
    std::array<unsigned char, 128> session;
  };
  struct Frame {
    uint64_t magic, sequence, operation, bytes;
  };
  int rank_, ranks_, timeoutMs_;
  uint64_t sequence_ = 0;
  std::vector<Fd> peers_;
  Clock::time_point deadline() const {
    return Clock::now() + std::chrono::milliseconds(timeoutMs_);
  }
  [[noreturn]] static void fail(char const* message) {
    throw std::runtime_error(std::string("CPU control: ") + message);
  }
  static void ready(int fd, short events, Clock::time_point end,
                    bool connecting = false) {
    for (;;) {
      auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end - Clock::now())
                      .count();
      if (left <= 0) fail("timed out");
      pollfd p{fd, events, 0};
      int rc = poll(&p, 1, int(std::min<int64_t>(left, 30000)));
      if (rc < 0 && errno == EINTR) continue;
      if (rc < 0) fail("poll failed");
      if (!rc) continue;
      if (p.revents & events) return;
      if (connecting && (p.revents & (POLLERR | POLLHUP))) return;
      if (p.revents & (POLLERR | POLLHUP | POLLNVAL))
        fail("connection closed or failed");
    }
  }
  static void transfer(int fd, void* data, size_t bytes, bool send,
                       Clock::time_point end) {
    auto* p = static_cast<unsigned char*>(data);
    while (bytes) {
      ready(fd, send ? POLLOUT : POLLIN, end);
      auto n =
          send ? ::send(fd, p, bytes, MSG_NOSIGNAL) : ::recv(fd, p, bytes, 0);
      if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
        continue;
      if (n <= 0) fail("send/receive failed");
      p += n;
      bytes -= size_t(n);
    }
  }
  static Fd socket() {
    Fd fd(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    if (fd.value < 0) fail("socket failed");
    int one = 1;
    if (setsockopt(fd.value, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)))
      fail("TCP_NODELAY failed");
    return fd;
  }
  void frame(int fd, uint64_t op, size_t bytes, bool send,
             Clock::time_point end) {
    Frame f{magic, sequence_, op, bytes};
    transfer(fd, &f, sizeof(f), send, end);
    if (!send && (f.magic != magic || f.sequence != sequence_ ||
                  f.operation != op || f.bytes != bytes))
      fail("collective framing/order mismatch");
  }
  void extent(size_t bytes) const {
    if (!bytes || bytes > (1UL << 20) / size_t(ranks_))
      fail("invalid control payload extent");
  }

 public:
  HostControl(char const* root, void const* session, size_t sessionBytes,
              int rank, int ranks, int timeoutMs = 30000)
      : rank_(rank),
        ranks_(ranks),
        timeoutMs_(timeoutMs),
        peers_(ranks >= 2 && ranks <= 8 ? ranks : 0) {
    if (sessionBytes != 128 || !session || ranks < 2 || ranks > 8 || rank < 0 ||
        rank >= ranks || timeoutMs <= 0)
      fail("invalid configuration");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    if (!root || inet_pton(AF_INET, root, &address.sin_addr) != 1)
      fail("CONTROL_ROOT must be a common root IPv4 address");
    uint16_t firstPort = hostControlInitialPort(session, sessionBytes);
    auto choosePort = [&](unsigned attempt) {
      address.sin_port = htons(
          uint16_t(20000 + (unsigned(firstPort) - 20000 + attempt) % 10000));
    };
    Hello hello{magic, 2, uint32_t(ranks), uint32_t(rank), 0, {}};
    std::memcpy(hello.session.data(), session, sessionBytes);
    auto end = deadline();
    if (rank == 0) {
      Fd listener;
      int one = 1;
      for (unsigned attempt = 0; attempt < portAttempts; ++attempt) {
        auto candidate = socket();
        if (setsockopt(candidate.value, SOL_SOCKET, SO_REUSEADDR, &one,
                       sizeof(one)))
          fail("root socket options failed");
        choosePort(attempt);
        if (!bind(candidate.value, reinterpret_cast<sockaddr*>(&address),
                  sizeof(address))) {
          if (listen(candidate.value, ranks)) fail("root listen failed");
          listener = std::move(candidate);
          break;
        }
        if (errno != EADDRINUSE) fail("root bind failed");
      }
      if (listener.value < 0) fail("control session ports unavailable");
      for (int accepted = 0; accepted < ranks - 1;) {
        ready(listener.value, POLLIN, end);
        Fd fd(accept4(listener.value, nullptr, nullptr,
                      SOCK_CLOEXEC | SOCK_NONBLOCK));
        if (fd.value < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (fd.value < 0) fail("accept failed");
        if (setsockopt(fd.value, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)))
          fail("accepted TCP_NODELAY failed");
        Hello peer{};
        transfer(fd.value, &peer, sizeof(peer), false, end);
        // A port collision must never mix communicator sessions. Reject the
        // unrelated connection; its root can try the next session port.
        if (peer.magic != magic || peer.version != 2 ||
            peer.ranks != uint32_t(ranks) || peer.rank == 0 ||
            peer.rank >= uint32_t(ranks) || peer.session != hello.session)
          continue;
        if (peers_[peer.rank].value >= 0) fail("duplicate control rank");
        uint64_t ack = acknowledgement;
        transfer(fd.value, &ack, sizeof(ack), true, end);
        peers_[peer.rank] = std::move(fd);
        ++accepted;
      }
    } else {
      for (unsigned attempt = 0;; ++attempt) {
        auto fd = socket();
        choosePort(attempt % portAttempts);
        auto handshakeEnd =
            std::min(end, Clock::now() + std::chrono::milliseconds(250));
        try {
          int rc = connect(fd.value, reinterpret_cast<sockaddr*>(&address),
                           sizeof(address));
          int error = rc == 0 ? 0 : errno;
          if (rc < 0 && error == EINPROGRESS) {
            ready(fd.value, POLLOUT, handshakeEnd, true);
            socklen_t length = sizeof(error);
            if (getsockopt(fd.value, SOL_SOCKET, SO_ERROR, &error, &length))
              fail("connect status failed");
          }
          if (!error) {
            transfer(fd.value, &hello, sizeof(hello), true, handshakeEnd);
            uint64_t ack = 0;
            transfer(fd.value, &ack, sizeof(ack), false, handshakeEnd);
            // A Linux TCP self-connect must not accept our own Hello magic.
            if (ack == acknowledgement) {
              peers_[0] = std::move(fd);
              break;
            }
          }
        } catch (std::runtime_error const&) {
          // A port may belong to another session/service. Its EOF, reply or
          // short handshake timeout is not proof this session's root failed.
        }
        if (Clock::now() >= end) fail("root connect/handshake timed out");
        if (attempt % portAttempts == portAttempts - 1)
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    }
  }
  HostControl(HostControl const&) = delete;
  HostControl& operator=(HostControl const&) = delete;
  void allGather(void const* local, void* all, size_t bytes) {
    extent(bytes);
    auto end = deadline();
    if (rank_ == 0) {
      std::memmove(all, local, bytes);
      for (int r = 1; r < ranks_; ++r) {
        frame(peers_[r].value, 1, bytes, false, end);
        transfer(peers_[r].value,
                 static_cast<unsigned char*>(all) + size_t(r) * bytes, bytes,
                 false, end);
      }
      for (int r = 1; r < ranks_; ++r) {
        frame(peers_[r].value, 1, bytes * ranks_, true, end);
        transfer(peers_[r].value, all, bytes * ranks_, true, end);
      }
    } else {
      frame(peers_[0].value, 1, bytes, true, end);
      transfer(peers_[0].value, const_cast<void*>(local), bytes, true, end);
      frame(peers_[0].value, 1, bytes * ranks_, false, end);
      transfer(peers_[0].value, all, bytes * ranks_, false, end);
    }
    ++sequence_;
  }
  void broadcast(void* data, size_t bytes) {
    extent(bytes);
    auto end = deadline();
    if (rank_ == 0) {
      for (int r = 1; r < ranks_; ++r) {
        frame(peers_[r].value, 2, bytes, true, end);
        transfer(peers_[r].value, data, bytes, true, end);
      }
      for (int r = 1; r < ranks_; ++r) frame(peers_[r].value, 3, 0, false, end);
    } else {
      frame(peers_[0].value, 2, bytes, false, end);
      transfer(peers_[0].value, data, bytes, false, end);
      frame(peers_[0].value, 3, 0, true, end);
    }
    ++sequence_;
  }
};
}  // namespace uccl_lite::nccl_detail
