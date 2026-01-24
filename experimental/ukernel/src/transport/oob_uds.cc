#include "oob.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <errno.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {
constexpr uint32_t kHelloMagic = 0xC0DEF00D;
constexpr uint32_t kMsgMagic = 0x55445331;  // "UDS1"
constexpr uint16_t kVersion = 1;

static inline std::string dirname_of(std::string const& path) {
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos) return {};
  return path.substr(0, pos);
}

static inline void mkdir_best_effort(std::string const& dir) {
  if (dir.empty()) return;
  ::mkdir(dir.c_str(), 0700);
}

}  // namespace

UdsExchanger::UdsExchanger(int self_rank) : self_rank_(self_rank) {}

UdsExchanger::~UdsExchanger() {
  running_.store(false, std::memory_order_relaxed);

  // close peers
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& kv : rank_to_fd_) {
      ::shutdown(kv.second, SHUT_RDWR);
      ::close(kv.second);
    }
    rank_to_fd_.clear();
    rank_send_mu_.clear();
  }

  // close listen
  if (listen_fd_ != -1) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }

  // unlink sock path
  if (!local_path_.empty()) {
    ::unlink(local_path_.c_str());
  }
}

bool UdsExchanger::ensure_server_started() {
  if (running_.load(std::memory_order_relaxed)) return true;

  std::lock_guard<std::mutex> lk(mu_);
  if (running_.load(std::memory_order_relaxed)) return true;

  local_path_ = path_for_rank(self_rank_);
  mkdir_best_effort(dirname_of(local_path_));
  ::unlink(local_path_.c_str());

  listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    std::cerr << "[UDS] socket() failed: " << std::strerror(errno) << "\n";
    listen_fd_ = -1;
    return false;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s",
                local_path_.c_str());

  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
      0) {
    std::cerr << "[UDS] bind(" << local_path_
              << ") failed: " << std::strerror(errno) << "\n";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  if (::listen(listen_fd_, 128) < 0) {
    std::cerr << "[UDS] listen() failed: " << std::strerror(errno) << "\n";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  running_.store(true, std::memory_order_relaxed);

  std::cout << "[UDS] listen() at: " << local_path_ << std::endl;
  return true;
}

bool UdsExchanger::connect_to(int peer_rank, int timeout_ms) {
  if (peer_rank == self_rank_) return true;
  if (!ensure_server_started()) return false;

  {
    std::lock_guard<std::mutex> lk(mu_);
    if (rank_to_fd_.count(peer_rank)) return true;
  }

  const std::string peer_path = path_for_rank(peer_rank);
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

  while (std::chrono::steady_clock::now() < deadline) {
    int fd = -1;
    if (connect_once(peer_path, fd)) {
      Hello h{kHelloMagic, self_rank_, peer_rank, kVersion};
      if (!send_all(fd, reinterpret_cast<char const*>(&h), sizeof(h))) {
        ::close(fd);
        return false;
      }

      {
        std::lock_guard<std::mutex> lk(mu_);
        // In case another thread already connected
        if (rank_to_fd_.count(peer_rank)) {
          ::shutdown(fd, SHUT_RDWR);
          ::close(fd);
          return true;
        }
        rank_to_fd_[peer_rank] = fd;
        rank_send_mu_[peer_rank] = std::make_unique<std::mutex>();
        rank_recv_mu_[peer_rank] = std::make_unique<std::mutex>();
      }
      return true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  return false;
}

bool UdsExchanger::accept_from(int peer_rank, int timeout_ms) {
  if (peer_rank == self_rank_) return true;
  if (!ensure_server_started()) return false;

  {
    std::lock_guard<std::mutex> lk(mu_);
    if (rank_to_fd_.count(peer_rank)) return true;
  }

  // Ensure only one accept loop at a time
  std::unique_lock<std::mutex> accept_lk(accept_mu_);

  // re-check after acquiring lock
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (rank_to_fd_.count(peer_rank)) return true;
  }

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

  while (std::chrono::steady_clock::now() < deadline) {
    int remaining_ms =
        (int)std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now())
            .count();
    if (remaining_ms <= 0) break;

    int fd = accept_with_timeout(remaining_ms);
    if (fd == -1) {  // timeout
      break;
    }
    if (fd == -2) {  // fatal
      return false;
    }

    // read hello
    Hello h{};
    if (!recv_all(fd, reinterpret_cast<char*>(&h), sizeof(h))) {
      ::close(fd);
      continue;
    }
    if (h.magic != kHelloMagic || h.version != kVersion ||
        h.to_rank != self_rank_) {
      ::close(fd);
      continue;
    }

    // cache connection (even if not the expected peer)
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = rank_to_fd_.find(h.from_rank);
      if (it != rank_to_fd_.end()) {
        // already have; keep existing
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
      } else {
        rank_to_fd_[h.from_rank] = fd;
        rank_send_mu_[h.from_rank] = std::make_unique<std::mutex>();
        rank_recv_mu_[h.from_rank] = std::make_unique<std::mutex>();
      }
    }

    // done if expected is present
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (rank_to_fd_.count(peer_rank)) return true;
    }
  }

  return false;
}

bool UdsExchanger::send(int peer_rank, uint16_t type, uint64_t seq,
                        void const* payload, uint32_t bytes) {
  int fd = -1;
  std::mutex* smu = nullptr;

  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = rank_to_fd_.find(peer_rank);
    if (it == rank_to_fd_.end()) return false;
    fd = it->second;

    auto itmu = rank_send_mu_.find(peer_rank);
    if (itmu == rank_send_mu_.end() || !itmu->second) return false;
    smu = itmu->second.get();
  }

  MsgHdr hdr{};
  hdr.magic = kMsgMagic;
  hdr.version = kVersion;
  hdr.type = type;
  hdr.bytes = bytes;
  hdr.from_rank = self_rank_;
  hdr.to_rank = peer_rank;
  hdr.seq = seq;

  std::lock_guard<std::mutex> lk(*smu);
  if (!send_all(fd, reinterpret_cast<char const*>(&hdr), sizeof(hdr)))
    return false;
  if (bytes > 0 && payload != nullptr) {
    if (!send_all(fd, reinterpret_cast<char const*>(payload), bytes))
      return false;
  }
  return true;
}

bool UdsExchanger::send_ipc_cache(int peer_rank, uint64_t seq,
                                  IpcCacheWire const& cache) {
  return send(peer_rank, kTypeIpcCache, seq, &cache,
              (uint32_t)sizeof(IpcCacheWire));
}

bool UdsExchanger::recv_ipc_cache(int peer_rank, IpcCacheWire& out_cache,
                                  uint64_t* out_seq, int timeout_ms) {
  int fd = -1;
  std::mutex* rmu = nullptr;

  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = rank_to_fd_.find(peer_rank);
    if (it == rank_to_fd_.end()) return false;
    fd = it->second;

    auto itmu = rank_recv_mu_.find(peer_rank);
    if (itmu == rank_recv_mu_.end() || !itmu->second) return false;
    rmu = itmu->second.get();
  }

  // Serialize all recv operations for this peer.
  std::unique_lock<std::mutex> rlk(*rmu);

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);

  auto drain_bytes = [&](uint32_t nbytes) -> bool {
    // Drain without allocating huge buffers.
    char buf[4096];
    uint32_t left = nbytes;
    while (left > 0) {
      uint32_t chunk = left > sizeof(buf) ? (uint32_t)sizeof(buf) : left;
      if (!recv_all(fd, buf, chunk)) return false;
      left -= chunk;
    }
    return true;
  };

  while (true) {
    // wait for readable with timeout
    int wait_ms = 0;
    if (timeout_ms < 0) {
      wait_ms = 1000;  // "forever" mode, 1s polling chunks
    } else {
      auto now = std::chrono::steady_clock::now();
      if (now >= deadline) return false;
      wait_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - now)
                    .count();
      if (wait_ms <= 0) return false;
    }

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);

    timeval tv{};
    tv.tv_sec = wait_ms / 1000;
    tv.tv_usec = (wait_ms % 1000) * 1000;

    int r = ::select(fd + 1, &rfds, nullptr, nullptr, &tv);
    if (r == 0) {
      if (timeout_ms < 0) continue;  // keep waiting
      return false;                  // timeout
    }
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }

    // read header
    MsgHdr hdr{};
    if (!recv_all(fd, reinterpret_cast<char*>(&hdr), sizeof(hdr))) {
      return false;
    }

    // Only accept IPC cache messages; otherwise drain and continue.
    if (hdr.type != kTypeIpcCache) {
      if (hdr.bytes > 0 && !drain_bytes(hdr.bytes)) return false;
      continue;
    }

    if (hdr.bytes != sizeof(IpcCacheWire)) {
      if (hdr.bytes > 0 && !drain_bytes(hdr.bytes)) return false;
      continue;
    }

    // read payload
    if (!recv_all(fd, reinterpret_cast<char*>(&out_cache), sizeof(out_cache))) {
      return false;
    }

    if (out_seq) *out_seq = hdr.seq;
    return true;
  }
}

bool UdsExchanger::send_ack(int peer_rank, uint64_t seq, uint32_t status) {
  AckWire ack{};
  ack.status = status;
  ack.reserved = 0;
  return send(peer_rank, kTypeAck, seq, &ack, (uint32_t)sizeof(AckWire));
}

bool UdsExchanger::recv_ack(int peer_rank, uint32_t* out_status,
                            uint64_t* out_seq, int timeout_ms,
                            uint64_t expected_seq) {
  int fd = -1;
  std::mutex* rmu = nullptr;

  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = rank_to_fd_.find(peer_rank);
    if (it == rank_to_fd_.end()) return false;
    fd = it->second;

    auto itmu = rank_recv_mu_.find(peer_rank);
    if (itmu == rank_recv_mu_.end() || !itmu->second) return false;
    rmu = itmu->second.get();
  }

  // Serialize all recv operations for this peer.
  std::unique_lock<std::mutex> rlk(*rmu);

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);

  auto drain_bytes = [&](uint32_t nbytes) -> bool {
    char buf[4096];
    uint32_t left = nbytes;
    while (left > 0) {
      uint32_t chunk = left > sizeof(buf) ? (uint32_t)sizeof(buf) : left;
      if (!recv_all(fd, buf, chunk)) return false;
      left -= chunk;
    }
    return true;
  };

  while (true) {
    // wait readable with timeout
    int wait_ms = 0;
    if (timeout_ms < 0) {
      wait_ms = 1000;  // forever mode
    } else {
      auto now = std::chrono::steady_clock::now();
      if (now >= deadline) return false;
      wait_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - now)
                    .count();
      if (wait_ms <= 0) return false;
    }

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);

    timeval tv{};
    tv.tv_sec = wait_ms / 1000;
    tv.tv_usec = (wait_ms % 1000) * 1000;

    int r = ::select(fd + 1, &rfds, nullptr, nullptr, &tv);
    if (r == 0) {
      if (timeout_ms < 0) continue;
      return false;  // timeout
    }
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }

    // read header
    MsgHdr hdr{};
    if (!recv_all(fd, reinterpret_cast<char*>(&hdr), sizeof(hdr))) return false;

    // If not ack, drain and continue
    if (hdr.type != kTypeAck) {
      if (hdr.bytes > 0 && !drain_bytes(hdr.bytes)) return false;
      continue;
    }

    // optional seq check
    if (expected_seq != UINT64_MAX && hdr.seq != expected_seq) {
      if (hdr.bytes > 0 && !drain_bytes(hdr.bytes)) return false;
      continue;
    }

    // validate payload size
    if (hdr.bytes != sizeof(AckWire)) {
      if (hdr.bytes > 0 && !drain_bytes(hdr.bytes)) return false;
      continue;
    }

    AckWire ack{};
    if (!recv_all(fd, reinterpret_cast<char*>(&ack), sizeof(ack))) return false;

    if (out_status) *out_status = ack.status;
    if (out_seq) *out_seq = hdr.seq;
    return true;
  }
}

int UdsExchanger::get_fd(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = rank_to_fd_.find(peer_rank);
  return it == rank_to_fd_.end() ? -1 : it->second;
}

void UdsExchanger::close_peer(int peer_rank) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = rank_to_fd_.find(peer_rank);
  if (it == rank_to_fd_.end()) return;
  int fd = it->second;
  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);
  rank_to_fd_.erase(it);
  rank_send_mu_.erase(peer_rank);
  rank_recv_mu_.erase(peer_rank);
}

bool UdsExchanger::connect_once(std::string const& peer_path, int& out_fd) {
  out_fd = -1;
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return false;

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", peer_path.c_str());

  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    out_fd = fd;
    return true;
  }

  ::close(fd);
  return false;
}

bool UdsExchanger::send_all(int fd, char const* buf, size_t len) {
  size_t off = 0;
  while (off < len) {
    ssize_t n = ::send(fd, buf + off, len - off, 0);
    if (n > 0) {
      off += (size_t)n;
      continue;
    }
    if (n == 0) return false;
    if (errno == EINTR) continue;
    return false;
  }
  return true;
}

bool UdsExchanger::recv_all(int fd, char* buf, size_t len) {
  size_t off = 0;
  while (off < len) {
    ssize_t n = ::recv(fd, buf + off, len - off, 0);
    if (n > 0) {
      off += (size_t)n;
      continue;
    }
    if (n == 0) return false;
    if (errno == EINTR) continue;
    return false;
  }
  return true;
}

int UdsExchanger::accept_with_timeout(int timeout_ms) {
  if (listen_fd_ < 0) return -2;

  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(listen_fd_, &rfds);

  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;

  int r = ::select(listen_fd_ + 1, &rfds, nullptr, nullptr, &tv);
  if (r == 0) return -1;  // timeout
  if (r < 0) {
    if (errno == EINTR) return -1;
    return -2;
  }

  int fd = ::accept(listen_fd_, nullptr, nullptr);
  if (fd < 0) {
    if (errno == EINTR) return -1;
    return -2;
  }
  return fd;
}

std::string UdsExchanger::path_for_rank(int rank) {
  // Keep path short: /tmp/ukernel_oob_uds/r<rank>.sock
  char buf[128];
  std::snprintf(buf, sizeof(buf), "/tmp/ukernel_oob_uds/r%d.sock", rank);

  std::string path(buf);
  mkdir_best_effort(dirname_of(path));
  return path;
}

}  // namespace Transport
}  // namespace UKernel