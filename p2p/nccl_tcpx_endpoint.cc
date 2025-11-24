#include "nccl_tcpx_endpoint.h"
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <cuda_runtime_api.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

namespace nccl_tcpx {
namespace {
// Control-plane magic/version. Exchange uid and GPU index over TCP before
// initializing ncclComm.
constexpr uint32_t kMagic = 0x4e43434c;  // "NCCL"
constexpr uint32_t kVersion = 1;

struct ClientHello {
  uint32_t magic;
  uint32_t version;
  int gpu_idx;
};

struct ServerHello {
  uint32_t magic;
  uint32_t version;
  int gpu_idx;
  ncclUniqueId uid;
};

int get_env_int(char const* key, int def) {
  char const* v = std::getenv(key);
  if (!v || !*v) return def;
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v) return def;
  return static_cast<int>(parsed);
}

std::string get_local_ip() {
  char const* env_ip = std::getenv("UCCL_TCPX_CTRL_IP");
  if (env_ip && *env_ip) return std::string(env_ip);

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) == 0) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(hostname, nullptr, &hints, &res) == 0) {
      for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        sockaddr_in* addr = reinterpret_cast<sockaddr_in*>(p->ai_addr);
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &(addr->sin_addr), buf, sizeof(buf))) {
          freeaddrinfo(res);
          return std::string(buf);
        }
      }
      freeaddrinfo(res);
    }
  }
  return "127.0.0.1";
}
}  // namespace

Endpoint::Endpoint(int /*num_cpus*/) {
  local_gpu_idx_ =
      get_env_int("UCCL_NCCL_DEVICE", get_env_int("UCCL_TCPX_LOCAL_DEVICE", 0));
  ctrl_port_ = get_env_int("UCCL_TCPX_OOB_PORT", 28900);
  setup_listener_();
}

Endpoint::~Endpoint() {
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    for (auto& kv : conn_map_) {
      if (kv.second.stream) cudaStreamDestroy(kv.second.stream);
      if (kv.second.comm) ncclCommDestroy(kv.second.comm);
      if (kv.second.sock_fd >= 0) ::close(kv.second.sock_fd);
    }
    conn_map_.clear();
  }
  if (ctrl_listen_fd_ >= 0) {
    ::close(ctrl_listen_fd_);
    ctrl_listen_fd_ = -1;
  }
}

bool Endpoint::setup_listener_() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return false;

  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(ctrl_port_));
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::cerr << "[nccl-tcpx] bind failed: " << strerror(errno)
              << " port=" << ctrl_port_ << std::endl;
    ::close(fd);
    return false;
  }
  if (listen(fd, 8) < 0) {
    std::cerr << "[nccl-tcpx] listen failed: " << strerror(errno) << std::endl;
    ::close(fd);
    return false;
  }
  ctrl_listen_fd_ = fd;
  return true;
}

bool Endpoint::send_all_(int fd, void const* buf, size_t len) {
  char const* p = static_cast<char const*>(buf);
  size_t sent = 0;
  while (sent < len) {
    ssize_t rc = ::send(fd, p + sent, len - sent, 0);
    if (rc <= 0) return false;
    sent += static_cast<size_t>(rc);
  }
  return true;
}

bool Endpoint::recv_all_(int fd, void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  size_t recvd = 0;
  while (recvd < len) {
    ssize_t rc = ::recv(fd, p + recvd, len - recvd, 0);
    if (rc <= 0) return false;
    recvd += static_cast<size_t>(rc);
  }
  return true;
}

bool Endpoint::init_comm_(Conn& conn, ncclUniqueId const& uid, int rank) {
  // Encapsulate NCCL control/data lifetime per connection so uccl_engine does
  // not need extra state.
  if (cudaSetDevice(conn.local_gpu_idx) != cudaSuccess) return false;
  if (cudaStreamCreateWithFlags(&conn.stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    return false;
  }
  ncclResult_t rc = ncclCommInitRank(&conn.comm, 2, uid, rank);
  if (rc != ncclSuccess) {
    std::cerr << "[nccl-tcpx] ncclCommInitRank failed: "
              << ncclGetErrorString(rc) << std::endl;
    return false;
  }
  return true;
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
                       uint64_t& conn_id) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return false;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(remote_port));
  if (inet_pton(AF_INET, ip_addr.c_str(), &addr.sin_addr) <= 0) {
    ::close(fd);
    return false;
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return false;
  }

  ClientHello ch{
      .magic = kMagic, .version = kVersion, .gpu_idx = local_gpu_idx_};
  if (!send_all_(fd, &ch, sizeof(ch))) {
    ::close(fd);
    return false;
  }

  ServerHello sh{};
  if (!recv_all_(fd, &sh, sizeof(sh))) {
    ::close(fd);
    return false;
  }
  if (sh.magic != kMagic) {
    ::close(fd);
    return false;
  }

  Conn conn{};
  conn.sock_fd = fd;
  conn.rank = 1;
  conn.remote_rank = 0;
  conn.local_gpu_idx = local_gpu_idx_;
  conn.remote_gpu_idx = remote_gpu_idx;

  if (!init_comm_(conn, sh.uid, conn.rank)) {
    ::close(fd);
    return false;
  }

  conn_id = next_conn_id_.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    conn_map_.emplace(conn_id, conn);
  }
  return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  if (ctrl_listen_fd_ < 0) return false;
  sockaddr_in cli_addr{};
  socklen_t len = sizeof(cli_addr);
  int fd =
      ::accept(ctrl_listen_fd_, reinterpret_cast<sockaddr*>(&cli_addr), &len);
  if (fd < 0) return false;
  char ip_buf[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &cli_addr.sin_addr, ip_buf, sizeof(ip_buf));
  ip_addr = std::string(ip_buf);

  ClientHello ch{};
  if (!recv_all_(fd, &ch, sizeof(ch))) {
    ::close(fd);
    return false;
  }
  if (ch.magic != kMagic) {
    ::close(fd);
    return false;
  }
  remote_gpu_idx = ch.gpu_idx;

  ncclUniqueId uid;
  ncclGetUniqueId(&uid);

  ServerHello sh{};
  sh.magic = kMagic;
  sh.version = kVersion;
  sh.gpu_idx = local_gpu_idx_;
  sh.uid = uid;
  if (!send_all_(fd, &sh, sizeof(sh))) {
    ::close(fd);
    return false;
  }

  Conn conn{};
  conn.sock_fd = fd;
  conn.rank = 0;
  conn.remote_rank = 1;
  conn.local_gpu_idx = local_gpu_idx_;
  conn.remote_gpu_idx = remote_gpu_idx;

  if (!init_comm_(conn, uid, conn.rank)) {
    ::close(fd);
    return false;
  }

  conn_id = next_conn_id_.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    conn_map_.emplace(conn_id, conn);
  }
  return true;
}

int Endpoint::get_sock_fd(uint64_t conn_id) const {
  std::lock_guard<std::mutex> lock(conn_mu_);
  auto it = conn_map_.find(conn_id);
  if (it == conn_map_.end()) return -1;
  return it->second.sock_fd;
}

std::vector<uint8_t> Endpoint::get_unified_metadata() {
  std::vector<uint8_t> meta(10);
  std::string ip = get_local_ip();
  in_addr ipv4{};
  inet_pton(AF_INET, ip.c_str(), &ipv4);
  std::memcpy(meta.data(), &ipv4, sizeof(ipv4));
  uint16_t port_be = htons(static_cast<uint16_t>(ctrl_port_));
  meta[4] = static_cast<uint8_t>((port_be >> 8) & 0xFF);
  meta[5] = static_cast<uint8_t>(port_be & 0xFF);
  std::memcpy(meta.data() + 6, &local_gpu_idx_, sizeof(int));
  return meta;
}

std::tuple<std::string, uint16_t, int> Endpoint::parse_metadata(
    std::vector<uint8_t> const& metadata) {
  if (metadata.size() != 10) return std::make_tuple("", 0, 0);
  std::string ip =
      std::to_string(metadata[0]) + "." + std::to_string(metadata[1]) + "." +
      std::to_string(metadata[2]) + "." + std::to_string(metadata[3]);
  uint16_t net_port =
      static_cast<uint16_t>((static_cast<uint16_t>(metadata[4]) << 8) |
                            static_cast<uint16_t>(metadata[5]));
  uint16_t port = ntohs(net_port);
  int gpu_idx = 0;
  std::memcpy(&gpu_idx, metadata.data() + 6, sizeof(int));
  return std::make_tuple(ip, port, gpu_idx);
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  if (!data || size == 0) return false;
  mr_id = next_mr_id_.fetch_add(1);
  std::lock_guard<std::mutex> lock(mr_mu_);
  mr_map_[mr_id] = MrEntry{const_cast<void*>(data), size};
  return true;
}

bool Endpoint::dereg(uint64_t mr_id) {
  std::lock_guard<std::mutex> lock(mr_mu_);
  auto it = mr_map_.find(mr_id);
  if (it == mr_map_.end()) return false;
  mr_map_.erase(it);
  return true;
}

bool Endpoint::advertise(uint64_t /*conn_id*/, uint64_t mr_id, void const* addr,
                         size_t len, void* out_buf) {
  // Preserve the TCPX-style advertise API: return a 64B FIFO descriptor; tag is
  // always 0.
  MrEntry mr{};
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }
  uintptr_t base = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t addr_u = reinterpret_cast<uintptr_t>(addr);
  if (addr_u < base || addr_u + len > base + mr.size) return false;

  FifoItem item{};
  item.mr_id = mr_id;
  item.size = static_cast<uint32_t>(len);
  item.offset = static_cast<uint64_t>(addr_u - base);
  item.tag = 0;
  item.token = 0;
  std::memcpy(out_buf, &item, sizeof(item));
  return true;
}

bool Endpoint::send_internal_(Conn& conn, void const* data, size_t size,
                              uint64_t& transfer_id) {
  // Actual data-plane call: ncclSend plus a cudaEvent on the stream for
  // poll_async to poll.
  if (cudaSetDevice(conn.local_gpu_idx) != cudaSuccess) return false;
  ncclResult_t rc =
      ncclSend(data, size, ncclChar, conn.remote_rank, conn.comm, conn.stream);
  if (rc != ncclSuccess) {
    std::cerr << "[nccl-tcpx] ncclSend failed: " << ncclGetErrorString(rc)
              << std::endl;
    return false;
  }

  cudaEvent_t evt = nullptr;
  if (cudaEventCreateWithFlags(&evt, cudaEventDisableTiming) != cudaSuccess)
    return false;
  if (cudaEventRecord(evt, conn.stream) != cudaSuccess) {
    cudaEventDestroy(evt);
    return false;
  }

  transfer_id = next_transfer_id_.fetch_add(1);
  std::lock_guard<std::mutex> lock(transfer_mu_);
  transfer_map_[transfer_id] = Transfer{.event = evt};
  return true;
}

bool Endpoint::recv_internal_(Conn& conn, void* data, size_t size,
                              uint64_t& transfer_id) {
  // Symmetric receive path, also using an event to mark completion.
  if (cudaSetDevice(conn.local_gpu_idx) != cudaSuccess) return false;
  ncclResult_t rc =
      ncclRecv(data, size, ncclChar, conn.remote_rank, conn.comm, conn.stream);
  if (rc != ncclSuccess) {
    std::cerr << "[nccl-tcpx] ncclRecv failed: " << ncclGetErrorString(rc)
              << std::endl;
    return false;
  }

  cudaEvent_t evt = nullptr;
  if (cudaEventCreateWithFlags(&evt, cudaEventDisableTiming) != cudaSuccess)
    return false;
  if (cudaEventRecord(evt, conn.stream) != cudaSuccess) {
    cudaEventDestroy(evt);
    return false;
  }

  transfer_id = next_transfer_id_.fetch_add(1);
  std::lock_guard<std::mutex> lock(transfer_mu_);
  transfer_map_[transfer_id] = Transfer{.event = evt};
  return true;
}

bool Endpoint::send_async(uint64_t conn_id, uint64_t /*mr_id*/,
                          void const* data, size_t size,
                          uint64_t* transfer_id) {
  if (!data || size == 0 || !transfer_id) return false;
  Conn conn{};
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn = it->second;
  }
  return send_internal_(conn, data, size, *transfer_id);
}

bool Endpoint::recv_async(uint64_t conn_id, uint64_t /*mr_id*/, void* data,
                          size_t size, uint64_t* transfer_id) {
  if (!data || size == 0 || !transfer_id) return false;
  Conn conn{};
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn = it->second;
  }
  return recv_internal_(conn, data, size, *transfer_id);
}

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst,
                          size_t size, FifoItem const& slot_item,
                          uint64_t* transfer_id) {
  // Compatibility read_async entry: in the NCCL path we ignore tag and receive
  // min(slot_item.size, size).
  size_t recv_size = size;
  if (slot_item.size > 0 && slot_item.size < recv_size) {
    recv_size = slot_item.size;
  }
  return recv_async(conn_id, mr_id, dst, recv_size, transfer_id);
}

bool Endpoint::queue_read_response(uint64_t conn_id,
                                   FifoItem const& fifo_item) {
  // FIFO callback directly issues an ncclSend; no bounce buffer, just keep the
  // API shape.
  Conn conn{};
  {
    std::lock_guard<std::mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn = it->second;
  }

  MrEntry mr{};
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(fifo_item.mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }
  if (fifo_item.offset + fifo_item.size > mr.size) return false;
  char* base = static_cast<char*>(mr.base) + fifo_item.offset;

  uint64_t tid = 0;
  if (!send_internal_(conn, base, fifo_item.size, tid)) return false;

  // Wait for the send to complete so we do not leak events in transfer_map_.
  bool done = false;
  while (!done) {
    if (!poll_async(tid, &done)) return false;
    if (!done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  return true;
}

bool Endpoint::recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size) {
  uint64_t tid = 0;
  if (!recv_async(conn_id, mr_id, data, size, &tid)) return false;
  bool done = false;
  while (!done) {
    if (!poll_async(tid, &done)) return false;
    if (!done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  if (!is_done) return false;
  Transfer tr{};
  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    auto it = transfer_map_.find(transfer_id);
    if (it == transfer_map_.end()) {
      *is_done = true;
      return true;
    }
    tr = it->second;
  }
  cudaError_t rc = cudaEventQuery(tr.event);
  if (rc == cudaSuccess) {
    cudaEventDestroy(tr.event);
    std::lock_guard<std::mutex> lock(transfer_mu_);
    transfer_map_.erase(transfer_id);
    *is_done = true;
    return true;
  }
  if (rc == cudaErrorNotReady) {
    *is_done = false;
    return true;
  }
  return false;
}

}  // namespace nccl_tcpx
