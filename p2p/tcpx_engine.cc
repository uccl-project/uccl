#include "tcpx_engine.h"
#include "tcpx/include/bootstrap.h"
#include "tcpx/include/unpack_descriptor.h"
#include <algorithm>
#include <array>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

thread_local bool inside_python = false;

namespace {

constexpr int kDefaultOobPort = 28900;
constexpr int kCtrlBacklog = 128;
constexpr size_t kDefaultChunkBytes = 512 * 1024;

// Payloads exchanged on the lightweight TCP control plane.
struct EndpointInfo {
  char ip[INET_ADDRSTRLEN];
  uint16_t port;
  int gpu;
  int reserved;
};
static_assert(std::is_trivially_copyable<EndpointInfo>::value,
              "EndpointInfo must be trivially copyable");

enum CtrlMsgType : uint16_t {
  CTRL_ACK = 0x01,
  CTRL_STRUCT = 0x02,
};

struct CtrlMsgHeader {
  uint16_t type;
  uint16_t flags;
  uint32_t length;
};

bool send_all(int fd, void const* buf, size_t len) {
  char const* ptr = static_cast<char const*>(buf);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, ptr + sent, len - sent, 0);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool recv_all(int fd, void* buf, size_t len) {
  char* ptr = static_cast<char*>(buf);
  size_t received = 0;
  while (received < len) {
    ssize_t n = ::recv(fd, ptr + received, len - received, 0);
    if (n <= 0) return false;
    received += static_cast<size_t>(n);
  }
  return true;
}

bool send_ctrl_header(int fd, CtrlMsgType type, size_t payload_len) {
  CtrlMsgHeader hdr{static_cast<uint16_t>(type), 0,
                    static_cast<uint32_t>(payload_len)};
  return send_all(fd, &hdr, sizeof(hdr));
}

bool recv_ctrl_header(int fd, CtrlMsgHeader& hdr) {
  if (!recv_all(fd, &hdr, sizeof(hdr))) return false;
  return true;
}

template <typename T>
bool send_ctrl_struct(int fd, T const& pod) {
  static_assert(std::is_trivially_copyable<T>::value, "payload must be POD");
  if (!send_ctrl_header(fd, CTRL_STRUCT, sizeof(T))) return false;
  return send_all(fd, &pod, sizeof(T));
}

template <typename T>
bool recv_ctrl_struct(int fd, T& pod) {
  static_assert(std::is_trivially_copyable<T>::value, "payload must be POD");
  CtrlMsgHeader hdr{};
  if (!recv_ctrl_header(fd, hdr)) return false;
  if (hdr.type != CTRL_STRUCT || hdr.length != sizeof(T)) {
    return false;
  }
  return recv_all(fd, &pod, sizeof(T));
}

bool send_ctrl_ack(int fd) { return send_ctrl_header(fd, CTRL_ACK, 0); }

bool recv_ctrl_ack(int fd) {
  CtrlMsgHeader hdr{};
  if (!recv_ctrl_header(fd, hdr)) return false;
  return hdr.type == CTRL_ACK;
}

struct ChannelHandleMsg {
  uint32_t num_channels;
  uint32_t reserved;
  std::array<ncclNetHandle_v7, 1> handles;
};
static_assert(std::is_trivially_copyable<ChannelHandleMsg>::value,
              "ChannelHandleMsg must be trivially copyable");

// Resolve a best effort control IP; used only to seed the metadata exchange.
std::string get_local_ip() {
  char const* env_ip = std::getenv("UCCL_TCPX_CTRL_IP");
  if (env_ip && *env_ip) return env_ip;

  // Attempt to use hostname resolution
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

  // Fallback
  return "127.0.0.1";
}

int get_env_int(char const* name, int def_val) {
  char const* value = std::getenv(name);
  if (!value || !*value) return def_val;
  return std::atoi(value);
}

size_t get_env_size_t(char const* name, size_t def_val) {
  char const* value = std::getenv(name);
  if (!value || !*value) return def_val;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value) return def_val;
  return static_cast<size_t>(parsed);
}

// Score available TCPX NICs to find the best attachment for this GPU.
int find_best_dev(uint32_t local_gpu_idx) {
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cerr << "[TCPX] No devices detected by plugin" << std::endl;
    return -1;
  }

  int best_dev = 0;
  float best_score = -std::numeric_limits<float>::infinity();

  for (int dev = 0; dev < device_count; ++dev) {
    tcpx_net_properties props{};
    if (tcpx_get_properties(dev, &props) != 0) continue;
    float speed = static_cast<float>(props.speed);
    float latency = props.latency <= 0.0f ? 1.0f : props.latency;
    float score = speed / latency;
    if (score > best_score) {
      best_score = score;
      best_dev = dev;
    }
  }
  return best_dev;
}

}  // namespace

namespace tcpx {

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const /*num_cpus*/) {
  local_gpu_idx_ = local_gpu_idx;
  int override_gpu = get_env_int("UCCL_TCPX_LOCAL_DEVICE", -1);
  if (override_gpu < 0) {
    override_gpu = get_env_int("UCCL_TCPX_DEVICE_IDX", -1);
  }
  if (override_gpu >= 0) {
    local_gpu_idx_ = static_cast<uint32_t>(override_gpu);
  }

  // Bring up the control socket, pre-create the TCPX listen handle, and warm up
  // the CUDA stream used by bounce-buffer unpack launches.
  dev_id_ = find_best_dev(local_gpu_idx_);
  if (dev_id_ < 0) {
    throw std::runtime_error("tcpx: no available device");
  }

  // Allow overriding the out-of-band control port via either the new
  // UCCL_TCPX_OOB_PORT or the legacy UCCL_TCPX_CTRL_PORT environment variable.
  ctrl_port_ = get_env_int("UCCL_TCPX_OOB_PORT",
                           get_env_int("UCCL_TCPX_CTRL_PORT", kDefaultOobPort));

  chunk_bytes_ = get_env_size_t("UCCL_TCPX_CHUNK_BYTES", kDefaultChunkBytes);
  if (chunk_bytes_ == 0) {
    chunk_bytes_ = kDefaultChunkBytes;
  }
  std::cerr << "[tcpx] chunk size configured to " << chunk_bytes_
            << " bytes per transfer chunk" << std::endl;

  // Prepare TCP control listener
  ctrl_listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (ctrl_listen_fd_ < 0) {
    throw std::runtime_error("tcpx: failed to create control socket");
  }

  int opt = 1;
  setsockopt(ctrl_listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  setsockopt(ctrl_listen_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;

  int const max_attempts =
      std::max(1, get_env_int("UCCL_TCPX_PORT_RETRIES", 16));
  bool bound = false;
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    int try_port = ctrl_port_ + attempt;
    addr.sin_port = htons(try_port);
    if (::bind(ctrl_listen_fd_, reinterpret_cast<sockaddr*>(&addr),
               sizeof(addr)) == 0) {
      ctrl_port_ = try_port;
      bound = true;
      std::cerr << "[tcpx] control socket bound on port " << ctrl_port_ << " (attempt "
                << (attempt + 1) << "/" << max_attempts << ")" << std::endl;
      break;
    }
    if (errno != EADDRINUSE) {
      std::cerr << "[tcpx] control bind failed on port " << try_port
                << " with errno=" << errno << " (" << std::strerror(errno) << ")"
                << std::endl;
      break;
    }
    std::cerr << "[tcpx] control port " << try_port
              << " busy, retrying next port (attempt " << (attempt + 1) << "/"
              << max_attempts << ")" << std::endl;
  }
  if (!bound) {
    int err = errno;
    ::close(ctrl_listen_fd_);
    std::string msg = "tcpx: failed to bind control socket on ports [" +
                      std::to_string(ctrl_port_) + "," +
                      std::to_string(ctrl_port_ + max_attempts - 1) +
                      "]: " + std::strerror(err);
    throw std::runtime_error(msg);
  }
  if (::listen(ctrl_listen_fd_, kCtrlBacklog) < 0) {
    ::close(ctrl_listen_fd_);
    throw std::runtime_error("tcpx: failed to listen on control socket");
  }

  // Prepare TCPX listen comm for inbound connections
  if (tcpx_listen(dev_id_, &listen_handle_, &listen_comms_) != 0 ||
      !listen_comms_) {
    throw std::runtime_error("tcpx: listen failed");
  }

  // Initialize CUDA context for unpack kernels
  if (cudaSetDevice(static_cast<int>(local_gpu_idx_)) != cudaSuccess) {
    throw std::runtime_error("tcpx: cudaSetDevice failed");
  }
  if (cudaStreamCreateWithFlags(&unpack_stream_, cudaStreamNonBlocking) !=
      cudaSuccess) {
    throw std::runtime_error("tcpx: failed to create CUDA stream");
  }
  device::UnpackLaunchConfig cfg;
  cfg.stream = unpack_stream_;
  cfg.use_small_kernel = true;
  cfg.enable_profiling = false;
  unpack_launcher_ = std::make_unique<device::UnpackLauncher>(cfg);
}

Endpoint::~Endpoint() {
  // Connections may still be cached in conn_map_; tear them down before closing
  // shared resources.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    for (auto& kv : conn_map_) {
      free_conn_(kv.second);
    }
    conn_map_.clear();
  }

  {
    std::lock_guard<std::mutex> lk(mr_mu_);
    mr_map_.clear();
  }

  if (listen_comms_) {
    tcpx_close_listen(listen_comms_);
    listen_comms_ = nullptr;
  }
  if (ctrl_listen_fd_ >= 0) {
    ::close(ctrl_listen_fd_);
    ctrl_listen_fd_ = -1;
  }

  if (unpack_launcher_) {
    unpack_launcher_.reset();
  }
  if (unpack_stream_) {
    cudaStreamDestroy(unpack_stream_);
    unpack_stream_ = nullptr;
  }
}

std::vector<uint8_t> Endpoint::get_metadata() {
  EndpointInfo info{};
  std::string ip = get_local_ip();
  std::snprintf(info.ip, sizeof(info.ip), "%s", ip.c_str());
  info.port = static_cast<uint16_t>(ctrl_port_);
  info.gpu = static_cast<int>(local_gpu_idx_);

  in_addr ipv4{};
  if (inet_pton(AF_INET, ip.c_str(), &ipv4) == 1) {
    std::vector<uint8_t> meta(10);
    std::memcpy(meta.data(), &ipv4, sizeof(ipv4));
    uint16_t port_be = htons(info.port);
    meta[4] = static_cast<uint8_t>((port_be >> 8) & 0xFF);
    meta[5] = static_cast<uint8_t>(port_be & 0xFF);
    std::memcpy(meta.data() + 6, &info.gpu, sizeof(info.gpu));
    return meta;
  }

  in6_addr ipv6{};
  if (inet_pton(AF_INET6, ip.c_str(), &ipv6) == 1) {
    std::vector<uint8_t> meta(22);
    std::memcpy(meta.data(), &ipv6, sizeof(ipv6));
    uint16_t port_be = htons(info.port);
    meta[16] = static_cast<uint8_t>((port_be >> 8) & 0xFF);
    meta[17] = static_cast<uint8_t>(port_be & 0xFF);
    std::memcpy(meta.data() + 18, &info.gpu, sizeof(info.gpu));
    return meta;
  }

  std::vector<uint8_t> meta(sizeof(EndpointInfo));
  std::memcpy(meta.data(), &info, sizeof(info));
  return meta;
}

std::tuple<std::string, uint16_t, int> Endpoint::parse_metadata(
    std::vector<uint8_t> const& metadata) {
  if (metadata.size() == 10) {
    std::string ip = std::to_string(metadata[0]) + "." +
                     std::to_string(metadata[1]) + "." +
                     std::to_string(metadata[2]) + "." +
                     std::to_string(metadata[3]);
    uint16_t net_port = static_cast<uint16_t>(
        (static_cast<uint16_t>(metadata[4]) << 8) |
        static_cast<uint16_t>(metadata[5]));
    uint16_t port = ntohs(net_port);
    int gpu_idx = 0;
    std::memcpy(&gpu_idx, metadata.data() + 6, sizeof(int));
    return std::make_tuple(ip, port, gpu_idx);
  }

  if (metadata.size() == 22) {
    char ip6_str[INET6_ADDRSTRLEN];
    std::memset(ip6_str, 0, sizeof(ip6_str));
    in6_addr ip6_addr{};
    std::memcpy(&ip6_addr, metadata.data(), sizeof(ip6_addr));
    if (!inet_ntop(AF_INET6, &ip6_addr, ip6_str, sizeof(ip6_str))) {
      throw std::runtime_error("tcpx metadata IPv6 decode failed");
    }
    uint16_t net_port = static_cast<uint16_t>(
        (static_cast<uint16_t>(metadata[16]) << 8) |
        static_cast<uint16_t>(metadata[17]));
    uint16_t port = ntohs(net_port);
    int gpu_idx = 0;
    std::memcpy(&gpu_idx, metadata.data() + 18, sizeof(int));
    return std::make_tuple(std::string(ip6_str), port, gpu_idx);
  }

  if (metadata.size() == sizeof(EndpointInfo)) {
    EndpointInfo info{};
    std::memcpy(&info, metadata.data(), sizeof(info));
    return std::make_tuple(std::string(info.ip), info.port, info.gpu);
  }

  throw std::runtime_error("tcpx metadata size mismatch");
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
                       uint64_t& conn_id) {
  (void)remote_gpu_idx;

  if (remote_port < 0) remote_port = ctrl_port_;

  auto conn = std::make_unique<Conn>();

  void* reverse_listen = nullptr;
  ncclNetHandle_v7 reverse_handle{};
  if (tcpx_listen(dev_id_, &reverse_handle, &reverse_listen) != 0 ||
      !reverse_listen) {
    std::cerr << "[tcpx] tcpx_listen (reverse) failed" << std::endl;
    return false;
  }
  auto close_reverse_listen = [&]() {
    if (reverse_listen) {
      tcpx_close_listen(reverse_listen);
      reverse_listen = nullptr;
    }
  };

  int sock_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    close_reverse_listen();
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(remote_port);
  if (inet_pton(AF_INET, ip_addr.c_str(), &addr.sin_addr) != 1) {
    std::cerr << "[tcpx] invalid IP address " << ip_addr << std::endl;
    ::close(sock_fd);
    close_reverse_listen();
    return false;
  }

  if (::connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
      0) {
    std::cerr << "[tcpx] connect() to " << ip_addr << ":" << remote_port
              << " failed: " << strerror(errno) << std::endl;
    ::close(sock_fd);
    close_reverse_listen();
    return false;
  }

  EndpointInfo local_info{};
  std::string local_ip = get_local_ip();
  std::snprintf(local_info.ip, sizeof(local_info.ip), "%s", local_ip.c_str());
  local_info.port = static_cast<uint16_t>(ctrl_port_);
  local_info.gpu = static_cast<int>(local_gpu_idx_);

  // Exchange basic endpoint metadata over the control socket before touching
  // the TCPX handles.
  if (!send_ctrl_struct(sock_fd, local_info)) {
    ::close(sock_fd);
    close_reverse_listen();
    return false;
  }

  EndpointInfo remote_info{};
  if (!recv_ctrl_struct(sock_fd, remote_info)) {
    ::close(sock_fd);
    close_reverse_listen();
    return false;
  }

  ChannelHandleMsg server_handles{};
  if (!recv_ctrl_struct(sock_fd, server_handles)) {
    ::close(sock_fd);
    close_reverse_listen();
    return false;
  }
  if (server_handles.num_channels != 1) {
    std::cerr << "[tcpx] unexpected channel count from server: "
              << server_handles.num_channels << std::endl;
    ::close(sock_fd);
    close_reverse_listen();
    return false;
  }

  ncclNetHandle_v7 server_handle_copy = server_handles.handles[0];
  if (tcpx_connect_v5(dev_id_, &server_handle_copy, &conn->send_comm,
                      &conn->send_dev_handle) != 0 ||
      !conn->send_comm) {
    std::cerr << "[tcpx] tcpx_connect_v5 failed (client)" << std::endl;
    ::close(sock_fd);
    close_reverse_listen();
    free_conn_(conn);
    return false;
  }

  ChannelHandleMsg client_handles{};
  client_handles.num_channels = 1;
  client_handles.reserved = 0;
  client_handles.handles[0] = reverse_handle;
  if (!send_ctrl_struct(sock_fd, client_handles)) {
    ::close(sock_fd);
    tcpx_close_send(conn->send_comm);
    close_reverse_listen();
    free_conn_(conn);
    return false;
  }

  {
    constexpr int kMaxRetries = 100;
    constexpr int kRetryMs = 50;
    bool accepted = false;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
      int rc = tcpx_accept_v5(reverse_listen, &conn->recv_comm,
                              &conn->recv_dev_handle);
      if (rc == 0 && conn->recv_comm) {
        accepted = true;
        break;
      }
      std::cout << "[tcpx] client reverse accept retry rc=" << rc
                << " attempt=" << attempt + 1 << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
    }
    if (!accepted) {
      std::cerr << "[tcpx] tcpx_accept_v5 failed (client reverse)" << std::endl;
      ::close(sock_fd);
      tcpx_close_send(conn->send_comm);
      close_reverse_listen();
      free_conn_(conn);
      return false;
    }
  }
  close_reverse_listen();

  if (!recv_ctrl_ack(sock_fd)) {
    std::cerr << "[tcpx] missing ACK from server" << std::endl;
    ::close(sock_fd);
    tcpx_close_send(conn->send_comm);
    tcpx_close_recv(conn->recv_comm);
    free_conn_(conn);
    return false;
  }

  conn_id = next_conn_id_.fetch_add(1);
  conn->conn_id = conn_id;
  conn->ip_addr = ip_addr;
  conn->remote_gpu_idx = remote_info.gpu;
  conn->remote_port = remote_port;
  conn->ctrl_sock_fd = sock_fd;

  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_map_.emplace(conn_id, std::move(conn));
  }
  return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  // Symmetric handshake to connect(): accept the control socket, exchange
  // endpoint info, push back our listen handle, then finish the send/recv comm
  // pair using handle payloads from the client.
  sockaddr_in client_addr{};
  socklen_t addrlen = sizeof(client_addr);
  int sock_fd = ::accept(ctrl_listen_fd_,
                         reinterpret_cast<sockaddr*>(&client_addr), &addrlen);
  if (sock_fd < 0) {
    std::cerr << "[tcpx] accept() failed: " << strerror(errno) << std::endl;
    return false;
  }
  char ip_buf[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &client_addr.sin_addr, ip_buf, sizeof(ip_buf));
  ip_addr = ip_buf;

  EndpointInfo client_info{};
  if (!recv_ctrl_struct(sock_fd, client_info)) {
    ::close(sock_fd);
    return false;
  }
  remote_gpu_idx = client_info.gpu;

  EndpointInfo local_info{};
  std::string local_ip = get_local_ip();
  std::snprintf(local_info.ip, sizeof(local_info.ip), "%s", local_ip.c_str());
  local_info.port = static_cast<uint16_t>(ctrl_port_);
  local_info.gpu = static_cast<int>(local_gpu_idx_);

  // Return our metadata first so the client knows which GPU and port to expect
  // on the reverse connection.
  if (!send_ctrl_struct(sock_fd, local_info)) {
    ::close(sock_fd);
    return false;
  }

  auto conn = std::make_unique<Conn>();

  ChannelHandleMsg server_handles{};
  server_handles.num_channels = 1;
  server_handles.reserved = 0;
  server_handles.handles[0] = listen_handle_;
  if (!send_ctrl_struct(sock_fd, server_handles)) {
    ::close(sock_fd);
    return false;
  }

  {
    constexpr int kMaxRetries = 100;
    constexpr int kRetryMs = 50;
    bool accepted = false;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
      int rc = tcpx_accept_v5(listen_comms_, &conn->recv_comm,
                              &conn->recv_dev_handle);
      if (rc == 0 && conn->recv_comm) {
        accepted = true;
        break;
      }
      std::cout << "[tcpx] server accept retry rc=" << rc
                << " attempt=" << attempt + 1 << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
    }
    if (!accepted) {
      std::cerr << "[tcpx] tcpx_accept_v5 failed (server)" << std::endl;
      ::close(sock_fd);
      return false;
    }
  }

  ChannelHandleMsg client_handles{};
  if (!recv_ctrl_struct(sock_fd, client_handles)) {
    ::close(sock_fd);
    tcpx_close_recv(conn->recv_comm);
    return false;
  }
  if (client_handles.num_channels != 1) {
    std::cerr << "[tcpx] unexpected channel count from client: "
              << client_handles.num_channels << std::endl;
    ::close(sock_fd);
    tcpx_close_recv(conn->recv_comm);
    return false;
  }

  ncclNetHandle_v7 client_handle_copy = client_handles.handles[0];
  if (tcpx_connect_v5(dev_id_, &client_handle_copy, &conn->send_comm,
                      &conn->send_dev_handle) != 0 ||
      !conn->send_comm) {
    std::cerr << "[tcpx] tcpx_connect_v5 failed (server->client)" << std::endl;
    ::close(sock_fd);
    tcpx_close_recv(conn->recv_comm);
    return false;
  }

  if (!send_ctrl_ack(sock_fd)) {
    ::close(sock_fd);
    tcpx_close_recv(conn->recv_comm);
    tcpx_close_send(conn->send_comm);
    return false;
  }

  conn_id = next_conn_id_.fetch_add(1);
  conn->conn_id = conn_id;
  conn->ip_addr = ip_addr;
  conn->remote_gpu_idx = client_info.gpu;
  conn->remote_port = ntohs(client_addr.sin_port);
  conn->ctrl_sock_fd = sock_fd;

  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_map_.emplace(conn_id, std::move(conn));
  }

  return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  if (!data || size == 0) return false;
  uint64_t id = next_mr_id_.fetch_add(1);
  MrEntry entry;
  entry.base = const_cast<void*>(data);
  entry.size = size;
  entry.ptr_type = NCCL_PTR_CUDA;
  entry.is_recv = true;  // default; exact usage determined later

  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    mr_map_[id] = entry;
  }
  mr_id = id;
  return true;
}

bool Endpoint::dereg(uint64_t mr_id) {
  std::vector<std::pair<uint64_t, void*>> send_handles;
  std::vector<std::pair<uint64_t, void*>> recv_handles;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    send_handles.reserve(it->second.send_handles.size());
    recv_handles.reserve(it->second.recv_handles.size());
    for (auto const& kv : it->second.send_handles) {
      send_handles.emplace_back(kv.first, kv.second);
    }
    for (auto const& kv : it->second.recv_handles) {
      recv_handles.emplace_back(kv.first, kv.second);
    }
    mr_map_.erase(it);
  }

  auto dereg_for_conn = [&](std::pair<uint64_t, void*> const& item,
                            bool is_send) {
    std::shared_lock<std::shared_mutex> conn_lock(conn_mu_);
    auto conn_it = conn_map_.find(item.first);
    if (conn_it == conn_map_.end()) return;
    Conn* conn = conn_it->second.get();
    void* comm = is_send ? conn->send_comm : conn->recv_comm;
    if (comm) {
      tcpx_dereg_mr(comm, item.second);
    }
  };

  for (auto const& kv : send_handles) {
    dereg_for_conn(kv, /*is_send=*/true);
  }
  for (auto const& kv : recv_handles) {
    dereg_for_conn(kv, /*is_send=*/false);
  }

  return true;
}

bool Endpoint::find_mr_by_addr(uintptr_t addr, size_t size,
                               uint64_t* mr_id) const {
  if (!mr_id || size == 0) return false;
  std::lock_guard<std::mutex> lock(mr_mu_);
  for (auto const& kv : mr_map_) {
    uintptr_t base = reinterpret_cast<uintptr_t>(kv.second.base);
    if (addr < base) continue;
    size_t offset = static_cast<size_t>(addr - base);
    if (offset + size <= kv.second.size) {
      *mr_id = kv.first;
      return true;
    }
  }
  return false;
}

bool Endpoint::populate_conn_handles_(Conn& conn, uint64_t mr_id, bool is_recv,
                                      void** mhandle_out) {
  void* comm = is_recv ? conn.recv_comm : conn.send_comm;
  if (!comm) return false;

  void* existing = nullptr;
  void* base = nullptr;
  size_t size = 0;
  int ptr_type = NCCL_PTR_CUDA;

  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    auto& table = is_recv ? it->second.recv_handles : it->second.send_handles;
    auto handle_it = table.find(conn.conn_id);
    if (handle_it != table.end()) {
      existing = handle_it->second;
    } else {
      base = it->second.base;
      size = it->second.size;
      ptr_type = it->second.ptr_type;
    }
  }

  if (existing) {
    *mhandle_out = existing;
    return true;
  }

  void* mhandle = nullptr;
  int rc = tcpx_reg_mr(comm, base, size, ptr_type, &mhandle);
  if (rc != 0 || !mhandle) {
    std::cerr << "[tcpx] tcpx_reg_mr failed rc=" << rc << std::endl;
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) {
      tcpx_dereg_mr(comm, mhandle);
      return false;
    }
    auto& table = is_recv ? it->second.recv_handles : it->second.send_handles;
    auto [insert_it, inserted] = table.emplace(conn.conn_id, mhandle);
    if (!inserted) {
      tcpx_dereg_mr(comm, mhandle);
      *mhandle_out = insert_it->second;
      return true;
    }
  }

  *mhandle_out = mhandle;
  return true;
}

bool Endpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr,
                         size_t len, char* out_buf) {
  // Produce a fixed-width description of a registered buffer so the peer can
  // issue a tagged receive or read.
  if (!out_buf) return false;

  MrEntry mr;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t addr_u = reinterpret_cast<uintptr_t>(addr);
  if (addr_u < base_addr || addr_u + len > base_addr + mr.size) {
    return false;
  }

  FifoItem item{};
  item.mr_id = mr_id;
  item.size = static_cast<uint32_t>(len);
  item.offset = static_cast<uint64_t>(addr_u - base_addr);
  item.tag = next_tag_.fetch_add(1);
  // Fifo item travels over the uccl control socket; the peer rehydrates it into
  // tag/offset parameters for TCPX operations.
  std::memcpy(out_buf, &item, sizeof(FifoItem));
  return true;
}

bool Endpoint::queue_read_response(uint64_t conn_id,
                                   FifoItem const& fifo_item) {
  // Active side of a read: locate the advertised slice and push it using the
  // tag recorded in the FIFO.
  Conn* conn_ptr = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second.get();
  }

  MrEntry mr;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(fifo_item.mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }

  if (fifo_item.offset + fifo_item.size > mr.size) return false;
  char* base_ptr =
      static_cast<char*>(mr.base) + static_cast<size_t>(fifo_item.offset);
  void const* base = static_cast<void const*>(base_ptr);

  uint64_t tid = 0;
  if (!post_send_(*conn_ptr, fifo_item.mr_id, mr, base, fifo_item.size,
                  static_cast<int>(fifo_item.tag), tid)) {
    return false;
  }

  // Wait for completion to avoid leaking outstanding transfers.
  // Reads appear synchronous to the uccl engine, so a short polling loop here
  // is acceptable.
  bool done = false;
  while (!done) {
    if (!poll_async(tid, &done)) return false;
    if (!done) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }
  return true;
}

bool Endpoint::send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                          size_t size, uint64_t* transfer_id) {
  return send_async_with_tag(conn_id, mr_id, data, size, 0, transfer_id);
}

bool Endpoint::send_async_with_tag(uint64_t conn_id, uint64_t mr_id,
                                   void const* data, size_t size, uint32_t tag,
                                   uint64_t* transfer_id) {
  // Variant used by queue_read_response / read_async where the caller controls
  // the tag.
  if (!data || size == 0) return false;

  Conn* conn_ptr = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second.get();
  }

  MrEntry mr;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t data_addr = reinterpret_cast<uintptr_t>(data);
  if (data_addr < base_addr || data_addr + size > base_addr + mr.size) {
    return false;
  }

  uint64_t tid = 0;
  if (!post_send_(*conn_ptr, mr_id, mr, data, size, static_cast<int>(tag),
                  tid)) {
    return false;
  }

  if (transfer_id) *transfer_id = tid;
  return true;
}

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst,
                          size_t size, FifoItem const& slot_item,
                          uint64_t* transfer_id) {
  if (!dst || size == 0) return false;

  Conn* conn_ptr = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second.get();
  }

  MrEntry mr;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }

  if (slot_item.size > mr.size) return false;
  if (size < slot_item.size) return false;

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  if (dst_addr < base_addr || dst_addr + slot_item.size > base_addr + mr.size) {
    return false;
  }

  // Reuse the advertised tag so the active peer can match this recv.
  return recv_async_with_tag(conn_id, mr_id, dst, slot_item.size, slot_item.tag,
                             transfer_id);
}

bool Endpoint::recv_async(uint64_t conn_id, uint64_t mr_id, void* data,
                          size_t size, uint64_t* transfer_id) {
  return recv_async_with_tag(conn_id, mr_id, data, size, 0, transfer_id);
}

bool Endpoint::recv_async_with_tag(uint64_t conn_id, uint64_t mr_id, void* data,
                                   size_t size, uint32_t tag,
                                   uint64_t* transfer_id) {
  // Allow callers (e.g. FIFO-driven reads) to enforce the tag that the sender
  // will use.
  if (!data || size == 0) return false;

  Conn* conn_ptr = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second.get();
  }

  MrEntry mr;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(data);
  if (dst_addr < base_addr || dst_addr + size > base_addr + mr.size) {
    return false;
  }

  uint64_t tid = 0;
  if (!post_recv_(*conn_ptr, mr_id, mr, data, size, static_cast<int>(tag), tid,
                  /*needs_unpack=*/true)) {
    return false;
  }
  if (transfer_id) *transfer_id = tid;
  return true;
}

bool Endpoint::post_send_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                          void const* data, size_t size, int tag,
                          uint64_t& transfer_id) {
  // Internal helper shared by send_async/queue_read_response. Ensures the MR is
  // registered (lazy) and tracks the outstanding TCPX request in transfer_map_.
  if (!data || size == 0) return false;

  void* mhandle = nullptr;
  if (!populate_conn_handles_(conn, mr_id, /*is_recv=*/false, &mhandle))
    return false;

  void* request = nullptr;
  int rc = tcpx_isend(conn.send_comm, const_cast<void*>(data),
                      static_cast<int>(size), tag, mhandle, &request);
  if (rc != 0 || !request) {
    std::cerr << "[tcpx] tcpx_isend failed rc=" << rc << std::endl;
    return false;
  }

  PendingTransfer transfer;
  transfer.kind = PendingTransfer::Kind::kSend;
  transfer.transfer_id = next_transfer_id_.fetch_add(1);
  transfer.conn_id = conn.conn_id;
  transfer.mr_id = mr_id;
  transfer.size = size;
  transfer.tag = tag;  // Persist tag for debugging and potential retries.
  transfer.request = request;
  transfer.needs_unpack = false;
  transfer.event_recorded = false;
  transfer.completion_event = nullptr;

  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    transfer_map_[transfer.transfer_id] = transfer;
  }

  transfer_id = transfer.transfer_id;
  return true;
}

bool Endpoint::post_recv_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                          void* data, size_t size, int tag,
                          uint64_t& transfer_id, bool needs_unpack) {
  // Mirrors post_send_: queue a TCPX irecv, then stash bookkeeping so
  // poll_async can match the request, trigger unpack, and free resources later.
  if (!data || size == 0) return false;

  void* mhandle = nullptr;
  if (!populate_conn_handles_(conn, mr_id, /*is_recv=*/true, &mhandle))
    return false;

  void* buffers[1] = {data};
  int sizes[1] = {static_cast<int>(size)};
  int tags[1] = {tag};
  void* mhandles[1] = {mhandle};
  void* requests[1] = {nullptr};

  int rc =
      tcpx_irecv(conn.recv_comm, 1, buffers, sizes, tags, mhandles, requests);
  if (rc != 0 || !requests[0]) {
    std::cerr << "[tcpx] tcpx_irecv failed rc=" << rc << std::endl;
    return false;
  }

  PendingTransfer transfer;
  transfer.kind = PendingTransfer::Kind::kRecv;
  transfer.transfer_id = next_transfer_id_.fetch_add(1);
  transfer.conn_id = conn.conn_id;
  transfer.mr_id = mr_id;
  transfer.size = size;
  transfer.tag = tag;  // Tag must match the sender's isend to complete.
  transfer.request = requests[0];
  transfer.dst_ptr = data;
  transfer.needs_unpack =
      needs_unpack;  // Bounce-buffer unpack handled later if true.
  transfer.event_recorded = false;
  transfer.completion_event = nullptr;

  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    transfer_map_[transfer.transfer_id] = transfer;
  }

  transfer_id = transfer.transfer_id;
  return true;
}

bool Endpoint::poll_request_(PendingTransfer& transfer, bool* done,
                             int* received_size) {
  *done = false;
  if (!transfer.request) return false;

  int completed = 0;
  int size = 0;
  // Stage 1: query TCPX progress. tcpx_test returns immediately and sets
  // `completed` when the network transfer has finished populating the bounce
  // buffer.
  int rc = tcpx_test(transfer.request, &completed, &size);
  if (rc != 0) {
    std::cerr << "[tcpx] tcpx_test failed rc=" << rc << std::endl;
    return false;
  }
  std::cerr << "[tcpx] tcpx_test transfer_id=" << transfer.transfer_id
            << " completed=" << completed << " size=" << size << std::endl;
  if (!completed) return true;

  *done = true;
  if (received_size) *received_size = size;
  return true;
}

bool Endpoint::enqueue_unpack_(PendingTransfer& transfer,
                               tcpx::plugin::tcpxRequest* request, Conn& conn) {
  auto* dev_handle_struct =
      reinterpret_cast<tcpx::plugin::NcclNetDeviceHandle*>(
          conn.recv_dev_handle);
  if (!dev_handle_struct || !request) return false;

  uint64_t frag_cnt = 0;
  if (!request->unpack_slot.cnt) {
    std::cerr << "[tcpx] missing unpack counter" << std::endl;
    return false;
  }
  frag_cnt = *(request->unpack_slot.cnt);
  std::cerr << "[tcpx] unpack transfer_id=" << transfer.transfer_id
            << " frag_cnt=" << frag_cnt << " dst_ptr=" << transfer.dst_ptr
            << std::endl;
  if (frag_cnt == 0 || frag_cnt > MAX_UNPACK_DESCRIPTORS) {
    std::cerr << "[tcpx] invalid fragment count " << frag_cnt << std::endl;
    return false;
  }

  tcpx::plugin::unpackNetDeviceHandle dev_handle{};
  CUresult cu_rc = cuMemcpyDtoH(
      &dev_handle, reinterpret_cast<CUdeviceptr>(dev_handle_struct->handle),
      sizeof(dev_handle));
  if (cu_rc != CUDA_SUCCESS) {
    std::cerr << "[tcpx] cuMemcpyDtoH failed " << cu_rc << std::endl;
    return false;
  }
  std::cerr << "[tcpx] dev_handle bounce_buf=" << dev_handle.bounce_buf
            << std::endl;

  auto* meta_entries =
      static_cast<tcpx::plugin::loadMeta*>(request->unpack_slot.mem);
  tcpx::rx::UnpackDescriptorBlock block;
  // Translate the plugin-provided metadata into the format expected by the CUDA
  // unpack kernel.
  tcpx::rx::buildDescriptorBlock(meta_entries, static_cast<uint32_t>(frag_cnt),
                                 dev_handle.bounce_buf, transfer.dst_ptr,
                                 block);
  if (frag_cnt > 0) {
    auto const& m0 = block.descriptors[0];
    size_t probe_len = std::min<size_t>(m0.len, 64);
    std::vector<uint8_t> sample(probe_len);
    CUresult sample_rc = cuMemcpyDtoH(
        sample.data(),
        reinterpret_cast<CUdeviceptr>(dev_handle.bounce_buf + m0.src_off),
        probe_len);
    if (sample_rc == CUDA_SUCCESS) {
      std::cerr << "[tcpx] bounce sample @src_off=" << m0.src_off << ":";
      for (size_t i = 0; i < probe_len; ++i) {
        std::cerr << " " << static_cast<int>(sample[i]);
      }
      std::cerr << std::endl;
    } else {
      std::cerr << "[tcpx] bounce sample copy failed rc=" << sample_rc
                << std::endl;
    }
  }
  for (uint32_t i = 0; i < std::min<uint32_t>(frag_cnt, 4); ++i) {
    auto const& m = block.descriptors[i];
    std::cerr << "  meta[" << i << "] src_off=" << m.src_off << " len=" << m.len
              << " dst_off=" << m.dst_off << std::endl;
  }
  if (frag_cnt > 4) {
    for (uint32_t i = frag_cnt - std::min<uint32_t>(frag_cnt, 4); i < frag_cnt;
         ++i) {
      auto const& m = block.descriptors[i];
      std::cerr << "  meta[" << i << "] src_off=" << m.src_off
                << " len=" << m.len << " dst_off=" << m.dst_off << " (tail)"
                << std::endl;
    }
  }
  block.ready_flag = request->unpack_slot.cnt;
  block.ready_threshold = frag_cnt;

  std::cerr << "[tcpx] launch unpack: count=" << block.count
            << " total_bytes=" << block.total_bytes << std::endl;
  int launch_rc = unpack_launcher_->launch(block, unpack_stream_);
  if (launch_rc != 0) {
    std::cerr << "[tcpx] unpack kernel launch failed rc=" << launch_rc
              << std::endl;
    return false;
  }

  if (!transfer.completion_event) {
    cudaEventCreateWithFlags(&transfer.completion_event,
                             cudaEventDisableTiming);
  }
  cudaEventRecord(transfer.completion_event, unpack_stream_);
  transfer.event_recorded = true;
  transfer.desc_block = block;
  return true;
}

bool Endpoint::complete_pending_transfer_(PendingTransfer& transfer,
                                          bool success) {
  // Finalise bookkeeping for a finished transfer and return resources to the
  // TCPX plugin if appropriate.
  if (transfer.kind == PendingTransfer::Kind::kRecv && transfer.request) {
    Conn* conn_ptr = nullptr;
    {
      std::shared_lock<std::shared_mutex> lock(conn_mu_);
      auto it = conn_map_.find(transfer.conn_id);
      if (it != conn_map_.end()) {
        conn_ptr = it->second.get();
      }
    }
    if (conn_ptr && success && transfer.request) {
      tcpx_irecv_consumed(conn_ptr->recv_comm, 1, transfer.request);
    }
  }

  if (transfer.completion_event) {
    cudaEventDestroy(transfer.completion_event);
    transfer.completion_event = nullptr;
  }
  transfer.request = nullptr;
  return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  if (!is_done) return false;
  *is_done = false;

  PendingTransfer transfer;
  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    auto it = transfer_map_.find(transfer_id);
    if (it == transfer_map_.end()) return false;
    transfer = it->second;
  }

  bool completed = false;
  int size = 0;
  // Completion happens in two stages:
  //  1) tcpx_test indicates the TCP stack finished the transfer.
  //  2) If we received data via the bounce buffer, launch the CUDA unpack and
  //     wait for the event before we report completion to the caller.
  if (!poll_request_(transfer, &completed, &size)) return false;

  if (!completed) return true;

  if (transfer.kind == PendingTransfer::Kind::kRecv && transfer.needs_unpack) {
    Conn* conn_ptr = nullptr;
    {
      std::shared_lock<std::shared_mutex> lock(conn_mu_);
      auto it = conn_map_.find(transfer.conn_id);
      if (it != conn_map_.end()) {
        conn_ptr = it->second.get();
      }
    }
    if (!conn_ptr) return false;

    auto* rx_req =
        reinterpret_cast<tcpx::plugin::tcpxRequest*>(transfer.request);
    if (!enqueue_unpack_(transfer, rx_req, *conn_ptr)) return false;

    {
      std::lock_guard<std::mutex> lock(transfer_mu_);
      // Store the updated transfer (with CUDA event) so subsequent polling sees
      // the event handle while we wait for the kernel to finish.
      transfer_map_[transfer_id] = transfer;
    }

    if (cudaEventSynchronize(transfer.completion_event) != cudaSuccess) {
      std::cerr << "[tcpx] cudaEventSynchronize failed" << std::endl;
      return false;
    }
  }

  complete_pending_transfer_(transfer, /*success=*/true);

  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    transfer_map_.erase(transfer_id);
  }

  *is_done = true;
  return true;
}

// Helper invoked during teardown to make sure every comm and its registrations
// are returned to the TCPX runtime.
void Endpoint::free_conn_(std::unique_ptr<Conn>& conn) {
  if (!conn) return;
  std::vector<void*> send_handles;
  std::vector<void*> recv_handles;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    for (auto& kv : mr_map_) {
      auto& mr = kv.second;
      auto send_it = mr.send_handles.find(conn->conn_id);
      if (send_it != mr.send_handles.end()) {
        send_handles.push_back(send_it->second);
        mr.send_handles.erase(send_it);
      }
      auto recv_it = mr.recv_handles.find(conn->conn_id);
      if (recv_it != mr.recv_handles.end()) {
        recv_handles.push_back(recv_it->second);
        mr.recv_handles.erase(recv_it);
      }
    }
  }
  for (void* handle : send_handles) {
    if (conn->send_comm) tcpx_dereg_mr(conn->send_comm, handle);
  }
  for (void* handle : recv_handles) {
    if (conn->recv_comm) tcpx_dereg_mr(conn->recv_comm, handle);
  }
  if (conn->send_comm) {
    tcpx_close_send(conn->send_comm);
    conn->send_comm = nullptr;
  }
  if (conn->recv_comm) {
    tcpx_close_recv(conn->recv_comm);
    conn->recv_comm = nullptr;
  }
  if (conn->ctrl_sock_fd >= 0) {
    ::close(conn->ctrl_sock_fd);
    conn->ctrl_sock_fd = -1;
  }
}

}  // namespace tcpx
