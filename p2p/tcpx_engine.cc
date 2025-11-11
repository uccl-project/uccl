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
constexpr int kDefaultSendInflight = 12;
constexpr int kDefaultRecvInflight = 16;
constexpr int kTcpxBusy = 3;  // tcpx_isend/irecv temporary no-resource

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

class ScopedCudaContext {
 public:
  ScopedCudaContext(CUcontext ctx, int device_ordinal)
      : pushed_(false), context_(ctx) {
    if (!context_) return;
    CUresult cu_rc = cuCtxPushCurrent(context_);
    if (cu_rc != CUDA_SUCCESS) {
      std::cerr << "[tcpx] cuCtxPushCurrent failed rc=" << cu_rc << std::endl;
      context_ = nullptr;
      return;
    }
    pushed_ = true;
    if (device_ordinal >= 0) {
      cudaError_t cuda_rc = cudaSetDevice(device_ordinal);
      if (cuda_rc != cudaSuccess) {
        std::cerr << "[tcpx] cudaSetDevice failed inside ScopedCudaContext: "
                  << cudaGetErrorString(cuda_rc) << std::endl;
      }
    }
  }

  ~ScopedCudaContext() {
    if (!pushed_) return;
    CUcontext popped = nullptr;
    CUresult cu_rc = cuCtxPopCurrent(&popped);
    if (cu_rc != CUDA_SUCCESS) {
      std::cerr << "[tcpx] cuCtxPopCurrent failed rc=" << cu_rc << std::endl;
    }
  }

  ScopedCudaContext(ScopedCudaContext const&) = delete;
  ScopedCudaContext& operator=(ScopedCudaContext const&) = delete;

 private:
  bool pushed_;
  CUcontext context_;
};

// Control-plane payload for advertising TCPX listen handles.
// Note: current implementation is single-channel only; the extra fields
// (num_channels/reserved) are kept for forward-compat but always set to 1/0.
struct ChannelHandleMsg {
  uint32_t num_channels;                        // always 1 for now
  uint32_t reserved;                            // reserved for future use
  std::array<ncclNetHandle_v7, 1> handles;      // single handle payload
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

}  // namespace

namespace tcpx {

// ============================================================================
// Endpoint Construction & Control Plane
// ============================================================================

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

  // Fix per-chunk size once so later calculations stay consistent.
  chunk_bytes_ = get_env_size_t("UCCL_TCPX_CHUNK_BYTES", kDefaultChunkBytes);
  if (chunk_bytes_ == 0) {
    chunk_bytes_ = kDefaultChunkBytes;
  }
  int max_send_env =
      get_env_int("UCCL_TCPX_MAX_SEND_INFLIGHT", kDefaultSendInflight);
  if (max_send_env <= 0) max_send_env = kDefaultSendInflight;
  max_send_inflight_ = static_cast<size_t>(max_send_env);

  int max_recv_env =
      get_env_int("UCCL_TCPX_MAX_RECV_INFLIGHT", kDefaultRecvInflight);
  if (max_recv_env <= 0) max_recv_env = kDefaultRecvInflight;
  max_recv_inflight_ = static_cast<size_t>(max_recv_env);
  debug_enabled_ = get_env_int("UCCL_TCPX_DEBUG", 0) != 0;
  if (debug_enabled_) {
    std::cerr << "[tcpx] chunk size configured to " << chunk_bytes_
              << " bytes per transfer chunk" << std::endl;
  }

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

  // Initialize CUDA primary context for unpack kernels and driver interactions.
  CUresult cu_rc = cuInit(0);
  if (cu_rc != CUDA_SUCCESS) {
    throw std::runtime_error("tcpx: cuInit failed with rc=" +
                             std::to_string(static_cast<int>(cu_rc)));
  }
  cu_rc = cuDeviceGet(&cu_device_, static_cast<int>(local_gpu_idx_));
  if (cu_rc != CUDA_SUCCESS) {
    throw std::runtime_error("tcpx: cuDeviceGet failed with rc=" +
                             std::to_string(static_cast<int>(cu_rc)));
  }
  cu_rc = cuDevicePrimaryCtxRetain(&cu_context_, cu_device_);
  if (cu_rc != CUDA_SUCCESS) {
    throw std::runtime_error("tcpx: cuDevicePrimaryCtxRetain failed with rc=" +
                             std::to_string(static_cast<int>(cu_rc)));
  }
  cu_rc = cuCtxSetCurrent(cu_context_);
  if (cu_rc != CUDA_SUCCESS) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
    throw std::runtime_error("tcpx: cuCtxSetCurrent failed with rc=" +
                             std::to_string(static_cast<int>(cu_rc)));
  }
  if (cudaSetDevice(static_cast<int>(local_gpu_idx_)) != cudaSuccess) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
    throw std::runtime_error("tcpx: cudaSetDevice failed");
  }
  if (cudaStreamCreateWithFlags(&unpack_stream_, cudaStreamNonBlocking) !=
      cudaSuccess) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
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
  if (cu_context_) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
  }
}

// ---------------------------------------------------------------------------
// Metadata helpers
// ---------------------------------------------------------------------------
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
  ScopedCudaContext ctx_guard(cu_context_, static_cast<int>(local_gpu_idx_));
  (void)remote_gpu_idx;

  if (remote_port < 0) remote_port = ctrl_port_;

  auto conn = std::make_shared<Conn>();

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

  // Read the listen handle(s) the server advertised over the control socket.
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

  // Client returns the reverse-path handle using the same compact framing.
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

  if (!ensure_recv_dev_handle_cached_(*conn)) {
    std::cerr << "[tcpx] failed to cache recv device handle (client)" << std::endl;
    free_conn_(conn);
    return false;
  }

  // ==========================================================================
  // 初始化 CUDA Event Pool
  // ==========================================================================
  // 设计要点：
  // - 预分配固定数量的 events（等于 max_recv_inflight）
  // - 整个连接生命周期内循环复用（round-robin）
  // - 避免在热路径上创建/销毁 events（性能优化）
  // - 使用 cudaEventDisableTiming 标志（不需要计时功能）
  int max_recv_inflight =
      get_env_int("UCCL_TCPX_MAX_RECV_INFLIGHT", kDefaultRecvInflight);
  if (max_recv_inflight <= 0) max_recv_inflight = kDefaultRecvInflight;
  size_t pool_size = static_cast<size_t>(max_recv_inflight);
  conn->recv_events.resize(pool_size);

  // 预分配所有 events
  for (size_t i = 0; i < pool_size; ++i) {
    cudaError_t rc = cudaEventCreateWithFlags(&conn->recv_events[i],
                                              cudaEventDisableTiming);
    if (rc != cudaSuccess) {
      std::cerr << "[tcpx] cudaEventCreateWithFlags failed for event " << i
                << ": " << cudaGetErrorString(rc) << std::endl;

      // 清理已创建的 events
      for (size_t j = 0; j < i; ++j) {
        cudaEventDestroy(conn->recv_events[j]);
      }

      // 清理连接资源
      ::close(sock_fd);
      tcpx_close_send(conn->send_comm);
      tcpx_close_recv(conn->recv_comm);
      free_conn_(conn);
      return false;
    }
  }

  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_map_.emplace(conn_id, conn);
  }
  return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  ScopedCudaContext ctx_guard(cu_context_, static_cast<int>(local_gpu_idx_));
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

  auto conn = std::make_shared<Conn>();

  ChannelHandleMsg server_handles{};  // single-channel
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

  ncclNetHandle_v7 client_handle_copy = client_handles.handles[0];  // single-channel
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

  if (!ensure_recv_dev_handle_cached_(*conn)) {
    std::cerr << "[tcpx] failed to cache recv device handle (server)" << std::endl;
    free_conn_(conn);
    return false;
  }

  // 初始化 CUDA Event Pool（对齐原来的 ChannelWindow::events 设计）
  // 预分配固定数量的 events，整个连接生命周期内循环复用
  int max_recv_inflight_acc =
      get_env_int("UCCL_TCPX_MAX_RECV_INFLIGHT", kDefaultRecvInflight);
  if (max_recv_inflight_acc <= 0) max_recv_inflight_acc = kDefaultRecvInflight;
  size_t pool_size_acc = static_cast<size_t>(max_recv_inflight_acc);
  conn->recv_events.resize(pool_size_acc);
  for (size_t i = 0; i < pool_size_acc; ++i) {
    cudaError_t rc = cudaEventCreateWithFlags(&conn->recv_events[i],
                                              cudaEventDisableTiming);
    if (rc != cudaSuccess) {
      std::cerr << "[tcpx] cudaEventCreateWithFlags failed for event " << i
                << ": " << cudaGetErrorString(rc) << std::endl;
      // 清理已创建的 events
      for (size_t j = 0; j < i; ++j) {
        cudaEventDestroy(conn->recv_events[j]);
      }
      ::close(sock_fd);
      tcpx_close_recv(conn->recv_comm);
      tcpx_close_send(conn->send_comm);
      return false;
    }
  }

  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_map_.emplace(conn_id, conn);
  }

  return true;
}

// ---------------------------------------------------------------------------
// Memory registration and lookup
// ---------------------------------------------------------------------------
bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  if (!data || size == 0) return false;
  uint64_t id = next_mr_id_.fetch_add(1);
  MrEntry entry;
  entry.base = const_cast<void*>(data);
  entry.size = size;
  entry.ptr_type = NCCL_PTR_CUDA;

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
    std::shared_ptr<Conn> conn = conn_it->second;
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
  ScopedCudaContext ctx_guard(cu_context_, static_cast<int>(local_gpu_idx_));
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
    // Reuse cached registration if this connection already touched it.
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

bool Endpoint::ensure_recv_dev_handle_cached_(Conn& conn) {
  if (conn.recv_dev_handle_cached) return true;
  if (!conn.recv_dev_handle) return false;

  auto* dev_handle_struct =
      reinterpret_cast<tcpx::plugin::NcclNetDeviceHandle*>(
          conn.recv_dev_handle);
  if (!dev_handle_struct || !dev_handle_struct->handle) return false;

  ScopedCudaContext ctx_guard(cu_context_, static_cast<int>(local_gpu_idx_));
  CUresult cu_rc = cuMemcpyDtoH(
      &conn.recv_dev_handle_host,
      reinterpret_cast<CUdeviceptr>(dev_handle_struct->handle),
      sizeof(conn.recv_dev_handle_host));
  if (cu_rc != CUDA_SUCCESS) {
    std::cerr << "[tcpx] cuMemcpyDtoH (cache recv handle) failed rc=" << cu_rc
              << std::endl;
    return false;
  }
  conn.recv_dev_handle_cached = true;
  return true;
}

// ============================================================================
// 传输推进的核心辅助函数
// ============================================================================
// 以下两个函数实现了传输推进的共享路径：
//
// 1. advance_transfer_locked：
//    - 统一的传输推进函数，被 poll_async 调用
//    - 假设 transfer_mu_ 已持锁
//    - 执行 Stage 0（调度新 chunks）和 Stage 1/2（推进已提交的 chunks）
//    - 返回传输是否完成
//
// 2. finalize_transfer_locked：
//    - 从 transfer_map_ 中移除已完成的传输
//    - 假设 transfer_mu_ 已持锁
//
// 设计要点：
// - 用户通过 poll_async 主动推进传输
// - 所有传输推进逻辑集中在 advance_transfer_locked，便于维护和调试
// - 完成后重置窗口计数器，确保后续传输不会因窗口满而阻塞
// ============================================================================

/**
 * 统一的传输推进函数（持锁调用）
 *
 * 执行流程：
 * 1. Stage 0：如果还有未提交的 chunks，调用 schedule_*_chunks_locked 调度
 * 2. Stage 1/2：调用 progress_transfer_locked 推进已提交的 chunks
 * 3. 如果 Stage 1/2 有进度，再次尝试调度（充分利用窗口）
 * 4. 检查是否所有 chunks 都已完成（chunks_completed >= chunks.size()）
 *
 * @param conn 连接对象
 * @param transfer 传输对象
 * @param transfer_complete 输出：传输是否完成
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::advance_transfer_locked(Conn& conn, PendingTransfer& transfer,
                                       bool* transfer_complete) {
  if (!transfer_complete) return false;
  if (transfer.next_chunk_to_post < transfer.chunks.size()) {
    ScheduleOutcome outcome = ScheduleOutcome::kNoProgress;
    if (transfer.kind == PendingTransfer::Kind::kSend) {
      outcome = schedule_send_chunks_locked(conn, transfer);
    } else {
      outcome = schedule_recv_chunks_locked(conn, transfer);
    }
    if (outcome == ScheduleOutcome::kError) return false;
  }

  bool schedule_send = false;
  bool schedule_recv = false;
  if (!progress_transfer_locked(conn, transfer, &schedule_send, &schedule_recv))
    return false;

  if (schedule_send && transfer.kind == PendingTransfer::Kind::kSend &&
      transfer.next_chunk_to_post < transfer.chunks.size()) {
    auto outcome = schedule_send_chunks_locked(conn, transfer);
    if (outcome == ScheduleOutcome::kError) return false;
  }
  if (schedule_recv && transfer.kind != PendingTransfer::Kind::kSend &&
      transfer.next_chunk_to_post < transfer.chunks.size()) {
    auto outcome = schedule_recv_chunks_locked(conn, transfer);
    if (outcome == ScheduleOutcome::kError) return false;
  }

  *transfer_complete = (transfer.chunks_completed >= transfer.chunks.size());
  return true;
}

/**
 * 完成传输并清理资源（持锁调用）
 *
 * 从 transfer_map_ 中移除传输，释放内存
 *
 * @param it 传输迭代器
 */
void Endpoint::finalize_transfer_locked(
    std::unordered_map<uint64_t, PendingTransfer>::iterator it) {
  transfer_map_.erase(it);
}

/**
 * 重置连接的窗口计数器
 *
 * 清空 send_inflight_chunks_ 和 recv_inflight_chunks_ 中的条目
 *
 * @param conn_id 连接 ID
 */
void Endpoint::reset_conn_window_counters_(uint64_t conn_id) {
  std::lock_guard<std::mutex> lock(window_mu_);
  send_inflight_chunks_.erase(conn_id);
  recv_inflight_chunks_.erase(conn_id);
}



// ---------------------------------------------------------------------------
// Data-plane entry points (advertise / send / recv / read)
// ---------------------------------------------------------------------------
bool Endpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr,
                         size_t len, char* out_buf) {
  (void)conn_id;  // kept for API symmetry with higher layers; unused here
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
  item.token = 0;
  // Fifo item travels over the uccl control socket; the peer rehydrates it into
  // tag/offset parameters for TCPX operations.
  std::memcpy(out_buf, &item, sizeof(FifoItem));
  return true;
}

/**
 * 响应 READ 请求（主动端）
 *
 * 执行流程：
 * 1. 根据 conn_id 查找连接对象
 * 2. 根据 fifo_item.mr_id 查找内存注册条目
 * 3. 计算数据地址（base + offset）
 * 4. 调用 post_send_ 提交发送传输
 * 5. 用户通过 poll_async 推进传输
 *
 * 设计要点：
 * - READ 回复路径完全异步，避免阻塞 UCCL listener 线程
 * - 使用 FifoItem 中的 tag 确保与被动端的接收匹配
 * - 传输由用户线程通过 poll_async 推进
 *
 * @param conn_id 连接 ID
 * @param fifo_item FIFO 元数据项
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::queue_read_response(uint64_t conn_id,
                                   FifoItem const& fifo_item) {
  // 查找连接对象
  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second;
  }

  // 查找内存注册条目
  MrEntry mr;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(fifo_item.mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }

  // 验证偏移量和大小
  if (fifo_item.offset + fifo_item.size > mr.size) return false;

  // 计算数据地址
  char* base_ptr =
      static_cast<char*>(mr.base) + static_cast<size_t>(fifo_item.offset);
  void const* base = static_cast<void const*>(base_ptr);

  // 提交发送传输
  uint64_t tid = 0;
  if (!post_send_(*conn_ptr, fifo_item.mr_id, mr, base, fifo_item.size,
                  static_cast<int>(fifo_item.tag), tid)) {
    return false;
  }

  // 传输已提交，用户需要通过 poll_async 推进
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

  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second;
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

  // 提交发送传输
  uint64_t tid = 0;
  if (!post_send_(*conn_ptr, mr_id, mr, data, size, static_cast<int>(tag),
                  tid)) {
    return false;
  }

  // 传输已提交，用户需要通过 poll_async 推进
  if (transfer_id) *transfer_id = tid;
  return true;
}

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst,
                          size_t size, FifoItem const& slot_item,
                          uint64_t* transfer_id) {
  if (!dst || size == 0) return false;

  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second;
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

  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second;
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

  // 提交接收传输
  uint64_t tid = 0;
  if (!post_recv_(*conn_ptr, mr_id, mr, data, size, static_cast<int>(tag), tid,
                  /*needs_unpack=*/true)) {
    return false;
  }

  // 传输已提交，用户需要通过 poll_async 推进
  if (transfer_id) *transfer_id = tid;
  return true;
}

// ============================================================================
// 滑动窗口流控
// ============================================================================
//
// 设计目标：
// - 限制同时在途的 chunks 数量，避免资源耗尽
// - 发送端和接收端分别限制（max_send_inflight_ 和 max_recv_inflight_）
// - 每个连接独立计数（send_inflight_chunks_ 和 recv_inflight_chunks_）
//
// 工作原理：
// 1. reserve_*_slot: 在调度 chunk 前调用，检查是否有空闲配额
//    - 如果 counter < limit，递增 counter 并返回 true
//    - 如果 counter >= limit，返回 false（窗口已满）
// 2. release_*_slot: 在 chunk 完成后调用，释放配额
//    - 递减 counter
//    - 唤醒等待的调用者（window_cv_.notify_all）
//
// 流控效果：
// - 防止内存耗尽：限制 TCPX bounce buffer 使用量
// - 防止 GPU 资源耗尽：限制同时在途的 CUDA events 数量
// - 提高吞吐量：窗口大小调优可以平衡延迟和吞吐量
//
// 默认窗口大小：
// - max_send_inflight_ = 128（可通过环境变量 UCCL_TCPX_MAX_SEND_INFLIGHT 配置）
// - max_recv_inflight_ = 128（可通过环境变量 UCCL_TCPX_MAX_RECV_INFLIGHT 配置）
// ============================================================================

/**
 * 【滑动窗口】预留发送 slot（调度前调用）
 *
 * @param conn_id 连接 ID
 * @param limit 窗口大小限制（max_send_inflight_）
 * @return 成功返回 true，窗口已满返回 false
 */
bool Endpoint::reserve_send_slot(uint64_t conn_id, size_t limit) {
  std::lock_guard<std::mutex> lock(window_mu_);
  size_t& counter = send_inflight_chunks_[conn_id];
  if (counter >= limit) return false;  // 窗口已满
  ++counter;
  return true;
}

/**
 * 【滑动窗口】预留接收 slot（调度前调用）
 *
 * @param conn_id 连接 ID
 * @param limit 窗口大小限制（max_recv_inflight_）
 * @return 成功返回 true，窗口已满返回 false
 */
bool Endpoint::reserve_recv_slot(uint64_t conn_id, size_t limit) {
  std::lock_guard<std::mutex> lock(window_mu_);
  size_t& counter = recv_inflight_chunks_[conn_id];
  if (counter >= limit) return false;  // 窗口已满
  ++counter;
  return true;
}

/**
 * 【滑动窗口】释放发送 slot（chunk 完成后调用）
 *
 * 执行流程：
 * 1. 递减 send_inflight_chunks_[conn_id]
 * 2. 如果 counter 降为 0，从 map 中移除条目
 *
 * @param conn_id 连接 ID
 */
void Endpoint::release_send_slot(uint64_t conn_id) {
  std::lock_guard<std::mutex> lock(window_mu_);
  auto it = send_inflight_chunks_.find(conn_id);
  if (it == send_inflight_chunks_.end()) return;
  if (it->second <= 1) {
    send_inflight_chunks_.erase(it);  // counter 降为 0，移除条目
  } else {
    --(it->second);
  }
}

/**
 * 【滑动窗口】释放接收 slot（chunk 完成后调用）
 *
 * 执行流程：
 * 1. 递减 recv_inflight_chunks_[conn_id]
 * 2. 如果 counter 降为 0，从 map 中移除条目
 *
 * @param conn_id 连接 ID
 */
void Endpoint::release_recv_slot(uint64_t conn_id) {
  std::lock_guard<std::mutex> lock(window_mu_);
  auto it = recv_inflight_chunks_.find(conn_id);
  if (it == recv_inflight_chunks_.end()) return;
  if (it->second <= 1) {
    recv_inflight_chunks_.erase(it);  // counter 降为 0，移除条目
  } else {
    --(it->second);
  }
}

/**
 * 调度发送 chunks（Stage 0，持锁调用）
 *
 * 执行流程：
 * 1. 从 next_chunk_to_post 开始遍历未提交的 chunks
 * 2. 调用 reserve_send_slot 检查滑动窗口是否有空闲配额
 * 3. 如果窗口已满，停止调度（返回 kNoProgress）
 * 4. 调用 tcpx_isend 提交 chunk
 * 5. 如果返回 kTcpxBusy 或 request 为空，短暂休眠后重试（最多 512 次）
 * 6. 成功后标记 chunk.posted = true 并加入 send_queue
 * 7. 递增 next_chunk_to_post
 *
 * 设计要点：
 * - 滑动窗口流控：限制同时在途的 chunks 数量（max_send_inflight_）
 * - 重试机制：如果 TCPX 返回 kTcpxBusy，休眠 5 微秒后重试
 * - 批量调度：尽可能多地调度 chunks，直到窗口满或所有 chunks 已提交
 * - 错误处理：如果 tcpx_isend 失败，释放窗口配额并返回 kError
 *
 * @param conn 连接对象
 * @param transfer 传输对象
 * @return 调度结果（kNoProgress/kProgress/kError）
 */
Endpoint::ScheduleOutcome Endpoint::schedule_send_chunks_locked(
    Conn& conn, PendingTransfer& transfer) {
  size_t limit = max_send_inflight_;

  constexpr int kBusyRetryMax = 512;
  constexpr int kBusySleepMicros = 5;

  bool posted_any = false;

  // 从 next_chunk_to_post 开始遍历未提交的 chunks
  while (transfer.next_chunk_to_post < transfer.chunks.size()) {
    // 【滑动窗口流控】检查是否有空闲配额
    if (!reserve_send_slot(conn.conn_id, limit)) break;

    auto& chunk = transfer.chunks[transfer.next_chunk_to_post];
    bool chunk_posted = false;
    int attempt = 0;

    // 【重试机制】如果返回 kTcpxBusy，短暂休眠后重试
    while (attempt < kBusyRetryMax) {
      void* request = nullptr;

      // 调用 tcpx_isend 提交 chunk
      int rc = tcpx_isend(conn.send_comm, chunk.dst_ptr,
                          static_cast<int>(chunk.bytes),
                          static_cast<int>(chunk.tag), transfer.mhandle,
                          &request);
      if (debug_enabled_) {
        std::cerr << "[tcpx] isend rc=" << rc << " req=" << request
                  << " chunk_off=" << chunk.offset
                  << " chunk_bytes=" << chunk.bytes
                  << " tag=" << chunk.tag << std::endl;
      }

      // 成功：request 非空
      if (rc == 0 && request) {
        chunk.request = request;
        chunk.posted = true;
        chunk_posted = true;
        break;
      }

      // 失败：释放窗口配额
      release_send_slot(conn.conn_id);

      // kTcpxBusy 或 request 为空：重试
      if (rc == kTcpxBusy || (rc == 0 && !request)) {
        ++attempt;
        std::this_thread::sleep_for(
            std::chrono::microseconds(kBusySleepMicros));
        // 重新获取窗口配额
        if (!reserve_send_slot(conn.conn_id, limit)) break;
        continue;
      }

      // 其他错误：返回 kError
      std::cerr << "[tcpx] tcpx_isend failed rc=" << rc
                << " chunk_offset=" << chunk.offset
                << " chunk_bytes=" << chunk.bytes << std::endl;
      return ScheduleOutcome::kError;
    }

    // 如果重试次数耗尽，停止调度
    if (!chunk_posted) break;

    // 加入 send_queue 并递增 next_chunk_to_post
    transfer.send_queue.push_back(transfer.next_chunk_to_post);
    transfer.next_chunk_to_post++;
    posted_any = true;
  }

  return posted_any ? ScheduleOutcome::kProgress
                    : ScheduleOutcome::kNoProgress;
}

/**
 * 调度接收 chunks（Stage 0，持锁调用）
 *
 * 执行流程：
 * 1. 从 next_chunk_to_post 开始遍历未提交的 chunks
 * 2. 调用 reserve_recv_slot 检查滑动窗口是否有空闲配额
 * 3. 如果窗口已满，停止调度（返回 kNoProgress）
 * 4. 调用 tcpx_irecv 提交 chunk
 * 5. 如果返回 kTcpxBusy 或 request 为空，短暂休眠后重试（最多 512 次）
 * 6. 成功后标记 chunk.posted = true 并加入 recv_stage1_queue
 * 7. 递增 next_chunk_to_post
 *
 * 设计要点：
 * - 滑动窗口流控：限制同时在途的 chunks 数量（max_recv_inflight_）
 * - 重试机制：如果 TCPX 返回 kTcpxBusy，休眠 5 微秒后重试
 * - 批量调度：尽可能多地调度 chunks，直到窗口满或所有 chunks 已提交
 * - 错误处理：如果 tcpx_irecv 失败，释放窗口配额并返回 kError
 * - 接收路径特殊性：数据到达 bounce buffer，需要后续 GPU unpack（Stage 2）
 *
 * @param conn 连接对象
 * @param transfer 传输对象
 * @return 调度结果（kNoProgress/kProgress/kError）
 */
Endpoint::ScheduleOutcome Endpoint::schedule_recv_chunks_locked(
    Conn& conn, PendingTransfer& transfer) {
  size_t limit = max_recv_inflight_;

  constexpr int kBusyRetryMax = 512;
  constexpr int kBusySleepMicros = 5;

  bool posted_any = false;

  // 从 next_chunk_to_post 开始遍历未提交的 chunks
  while (transfer.next_chunk_to_post < transfer.chunks.size()) {
    // 【滑动窗口流控】检查是否有空闲配额
    if (!reserve_recv_slot(conn.conn_id, limit)) break;

    auto& chunk = transfer.chunks[transfer.next_chunk_to_post];
    bool chunk_posted = false;
    int attempt = 0;

    // 【重试机制】如果返回 kTcpxBusy，短暂休眠后重试
    while (attempt < kBusyRetryMax) {
      // tcpx_irecv 使用数组接口（支持批量接收）
      void* buffers[1] = {chunk.dst_ptr};
      int sizes[1] = {static_cast<int>(chunk.bytes)};
      int tags[1] = {static_cast<int>(chunk.tag)};
      void* mhandles[1] = {transfer.mhandle};
      void* requests[1] = {nullptr};

      // 调用 tcpx_irecv 提交 chunk
      int rc = tcpx_irecv(conn.recv_comm, 1, buffers, sizes, tags, mhandles,
                          requests);
      if (debug_enabled_) {
        std::cerr << "[tcpx] irecv rc=" << rc << " req=" << requests[0]
                  << " chunk_off=" << chunk.offset
                  << " chunk_bytes=" << chunk.bytes
                  << " tag=" << chunk.tag << std::endl;
      }

      // 成功：request 非空
      if (rc == 0 && requests[0]) {
        chunk.request = requests[0];
        chunk.posted = true;
        chunk_posted = true;
        break;
      }

      // 失败：释放窗口配额
      release_recv_slot(conn.conn_id);

      // kTcpxBusy 或 request 为空：重试
      if (rc == kTcpxBusy || (rc == 0 && !requests[0])) {
        ++attempt;
        std::this_thread::sleep_for(
            std::chrono::microseconds(kBusySleepMicros));
        // 重新获取窗口配额
        if (!reserve_recv_slot(conn.conn_id, limit)) break;
        continue;
      }

      // 其他错误：返回 kError
      std::cerr << "[tcpx] tcpx_irecv failed rc=" << rc
                << " chunk_offset=" << chunk.offset
                << " chunk_bytes=" << chunk.bytes << std::endl;
      return ScheduleOutcome::kError;
    }

    // 如果重试次数耗尽，停止调度
    if (!chunk_posted) break;

    // 加入 recv_stage1_queue（等待网络完成）并递增 next_chunk_to_post
    transfer.recv_stage1_queue.push_back(transfer.next_chunk_to_post);
    transfer.next_chunk_to_post++;
    posted_any = true;
  }

  return posted_any ? ScheduleOutcome::kProgress
                    : ScheduleOutcome::kNoProgress;
}

// ---------------------------------------------------------------------------
// Transfer scheduling helpers (chunk partition, sliding window posting)
// ---------------------------------------------------------------------------
/**
 * 提交发送传输（send_async 的内部实现）
 *
 * 执行流程：
 * 1. 获取 TCPX 内存注册句柄（mhandle）
 * 2. 将数据分割成多个 chunks（默认 512KB 每个）
 * 3. 创建 PendingTransfer 对象并加入 transfer_map_
 * 4. 调用 schedule_send_chunks_locked 开始调度第一批 chunks
 * 5. 返回 transfer_id 给调用者
 *
 * 数据分割策略：
 * - chunk_bytes = 512KB（默认）
 * - chunk_count = ceil(size / chunk_bytes)
 * - 每个 chunk 有独立的 tag = base_tag + chunk_index
 *
 * 流水线设计：
 * - 发送端是单阶段流水线（只有网络传输，无 GPU unpack）
 * - chunks 被加入 send_queue，等待 tcpx_isend 提交
 * - 用户通过 poll_async 推进传输，直到所有 chunks 完成
 *
 * @param conn 连接对象
 * @param mr_id 内存注册 ID
 * @param mr 内存注册条目
 * @param data 发送数据指针（必须在 mr 范围内）
 * @param size 发送数据大小
 * @param tag 基础 tag（每个 chunk 的 tag = base_tag + chunk_index）
 * @param transfer_id 输出：传输 ID
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::post_send_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                          void const* data, size_t size, int tag,
                          uint64_t& transfer_id) {
  if (!data || size == 0) return false;

  // 【步骤 1】获取 TCPX 内存注册句柄
  void* mhandle = nullptr;
  if (!populate_conn_handles_(conn, mr_id, /*is_recv=*/false, &mhandle))
    return false;

  // 【步骤 2】计算 chunk 分割参数
  size_t chunk_bytes = std::max<size_t>(1, chunk_bytes_);
  size_t chunk_count = (size + chunk_bytes - 1) / chunk_bytes;

  // 【步骤 3】边界检查：确保数据在 MR 范围内
  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t data_addr = reinterpret_cast<uintptr_t>(data);
  if (data_addr < base_addr || data_addr + size > base_addr + mr.size) {
    std::cerr << "[tcpx] send_async data out of bounds" << std::endl;
    return false;
  }

  // 【步骤 4】创建 PendingTransfer 对象
  PendingTransfer transfer;
  transfer.kind = PendingTransfer::Kind::kSend;
  transfer.transfer_id = next_transfer_id_.fetch_add(1);
  transfer.conn_id = conn.conn_id;
  transfer.mr_id = mr_id;
  transfer.total_bytes = size;
  transfer.base_tag = static_cast<uint32_t>(tag);
  transfer.mhandle = mhandle;
  transfer.chunks.reserve(chunk_count);

  // 【步骤 5】分割成 chunks
  auto base_ptr = static_cast<char const*>(data);
  for (size_t idx = 0; idx < chunk_count; ++idx) {
    size_t offset = idx * chunk_bytes;
    size_t chunk_len = std::min(chunk_bytes, size - offset);
    PendingTransfer::ChunkState chunk;
    chunk.offset = offset;
    chunk.bytes = chunk_len;
    chunk.tag = static_cast<uint32_t>(tag) + static_cast<uint32_t>(idx);
    chunk.needs_unpack = false;  // 发送端不需要 unpack
    chunk.dst_ptr = const_cast<char*>(base_ptr) + offset;
    transfer.chunks.push_back(std::move(chunk));
  }

  // 【步骤 6】加入 transfer_map_ 并开始调度
  uint64_t tid = transfer.transfer_id;
  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    auto [it, inserted] = transfer_map_.emplace(tid, std::move(transfer));
    PendingTransfer& stored = it->second;

    // 立即调度第一批 chunks（受滑动窗口限制）
    auto outcome = schedule_send_chunks_locked(conn, stored);
    if (outcome == ScheduleOutcome::kError) {
      transfer_map_.erase(it);
      return false;
    }
  }

  transfer_id = tid;
  return true;
}

/**
 * 提交接收传输（recv_async 的内部实现）
 *
 * 执行流程：
 * 1. 获取 TCPX 内存注册句柄（mhandle）
 * 2. 将数据分割成多个 chunks（默认 512KB 每个）
 * 3. 创建 PendingTransfer 对象并加入 transfer_map_
 * 4. 调用 schedule_recv_chunks_locked 开始调度第一批 chunks
 * 5. 返回 transfer_id 给调用者
 *
 * 数据分割策略：
 * - chunk_bytes = 512KB（默认）
 * - chunk_count = ceil(size / chunk_bytes)
 * - 每个 chunk 有独立的 tag = base_tag + chunk_index
 *
 * 流水线设计：
 * - 接收端是两阶段流水线（网络传输 + GPU unpack）
 * - Stage 1: chunks 被加入 recv_stage1_queue，等待 tcpx_irecv 提交
 * - Stage 2: 网络完成后，chunks 被移入 recv_stage2_queue，等待 GPU unpack 完成
 * - 用户通过 poll_async 推进传输，直到所有 chunks 完成
 *
 * needs_unpack 参数：
 * - true: 需要 GPU unpack kernel（recv_async 调用）
 * - false: 不需要 unpack（READ 操作，数据直接写入目标内存）
 *
 * @param conn 连接对象
 * @param mr_id 内存注册 ID
 * @param mr 内存注册条目
 * @param data 接收数据指针（必须在 mr 范围内）
 * @param size 接收数据大小
 * @param tag 基础 tag（每个 chunk 的 tag = base_tag + chunk_index）
 * @param transfer_id 输出：传输 ID
 * @param needs_unpack 是否需要 GPU unpack kernel
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::post_recv_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                          void* data, size_t size, int tag,
                          uint64_t& transfer_id, bool needs_unpack) {
  if (!data || size == 0) return false;

  // 【步骤 1】获取 TCPX 内存注册句柄
  void* mhandle = nullptr;
  if (!populate_conn_handles_(conn, mr_id, /*is_recv=*/true, &mhandle))
    return false;

  // 【步骤 2】计算 chunk 分割参数
  size_t chunk_bytes = std::max<size_t>(1, chunk_bytes_);
  size_t chunk_count = (size + chunk_bytes - 1) / chunk_bytes;

  // 【步骤 3】边界检查：确保数据在 MR 范围内
  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(data);
  if (dst_addr < base_addr || dst_addr + size > base_addr + mr.size) {
    std::cerr << "[tcpx] recv_async destination out of bounds" << std::endl;
    return false;
  }

  // 【步骤 4】创建 PendingTransfer 对象
  PendingTransfer transfer;
  transfer.kind = needs_unpack ? PendingTransfer::Kind::kRecv
                               : PendingTransfer::Kind::kRead;
  transfer.transfer_id = next_transfer_id_.fetch_add(1);
  transfer.conn_id = conn.conn_id;
  transfer.mr_id = mr_id;
  transfer.total_bytes = size;
  transfer.base_tag = static_cast<uint32_t>(tag);
  transfer.mhandle = mhandle;
  transfer.chunks.reserve(chunk_count);

  // 【步骤 5】分割成 chunks
  auto base_ptr = static_cast<char*>(data);
  for (size_t idx = 0; idx < chunk_count; ++idx) {
    size_t offset = idx * chunk_bytes;
    size_t chunk_len = std::min(chunk_bytes, size - offset);
    PendingTransfer::ChunkState chunk;
    chunk.offset = offset;
    chunk.bytes = chunk_len;
    chunk.tag = static_cast<uint32_t>(tag) + static_cast<uint32_t>(idx);
    chunk.needs_unpack = needs_unpack;  // 接收端可能需要 GPU unpack
    chunk.dst_ptr = base_ptr + offset;
    transfer.chunks.push_back(std::move(chunk));
  }

  // 【步骤 6】加入 transfer_map_ 并开始调度
  uint64_t tid = transfer.transfer_id;
  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    auto [it, inserted] = transfer_map_.emplace(tid, std::move(transfer));
    PendingTransfer& stored = it->second;

    // 立即调度第一批 chunks（受滑动窗口限制）
    auto outcome = schedule_recv_chunks_locked(conn, stored);
    if (outcome == ScheduleOutcome::kError) {
      transfer_map_.erase(it);
      return false;
    }
  }

  transfer_id = tid;
  return true;
}

/**
 * 【Stage 1】轮询 TCPX 网络传输完成状态
 *
 * 功能：
 * - 调用 tcpx_test 查询 chunk 的网络传输是否完成
 * - 网络完成 = 数据已到达 bounce buffer（CPU 端 pinned memory）
 * - 对于接收端，还需要 Stage 2（GPU unpack）才能真正完成
 *
 * tcpx_test 返回值：
 * - 0: 成功，检查 completed 标志
 * - kTcpxBusy: TCPX 运行时还没有进度，继续等待
 * - 2: 对端关闭连接（如果 completed=1 则忽略，否则继续等待）
 * - 其他: 错误
 *
 * 设计要点：
 * - 非阻塞：tcpx_test 立即返回，不等待
 * - 幂等：可以多次调用，直到 completed=1
 * - 错误处理：区分暂时性错误（kTcpxBusy）和永久性错误
 *
 * @param transfer 传输对象
 * @param chunk chunk 状态
 * @param done 输出：是否完成
 * @param received_size 输出：接收到的字节数（可选）
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::poll_chunk_request_(PendingTransfer& transfer,
                                   PendingTransfer::ChunkState& chunk,
                                   bool* done, int* received_size) {
  *done = false;
  if (!chunk.request) return false;

  int completed = 0;
  int size = 0;

  // 【Stage 1】查询 TCPX 网络传输进度
  // tcpx_test 立即返回，completed=1 表示数据已到达 bounce buffer
  int rc = tcpx_test(chunk.request, &completed, &size);

  if (debug_enabled_ && rc != kTcpxBusy) {
    std::cerr << "[tcpx] test transfer=" << transfer.transfer_id
              << " chunk_off=" << chunk.offset << " rc=" << rc
              << " completed=" << completed << " size=" << size << std::endl;
  }

  // kTcpxBusy: TCPX 运行时还没有进度，继续等待
  if (rc == kTcpxBusy) {
    return true;
  }

  // rc=2: 对端关闭连接
  // 如果 chunk 已完成则忽略，否则继续等待（对齐原来的 TcpxTransfer 逻辑）
  if (rc == 2) {
    if (!completed) {
      if (debug_enabled_) {
        std::cerr << "[tcpx] tcpx_test reported connection close (rc=2) for "
                  << "transfer_id=" << transfer.transfer_id
                  << " offset=" << chunk.offset << " (not ready)" << std::endl;
      }
      return true;
    }
  } else if (rc != 0) {
    // 其他错误：永久性失败
    std::cerr << "[tcpx] tcpx_test failed rc=" << rc
              << " transfer_id=" << transfer.transfer_id
              << " offset=" << chunk.offset << std::endl;
    return false;
  }

  // completed=0: 还没完成，继续等待
  if (!completed) return true;

  // completed=1: 网络传输完成
  *done = true;
  if (received_size) *received_size = size;
  return true;
}

/**
 * 【Stage 2 启动】启动 GPU unpack kernel（接收端专用）
 *
 * 功能：
 * - 将数据从 bounce buffer（CPU pinned memory）拷贝到目标 GPU 内存
 * - 使用 CUDA kernel 执行异步拷贝（比 cudaMemcpy 更高效）
 * - 记录 CUDA event 用于跟踪 kernel 完成状态
 *
 * 执行流程：
 * 1. 从 TCPX request 中提取 fragment 元数据（src_off, len, dst_off）
 * 2. 构建 UnpackDescriptorBlock（描述所有 fragments）
 * 3. 启动 GPU unpack kernel（在 unpack_stream_ 上）
 * 4. 记录 CUDA event（用于 Stage 2 完成检测）
 * 5. 保存 event 引用到 chunk（不拥有，由 Conn::recv_events pool 管理）
 *
 * Bounce Buffer 设计：
 * - TCPX 使用 bounce buffer 作为中间存储（CPU pinned memory）
 * - 网络数据先到达 bounce buffer（Stage 1 完成）
 * - GPU kernel 再从 bounce buffer 拷贝到目标 GPU 内存（Stage 2）
 * - 这样可以解耦网络传输和 GPU 操作，提高流水线效率
 *
 * Fragment 元数据：
 * - frag_cnt: fragment 数量（一个 chunk 可能被分成多个 fragments）
 * - src_off: 在 bounce buffer 中的偏移
 * - len: fragment 长度
 * - dst_off: 在目标 GPU 内存中的偏移
 *
 * CUDA Event Pool 设计：
 * - 对齐原来的 ChannelWindow::events 设计
 * - 预分配固定数量的 events（MAX_INFLIGHT_PER_CHANNEL）
 * - 循环复用：event_idx = event_counter % pool_size
 * - 避免频繁创建/销毁 events（性能优化）
 *
 * @param transfer 传输对象
 * @param chunk chunk 状态
 * @param request TCPX request（包含 fragment 元数据）
 * @param conn 连接对象
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::enqueue_chunk_unpack_(PendingTransfer& transfer,
                                     PendingTransfer::ChunkState& chunk,
                                     tcpx::plugin::tcpxRequest* request,
                                     Conn& conn) {
  ScopedCudaContext ctx_guard(cu_context_, static_cast<int>(local_gpu_idx_));

  if (!request) return false;

  // 【步骤 1】确保 recv_dev_handle 已缓存（包含 bounce_buf 指针）
  if (!conn.recv_dev_handle_cached) {
    if (!ensure_recv_dev_handle_cached_(conn)) return false;
  }
  auto const& dev_handle = conn.recv_dev_handle_host;

  // 【步骤 2】提取 fragment 数量
  uint64_t frag_cnt = 0;
  if (!request->unpack_slot.cnt) {
    std::cerr << "[tcpx] missing unpack counter" << std::endl;
    return false;
  }
  frag_cnt = *(request->unpack_slot.cnt);

  if (debug_enabled_) {
    std::cerr << "[tcpx] unpack transfer_id=" << transfer.transfer_id
              << " chunk_off=" << chunk.offset << " frag_cnt=" << frag_cnt
              << std::endl;
  }

  if (frag_cnt == 0 || frag_cnt > MAX_UNPACK_DESCRIPTORS) {
    std::cerr << "[tcpx] invalid fragment count " << frag_cnt << std::endl;
    return false;
  }

  if (debug_enabled_) {
    std::cerr << "[tcpx] dev_handle bounce_buf=" << dev_handle.bounce_buf
              << std::endl;
  }

  // 【步骤 3】构建 UnpackDescriptorBlock
  // 将 TCPX plugin 提供的元数据转换为 CUDA kernel 期望的格式
  auto* meta_entries =
      static_cast<tcpx::plugin::loadMeta*>(request->unpack_slot.mem);
  tcpx::rx::UnpackDescriptorBlock block;
  tcpx::rx::buildDescriptorBlock(meta_entries, static_cast<uint32_t>(frag_cnt),
                                 dev_handle.bounce_buf, chunk.dst_ptr,
                                 block);

  // 调试采样（cuMemcpyDtoH）已移除：避免在大规模传输下带来额外同步与日志开销。

  // 【调试】打印 fragment 元数据
  if (debug_enabled_) {
    for (uint32_t i = 0; i < std::min<uint32_t>(frag_cnt, 4); ++i) {
      auto const& m = block.descriptors[i];
      std::cerr << "  meta[" << i << "] src_off=" << m.src_off << " len="
                << m.len << " dst_off=" << m.dst_off << " (chunk_off="
                << chunk.offset << ")" << std::endl;
    }
    if (frag_cnt > 4) {
      for (uint32_t i = frag_cnt - std::min<uint32_t>(frag_cnt, 4); i < frag_cnt;
           ++i) {
        auto const& m = block.descriptors[i];
        std::cerr << "  meta[" << i << "] src_off=" << m.src_off
                  << " len=" << m.len << " dst_off=" << m.dst_off
                  << " (tail)" << std::endl;
      }
    }
  }

  block.ready_flag = request->unpack_slot.cnt;
  block.ready_threshold = frag_cnt;

  // 【步骤 4】启动 GPU unpack kernel
  if (debug_enabled_) {
    std::cerr << "[tcpx] launch unpack: count=" << block.count
              << " total_bytes=" << block.total_bytes << std::endl;
  }
  int launch_rc = unpack_launcher_->launch(block, unpack_stream_);
  if (launch_rc != 0) {
    std::cerr << "[tcpx] unpack kernel launch failed rc=" << launch_rc
              << std::endl;
    return false;
  }

  // 【步骤 5】从 Event Pool 中获取 event（循环复用）
  // 对齐原来的 ChannelWindow::events 设计：
  //   int event_idx = win.chunk_counter % MAX_INFLIGHT_PER_CHANNEL;
  //   cudaEventRecord(win.events[event_idx], unpack_stream);
  size_t event_idx = conn.recv_events.empty()
                          ? 0
                          : (conn.event_counter % conn.recv_events.size());
  conn.event_counter++;

  cudaEvent_t event = conn.recv_events[event_idx];
  cudaError_t record_rc = cudaEventRecord(event, unpack_stream_);
  if (record_rc != cudaSuccess) {
    std::cerr << "[tcpx] cudaEventRecord failed: "
              << cudaGetErrorString(record_rc) << std::endl;
    return false;
  }

  // 【步骤 6】保存 event 引用（不拥有，由 Conn::recv_events pool 管理）
  chunk.event = event;
  chunk.event_idx = event_idx;
  chunk.desc_block = block;
  return true;
}

/**
 * 【Stage 2 完成】释放接收 chunk 的资源
 *
 * 功能：
 * - 通知 TCPX 运行时 bounce buffer 已被消费（可以复用）
 * - 清空 chunk 的 event 和 request 引用
 *
 * 执行流程：
 * 1. 调用 tcpx_irecv_consumed 释放 bounce buffer slot
 * 2. 清空 chunk.event 引用（不销毁，由 Event Pool 管理）
 * 3. 清空 chunk.request 引用
 *
 * 设计要点：
 * - tcpx_irecv_consumed: 告诉 TCPX 运行时 GPU unpack 已完成
 * - bounce buffer 可以被复用（用于新的 chunks）
 * - Event Pool 设计：events 预分配，循环复用，不在热路径上销毁
 * - 对齐原来的 ChannelWindow 设计
 *
 * @param conn 连接对象
 * @param chunk chunk 状态
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::finalize_recv_chunk_(Conn& conn,
                                    PendingTransfer::ChunkState& chunk) {
  // 【步骤 1】通知 TCPX 运行时 bounce buffer 已被消费
  if (conn.recv_comm) {
    int rc = tcpx_irecv_consumed(conn.recv_comm, 1, chunk.request);
    if (rc != 0) {
      std::cerr << "[tcpx] tcpx_irecv_consumed failed rc=" << rc
                << " chunk_offset=" << chunk.offset << std::endl;
      return false;
    }
  }

  // 【步骤 2】清空引用（不销毁）
  // ✅ 不销毁 event，只清空引用（event 由 Conn::recv_events pool 拥有）
  // 对齐原来的设计：events 在 ChannelWindow 中预分配，循环复用，不在热路径上销毁
  chunk.event = nullptr;
  chunk.request = nullptr;
  return true;
}

/**
 * 【核心流水线函数】推进传输的 Stage 1 和 Stage 2（持锁调用）
 *
 * ============================================================================
 * 发送端流程（单阶段流水线）
 * ============================================================================
 *
 * 队列：send_queue
 *
 * 流程：
 * 1. 遍历 send_queue（FIFO 顺序，只处理队列头部）
 * 2. 对每个 chunk 调用 poll_chunk_request_（查询 tcpx_test）
 * 3. 如果 tcpx_test 返回 completed=1：
 *    - 从 send_queue 中移除
 *    - 标记 chunk.stage2_done = true
 *    - 递增 transfer.chunks_completed
 *    - 释放窗口配额（release_send_slot）
 *    - 设置 trigger_send = true（触发重新调度）
 * 4. 如果 tcpx_test 返回 completed=0：
 *    - 停止处理（TCPX 要求 FIFO 顺序）
 *
 * 为什么是单阶段：
 * - 发送端不需要 GPU unpack（数据直接从 GPU 内存发送）
 * - 只需要等待网络传输完成（tcpx_test）
 *
 * ============================================================================
 * 接收端流程（两阶段流水线）
 * ============================================================================
 *
 * 队列：recv_stage1_queue -> recv_stage2_queue
 *
 * 【Stage 1】网络传输完成（数据到达 bounce buffer）
 *
 * 流程：
 * 1. 遍历 recv_stage1_queue（FIFO 顺序，只处理队列头部）
 * 2. 对每个 chunk 调用 poll_chunk_request_（查询 tcpx_test）
 * 3. 如果 tcpx_test 返回 completed=1：
 *    - 从 recv_stage1_queue 中移除
 *    - 标记 chunk.stage1_done = true
 *    - 如果 chunk.needs_unpack = true：
 *      a. 启动 GPU unpack kernel
 *      b. 将 chunk 索引加入 recv_stage2_queue
 *    - 如果 chunk.needs_unpack = false（READ 操作）：
 *      a. 调用 finalize_recv_chunk_ 释放资源
 *      b. 标记 chunk.stage2_done = true
 *      c. 递增 transfer.chunks_completed
 *      d. 释放窗口配额（release_recv_slot）
 *      e. 设置 trigger_recv = true
 * 4. 如果 tcpx_test 返回 completed=0：
 *    - 停止处理（TCPX 要求 FIFO 顺序）
 *
 * 【Stage 2】GPU unpack 完成（数据从 bounce buffer 拷贝到目标 GPU 内存）
 *
 * 流程：
 * 1. 遍历 recv_stage2_queue（遍历整个队列，处理所有已 ready 的 events）
 * 2. 对每个 chunk 调用 cudaEventQuery（查询 GPU kernel 是否完成）
 * 3. 如果 cudaEventQuery 返回 cudaSuccess：
 *    - 调用 finalize_recv_chunk_ 释放资源（bounce buffer slot）
 *    - 标记 chunk.stage2_done = true
 *    - 递增 transfer.chunks_completed
 *    - 释放窗口配额（release_recv_slot）
 *    - 从 recv_stage2_queue 中移除
 *    - 设置 trigger_recv = true
 * 4. 如果 cudaEventQuery 返回 cudaErrorNotReady：
 *    - 跳过该 chunk，继续检查下一个（允许乱序完成）
 *
 * ============================================================================
 * 设计要点
 * ============================================================================
 *
 * 1. FIFO 顺序（Stage 1）：
 *    - TCPX 要求按 FIFO 顺序 test requests（只能 test "next in line"）
 *    - send_queue 和 recv_stage1_queue 只处理队列头部
 *    - 如果头部未完成则停止处理
 *
 * 2. 乱序完成（Stage 2）：
 *    - GPU unpack kernels 可能乱序完成
 *    - recv_stage2_queue 遍历整个队列，处理所有已 ready 的 events
 *    - 修复了之前只检查队列头部的 bug
 *
 * 3. 释放窗口配额：
 *    - 每完成一个 chunk 就释放配额（release_*_slot）
 *    - 允许 Stage 0 调度新的 chunks
 *    - 充分利用滑动窗口，最大化吞吐量
 *
 * 4. 触发重新调度：
 *    - 如果有进度，设置 schedule_* 标志
 *    - 调用者（advance_transfer_locked）会再次调用 schedule_*_chunks_locked
 *    - 确保窗口始终保持满载
 *
 * 5. Bounce Buffer 设计：
 *    - 网络数据先到达 bounce buffer（CPU pinned memory）
 *    - GPU kernel 再从 bounce buffer 拷贝到目标 GPU 内存
 *    - 解耦网络传输和 GPU 操作，提高流水线效率
 *
 * @param conn 连接对象
 * @param transfer 传输对象
 * @param schedule_send 输出：是否需要调度更多发送 chunks
 * @param schedule_recv 输出：是否需要调度更多接收 chunks
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::progress_transfer_locked(Conn& conn, PendingTransfer& transfer,
                                        bool* schedule_send,
                                        bool* schedule_recv) {
  bool trigger_send = false;
  bool trigger_recv = false;

  // ========================================================================
  // 发送端：单阶段流水线
  // ========================================================================
  if (transfer.kind == PendingTransfer::Kind::kSend) {
    // TCPX 要求按 FIFO 顺序 test requests（只能 test "next in line"）
    // 只处理队列头部，如果头部未完成则停止
    while (!transfer.send_queue.empty()) {
      size_t idx = transfer.send_queue.front();

      // 安全检查：索引越界
      if (idx >= transfer.chunks.size()) {
        transfer.send_queue.pop_front();
        continue;
      }

      auto& chunk = transfer.chunks[idx];

      // 跳过未提交的 chunks
      if (!chunk.posted) {
        break;  // 头部未提交，停止处理
      }

      // 【Stage 1】轮询 TCPX 请求状态
      bool done = false;
      int received = 0;
      if (!poll_chunk_request_(transfer, chunk, &done, &received))
        return false;

      // 如果未完成，停止处理（TCPX 要求 FIFO 顺序）
      if (!done) {
        break;
      }

      // 【完成】从队列中移除，标记完成，释放窗口配额
      transfer.send_queue.pop_front();
      chunk.stage2_done = true;
      transfer.chunks_completed++;
      chunk.request = nullptr;
      release_send_slot(transfer.conn_id);
      trigger_send = true;  // 触发重新调度
    }

    if (schedule_send) *schedule_send = trigger_send;
    if (schedule_recv) *schedule_recv = false;
    return true;
  }

  // ========================================================================
  // 接收端：两阶段流水线
  // ========================================================================
  else {
    // ------------------------------------------------------------------------
    // Stage 1: 推进网络传输完成（数据到达 bounce buffer）
    // ------------------------------------------------------------------------
    // TCPX 要求按 FIFO 顺序 test requests（只能 test "next in line"）
    // 只处理队列头部，如果头部未完成则停止
    while (!transfer.recv_stage1_queue.empty()) {
      size_t idx = transfer.recv_stage1_queue.front();
      if (idx >= transfer.chunks.size()) {
        transfer.recv_stage1_queue.pop_front();
        continue;
      }
      auto& chunk = transfer.chunks[idx];
      if (!chunk.posted) {
        break;  // 头部未提交，停止处理
      }
      if (chunk.stage1_done) {
        transfer.recv_stage1_queue.pop_front();
        continue;
      }

      // 对这个chunk调用一次tcpx_test
      bool done = false;
      int received = 0;
      if (!poll_chunk_request_(transfer, chunk, &done, &received))
        return false;

      if (!done) {
        // chunk还没完成，停止处理（TCPX 要求 FIFO 顺序）
        break;
      }

      // chunk已完成网络传输（数据已到达bounce buffer）
      chunk.stage1_done = true;
      transfer.recv_stage1_queue.pop_front();

      if (chunk.needs_unpack) {
        // 需要GPU unpack kernel：将数据从bounce buffer拷贝到目标GPU内存
        auto* rx_req = reinterpret_cast<tcpx::plugin::tcpxRequest*>(chunk.request);
        if (!enqueue_chunk_unpack_(transfer, chunk, rx_req, conn)) return false;
        transfer.recv_stage2_queue.push_back(idx);
      } else {
        // 不需要unpack（例如READ操作），直接完成
        if (!finalize_recv_chunk_(conn, chunk)) return false;
        chunk.stage2_done = true;
        transfer.chunks_completed++;
        chunk.request = nullptr;
        release_recv_slot(transfer.conn_id);
        trigger_recv = true;
      }
    }

    // Stage 2: 等待GPU unpack完成
    //
    // BUG修复：遍历整个队列，处理所有已ready的events
    //
    // 原来的错误逻辑：
    //   - 遇到第一个未ready的event就break
    //   - 问题：后面已ready的events会被忽略
    //
    // 正确做法：
    //   - 遍历整个recv_stage2_queue
    //   - 对每个event调用cudaEventQuery
    //   - 处理所有已ready的events
    //   - 未ready的events留在队列中
    auto stage2_it = transfer.recv_stage2_queue.begin();
    while (stage2_it != transfer.recv_stage2_queue.end()) {
      size_t idx = *stage2_it;
      if (idx >= transfer.chunks.size()) {
        stage2_it = transfer.recv_stage2_queue.erase(stage2_it);
        continue;
      }
      auto& chunk = transfer.chunks[idx];
      if (chunk.stage2_done) {
        stage2_it = transfer.recv_stage2_queue.erase(stage2_it);
        continue;
      }
      if (!chunk.event) {
        std::cerr << "[tcpx] missing CUDA event for chunk offset "
                  << chunk.offset << std::endl;
        if (!finalize_recv_chunk_(conn, chunk)) return false;
        chunk.stage2_done = true;
        transfer.chunks_completed++;
        chunk.request = nullptr;
        release_recv_slot(transfer.conn_id);
        trigger_recv = true;
        stage2_it = transfer.recv_stage2_queue.erase(stage2_it);
        continue;
      }

      cudaError_t err = cudaEventQuery(chunk.event);
      if (err == cudaErrorNotReady) {
        // Event还没ready，跳过它，继续检查下一个event
        ++stage2_it;
        continue;
      }
      if (err != cudaSuccess) {
        std::cerr << "[tcpx] cudaEventQuery failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
      }

      // Event已ready，完成这个chunk
      if (!finalize_recv_chunk_(conn, chunk)) return false;
      chunk.stage2_done = true;
      transfer.chunks_completed++;
      chunk.request = nullptr;
      release_recv_slot(transfer.conn_id);
      trigger_recv = true;
      stage2_it = transfer.recv_stage2_queue.erase(stage2_it);
    }
  }

  if (schedule_send) *schedule_send = trigger_send;
  if (schedule_recv) *schedule_recv = trigger_recv;
  return true;
}

/**
 * 【用户接口】轮询传输是否完成
 *
 * 执行流程：
 * 1. 持 transfer_mu_ 查找传输
 * 2. 如果不存在，说明已完成，返回 true
 * 3. 持 conn_mu_ 查找连接对象（嵌套锁）
 * 4. 调用 advance_transfer_locked 推进传输
 * 5. 如果完成，清理并释放锁，然后重置窗口计数器
 *
 * 【锁使用详解】
 *
 * 1. 锁顺序（避免死锁）：
 *    - 先获取 transfer_mu_（外层锁）
 *    - 再获取 conn_mu_（内层锁，使用嵌套作用域）
 *    - 这个顺序与 drive_transfer_ 一致
 *
 * 2. 为什么使用嵌套作用域获取 conn_mu_：
 *    - conn_mu_ 只需要持锁查找连接对象
 *    - 查找完成后立即释放 conn_mu_（离开作用域）
 *    - 减少锁持有时间，降低锁竞争
 *
 * 3. 为什么使用 shared_lock 而非 unique_lock：
 *    - conn_mu_ 是读写锁（shared_mutex）
 *    - 只需要读取 conn_map_，不需要修改
 *    - 使用 shared_lock 允许多个线程并发读取
 *
 * 4. 为什么先释放 transfer_mu_ 再调用 reset_conn_window_counters_：
 *    - 与 drive_transfer_ 相同的原因
 *    - 避免同时持有 transfer_mu_ 和 window_mu_
 *    - 减少锁竞争，提高并发性能
 *
 * @param transfer_id 传输 ID
 * @param is_done 输出：传输是否完成
 * @return 成功返回 true，失败返回 false
 */
bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  if (!is_done) return false;
  *is_done = false;

  // 【持锁 1】获取 transfer_mu_，保护 transfer_map_
  std::unique_lock<std::mutex> lock(transfer_mu_);

  auto it = transfer_map_.find(transfer_id);
  if (it == transfer_map_.end()) {
    // 传输已完成并移除
    if (debug_enabled_) {
      std::cerr << "[tcpx] poll transfer_id=" << transfer_id
                << " c" << std::endl;
    }
    *is_done = true;
    return true;
  }
  PendingTransfer& transfer = it->second;

  // 【持锁 2】嵌套获取 conn_mu_（读锁），查找连接对象
  // 使用嵌套作用域，查找完成后立即释放 conn_mu_
  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> conn_lock(conn_mu_);
    auto conn_it = conn_map_.find(transfer.conn_id);
    if (conn_it == conn_map_.end()) return false;
    conn_ptr = conn_it->second;
  }
  // 【释放锁 2】离开作用域，conn_mu_ 自动释放

  // 【持锁推进】调用 advance_transfer_locked 推进传输
  // 此时只持有 transfer_mu_，conn_mu_ 已释放
  bool transfer_complete = false;
  if (!advance_transfer_locked(*conn_ptr, transfer, &transfer_complete))
    return false;

  if (transfer_complete) {
    // 【持锁清理】从 transfer_map_ 中移除传输
    finalize_transfer_locked(it);

    // 【释放锁 1】在调用 reset_conn_window_counters_ 前释放 transfer_mu_
    lock.unlock();

    // 【持锁 3】reset_conn_window_counters_ 内部会持 window_mu_
    reset_conn_window_counters_(conn_ptr->conn_id);

    *is_done = true;
    return true;
  }

  if (debug_enabled_) {
    size_t posted = transfer.next_chunk_to_post;
    size_t completed = transfer.chunks_completed;
    size_t inflight = 0;
    for (auto const& chunk : transfer.chunks) {
      if (chunk.posted && !chunk.stage2_done) ++inflight;
    }
    std::cerr << "[tcpx] poll transfer=" << transfer.transfer_id
              << " state=pending"
              << " posted=" << posted << " inflight=" << inflight
              << " completed=" << completed << std::endl;
  }

  lock.unlock();
  return true;
}

bool Endpoint::progress_conn(uint64_t conn_id) {
  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second;
  }

  std::vector<uint64_t> pending;
  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    for (auto const& kv : transfer_map_) {
      if (kv.second.conn_id == conn_id) {
        pending.push_back(kv.first);
      }
    }
  }

  bool progressed = false;
  for (auto transfer_id : pending) {
    bool is_done = false;
    if (!poll_async(transfer_id, &is_done)) continue;
    progressed = true;
  }
  return progressed;
}







/**
 * 释放连接资源（在析构或连接关闭时调用）
 *
 * 清理顺序（重要！）：
 * 1. 收集该连接的所有 TCPX 内存注册句柄
 * 2. 重置窗口计数器
 * 3. 调用 tcpx_dereg_mr 释放所有句柄
 * 4. 销毁 CUDA Event Pool
 * 5. 关闭 TCPX send_comm 和 recv_comm
 * 6. 关闭 TCP 控制套接字
 *
 * 设计要点：
 * - 先收集句柄：避免持锁时间过长（tcpx_dereg_mr 可能阻塞）
 * - Event Pool 清理：对齐原来的 ChannelWindow 析构逻辑
 *
 * 【锁使用详解】
 *
 * 1. mr_mu_ 的持锁范围：
 *    - 只在收集句柄时持锁
 *    - 收集完成后立即释放锁（离开作用域）
 *    - 避免在持锁时调用 tcpx_dereg_mr（可能阻塞）
 *
 * 2. 为什么先收集句柄再释放锁：
 *    - tcpx_dereg_mr 可能阻塞（等待 TCPX 运行时完成）
 *    - 如果持锁调用，会阻塞其他线程访问 mr_map_
 *    - 先收集到 vector，释放锁后再调用 tcpx_dereg_mr
 *
 * 3. reset_conn_window_counters_ 的锁：
 *    - 内部会持 window_mu_
 *    - 此时不持 mr_mu_，避免同时持有多个锁
 *
 * 4. 清理顺序的重要性：
 *    - 先收集句柄（持锁时间短）
 *    - 再释放句柄（无锁，可能阻塞）
 *    - 最后销毁 CUDA events 和关闭 TCPX 通道
 *
 * @param conn 连接对象
 */
void Endpoint::free_conn_(std::shared_ptr<Conn> const& conn) {
  if (!conn) return;

  // 【步骤 1】收集该连接的所有 TCPX 内存注册句柄
  // 【持锁收集】持 mr_mu_ 遍历 mr_map_ 并收集句柄
  std::vector<void*> send_handles;
  std::vector<void*> recv_handles;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);

    // 遍历所有 MR，移除该连接的缓存句柄
    for (auto& kv : mr_map_) {
      auto& mr = kv.second;

      // 发送句柄
      auto send_it = mr.send_handles.find(conn->conn_id);
      if (send_it != mr.send_handles.end()) {
        send_handles.push_back(send_it->second);
        mr.send_handles.erase(send_it);
      }

      // 接收句柄
      auto recv_it = mr.recv_handles.find(conn->conn_id);
      if (recv_it != mr.recv_handles.end()) {
        recv_handles.push_back(recv_it->second);
        mr.recv_handles.erase(recv_it);
      }
    }
  }
  // 【释放锁】离开作用域，mr_mu_ 自动释放

  // 【步骤 2】重置窗口计数器并唤醒等待的调用者
  // reset_conn_window_counters_ 内部会持 window_mu_
  reset_conn_window_counters_(conn->conn_id);

  // 【步骤 3】释放 TCPX 内存注册句柄
  // 【无锁调用】tcpx_dereg_mr 可能阻塞，不持锁
  for (void* handle : send_handles) {
    if (conn->send_comm) tcpx_dereg_mr(conn->send_comm, handle);
  }
  for (void* handle : recv_handles) {
    if (conn->recv_comm) tcpx_dereg_mr(conn->recv_comm, handle);
  }

  // 【步骤 4】销毁 CUDA Event Pool
  // 对齐原来的设计：events 在 ChannelWindow 析构时销毁
  for (auto& event : conn->recv_events) {
    if (event) {
      cudaEventDestroy(event);
    }
  }
  conn->recv_events.clear();

  // 【步骤 5】关闭 TCPX 通道
  if (conn->send_comm) {
    tcpx_close_send(conn->send_comm);
    conn->send_comm = nullptr;
  }
  if (conn->recv_comm) {
    tcpx_close_recv(conn->recv_comm);
    conn->recv_comm = nullptr;
  }

  // 【步骤 6】关闭 TCP 控制套接字
  if (conn->ctrl_sock_fd >= 0) {
    ::close(conn->ctrl_sock_fd);
    conn->ctrl_sock_fd = -1;
  }
}

}  // namespace tcpx
