#include "tcpx_engine.h"
#include "tcpx/include/unpack_descriptor.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <algorithm>
#include <array>
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
// Supports multi-channel: num_channels can be > 1.
constexpr size_t kMaxChannels = 16;
struct ChannelHandleMsg {
  uint32_t num_channels;
  uint32_t reserved;
  std::array<ncclNetHandle_v7, kMaxChannels> handles;
};
static_assert(std::is_trivially_copyable<ChannelHandleMsg>::value,
              "ChannelHandleMsg must be trivially copyable");

// Helper functions to send/recv channel handles
bool send_channel_handles(int fd, std::vector<ncclNetHandle_v7> const& handles) {
  ChannelHandleMsg msg{};
  msg.num_channels = static_cast<uint32_t>(handles.size());
  msg.reserved = 0;
  for (size_t i = 0; i < handles.size() && i < kMaxChannels; ++i) {
    msg.handles[i] = handles[i];
  }
  return send_ctrl_struct(fd, msg);
}

bool recv_channel_handles(int fd, std::vector<ncclNetHandle_v7>& handles) {
  ChannelHandleMsg msg{};
  if (!recv_ctrl_struct(fd, msg)) return false;
  if (msg.num_channels == 0 || msg.num_channels > kMaxChannels) return false;
  handles.resize(msg.num_channels);
  for (size_t i = 0; i < msg.num_channels; ++i) {
    handles[i] = msg.handles[i];
  }
  return true;
}

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

Endpoint::Endpoint(uint32_t const /*num_cpus*/) {
  // Resolve local GPU index from env (default 0) to align with RDMA-style ctor.
  local_gpu_idx_ = 0;
  int override_gpu = get_env_int("UCCL_TCPX_LOCAL_DEVICE", -1);
  if (override_gpu < 0) override_gpu = get_env_int("UCCL_TCPX_DEVICE_IDX", -1);
  if (override_gpu >= 0) local_gpu_idx_ = static_cast<uint32_t>(override_gpu);

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
      std::cerr << "[tcpx] control socket bound on port " << ctrl_port_
                << " (attempt " << (attempt + 1) << "/" << max_attempts << ")"
                << std::endl;
      break;
    }
    if (errno != EADDRINUSE) {
      std::cerr << "[tcpx] control bind failed on port " << try_port
                << " with errno=" << errno << " (" << std::strerror(errno)
                << ")" << std::endl;
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

  // Multi-channel setup
  num_channels_ = get_env_size_t("UCCL_TCPX_NUM_CHANNELS", 1);
  if (num_channels_ == 0) num_channels_ = 1;

  // Initialize channel device IDs (all use the same dev_id for now)
  channel_dev_ids_.resize(num_channels_, dev_id_);

  // Prepare TCPX listen comms for inbound connections (one per channel)
  listen_handles_.resize(num_channels_);
  listen_comms_.resize(num_channels_);

  for (size_t i = 0; i < num_channels_; ++i) {
    if (tcpx_listen(channel_dev_ids_[i], &listen_handles_[i],
                    &listen_comms_[i]) != 0 || !listen_comms_[i]) {
      std::cerr << "[tcpx] listen failed for channel " << i << std::endl;
      throw std::runtime_error("tcpx: listen failed for channel " +
                               std::to_string(i));
    }
  }

  if (debug_enabled_) {
    std::cerr << "[tcpx] initialized " << num_channels_ << " channels"
              << std::endl;
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

  for (auto& lc : listen_comms_) {
    if (lc) {
      tcpx_close_listen(lc);
      lc = nullptr;
    }
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
std::vector<uint8_t> Endpoint::get_unified_metadata() {
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

  if (metadata.size() == 22) {
    char ip6_str[INET6_ADDRSTRLEN];
    std::memset(ip6_str, 0, sizeof(ip6_str));
    in6_addr ip6_addr{};
    std::memcpy(&ip6_addr, metadata.data(), sizeof(ip6_addr));
    if (!inet_ntop(AF_INET6, &ip6_addr, ip6_str, sizeof(ip6_str))) {
      throw std::runtime_error("tcpx metadata IPv6 decode failed");
    }
    uint16_t net_port =
        static_cast<uint16_t>((static_cast<uint16_t>(metadata[16]) << 8) |
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

  // Connection blueprint (client):
  //   - Spin up a temporary reverse listen handle so the server can connect
  //     back once the send path is established.
  //   - Establish the control socket and swap EndpointInfo + listen handles.
  //   - Use tcpx_connect_v5/tcpx_accept_v5 to materialize send/recv comms.
  //   - Cache device handles / preload CUDA events before publishing the conn.

  if (remote_port < 0) remote_port = ctrl_port_;

  auto conn = std::make_shared<Conn>();
  conn->init_channels(num_channels_);

  // Create reverse listen handles (one per channel) for server to connect back
  std::vector<void*> reverse_listens(num_channels_, nullptr);
  std::vector<ncclNetHandle_v7> reverse_handles(num_channels_);

  for (size_t i = 0; i < num_channels_; ++i) {
    if (tcpx_listen(channel_dev_ids_[i], &reverse_handles[i],
                    &reverse_listens[i]) != 0 || !reverse_listens[i]) {
      std::cerr << "[tcpx] reverse listen failed (client) channel=" << i
                << std::endl;
      // Clean up previously created listens
      for (size_t j = 0; j < i; ++j) {
        if (reverse_listens[j]) tcpx_close_listen(reverse_listens[j]);
      }
      return false;
    }
  }

  auto close_reverse_listens = [&]() {
    for (auto& rl : reverse_listens) {
      if (rl) {
        tcpx_close_listen(rl);
        rl = nullptr;
      }
    }
  };

  int sock_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    close_reverse_listens();
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(remote_port);
  if (inet_pton(AF_INET, ip_addr.c_str(), &addr.sin_addr) != 1) {
    std::cerr << "[tcpx] invalid IP address " << ip_addr << std::endl;
    ::close(sock_fd);
    close_reverse_listens();
    return false;
  }

  if (::connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
      0) {
    std::cerr << "[tcpx] connect() to " << ip_addr << ":" << remote_port
              << " failed: " << strerror(errno) << std::endl;
    ::close(sock_fd);
    close_reverse_listens();
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
    close_reverse_listens();
    return false;
  }

  EndpointInfo remote_info{};
  if (!recv_ctrl_struct(sock_fd, remote_info)) {
    ::close(sock_fd);
    close_reverse_listens();
    return false;
  }

  // Read the listen handle(s) the server advertised over the control socket.
  std::vector<ncclNetHandle_v7> server_handles;
  if (!recv_channel_handles(sock_fd, server_handles)) {
    ::close(sock_fd);
    close_reverse_listens();
    return false;
  }
  if (server_handles.size() != num_channels_) {
    std::cerr << "[tcpx] unexpected channel count from server: "
              << server_handles.size() << " expected: " << num_channels_
              << std::endl;
    ::close(sock_fd);
    close_reverse_listens();
    return false;
  }

  // Connect to all server channels
  for (size_t idx = 0; idx < num_channels_; ++idx) {
    ncclNetHandle_v7 handle_copy = server_handles[idx];
    if (tcpx_connect_v5(channel_dev_ids_[idx], &handle_copy,
                        &conn->channels[idx].send_comm,
                        &conn->channels[idx].send_dev_handle) != 0 ||
        !conn->channels[idx].send_comm) {
      std::cerr << "[tcpx] tcpx_connect_v5 failed (client) channel=" << idx
                << std::endl;
      ::close(sock_fd);
      close_reverse_listens();
      free_conn_(conn);
      return false;
    }
  }

  // Client returns the reverse-path handles using the helper function.
  if (!send_channel_handles(sock_fd, reverse_handles)) {
    ::close(sock_fd);
    for (auto& ch : conn->channels) {
      if (ch.send_comm) tcpx_close_send(ch.send_comm);
    }
    close_reverse_listens();
    free_conn_(conn);
    return false;
  }

  // Accept incoming connections on all reverse listen handles
  {
    constexpr int kMaxRetries = 100;
    constexpr int kRetryMs = 50;

    for (size_t idx = 0; idx < num_channels_; ++idx) {
      bool accepted = false;
      for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        int rc = tcpx_accept_v5(reverse_listens[idx],
                                &conn->channels[idx].recv_comm,
                                &conn->channels[idx].recv_dev_handle);
        if (rc == 0 && conn->channels[idx].recv_comm) {
          accepted = true;
          break;
        }
        std::cout << "[tcpx] client reverse accept retry rc=" << rc
                  << " channel=" << idx << " attempt=" << attempt + 1
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
      }
      if (!accepted) {
        std::cerr << "[tcpx] tcpx_accept_v5 failed (client reverse) channel="
                  << idx << std::endl;
        ::close(sock_fd);
        for (auto& ch : conn->channels) {
          if (ch.send_comm) tcpx_close_send(ch.send_comm);
          if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
        }
        close_reverse_listens();
        free_conn_(conn);
        return false;
      }
    }
  }
  close_reverse_listens();

  if (!recv_ctrl_ack(sock_fd)) {
    std::cerr << "[tcpx] missing ACK from server" << std::endl;
    ::close(sock_fd);
    for (auto& ch : conn->channels) {
      if (ch.send_comm) tcpx_close_send(ch.send_comm);
      if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
    }
    free_conn_(conn);
    return false;
  }

  conn_id = next_conn_id_.fetch_add(1);
  conn->conn_id = conn_id;
  conn->ip_addr = ip_addr;
  conn->remote_gpu_idx = remote_info.gpu;
  conn->remote_port = remote_port;
  conn->ctrl_sock_fd = sock_fd;

  // Cache recv device handles for all channels
  for (size_t idx = 0; idx < num_channels_; ++idx) {
    if (!ensure_recv_dev_handle_cached_(*conn, idx)) {
      std::cerr << "[tcpx] failed to cache recv device handle (client) channel="
                << idx << std::endl;
      free_conn_(conn);
      return false;
    }
  }

  // Preallocate the recv-side CUDA event pool for this connection.
  int max_recv_inflight =
      get_env_int("UCCL_TCPX_MAX_RECV_INFLIGHT", kDefaultRecvInflight);
  if (max_recv_inflight <= 0) max_recv_inflight = kDefaultRecvInflight;
  size_t pool_size = static_cast<size_t>(max_recv_inflight);
  conn->recv_events.resize(pool_size);

  for (size_t i = 0; i < pool_size; ++i) {
    cudaError_t rc =
        cudaEventCreateWithFlags(&conn->recv_events[i], cudaEventDisableTiming);
    if (rc != cudaSuccess) {
      std::cerr << "[tcpx] cudaEventCreateWithFlags failed for event " << i
                << ": " << cudaGetErrorString(rc) << std::endl;

      for (size_t j = 0; j < i; ++j) {
        cudaEventDestroy(conn->recv_events[j]);
      }

      ::close(sock_fd);
      for (auto& ch : conn->channels) {
        if (ch.send_comm) tcpx_close_send(ch.send_comm);
        if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
      }
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
  // Server-side sequence mirrors connect():
  //   - Accept the control socket and learn about the client's GPU/IP.
  //   - Immediately send back local EndpointInfo and listen handle so the
  //     client can connect to our recv comm.
  //   - After tcpx_accept_v5 returns, read the client's reverse listen handle
  //     and connect our send comm.
  //   - Cache device handles / CUDA events before publishing the connection.
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
  conn->init_channels(num_channels_);

  // Send our listen handles to the client
  if (!send_channel_handles(sock_fd, listen_handles_)) {
    ::close(sock_fd);
    return false;
  }

  // Accept incoming connections on all channels
  {
    constexpr int kMaxRetries = 100;
    constexpr int kRetryMs = 50;

    for (size_t idx = 0; idx < num_channels_; ++idx) {
      bool accepted = false;
      for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        int rc = tcpx_accept_v5(listen_comms_[idx],
                                &conn->channels[idx].recv_comm,
                                &conn->channels[idx].recv_dev_handle);
        if (rc == 0 && conn->channels[idx].recv_comm) {
          accepted = true;
          break;
        }
        std::cout << "[tcpx] server accept retry rc=" << rc
                  << " channel=" << idx << " attempt=" << attempt + 1
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
      }
      if (!accepted) {
        std::cerr << "[tcpx] tcpx_accept_v5 failed (server) channel=" << idx
                  << std::endl;
        ::close(sock_fd);
        for (auto& ch : conn->channels) {
          if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
        }
        return false;
      }
    }
  }

  // Receive client's reverse listen handles
  std::vector<ncclNetHandle_v7> client_handles;
  if (!recv_channel_handles(sock_fd, client_handles)) {
    ::close(sock_fd);
    for (auto& ch : conn->channels) {
      if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
    }
    return false;
  }
  if (client_handles.size() != num_channels_) {
    std::cerr << "[tcpx] unexpected channel count from client: "
              << client_handles.size() << " expected: " << num_channels_
              << std::endl;
    ::close(sock_fd);
    for (auto& ch : conn->channels) {
      if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
    }
    return false;
  }

  // Connect to all client reverse listen handles
  for (size_t idx = 0; idx < num_channels_; ++idx) {
    ncclNetHandle_v7 client_handle_copy = client_handles[idx];
    if (tcpx_connect_v5(channel_dev_ids_[idx], &client_handle_copy,
                        &conn->channels[idx].send_comm,
                        &conn->channels[idx].send_dev_handle) != 0 ||
        !conn->channels[idx].send_comm) {
      std::cerr << "[tcpx] tcpx_connect_v5 failed (server->client) channel="
                << idx << std::endl;
      ::close(sock_fd);
      for (auto& ch : conn->channels) {
        if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
        if (ch.send_comm) tcpx_close_send(ch.send_comm);
      }
      return false;
    }
  }

  if (!send_ctrl_ack(sock_fd)) {
    ::close(sock_fd);
    for (auto& ch : conn->channels) {
      if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
      if (ch.send_comm) tcpx_close_send(ch.send_comm);
    }
    return false;
  }

  conn_id = next_conn_id_.fetch_add(1);
  conn->conn_id = conn_id;
  conn->ip_addr = ip_addr;
  conn->remote_gpu_idx = client_info.gpu;
  conn->remote_port = ntohs(client_addr.sin_port);
  conn->ctrl_sock_fd = sock_fd;

  // Cache recv device handles for all channels
  for (size_t idx = 0; idx < num_channels_; ++idx) {
    if (!ensure_recv_dev_handle_cached_(*conn, idx)) {
      std::cerr << "[tcpx] failed to cache recv device handle (server) channel="
                << idx << std::endl;
      free_conn_(conn);
      return false;
    }
  }

  // Server side event pool mirrors the client logic above.
  int max_recv_inflight_acc =
      get_env_int("UCCL_TCPX_MAX_RECV_INFLIGHT", kDefaultRecvInflight);
  if (max_recv_inflight_acc <= 0) max_recv_inflight_acc = kDefaultRecvInflight;
  size_t pool_size_acc = static_cast<size_t>(max_recv_inflight_acc);
  conn->recv_events.resize(pool_size_acc);
  for (size_t i = 0; i < pool_size_acc; ++i) {
    cudaError_t rc =
        cudaEventCreateWithFlags(&conn->recv_events[i], cudaEventDisableTiming);
    if (rc != cudaSuccess) {
      std::cerr << "[tcpx] cudaEventCreateWithFlags failed for event " << i
                << ": " << cudaGetErrorString(rc) << std::endl;
      // Destroy events that were created before the failure.
      for (size_t j = 0; j < i; ++j) {
        cudaEventDestroy(conn->recv_events[j]);
      }
      ::close(sock_fd);
      for (auto& ch : conn->channels) {
        if (ch.recv_comm) tcpx_close_recv(ch.recv_comm);
        if (ch.send_comm) tcpx_close_send(ch.send_comm);
      }
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
  // Only caches the descriptor in mr_map_; actual tcpx_reg_mr calls are
  // deferred until populate_conn_handles_ is invoked for a specific connection.
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
  // Two-stage teardown to minimize lock hold time:
  //   1) Copy out the per-connection handles while holding mr_mu_.
  //   2) Release mr_mu_ and call tcpx_dereg_mr for each connection.
  struct HandleRef {
    uint64_t conn_id;
    size_t channel_idx;
    void* handle;
  };
  std::vector<HandleRef> send_handles;
  std::vector<HandleRef> recv_handles;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;

    for (auto const& kv : it->second.send_handles) {
      uint64_t conn_id = kv.first;
      auto const& handles_vec = kv.second;
      for (size_t ch_idx = 0; ch_idx < handles_vec.size(); ++ch_idx) {
        if (handles_vec[ch_idx]) {
          send_handles.push_back({conn_id, ch_idx, handles_vec[ch_idx]});
        }
      }
    }
    for (auto const& kv : it->second.recv_handles) {
      uint64_t conn_id = kv.first;
      auto const& handles_vec = kv.second;
      for (size_t ch_idx = 0; ch_idx < handles_vec.size(); ++ch_idx) {
        if (handles_vec[ch_idx]) {
          recv_handles.push_back({conn_id, ch_idx, handles_vec[ch_idx]});
        }
      }
    }
    mr_map_.erase(it);
  }

  auto dereg_for_conn = [&](HandleRef const& ref, bool is_send) {
    std::shared_lock<std::shared_mutex> conn_lock(conn_mu_);
    auto conn_it = conn_map_.find(ref.conn_id);
    if (conn_it == conn_map_.end()) return;
    std::shared_ptr<Conn> conn = conn_it->second;
    if (ref.channel_idx >= conn->channels.size()) return;
    auto& channel = conn->channels[ref.channel_idx];
    void* comm = is_send ? channel.send_comm : channel.recv_comm;
    if (comm) {
      tcpx_dereg_mr(comm, ref.handle);
    }
  };

  for (auto const& ref : send_handles) {
    dereg_for_conn(ref, /*is_send=*/true);
  }
  for (auto const& ref : recv_handles) {
    dereg_for_conn(ref, /*is_send=*/false);
  }

  return true;
}

bool Endpoint::populate_conn_handles_(Conn& conn, uint64_t mr_id, bool is_recv,
                                      size_t channel_idx, void** mhandle_out) {
  ScopedCudaContext ctx_guard(cu_context_, static_cast<int>(local_gpu_idx_));
  if (channel_idx >= conn.channels.size()) return false;
  auto& channel = conn.channels[channel_idx];
  void* comm = is_recv ? channel.recv_comm : channel.send_comm;
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
      auto const& handles_vec = handle_it->second;
      if (channel_idx < handles_vec.size() && handles_vec[channel_idx]) {
        existing = handles_vec[channel_idx];
      }
    }
    if (!existing) {
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
    std::cerr << "[tcpx] tcpx_reg_mr failed rc=" << rc << " channel="
              << channel_idx << std::endl;
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
    auto [insert_it, inserted] = table.try_emplace(conn.conn_id);
    auto& handles_vec = insert_it->second;
    if (handles_vec.size() <= channel_idx) {
      handles_vec.resize(channel_idx + 1, nullptr);
    }
    if (handles_vec[channel_idx]) {
      // Another thread beat us to it
      tcpx_dereg_mr(comm, mhandle);
      *mhandle_out = handles_vec[channel_idx];
      return true;
    }
    handles_vec[channel_idx] = mhandle;
  }

  *mhandle_out = mhandle;
  return true;
}

bool Endpoint::ensure_recv_dev_handle_cached_(Conn& conn,
                                              size_t channel_idx) {
  if (channel_idx >= conn.channels.size()) return false;
  auto& channel = conn.channels[channel_idx];
  if (channel.recv_dev_handle_cached) return true;
  if (!channel.recv_dev_handle) return false;

  auto* dev_handle_struct =
      reinterpret_cast<tcpx::plugin::NcclNetDeviceHandle*>(
          channel.recv_dev_handle);
  if (!dev_handle_struct || !dev_handle_struct->handle) return false;

  ScopedCudaContext ctx_guard(cu_context_, static_cast<int>(local_gpu_idx_));
  CUresult cu_rc =
      cuMemcpyDtoH(&channel.recv_dev_handle_host,
                   reinterpret_cast<CUdeviceptr>(dev_handle_struct->handle),
                   sizeof(channel.recv_dev_handle_host));
  if (cu_rc != CUDA_SUCCESS) {
    std::cerr << "[tcpx] cuMemcpyDtoH (cache recv handle) failed rc=" << cu_rc
              << " channel=" << channel_idx << std::endl;
    return false;
  }
  channel.recv_dev_handle_cached = true;
  return true;
}

// Polling entry point used by poll_async to drive Stage 0/1/2 progression while
// transfer_mu_ is held.
// Central driver that runs Stage 0 followed by Stage 1/2. Called only while
// transfer_mu_ is held to ensure exclusive access to the transfer state.
//
// Pipeline stages:
//   Stage 0: Schedule new chunks (schedule_send/recv_chunks_locked)
//   Stage 1: Poll network completions (progress_transfer_locked)
//   Stage 2: Poll GPU completions (progress_transfer_locked, receive only)
//
// Flow control:
//   - Stage 0 runs first to fill the pipeline
//   - Stage 1/2 run via progress_transfer_locked()
//   - If Stage 1/2 free window slots, Stage 0 runs again (pipelining)
bool Endpoint::advance_transfer_locked(Conn& conn, PendingTransfer& transfer,
                                       bool* transfer_complete) {
  if (!transfer_complete) return false;

  // Stage 0: Try to post new chunks if any remain unscheduled
  if (transfer.next_chunk_to_post < transfer.chunks.size()) {
    ScheduleOutcome outcome = ScheduleOutcome::kNoProgress;
    if (transfer.kind == PendingTransfer::Kind::kSend) {
      outcome = schedule_send_chunks_locked(conn, transfer);
    } else {
      outcome = schedule_recv_chunks_locked(conn, transfer);
    }
    if (outcome == ScheduleOutcome::kError) return false;
  }

  // Stage 1/2: Advance in-flight chunks through the pipeline
  bool schedule_send = false;
  bool schedule_recv = false;
  if (!progress_transfer_locked(conn, transfer, &schedule_send, &schedule_recv))
    return false;

  // If Stage 1/2 freed window slots, retry Stage 0 to keep pipeline full
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

void Endpoint::finalize_transfer_locked(
    std::unordered_map<uint64_t, PendingTransfer>::iterator it) {
  // Remove the transfer from transfer_map_ (transfer_mu_ held). Window counters
  // are reset later by poll_async once it drops transfer_mu_.
  transfer_map_.erase(it);
}

void Endpoint::reset_conn_window_counters_(uint64_t conn_id) {
  // After a transfer completes we nuke the inflight counters for the owning
  // connection so future transfers do not inherit stale slot usage.
  std::lock_guard<std::mutex> lock(window_mu_);
  send_inflight_chunks_.erase(conn_id);
  recv_inflight_chunks_.erase(conn_id);
}

// ---------------------------------------------------------------------------
// Data-plane entry points (advertise / send / recv / read)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// FIFO/read helpers
// ---------------------------------------------------------------------------
// advertise() is called by the passive peer (server) to publish a slice of a
// registered buffer. The listener thread forwards the resulting FifoItem to the
// active peer so it can issue a tagged READ.
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

// Active-side handler that turns a FIFO item into a tagged send. This is the
// mirror of advertise(): callers invoke it after they dequeue a FifoItem from
// the listener thread so the data path can remain fully asynchronous.
bool Endpoint::queue_read_response(uint64_t conn_id,
                                   FifoItem const& fifo_item) {
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

  return true;
}

// ---------------------------------------------------------------------------
// Asynchronous data path helpers (Stage 0 producers)
// ---------------------------------------------------------------------------
bool Endpoint::send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                          size_t size, uint64_t* transfer_id) {
  // Fast path uses an auto-assigned tag, which keeps the rest of the pipeline
  // identical to the explicit-tag variant.
  return send_async_with_tag(conn_id, mr_id, data, size, 0, transfer_id);
}

bool Endpoint::send_async_with_tag(uint64_t conn_id, uint64_t mr_id,
                                   void const* data, size_t size, uint32_t tag,
                                   uint64_t* transfer_id) {
  // Variant used by queue_read_response / read_async where the caller controls
  // the tag. The function slices the payload into chunks, seeds a
  // PendingTransfer and lets schedule_send_chunks_locked (Stage 0) post the
  // first batch.
  if (!data || size == 0) return false;

  // Look up the connection under a shared lock (hot path).
  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second;
  }

  // Snapshot the MR entry so the lock can be dropped before touching TCPX.
  MrEntry mr;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }

  // Validate that the caller-supplied pointer falls within the MR range.
  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t data_addr = reinterpret_cast<uintptr_t>(data);
  if (data_addr < base_addr || data_addr + size > base_addr + mr.size) {
    return false;
  }

  // Post the chunked transfer; transfer_id is returned to the caller so they
  // can drive completion via poll_async.
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
  // READs originate from the active peer after it consumes a slot from the
  // server-side FIFO. The tag inside slot_item ties the recv to the server's
  // pending send, allowing multi-chunk transfers without extra control traffic.
  if (!dst || size == 0) return false;

  // Grab the connection and MR metadata before touching CUDA/TCPX state.
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

  // Ensure the destination buffer is large enough and MR-backed.
  if (slot_item.size > mr.size) return false;
  if (size < slot_item.size) return false;
  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  if (dst_addr < base_addr || dst_addr + slot_item.size > base_addr + mr.size) {
    return false;
  }

  // Reuse the advertised tag so the passive peer can match this recv slot.
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
  // will use. Much like send_async_with_tag, the transfer is chunked, queued,
  // and schedule_recv_chunks_locked handles initial postings.
  if (!data || size == 0) return false;

  // Locate the connection instance under shared ownership.
  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return false;
    conn_ptr = it->second;
  }

  // Grab the MR metadata while holding mr_mu_.
  MrEntry mr;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);
    auto it = mr_map_.find(mr_id);
    if (it == mr_map_.end()) return false;
    mr = it->second;
  }

  // Bounds-check the receive buffer before posting anything to TCPX.
  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(data);
  if (dst_addr < base_addr || dst_addr + size > base_addr + mr.size) {
    return false;
  }

  // Create the pending transfer, marking that GPU unpack is required.
  uint64_t tid = 0;
  if (!post_recv_(*conn_ptr, mr_id, mr, data, size, static_cast<int>(tag), tid,
                  /*needs_unpack=*/true)) {
    return false;
  }

  if (transfer_id) *transfer_id = tid;
  return true;
}

bool Endpoint::reserve_send_slot(uint64_t conn_id, size_t limit) {
  // Sliding window flow control: check if we can post another send chunk.
  // Returns false if the window is full, causing Stage 0 to stop scheduling.
  std::lock_guard<std::mutex> lock(window_mu_);
  size_t& counter = send_inflight_chunks_[conn_id];
  if (counter >= limit) return false;
  ++counter;
  return true;
}

bool Endpoint::reserve_recv_slot(uint64_t conn_id, size_t limit) {
  // Sliding window flow control: check if we can post another receive chunk.
  // For receives, the slot remains reserved until Stage 2 completes.
  std::lock_guard<std::mutex> lock(window_mu_);
  size_t& counter = recv_inflight_chunks_[conn_id];
  if (counter >= limit) return false;
  ++counter;
  return true;
}

void Endpoint::release_send_slot(uint64_t conn_id) {
  // Called when a send chunk completes Stage 1 (network done).
  // Frees a window slot so Stage 0 can post more chunks.
  std::lock_guard<std::mutex> lock(window_mu_);
  auto it = send_inflight_chunks_.find(conn_id);
  if (it == send_inflight_chunks_.end()) return;
  if (it->second <= 1) {
    send_inflight_chunks_.erase(it);
  } else {
    --(it->second);
  }
}

void Endpoint::release_recv_slot(uint64_t conn_id) {
  // Called when a receive chunk completes Stage 2 (GPU unpack done).
  // Frees a window slot so Stage 0 can post more chunks.
  std::lock_guard<std::mutex> lock(window_mu_);
  auto it = recv_inflight_chunks_.find(conn_id);
  if (it == recv_inflight_chunks_.end()) return;
  if (it->second <= 1) {
    recv_inflight_chunks_.erase(it);
  } else {
    --(it->second);
  }
}

Endpoint::ScheduleOutcome Endpoint::schedule_send_chunks_locked(
    Conn& conn, PendingTransfer& transfer) {
  // Stage 0 (TX): attempt to keep the pipeline full by posting as many chunks
  // as the per-channel send window allows. Each chunk inherits a unique tag
  // (base_tag + index) so the remote side can disambiguate completions.
  //
  // CRITICAL: Use per-channel windows (not global window) to allow all channels
  // to run in parallel. When a channel is full, skip it and try the next chunk
  // (which maps to a different channel via round-robin).
  //
  // Data flow:
  //   1. Determine target channel via round-robin (chunk_idx % num_channels)
  //   2. Check if that channel's window has capacity
  //   3. If full, skip this chunk and try next (different channel)
  //   4. If has capacity, call tcpx_isend() to post chunk
  //   5. Mark chunk.posted = true and add to per-channel send_queue
  //   6. Increment channel's send_inflight counter
  //   7. Repeat until all chunks posted or all channels full
  constexpr int kBusyRetryMax = 512;
  constexpr int kBusySleepMicros = 5;
  const size_t per_channel_limit = max_send_inflight_;  // e.g., 12

  bool posted_any = false;
  size_t num_channels = conn.channels.size();

  while (transfer.next_chunk_to_post < transfer.chunks.size()) {
    auto& chunk = transfer.chunks[transfer.next_chunk_to_post];

    // Assign channel using round-robin
    size_t channel_idx = transfer.next_chunk_to_post % num_channels;
    chunk.channel_idx = channel_idx;
    auto& channel = conn.channels[channel_idx];

    // Check per-channel window capacity
    if (channel.send_inflight >= per_channel_limit) {
      // This channel is full, stop posting for now
      // Stage 1 will free slots and trigger retry
      break;
    }

    // Get or create mhandle for this channel
    if (!populate_conn_handles_(conn, transfer.mr_id, /*is_recv=*/false,
                                 channel_idx, &chunk.mhandle)) {
      std::cerr << "[tcpx] populate_conn_handles_ failed (send) channel="
                << channel_idx << std::endl;
      return ScheduleOutcome::kError;
    }

    bool chunk_posted = false;
    int attempt = 0;

    while (attempt < kBusyRetryMax) {
      void* request = nullptr;

      int rc = tcpx_isend(
          channel.send_comm, chunk.dst_ptr, static_cast<int>(chunk.bytes),
          static_cast<int>(chunk.tag), chunk.mhandle, &request);
      if (debug_enabled_) {
        size_t chunk_idx = transfer.next_chunk_to_post;
        std::cerr << "[tcpx] isend rc=" << rc << " req=" << request
                  << " chunk=" << chunk_idx << " channel=" << channel_idx
                  << " chunk_bytes=" << chunk.bytes
                  << " tag=" << chunk.tag << std::endl;
      }

      if (rc == 0 && request) {
        chunk.request = request;
        chunk.posted = true;
        chunk_posted = true;
        break;
      }

      if (rc == kTcpxBusy || (rc == 0 && !request)) {
        ++attempt;
        std::this_thread::sleep_for(
            std::chrono::microseconds(kBusySleepMicros));
        continue;
      }

      std::cerr << "[tcpx] tcpx_isend failed rc=" << rc
                << " chunk_offset=" << chunk.offset
                << " chunk_bytes=" << chunk.bytes
                << " channel=" << channel_idx << std::endl;
      return ScheduleOutcome::kError;
    }

    if (!chunk_posted) {
      // tcpx_isend busy after max retries, stop for now
      break;
    }

    // Successfully posted chunk
    transfer.send_queues[channel_idx].push_back(transfer.next_chunk_to_post);
    channel.send_inflight++;  // Increment per-channel counter
    transfer.next_chunk_to_post++;
    posted_any = true;
  }

  return posted_any ? ScheduleOutcome::kProgress : ScheduleOutcome::kNoProgress;
}

Endpoint::ScheduleOutcome Endpoint::schedule_recv_chunks_locked(
    Conn& conn, PendingTransfer& transfer) {
  // Stage 0 (RX): mirrors the send scheduler but seeds recv_stage1_queue so
  // later phases can detect network completions and launch GPU unpack work.
  //
  // CRITICAL: Use per-channel windows (not global window) to allow all channels
  // to run in parallel. When a channel is full, stop and wait for Stage 1 to
  // free slots.
  //
  // Data flow:
  //   1. Determine target channel via round-robin (chunk_idx % num_channels)
  //   2. Check if that channel's window has capacity
  //   3. If full, stop for now (Stage 1 will free slots and trigger retry)
  //   4. If has capacity, call tcpx_irecv() to post chunk
  //   5. Mark chunk.posted = true and add to per-channel recv_stage1_queue
  //   6. Increment channel's recv_inflight counter
  //   7. Window quota remains reserved until Stage 2 completes
  constexpr int kBusyRetryMax = 512;
  constexpr int kBusySleepMicros = 5;
  const size_t per_channel_limit = max_recv_inflight_;  // e.g., 16

  bool posted_any = false;
  size_t num_channels = conn.channels.size();

  while (transfer.next_chunk_to_post < transfer.chunks.size()) {
    auto& chunk = transfer.chunks[transfer.next_chunk_to_post];

    // Assign channel using round-robin
    size_t channel_idx = transfer.next_chunk_to_post % num_channels;
    chunk.channel_idx = channel_idx;
    auto& channel = conn.channels[channel_idx];

    // Check per-channel window capacity
    if (channel.recv_inflight >= per_channel_limit) {
      // This channel is full, stop posting for now
      // Stage 1/2 will free slots and trigger retry
      break;
    }

    // Get or create mhandle for this channel
    if (!populate_conn_handles_(conn, transfer.mr_id, /*is_recv=*/true,
                                 channel_idx, &chunk.mhandle)) {
      std::cerr << "[tcpx] populate_conn_handles_ failed (recv) channel="
                << channel_idx << std::endl;
      return ScheduleOutcome::kError;
    }

    bool chunk_posted = false;
    int attempt = 0;

    while (attempt < kBusyRetryMax) {
      void* buffers[1] = {chunk.dst_ptr};
      int sizes[1] = {static_cast<int>(chunk.bytes)};
      int tags[1] = {static_cast<int>(chunk.tag)};
      void* mhandles[1] = {chunk.mhandle};
      void* requests[1] = {nullptr};

      int rc = tcpx_irecv(channel.recv_comm, 1, buffers, sizes, tags, mhandles,
                          requests);
      if (debug_enabled_) {
        size_t chunk_idx = transfer.next_chunk_to_post;
        std::cerr << "[tcpx] irecv rc=" << rc << " req=" << requests[0]
                  << " chunk=" << chunk_idx << " channel=" << channel_idx
                  << " chunk_bytes=" << chunk.bytes
                  << " tag=" << chunk.tag << std::endl;
      }

      if (rc == 0 && requests[0]) {
        chunk.request = requests[0];
        chunk.posted = true;
        chunk_posted = true;
        break;
      }

      if (rc == kTcpxBusy || (rc == 0 && !requests[0])) {
        ++attempt;
        std::this_thread::sleep_for(
            std::chrono::microseconds(kBusySleepMicros));
        continue;
      }

      std::cerr << "[tcpx] tcpx_irecv failed rc=" << rc
                << " chunk_offset=" << chunk.offset
                << " chunk_bytes=" << chunk.bytes
                << " channel=" << channel_idx << std::endl;
      return ScheduleOutcome::kError;
    }

    if (!chunk_posted) {
      // tcpx_irecv busy after max retries, stop for now
      break;
    }

    // Successfully posted chunk
    transfer.recv_stage1_queues[channel_idx].push_back(transfer.next_chunk_to_post);
    channel.recv_inflight++;  // Increment per-channel counter
    transfer.next_chunk_to_post++;
    posted_any = true;
  }

  return posted_any ? ScheduleOutcome::kProgress : ScheduleOutcome::kNoProgress;
}

// ---------------------------------------------------------------------------
// Transfer scheduling helpers (chunk partition, sliding window posting)
// ---------------------------------------------------------------------------
bool Endpoint::post_send_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                          void const* data, size_t size, int tag,
                          uint64_t& transfer_id) {
  // Shared implementation for send_async*(). Builds a PendingTransfer,
  // precomputes chunk metadata (offset/size/tag), and lets Stage 0 take over.
  //
  // Chunking strategy:
  //   - Split transfer into chunks of chunk_bytes_ (default 512KB)
  //   - Each chunk gets unique tag = base_tag + chunk_index
  //   - All chunks share the same TCPX memory handle (mhandle)
  //   - Chunks are posted to send_queue for Stage 1 to process
  if (!data || size == 0) return false;

  size_t chunk_bytes = std::max<size_t>(1, chunk_bytes_);
  size_t chunk_count = (size + chunk_bytes - 1) / chunk_bytes;

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t data_addr = reinterpret_cast<uintptr_t>(data);
  if (data_addr < base_addr || data_addr + size > base_addr + mr.size) {
    std::cerr << "[tcpx] send_async data out of bounds" << std::endl;
    return false;
  }

  PendingTransfer transfer;
  transfer.kind = PendingTransfer::Kind::kSend;
  transfer.transfer_id = next_transfer_id_.fetch_add(1);
  transfer.conn_id = conn.conn_id;
  transfer.mr_id = mr_id;
  transfer.total_bytes = size;
  transfer.base_tag = static_cast<uint32_t>(tag);
  transfer.chunks.reserve(chunk_count);

  // Initialize per-channel queues
  transfer.send_queues.resize(conn.channels.size());

  auto base_ptr = static_cast<char const*>(data);
  for (size_t idx = 0; idx < chunk_count; ++idx) {
    size_t offset = idx * chunk_bytes;
    size_t chunk_len = std::min(chunk_bytes, size - offset);
    PendingTransfer::ChunkState chunk;
    chunk.offset = offset;
    chunk.bytes = chunk_len;
    chunk.tag = static_cast<uint32_t>(tag) + static_cast<uint32_t>(idx);
    chunk.needs_unpack = false;
    chunk.dst_ptr = const_cast<char*>(base_ptr) + offset;
    transfer.chunks.push_back(std::move(chunk));
  }

  uint64_t tid = transfer.transfer_id;
  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    auto [it, inserted] = transfer_map_.emplace(tid, std::move(transfer));
    PendingTransfer& stored = it->second;

    auto outcome = schedule_send_chunks_locked(conn, stored);
    if (outcome == ScheduleOutcome::kError) {
      transfer_map_.erase(it);
      return false;
    }
  }

  transfer_id = tid;
  return true;
}

bool Endpoint::post_recv_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                          void* data, size_t size, int tag,
                          uint64_t& transfer_id, bool needs_unpack) {
  // Mirror of post_send_. The needs_unpack flag distinguishes between pure READ
  // completions (already in device memory) versus recv_async, which requires
  // the CUDA unpack stage to run after TCPX hands back bounce buffer fragments.
  //
  // Two-stage pipeline (if needs_unpack=true):
  //   Stage 1: tcpx_irecv() -> bounce buffer (recv_stage1_queue)
  //   Stage 2: GPU unpack kernel -> final destination (recv_stage2_queue)
  //
  // Single-stage (if needs_unpack=false, READ path):
  //   Stage 1: tcpx_irecv() -> device memory directly (no Stage 2)
  if (!data || size == 0) return false;

  size_t chunk_bytes = std::max<size_t>(1, chunk_bytes_);
  size_t chunk_count = (size + chunk_bytes - 1) / chunk_bytes;

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(mr.base);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(data);
  if (dst_addr < base_addr || dst_addr + size > base_addr + mr.size) {
    std::cerr << "[tcpx] recv_async destination out of bounds" << std::endl;
    return false;
  }

  PendingTransfer transfer;
  transfer.kind = needs_unpack ? PendingTransfer::Kind::kRecv
                               : PendingTransfer::Kind::kRead;
  transfer.transfer_id = next_transfer_id_.fetch_add(1);
  transfer.conn_id = conn.conn_id;
  transfer.mr_id = mr_id;
  transfer.total_bytes = size;
  transfer.base_tag = static_cast<uint32_t>(tag);
  transfer.chunks.reserve(chunk_count);

  // Initialize per-channel queues
  transfer.recv_stage1_queues.resize(conn.channels.size());
  transfer.recv_consume_queues.resize(conn.channels.size());

  auto base_ptr = static_cast<char*>(data);
  for (size_t idx = 0; idx < chunk_count; ++idx) {
    size_t offset = idx * chunk_bytes;
    size_t chunk_len = std::min(chunk_bytes, size - offset);
    PendingTransfer::ChunkState chunk;
    chunk.offset = offset;
    chunk.bytes = chunk_len;
    chunk.tag = static_cast<uint32_t>(tag) + static_cast<uint32_t>(idx);
    chunk.needs_unpack = needs_unpack;
    chunk.dst_ptr = base_ptr + offset;
    transfer.chunks.push_back(std::move(chunk));
  }

  uint64_t tid = transfer.transfer_id;
  {
    std::lock_guard<std::mutex> lock(transfer_mu_);
    auto [it, inserted] = transfer_map_.emplace(tid, std::move(transfer));
    PendingTransfer& stored = it->second;

    auto outcome = schedule_recv_chunks_locked(conn, stored);
    if (outcome == ScheduleOutcome::kError) {
      transfer_map_.erase(it);
      return false;
    }
  }

  transfer_id = tid;
  return true;
}

bool Endpoint::poll_chunk_request_(PendingTransfer& transfer,
                                   PendingTransfer::ChunkState& chunk,
                                   bool* done, int* received_size) {
  // Stage-1 helper for both TX and RX: poll tcpx_test and mark
  // chunk.stage1_done when the hardware reports completion. For RX this also
  // extracts descriptor metadata that Stage 2 uses to launch the unpack kernel.
  *done = false;
  if (!chunk.request) return false;

  int completed = 0;
  int size = 0;

  int rc = tcpx_test(chunk.request, &completed, &size);

  if (debug_enabled_ && rc != kTcpxBusy) {
    size_t chunk_idx = chunk_bytes_ ? (chunk.offset / chunk_bytes_) : 0;
    std::cerr << "[tcpx] test transfer=" << transfer.transfer_id
              << " chunk=" << chunk_idx << " rc=" << rc
              << " completed=" << completed << " size=" << size << std::endl;
  }

  if (rc == kTcpxBusy) {
    return true;
  }

  if (rc == 2) {
    if (!completed) {
      if (debug_enabled_) {
        size_t chunk_idx = chunk_bytes_ ? (chunk.offset / chunk_bytes_) : 0;
        std::cerr << "[tcpx] tcpx_test reported connection close (rc=2) for "
                  << "transfer_id=" << transfer.transfer_id
                  << " chunk=" << chunk_idx << " (not ready)" << std::endl;
      }
      return true;
    }
  } else if (rc != 0) {
    size_t chunk_idx = chunk_bytes_ ? (chunk.offset / chunk_bytes_) : 0;
    std::cerr << "[tcpx] tcpx_test failed rc=" << rc
              << " transfer_id=" << transfer.transfer_id
              << " chunk=" << chunk_idx << std::endl;
    return false;
  }

  if (!completed) return true;

  *done = true;
  if (received_size) *received_size = size;
  return true;
}

bool Endpoint::enqueue_chunk_unpack_(PendingTransfer& transfer,
                                     PendingTransfer::ChunkState& chunk,
                                     tcpx::plugin::tcpxRequest* request,
                                     Conn& conn) {
  // Convert TCPX's unpack metadata into a descriptor block, launch the CUDA
  // unpack kernel, and associate a recycled event so Stage 2 completion can be
  // detected cheaply inside poll_async.
  //
  // GPU unpack flow:
  //   1. Extract bounce buffer metadata from TCPX request
  //   2. Build UnpackDescriptorBlock (src=bounce_buf, dst=chunk.dst_ptr)
  //   3. Launch GPU kernel on unpack_stream_
  //   4. Record CUDA event for completion detection
  //   5. Recycle event from pool (event_counter % pool_size)
  ScopedCudaContext ctx_guard(cu_context_, static_cast<int>(local_gpu_idx_));

  if (!request) return false;

  // Get the channel for this chunk
  size_t channel_idx = chunk.channel_idx;
  if (channel_idx >= conn.channels.size()) {
    std::cerr << "[tcpx] invalid channel_idx=" << channel_idx << std::endl;
    return false;
  }
  auto& channel = conn.channels[channel_idx];

  if (!channel.recv_dev_handle_cached) {
    if (!ensure_recv_dev_handle_cached_(conn, channel_idx)) return false;
  }
  auto const& dev_handle = channel.recv_dev_handle_host;

  uint64_t frag_cnt = 0;
  if (!request->unpack_slot.cnt) {
    std::cerr << "[tcpx] missing unpack counter" << std::endl;
    return false;
  }
  frag_cnt = *(request->unpack_slot.cnt);

  if (debug_enabled_) {
    size_t chunk_idx = chunk_bytes_ ? (chunk.offset / chunk_bytes_) : 0;
    std::cerr << "[tcpx] unpack transfer=" << transfer.transfer_id
              << " chunk=" << chunk_idx << " frag_cnt=" << frag_cnt
              << std::endl;
  }

  if (frag_cnt == 0 || frag_cnt > MAX_UNPACK_DESCRIPTORS) {
    std::cerr << "[tcpx] invalid fragment count " << frag_cnt << std::endl;
    return false;
  }

  auto* meta_entries =
      static_cast<tcpx::plugin::loadMeta*>(request->unpack_slot.mem);
  tcpx::rx::UnpackDescriptorBlock block;
  tcpx::rx::buildDescriptorBlock(meta_entries, static_cast<uint32_t>(frag_cnt),
                                 dev_handle.bounce_buf, chunk.dst_ptr, block);

  block.ready_flag = request->unpack_slot.cnt;
  block.ready_threshold = frag_cnt;

  int launch_rc = unpack_launcher_->launch(block, unpack_stream_);
  if (launch_rc != 0) {
    std::cerr << "[tcpx] unpack kernel launch failed rc=" << launch_rc
              << std::endl;
    return false;
  }

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

  chunk.event = event;
  chunk.event_idx = event_idx;
  chunk.desc_block = block;
  return true;
}

bool Endpoint::consume_recv_chunk_(Conn& conn,
                                   PendingTransfer::ChunkState& chunk) {
  // Call tcpx_irecv_consumed() to release bounce buffer.
  // CRITICAL: This must be called in FIFO order per channel!
  size_t channel_idx = chunk.channel_idx;
  if (channel_idx < conn.channels.size()) {
    auto& channel = conn.channels[channel_idx];
    if (channel.recv_comm && chunk.request && !chunk.consumed) {
      int rc = tcpx_irecv_consumed(channel.recv_comm, 1, chunk.request);
      if (rc != 0) {
        std::cerr << "[tcpx] tcpx_irecv_consumed failed rc=" << rc
                  << " chunk_offset=" << chunk.offset
                  << " channel=" << channel_idx << std::endl;
        return false;
      }
      chunk.consumed = true;
    }
  }
  return true;
}

bool Endpoint::finalize_recv_chunk_(Conn& conn,
                                    PendingTransfer::ChunkState& chunk) {
  // Stage 2 epilogue for recv/read paths: release the per-chunk CUDA event
  // back to the pool.
  //
  // NOTE: We do NOT call tcpx_irecv_consumed here! That must be done in FIFO
  // order per channel, which is handled separately in progress_transfer_locked.
  chunk.event = nullptr;
  return true;
}

bool Endpoint::progress_transfer_locked(Conn& conn, PendingTransfer& transfer,
                                        bool* schedule_send,
                                        bool* schedule_recv) {
  // Stage 1 covers network completions; Stage 2 covers GPU completions for the
  // receive path. schedule_send/recv are used to signal that Stage 0 should
  // retry submission because window slots were freed during this pass.
  //
  // Pipeline overview:
  //   Send path (single-stage):
  //     send_queue -> tcpx_test() -> release_send_slot() -> done
  //
  //   Receive path (two-stage):
  //     recv_stage1_queue -> tcpx_test() -> enqueue_chunk_unpack_() ->
  //     recv_stage2_queue -> cudaEventQuery() -> release_recv_slot() -> done
  bool trigger_send = false;
  bool trigger_recv = false;

  // ========================================================================
  // Send path: Stage 1 only (no GPU unpack needed)
  // ========================================================================
  if (transfer.kind == PendingTransfer::Kind::kSend) {
    // CRITICAL: Test only the FIFO head of each channel's queue to match
    // TCPX's internal rq.next_transmitting constraint
    for (size_t ch_idx = 0; ch_idx < transfer.send_queues.size(); ++ch_idx) {
      auto& queue = transfer.send_queues[ch_idx];

      while (!queue.empty()) {
        size_t idx = queue.front();

        if (idx >= transfer.chunks.size()) {
          queue.pop_front();
          continue;
        }

        auto& chunk = transfer.chunks[idx];

        if (!chunk.posted) {
          break;
        }

        // Poll TCPX for network completion
        bool done = false;
        int received = 0;
        if (!poll_chunk_request_(transfer, chunk, &done, &received)) return false;

        if (!done) {
          break;  // Still in flight, must wait (FIFO ordering)
        }

        // Network send complete: release per-channel window slot and mark done
        queue.pop_front();
        chunk.stage2_done = true;
        transfer.chunks_completed++;
        chunk.request = nullptr;

        // Decrement per-channel inflight counter
        if (chunk.channel_idx < conn.channels.size()) {
          auto& channel = conn.channels[chunk.channel_idx];
          if (channel.send_inflight > 0) {
            channel.send_inflight--;
          }
        }

        trigger_send = true;  // Signal Stage 0 to post more chunks
      }
    }

    if (schedule_send) *schedule_send = trigger_send;
    if (schedule_recv) *schedule_recv = false;
    return true;
  }

  // ========================================================================
  // Receive path: Stage 1 (network) + Stage 2 (GPU unpack)
  // ========================================================================
  else {
    // Stage 1: Poll TCPX for network completions. When a chunk arrives in the
    // bounce buffer, extract unpack metadata and launch the GPU kernel.
    // CRITICAL: Test only the FIFO head of each channel's queue to match
    // TCPX's internal rq.next_transmitting constraint
    for (size_t ch_idx = 0; ch_idx < transfer.recv_stage1_queues.size(); ++ch_idx) {
      auto& queue = transfer.recv_stage1_queues[ch_idx];

      while (!queue.empty()) {
        size_t idx = queue.front();
        if (idx >= transfer.chunks.size()) {
          queue.pop_front();
          continue;
        }
        auto& chunk = transfer.chunks[idx];
        if (!chunk.posted) {
          break;
        }
        if (chunk.stage1_done) {
          queue.pop_front();
          continue;
        }

        // Poll TCPX for network completion
        bool done = false;
        int received = 0;
        if (!poll_chunk_request_(transfer, chunk, &done, &received)) return false;

        if (!done) {
          break;  // Still in flight, must wait (FIFO ordering)
        }

        // Network receive complete: data is now in bounce buffer
        chunk.stage1_done = true;
        queue.pop_front();

        if (chunk.needs_unpack) {
          // Launch GPU kernel to copy from bounce buffer to final destination
          auto* rx_req =
              reinterpret_cast<tcpx::plugin::tcpxRequest*>(chunk.request);
          if (!enqueue_chunk_unpack_(transfer, chunk, rx_req, conn)) return false;
          transfer.recv_stage2_queue.push_back(idx);
          // Window slot remains reserved until Stage 2 completes
        } else {
          // READ path: data already in device memory, no unpack needed
          if (!finalize_recv_chunk_(conn, chunk)) return false;
          chunk.stage2_done = true;
          transfer.chunks_completed++;

          // Add to consume queue (will be consumed in FIFO order below)
          // Only add once to prevent duplicates
          if (!chunk.queued_for_consume && chunk.channel_idx < transfer.recv_consume_queues.size()) {
            transfer.recv_consume_queues[chunk.channel_idx].push_back(idx);
            chunk.queued_for_consume = true;  // Mark as queued for consume
          }
        }
      }
    }

    // Stage 2: Poll CUDA events to detect GPU unpack completion. This stage
    // runs asynchronously with Stage 1, allowing network and GPU work to
    // overlap. Chunks may complete out-of-order, so we iterate the entire
    // queue.
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

        // Add to consume queue (will be consumed in FIFO order below)
        // Only add once to prevent duplicates
        if (!chunk.queued_for_consume && chunk.channel_idx < transfer.recv_consume_queues.size()) {
          transfer.recv_consume_queues[chunk.channel_idx].push_back(idx);
          chunk.queued_for_consume = true;  // Mark as queued for consume
        }

        stage2_it = transfer.recv_stage2_queue.erase(stage2_it);
        continue;
      }

      // Query CUDA event to check if GPU unpack kernel has finished
      cudaError_t err = cudaEventQuery(chunk.event);
      if (err == cudaErrorNotReady) {
        ++stage2_it;  // GPU still working, check next chunk
        continue;
      }
      if (err != cudaSuccess) {
        std::cerr << "[tcpx] cudaEventQuery failed: " << cudaGetErrorString(err)
                  << std::endl;
        return false;
      }

      // GPU unpack complete: finalize and add to consume queue
      if (!finalize_recv_chunk_(conn, chunk)) return false;
      chunk.stage2_done = true;
      transfer.chunks_completed++;

      // Add to consume queue (will be consumed in FIFO order below)
      // Only add once to prevent duplicates
      if (!chunk.queued_for_consume && chunk.channel_idx < transfer.recv_consume_queues.size()) {
        transfer.recv_consume_queues[chunk.channel_idx].push_back(idx);
        chunk.queued_for_consume = true;  // Mark as queued for consume
      }

      stage2_it = transfer.recv_stage2_queue.erase(stage2_it);
    }

    // Stage 3 (Consume): Call tcpx_irecv_consumed in FIFO order per channel.
    // CRITICAL: TCPX requires that irecv_consumed be called in the same order
    // as irecv was posted. We process each channel's consume queue and only
    // consume the head chunk if it's ready (stage2_done).
    for (size_t ch_idx = 0; ch_idx < transfer.recv_consume_queues.size(); ++ch_idx) {
      auto& consume_queue = transfer.recv_consume_queues[ch_idx];

      while (!consume_queue.empty()) {
        size_t idx = consume_queue.front();
        if (idx >= transfer.chunks.size()) {
          consume_queue.pop_front();
          continue;
        }

        auto& chunk = transfer.chunks[idx];

        // Only consume if stage2 is done
        if (!chunk.stage2_done) {
          break;  // Wait for this chunk to complete (FIFO ordering)
        }

        // Consume this chunk
        if (!consume_recv_chunk_(conn, chunk)) return false;
        chunk.request = nullptr;

        // Decrement per-channel inflight counter (now Stage 0 can post more chunks)
        if (ch_idx < conn.channels.size()) {
          auto& channel = conn.channels[ch_idx];
          if (channel.recv_inflight > 0) {
            channel.recv_inflight--;
          }
        }

        trigger_recv = true;
        consume_queue.pop_front();
      }
    }
  }

  if (schedule_send) *schedule_send = trigger_send;
  if (schedule_recv) *schedule_recv = trigger_recv;
  return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  // Main entry point for advancing a transfer through its pipeline stages.
  // Called repeatedly by the user until the transfer completes.
  //
  // Overall data flow:
  //   1. Lookup transfer and connection (under locks)
  //   2. Call advance_transfer_locked() to run Stage 0/1/2
  //   3. If complete, finalize and reset window counters
  //   4. Return completion status to caller
  if (!is_done) return false;
  *is_done = false;

  // Step 1: grab the transfer while holding transfer_mu_. This guarantees that
  // only a single thread is advancing the pipeline for a given transfer_id.
  std::unique_lock<std::mutex> lock(transfer_mu_);

  auto it = transfer_map_.find(transfer_id);
  if (it == transfer_map_.end()) {
    if (debug_enabled_) {
      std::cerr << "[tcpx] poll transfer_id=" << transfer_id << " c"
                << std::endl;
    }
    *is_done = true;
    return true;
  }
  PendingTransfer& transfer = it->second;

  // Step 2: lookup the owning connection under conn_mu_. The connection is
  // shared_ptr-managed so we can safely release conn_mu_ immediately after.
  std::shared_ptr<Conn> conn_ptr;
  {
    std::shared_lock<std::shared_mutex> conn_lock(conn_mu_);
    auto conn_it = conn_map_.find(transfer.conn_id);
    if (conn_it == conn_map_.end()) return false;
    conn_ptr = conn_it->second;
  }

  // Step 3: drive Stage 0/1/2 progression. On success transfer_complete tells
  // us whether finalize_transfer_locked should run.
  bool transfer_complete = false;
  if (!advance_transfer_locked(*conn_ptr, transfer, &transfer_complete))
    return false;

  if (transfer_complete) {
    finalize_transfer_locked(it);

    lock.unlock();

    // After the transfer is removed we can safely reset the per-connection
    // window counters (guarded by window_mu_).
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

void Endpoint::free_conn_(std::shared_ptr<Conn> const& conn) {
  // Connection teardown order matters: remove per-connection MR handles while
  // holding mr_mu_, reset inflight windows, then release hardware resources in
  // the reverse order of creation (events -> comms -> control socket).
  if (!conn) return;

  struct HandleRef {
    size_t channel_idx;
    void* handle;
  };
  std::vector<HandleRef> send_handles;
  std::vector<HandleRef> recv_handles;
  {
    std::lock_guard<std::mutex> lock(mr_mu_);

    for (auto& kv : mr_map_) {
      auto& mr = kv.second;

      auto send_it = mr.send_handles.find(conn->conn_id);
      if (send_it != mr.send_handles.end()) {
        auto const& handles_vec = send_it->second;
        for (size_t ch_idx = 0; ch_idx < handles_vec.size(); ++ch_idx) {
          if (handles_vec[ch_idx]) {
            send_handles.push_back({ch_idx, handles_vec[ch_idx]});
          }
        }
        mr.send_handles.erase(send_it);
      }

      auto recv_it = mr.recv_handles.find(conn->conn_id);
      if (recv_it != mr.recv_handles.end()) {
        auto const& handles_vec = recv_it->second;
        for (size_t ch_idx = 0; ch_idx < handles_vec.size(); ++ch_idx) {
          if (handles_vec[ch_idx]) {
            recv_handles.push_back({ch_idx, handles_vec[ch_idx]});
          }
        }
        mr.recv_handles.erase(recv_it);
      }
    }
  }

  reset_conn_window_counters_(conn->conn_id);

  for (auto const& ref : send_handles) {
    if (ref.channel_idx < conn->channels.size()) {
      auto& channel = conn->channels[ref.channel_idx];
      if (channel.send_comm) {
        tcpx_dereg_mr(channel.send_comm, ref.handle);
      }
    }
  }
  for (auto const& ref : recv_handles) {
    if (ref.channel_idx < conn->channels.size()) {
      auto& channel = conn->channels[ref.channel_idx];
      if (channel.recv_comm) {
        tcpx_dereg_mr(channel.recv_comm, ref.handle);
      }
    }
  }

  for (auto& event : conn->recv_events) {
    if (event) {
      cudaEventDestroy(event);
    }
  }
  conn->recv_events.clear();

  // Close all channels
  for (auto& channel : conn->channels) {
    if (channel.send_comm) {
      tcpx_close_send(channel.send_comm);
      channel.send_comm = nullptr;
    }
    if (channel.recv_comm) {
      tcpx_close_recv(channel.recv_comm);
      channel.recv_comm = nullptr;
    }
  }

  if (conn->ctrl_sock_fd >= 0) {
    ::close(conn->ctrl_sock_fd);
    conn->ctrl_sock_fd = -1;
  }
}

}  // namespace tcpx
