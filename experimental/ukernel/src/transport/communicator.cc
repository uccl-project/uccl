#include "communicator.h"
#include "adapter/ipc_adapter.h"
#include "utils.h"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr int kBootstrapPollDelayMs = 100;
constexpr int kDefaultBootstrapTimeoutMs = 30000;
constexpr int kDefaultMrTimeoutMs = 30000;

std::string get_local_ip() {
  if (char const* env_ip = std::getenv("UHM_LOCAL_IP")) {
    if (std::strlen(env_ip) > 0) return env_ip;
  }

  int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) return "127.0.0.1";

  sockaddr_in remote{};
  remote.sin_family = AF_INET;
  remote.sin_port = htons(80);
  ::inet_pton(AF_INET, "8.8.8.8", &remote.sin_addr);

  ::connect(sock, (sockaddr*)&remote, sizeof(remote));

  sockaddr_in local{};
  socklen_t len = sizeof(local);
  ::getsockname(sock, (sockaddr*)&local, &len);
  ::close(sock);

  char buf[INET_ADDRSTRLEN];
  ::inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf));
  return buf;
}

bool has_env_value(char const* name) {
  char const* value = std::getenv(name);
  return value != nullptr && value[0] != '\0';
}

bool is_unspecified_ip(std::string const& ip) {
  return ip.empty() || ip == "0.0.0.0" || ip == "127.0.0.1" ||
         ip == "localhost";
}

std::string find_ifname_for_local_ip(std::string const& ip) {
  if (is_unspecified_ip(ip)) return {};

  ifaddrs* ifs = nullptr;
  if (::getifaddrs(&ifs) != 0) return {};

  std::string ifname;
  for (auto* it = ifs; it != nullptr; it = it->ifa_next) {
    if (!it->ifa_addr || it->ifa_addr->sa_family != AF_INET) continue;

    char buf[INET_ADDRSTRLEN] = {};
    auto* addr = reinterpret_cast<sockaddr_in*>(it->ifa_addr);
    if (!::inet_ntop(AF_INET, &addr->sin_addr, buf, sizeof(buf))) continue;
    if (ip == buf) {
      ifname = it->ifa_name;
      break;
    }
  }

  ::freeifaddrs(ifs);
  return ifname;
}

void maybe_configure_uccl_socket_ifname(std::string const& remote_hint_ip,
                                        std::string const& local_hint_ip) {
  if (has_env_value("UCCL_SOCKET_IFNAME") ||
      has_env_value("NCCL_SOCKET_IFNAME")) {
    return;
  }

  std::string ifname = find_ifname_for_local_ip(local_hint_ip);
  if (ifname.empty()) return;

  ::setenv("UCCL_SOCKET_IFNAME", ifname.c_str(), 0);
  std::cout << "[INFO] Auto-selected UCCL_SOCKET_IFNAME=" << ifname
            << std::endl;
}

std::string get_uccl_remote_hint_ip(
    std::shared_ptr<CommunicatorConfig> const& config,
    CommunicatorMeta const& peer_meta) {
  if (config && !is_unspecified_ip(config->exchanger_ip)) {
    return config->exchanger_ip;
  }
  if (!is_unspecified_ip(peer_meta.ip)) {
    return peer_meta.ip;
  }
  return {};
}

int get_timeout_ms(char const* env_name, int default_ms) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') return default_ms;
  try {
    return std::stoi(value);
  } catch (...) {
    return default_ms;
  }
}

int timeout_to_retries(int timeout_ms, int delay_ms) {
  if (timeout_ms < 0) return -1;
  if (delay_ms <= 0) return timeout_ms > 0 ? timeout_ms : 1;
  return std::max(1, (timeout_ms + delay_ms - 1) / delay_ms);
}

int bootstrap_timeout_ms() {
  return get_timeout_ms("UHM_BOOTSTRAP_TIMEOUT_MS", kDefaultBootstrapTimeoutMs);
}

int mr_timeout_ms() {
  return get_timeout_ms("UHM_MR_TIMEOUT_MS", kDefaultMrTimeoutMs);
}

std::string uccl_p2p_key(int src_rank, int dst_rank) {
  return "uccl_p2p_info_" + std::to_string(src_rank) + "_to_" +
         std::to_string(dst_rank);
}

std::string tcp_p2p_key(int src_rank, int dst_rank) {
  return "tcp_p2p_info_" + std::to_string(src_rank) + "_to_" +
         std::to_string(dst_rank);
}

std::string ipc_buffer_key(int src_rank, int dst_rank, uint32_t ipc_id) {
  return "ipcbuf:" + std::to_string(src_rank) + "->" +
         std::to_string(dst_rank) + ":ipc:" + std::to_string(ipc_id);
}

std::string ipc_buffer_versioned_key(int src_rank, int dst_rank,
                                     uint32_t ipc_id,
                                     uint64_t binding_version) {
  return ipc_buffer_key(src_rank, dst_rank, ipc_id) + ":v" +
         std::to_string(binding_version);
}

void validate_dst_hint_for_transport(PeerTransportKind kind,
                                     std::optional<RemoteSlice> const& dst_hint,
                                     size_t src_bytes) {
  if (!dst_hint.has_value()) return;
  auto const& hint = *dst_hint;
  if (hint.mem_id == 0) {
    throw std::invalid_argument("dst_hint.mem_id must be non-zero");
  }
  // IPC/TCP use common (`mem_id`, `offset`) hint and ignore `write`.
  if (kind == PeerTransportKind::Uccl && hint.has_write_hint() &&
      hint.write.capacity != 0 && hint.write.capacity < src_bytes) {
    throw std::invalid_argument(
        "dst_hint.write.capacity is smaller than send size");
  }
}

template <typename Info>
bool publish_local_then_fetch_remote(Exchanger& exchanger,
                                     std::string const& local_key,
                                     Info const& local_info,
                                     std::string const& remote_key,
                                     Info& remote_info, int timeout_ms) {
  if (!exchanger.publish(local_key, local_info)) return false;
  return exchanger.wait_and_fetch(
      remote_key, remote_info,
      timeout_to_retries(timeout_ms, kBootstrapPollDelayMs),
      kBootstrapPollDelayMs);
}

template <typename Info>
bool fetch_remote_then_publish_local(Exchanger& exchanger,
                                     std::string const& remote_key,
                                     Info& remote_info,
                                     std::string const& local_key,
                                     Info const& local_info, int timeout_ms) {
  if (!exchanger.wait_and_fetch(
          remote_key, remote_info,
          timeout_to_retries(timeout_ms, kBootstrapPollDelayMs),
          kBootstrapPollDelayMs)) {
    return false;
  }
  return exchanger.publish(local_key, local_info);
}

bool detect_local_rdma_capable() {
  if (char const* env = std::getenv("UHM_RDMA_CAPABLE")) {
    return std::strcmp(env, "0") != 0;
  }

  int count = 0;
  ibv_device** devices = ibv_get_device_list(&count);
  if (!devices) return false;
  ibv_free_device_list(devices);
  return count > 0;
}

}  // namespace

Communicator::Communicator(int gpu_id, int rank, int world_size,
                           std::shared_ptr<CommunicatorConfig> config)
    : local_gpu_idx_(gpu_id),
      global_rank_(rank),
      world_size_(world_size),
      peer_states_(static_cast<size_t>(world_size)),
      config_(config) {
  shm_control_ = std::make_shared<ShmRingExchanger>(
      global_rank_, world_size_,
      generate_host_id() + "_p" + std::to_string(config_->exchanger_port),
      config_->local_id >= 0 ? config_->local_id : global_rank_);
  ipc_channel_ = std::make_shared<IpcChannel>(this);
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
  GPU_RT_CHECK(
      gpuStreamCreateWithFlags(&host_copy_stream_, gpuStreamNonBlocking));

  bool is_server = (global_rank_ == 0);
  if (!is_server && config_->exchanger_ip == "0.0.0.0")
    config_->exchanger_ip = "127.0.0.1";
  std::cout << "[INFO] Using socket-based bootstrap exchanger as "
            << (is_server ? "server" : "client") << " " << config_->exchanger_ip
            << std::endl;
  exchanger_client_ = std::make_shared<SockExchanger>(
      (global_rank_ == 0), config_->exchanger_ip, config_->exchanger_port);
  if (!exchanger_client_->valid()) {
    fprintf(stderr, "[ERROR] Failed to connect to Exchanger\n");
    return;
  }

  shm_manager_ = std::make_unique<SHMManager>();

  // Start background progress immediately so post-transport work, such as
  // host-bounce host->device copies, is not delayed until an external
  // poll()/wait_finish() call.
  progress_running_.store(true);
  progress_started_.store(true);
  progress_thread_ = std::thread(&Communicator::progress_loop, this);

  exchange_peer_metas();
  std::cout << "[INFO] Communicator " << global_rank_
            << " initialized: peer meta exchange success" << std::endl;
}

void Communicator::exchange_peer_metas() {
  CommunicatorMeta local;
  local.host_id = generate_host_id();
  local.local_id = config_->local_id >= 0 ? config_->local_id : global_rank_;
  local.rdma_capable = detect_local_rdma_capable();
  local.ip = get_local_ip();
  {
    std::lock_guard<std::mutex> lk(peer_mu_);
    auto& self = peer_states_.at(static_cast<size_t>(global_rank_));
    self.meta = local;
    self.has_meta = true;
    self.connected = true;
  }

  std::string meta_key = "meta:" + std::to_string(global_rank_);
  if (!exchanger_client_->publish(meta_key, local)) {
    fprintf(stderr, "[ERROR] Failed to publish local CommunicatorMeta \n");
  }

  CommunicatorMeta remote;
  std::vector<int> missing_ranks;
  for (int i = 0; i < world_size_; i++) {
    if (i == global_rank_) continue;
    std::string key = "meta:" + std::to_string(i);
    if (exchanger_client_->wait_and_fetch(
            key, remote,
            timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
            kBootstrapPollDelayMs)) {
      std::lock_guard<std::mutex> lk(peer_mu_);
      auto& peer = peer_states_.at(static_cast<size_t>(i));
      peer.meta = remote;
      peer.has_meta = true;
      shm_control_->set_peer_local_id(i, remote.local_id);
    } else {
      missing_ranks.push_back(i);
    }
  }

  if (!missing_ranks.empty()) {
    std::ostringstream oss;
    oss << "timeout waiting for remote CommunicatorMeta from ranks ";
    for (size_t i = 0; i < missing_ranks.size(); ++i) {
      if (i != 0) oss << ",";
      oss << missing_ranks[i];
    }
    throw std::runtime_error(oss.str());
  }
}

Communicator::~Communicator() {
  if (progress_started_.load()) {
    progress_running_.store(false);
    progress_cv_.notify_all();
    if (progress_thread_.joinable()) {
      progress_thread_.join();
    }
  }

  if (ipc_channel_) {
    ipc_channel_->shutdown();
  }
  if (host_copy_stream_ != nullptr) {
    int orig_device = -1;
    GPU_RT_CHECK(gpuGetDevice(&orig_device));
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
    GPU_RT_CHECK(gpuStreamDestroy(host_copy_stream_));
    GPU_RT_CHECK(gpuSetDevice(orig_device));
    host_copy_stream_ = nullptr;
  }

  std::shared_ptr<ShmRingExchanger> shm_control = shm_control_;
  std::vector<int> ipc_peers;

  {
    std::lock_guard<std::mutex> lk(peer_mu_);
    for (int rank = 0; rank < world_size_; ++rank) {
      if (rank == global_rank_) continue;
      auto const& peer = peer_states_.at(static_cast<size_t>(rank));
      bool uses_ipc = (peer.kind == PeerTransportKind::Ipc);
      if (uses_ipc && peer.connected) {
        ipc_peers.push_back(rank);
      }
    }
  }

  if (shm_control) {
    for (int rank : ipc_peers) {
      shm_control->close_peer(rank);
    }
  }

  for (auto const& [buf, item] : mr_manager_.list_local_mrs()) {
    (void)item;
    dereg_mr(buf);
  }

  for (int i = 0; i < world_size_; ++i) {
    if (i == global_rank_) continue;
    ipc_manager_.delete_ipc(i);
  }

  // Destroy bounce pool before uccl_adapter_ to avoid dangling references
  // in the deregister callback during pool teardown.
  shm_manager_.reset();
  uccl_adapter_.reset();
  tcp_adapter_.reset();
  ipc_channel_.reset();

  std::cout << "[INFO] Communicator " << global_rank_ << " resources released"
            << std::endl;
}

UcclTransportAdapter& Communicator::ensure_uccl_adapter(
    CommunicatorMeta const& local_meta, CommunicatorMeta const& peer_meta) {
  if (!uccl_adapter_) {
    maybe_configure_uccl_socket_ifname(
        get_uccl_remote_hint_ip(config_, peer_meta), local_meta.ip);
    UcclTransportConfig uccl_cfg;
    uccl_cfg.local_ip = local_meta.ip;
    uccl_adapter_ = std::make_unique<UcclTransportAdapter>(
        local_gpu_idx_, world_size_, std::move(uccl_cfg));

    mr_manager_.bind_backend(
        [this](uint32_t mr_id, void* ptr, size_t len) -> bool {
          if (!uccl_adapter_ || !uccl_adapter_->is_initialized()) return false;
          bool ok = uccl_adapter_->register_memory(mr_id, ptr, len);
          std::lock_guard<std::mutex> lk(uccl_reg_mu_);
          if (ok) {
            uccl_direct_reg_failed_mrs_.erase(mr_id);
            uccl_registered_mrs_.insert(mr_id);
          } else {
            uccl_direct_reg_failed_mrs_.insert(mr_id);
          }
          return ok;
        },
        [this](uint32_t mr_id) {
          if (uccl_adapter_ && uccl_adapter_->is_initialized()) {
            uccl_adapter_->deregister_memory(mr_id);
          }
          std::lock_guard<std::mutex> lk(uccl_reg_mu_);
          uccl_direct_reg_failed_mrs_.erase(mr_id);
          uccl_registered_mrs_.erase(mr_id);
        });
    mr_manager_.sync_local_backend();
  }
  return *uccl_adapter_;
}

TcpTransportAdapter& Communicator::ensure_tcp_adapter(
    CommunicatorMeta const& local_meta) {
  if (!tcp_adapter_) {
    tcp_adapter_ =
        std::make_unique<TcpTransportAdapter>(local_meta.ip, global_rank_);
  }
  return *tcp_adapter_;
}

Communicator::ResolvedPeer Communicator::resolve_peer(int rank) const {
  if (rank == global_rank_) {
    throw std::invalid_argument("transport peer rank cannot be self");
  }
  if (rank < 0 || rank >= world_size_) {
    throw std::invalid_argument("transport peer rank out of range");
  }

  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& local_peer = peer_states_.at(static_cast<size_t>(global_rank_));
  auto const& remote_peer = peer_states_.at(static_cast<size_t>(rank));
  if (!local_peer.has_meta || !remote_peer.has_meta) {
    throw std::runtime_error("transport peer metadata is not established");
  }

  ResolvedPeer resolved;
  resolved.local_meta = local_peer.meta;
  resolved.remote_meta = remote_peer.meta;
  resolved.kind = resolve_peer_transport_kind(*config_, resolved.local_meta,
                                              resolved.remote_meta);
  return resolved;
}

bool Communicator::try_fallback_tcp_connect(
    int rank, CommunicatorMeta const& local_meta) {
  if (config_->preferred_transport != PreferredTransport::Auto) return false;
  auto& tcp_adapter = ensure_tcp_adapter(local_meta);
  if (tcp_adapter.has_send_peer(rank)) {
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }

  TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                            tcp_adapter.get_listen_port());
  std::string p2p_key = tcp_p2p_key(global_rank_, rank);
  std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);
  if (!exchanger_client_->publish(p2p_key, local_p2p_info)) return false;

  TcpP2PInfo remote_p2p_info;
  if (!exchanger_client_->wait_and_fetch(
          peer_p2p_key, remote_p2p_info,
          timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
          kBootstrapPollDelayMs)) {
    return false;
  }

  if (!tcp_adapter.connect_to_peer(rank, remote_p2p_info.ip,
                                   remote_p2p_info.port)) {
    return false;
  }
  {
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    std::cout << "[INFO] Communicator " << global_rank_
              << " TCP fallback connect_to succeeded to rank " << rank
              << std::endl;
  }
  return true;
}

bool Communicator::try_fallback_tcp_accept(
    int rank, CommunicatorMeta const& local_meta,
    CommunicatorMeta const& remote_meta) {
  (void)remote_meta;
  if (config_->preferred_transport != PreferredTransport::Auto) return false;
  auto& tcp_adapter = ensure_tcp_adapter(local_meta);
  if (tcp_adapter.has_send_peer(rank) && tcp_adapter.has_recv_peer(rank)) {
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }

  std::string p2p_key = tcp_p2p_key(global_rank_, rank);
  std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);

  TcpP2PInfo remote_p2p_info;
  if (!exchanger_client_->wait_and_fetch(
          peer_p2p_key, remote_p2p_info,
          timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
          kBootstrapPollDelayMs)) {
    return false;
  }

  TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                            tcp_adapter.get_listen_port());
  if (!exchanger_client_->publish(p2p_key, local_p2p_info)) return false;

  if (!tcp_adapter.accept_from_peer(rank, remote_p2p_info.ip)) {
    return false;
  }
  {
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    std::cout << "[INFO] Communicator " << global_rank_
              << " TCP fallback accept_from succeeded from rank " << rank
              << std::endl;
  }
  return true;
}

bool Communicator::connect(int rank) { return do_connect(rank); }

bool Communicator::accept(int rank) { return do_accept(rank); }

bool Communicator::do_connect(int rank) {
  if (rank == global_rank_) return true;
  if (rank < 0 || rank >= world_size_) {
    std::cerr << "[ERROR] Communicator " << global_rank_ << " invalid rank "
              << rank << ", world_size=" << world_size_ << std::endl;
    return false;
  }
  if (has_peer_path(rank)) return true;

  ResolvedPeer resolved;
  try {
    resolved = resolve_peer(rank);
  } catch (std::exception const& ex) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " failed to resolve transport for rank " << rank << ": "
              << ex.what() << std::endl;
    return false;
  }

  if (resolved.kind == PeerTransportKind::Uccl) {
    auto& uccl_adapter =
        ensure_uccl_adapter(resolved.local_meta, resolved.remote_meta);
    if (!uccl_adapter.has_send_peer(rank) ||
        !uccl_adapter.has_recv_peer(rank)) {
      int dev_idx = uccl_adapter.get_best_dev_idx(local_gpu_idx_);
      if (dev_idx < 0) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " UCCL get_best_dev_idx failed for local gpu "
                  << local_gpu_idx_ << std::endl;
        return try_fallback_tcp_connect(rank, resolved.local_meta);
      }
      uint16_t local_port = uccl_adapter.get_p2p_listen_port(dev_idx);
      if (local_port == 0) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " UCCL local listen port is invalid for dev " << dev_idx
                  << std::endl;
        return try_fallback_tcp_connect(rank, resolved.local_meta);
      }
      std::string local_ip_addr = uccl_adapter.get_p2p_listen_ip(dev_idx);
      if (local_ip_addr.empty()) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " UCCL local listen ip is empty for dev " << dev_idx
                  << std::endl;
        return try_fallback_tcp_connect(rank, resolved.local_meta);
      }
      UCCLP2PInfo local_p2p_info(local_ip_addr, local_port, dev_idx,
                                 local_gpu_idx_);
      std::string p2p_key = uccl_p2p_key(global_rank_, rank);
      std::string peer_p2p_key = uccl_p2p_key(rank, global_rank_);
      UCCLP2PInfo remote_p2p_info;
      if (!publish_local_then_fetch_remote(
              *exchanger_client_, p2p_key, local_p2p_info, peer_p2p_key,
              remote_p2p_info, bootstrap_timeout_ms())) {
        return false;
      }
      // connect() path: connect first, then establish reverse flow.
      if ((!uccl_adapter.has_send_peer(rank) ||
           !uccl_adapter.has_recv_peer(rank)) &&
          !uccl_adapter.connect_to_peer(rank, remote_p2p_info.ip,
                                        remote_p2p_info.port, dev_idx,
                                        local_gpu_idx_, remote_p2p_info.dev_idx,
                                        remote_p2p_info.gpu_idx)) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " UCCL connect_to failed to rank " << rank << std::endl;
        return try_fallback_tcp_connect(rank, resolved.local_meta);
      }
    }
    mark_peer_path_ready(rank, PeerTransportKind::Uccl);
    register_existing_local_mrs_with_uccl();
    return true;
  }

  if (resolved.kind == PeerTransportKind::Ipc) {
    if (!ipc_channel_->connect_to(rank)) {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC connect_to failed to rank " << rank << std::endl;
      shm_control_->close_peer(rank);
      return false;
    }
    mark_peer_path_ready(rank, PeerTransportKind::Ipc);
    return true;
  }

  if (resolved.kind == PeerTransportKind::Tcp) {
    auto& tcp_adapter = ensure_tcp_adapter(resolved.local_meta);
    if (!tcp_adapter.has_send_peer(rank)) {
      TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                                tcp_adapter.get_listen_port());
      std::string p2p_key = tcp_p2p_key(global_rank_, rank);
      std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);
      TcpP2PInfo remote_p2p_info;
      if (!publish_local_then_fetch_remote(
              *exchanger_client_, p2p_key, local_p2p_info, peer_p2p_key,
              remote_p2p_info, bootstrap_timeout_ms())) {
        return false;
      }
      if (!tcp_adapter.connect_to_peer(rank, remote_p2p_info.ip,
                                       remote_p2p_info.port)) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " TCP connect_to failed to rank " << rank << std::endl;
        return false;
      }
    }
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }
  return false;
}

bool Communicator::do_accept(int rank) {
  if (rank == global_rank_) return true;
  if (rank < 0 || rank >= world_size_) {
    std::cerr << "[ERROR] Communicator " << global_rank_ << " invalid rank "
              << rank << ", world_size=" << world_size_ << std::endl;
    return false;
  }
  if (has_peer_path(rank)) return true;

  ResolvedPeer resolved;
  try {
    resolved = resolve_peer(rank);
  } catch (std::exception const& ex) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " failed to resolve transport for rank " << rank << ": "
              << ex.what() << std::endl;
    return false;
  }

  if (resolved.kind == PeerTransportKind::Uccl) {
    auto& uccl_adapter =
        ensure_uccl_adapter(resolved.local_meta, resolved.remote_meta);
    if (!uccl_adapter.has_recv_peer(rank) ||
        !uccl_adapter.has_send_peer(rank)) {
      int dev_idx = uccl_adapter.get_best_dev_idx(local_gpu_idx_);
      if (dev_idx < 0) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " UCCL get_best_dev_idx failed for local gpu "
                  << local_gpu_idx_ << std::endl;
        return try_fallback_tcp_accept(rank, resolved.local_meta,
                                       resolved.remote_meta);
      }
      uint16_t local_port = uccl_adapter.get_p2p_listen_port(dev_idx);
      if (local_port == 0) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " UCCL local listen port is invalid for dev " << dev_idx
                  << std::endl;
        return try_fallback_tcp_accept(rank, resolved.local_meta,
                                       resolved.remote_meta);
      }
      std::string local_ip_addr = uccl_adapter.get_p2p_listen_ip(dev_idx);
      if (local_ip_addr.empty()) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " UCCL local listen ip is empty for dev " << dev_idx
                  << std::endl;
        return try_fallback_tcp_accept(rank, resolved.local_meta,
                                       resolved.remote_meta);
      }
      UCCLP2PInfo local_p2p_info(local_ip_addr, local_port, dev_idx,
                                 local_gpu_idx_);
      std::string p2p_key = uccl_p2p_key(global_rank_, rank);
      std::string peer_p2p_key = uccl_p2p_key(rank, global_rank_);
      UCCLP2PInfo remote_p2p_info;
      if (!fetch_remote_then_publish_local(
              *exchanger_client_, peer_p2p_key, remote_p2p_info, p2p_key,
              local_p2p_info, bootstrap_timeout_ms())) {
        return false;
      }
      // accept() path: accept first, then establish reverse flow.
      if ((!uccl_adapter.has_recv_peer(rank) ||
           !uccl_adapter.has_send_peer(rank)) &&
          !uccl_adapter.accept_from_peer(
              rank, remote_p2p_info.ip, remote_p2p_info.dev_idx,
              remote_p2p_info.gpu_idx, remote_p2p_info.port)) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " UCCL accept_from failed from rank " << rank << std::endl;
        return try_fallback_tcp_accept(rank, resolved.local_meta,
                                       resolved.remote_meta);
      }
    }
    mark_peer_path_ready(rank, PeerTransportKind::Uccl);
    register_existing_local_mrs_with_uccl();
    return true;
  }

  if (resolved.kind == PeerTransportKind::Ipc) {
    if (!ipc_channel_->accept_from(rank)) {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC accept_from failed from rank " << rank << std::endl;
      return false;
    }
    mark_peer_path_ready(rank, PeerTransportKind::Ipc);
    return true;
  }

  if (resolved.kind == PeerTransportKind::Tcp) {
    auto& tcp_adapter = ensure_tcp_adapter(resolved.local_meta);
    if (!tcp_adapter.has_recv_peer(rank)) {
      std::string p2p_key = tcp_p2p_key(global_rank_, rank);
      std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);
      TcpP2PInfo remote_p2p_info;
      TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                                tcp_adapter.get_listen_port());
      if (!fetch_remote_then_publish_local(
              *exchanger_client_, peer_p2p_key, remote_p2p_info, p2p_key,
              local_p2p_info, bootstrap_timeout_ms())) {
        return false;
      }
      if (!tcp_adapter.accept_from_peer(rank, remote_p2p_info.ip)) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " TCP accept_from failed from rank " << rank << std::endl;
        return false;
      }
    }
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }
  return false;
}

std::shared_ptr<IpcChannel> Communicator::get_ipc_channel_by_rank(int rank) {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return nullptr;
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  bool uses_ipc = (peer.kind == PeerTransportKind::Ipc);
  if (!peer.connected || !uses_ipc) return nullptr;
  return ipc_channel_;
}

bool Communicator::has_peer_path(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return false;
  return peer_states_.at(static_cast<size_t>(rank)).connected;
}

void Communicator::mark_peer_path_ready(int rank, PeerTransportKind kind) {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto& peer = peer_states_.at(static_cast<size_t>(rank));
  peer.kind = kind;
  peer.connected = true;
}

PeerTransportKind Communicator::get_peer_transport_kind(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) {
    throw std::runtime_error("transport peer rank out of range");
  }
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (!peer.has_meta) {
    throw std::runtime_error("transport peer session is not established");
  }
  if (!peer.connected) {
    throw std::runtime_error("transport peer path is not established");
  }
  if (peer.kind == PeerTransportKind::Unknown) {
    throw std::runtime_error("transport peer kind is unknown");
  }
  return peer.kind;
}

PeerTransportKind Communicator::peer_transport_kind(int rank) const {
  return get_peer_transport_kind(rank);
}

TransportAdapter* Communicator::get_adapter(int rank) {
  auto kind = get_peer_transport_kind(rank);
  return get_adapter(kind);
}

TransportAdapter* Communicator::get_adapter(PeerTransportKind kind) {
  switch (kind) {
    case PeerTransportKind::Uccl:
      return uccl_adapter_.get();
    case PeerTransportKind::Tcp:
      return tcp_adapter_.get();
    case PeerTransportKind::Ipc:
      return ipc_channel_.get();
    default:
      return nullptr;
  }
}

bool Communicator::same_host(int rank) const {
  if (rank == global_rank_) return true;
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& local_peer = peer_states_.at(static_cast<size_t>(global_rank_));
  auto const& remote_peer = peer_states_.at(static_cast<size_t>(rank));
  if (!local_peer.has_meta || !remote_peer.has_meta) {
    throw std::runtime_error("transport peer metadata is not established");
  }
  return local_peer.meta.host_id == remote_peer.meta.host_id;
}

void Communicator::register_existing_local_mrs_with_uccl() {
  if (!uccl_adapter_ || !uccl_adapter_->is_initialized()) return;
  mr_manager_.sync_local_backend();
}

bool Communicator::ensure_uccl_memory_registered(uint64_t mr_id, void* ptr,
                                                 size_t len) {
  if (uccl_adapter_ && uccl_adapter_->is_initialized()) {
    if (uccl_adapter_->is_memory_registered(mr_id)) return true;

    void* base_ptr = ptr;
    size_t mr_len = len;
    bool is_direct_local_mr = false;

    // Normal device tensor/staging MRs are tracked in MemoryRegistry by id.
    // Host bounce buffers use synthetic ids that are not, so fall back to the
    // explicit ptr/len provided by the caller in that case.
    if (mr_id <= std::numeric_limits<uint32_t>::max()) {
      try {
        MR mr = get_local_mr(static_cast<uint32_t>(mr_id));
        base_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(mr.address));
        mr_len = static_cast<size_t>(mr.length);
        is_direct_local_mr = true;
      } catch (std::exception const&) {
      }
    }

    if (is_direct_local_mr) {
      std::lock_guard<std::mutex> lk(uccl_reg_mu_);
      if (uccl_direct_reg_failed_mrs_.find(mr_id) !=
          uccl_direct_reg_failed_mrs_.end()) {
        return false;
      }
    }

    if (base_ptr == nullptr || mr_len == 0) {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " has invalid base pointer or length for UCCL registration, "
                << "mr_id=" << mr_id << std::endl;
      return false;
    }

    bool ok = uccl_adapter_->register_memory(mr_id, base_ptr, mr_len);
    if (!ok) {
      if (is_direct_local_mr) {
        std::lock_guard<std::mutex> lk(uccl_reg_mu_);
        uccl_direct_reg_failed_mrs_.insert(mr_id);
        std::cerr << "[WARN] Communicator " << global_rank_
                  << " failed to register local GPU MR " << mr_id
                  << " with UCCL, base=" << base_ptr << " len=" << mr_len
                  << "; future requests will fallback to host bounce"
                  << std::endl;
      } else {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " failed to register host bounce MR " << mr_id
                  << " with UCCL, base=" << base_ptr << " len=" << mr_len
                  << std::endl;
      }
    } else {
      std::lock_guard<std::mutex> lk(uccl_reg_mu_);
      uccl_registered_mrs_.insert(mr_id);
    }
    return ok;
  }

  return true;
}

bool Communicator::complete_host_bounce_recv(TrackedRequest& tracked,
                                             bool blocking) {
  if (!tracked.needs_host_to_device_copy) return true;
  if (!tracked.bounce.ptr || !tracked.completion_buffer) return false;

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  if (!tracked.host_copy_submitted) {
    if (tracked.host_copy_event == nullptr) {
      GPU_RT_CHECK(gpuEventCreateWithFlags(&tracked.host_copy_event,
                                           gpuEventDisableTiming));
    }
    GPU_RT_CHECK(gpuMemcpyAsync(static_cast<char*>(tracked.completion_buffer) +
                                    tracked.completion_offset,
                                tracked.bounce.ptr, tracked.completion_bytes,
                                gpuMemcpyHostToDevice, host_copy_stream_));
    GPU_RT_CHECK(gpuEventRecord(tracked.host_copy_event, host_copy_stream_));
    tracked.host_copy_submitted = true;
    if (!blocking) return false;
  }

  if (blocking) {
    GPU_RT_CHECK(gpuEventSynchronize(tracked.host_copy_event));
  } else {
    gpuError_t query = gpuEventQuery(tracked.host_copy_event);
    if (query == gpuErrorNotReady) return false;
    if (query != gpuSuccess) {
      GPU_RT_CHECK(query);
    }
  }
  GPU_RT_CHECK(gpuEventDestroy(tracked.host_copy_event));
  tracked.host_copy_event = nullptr;
  tracked.host_copy_submitted = false;
  tracked.needs_host_to_device_copy = false;
  return true;
}

void Communicator::cleanup_tracked_request(TrackedRequest& tracked) {
  unsigned adapter_id = tracked.adapter_request_id;
  if (tracked.kind == PeerTransportKind::Uccl && uccl_adapter_) {
    uccl_adapter_->release_request(adapter_id);
  }
  if (tracked.kind == PeerTransportKind::Tcp && tcp_adapter_) {
    tcp_adapter_->release_request(adapter_id);
  }
  if (tracked.kind == PeerTransportKind::Ipc && ipc_channel_) {
    ipc_channel_->release_request(adapter_id);
  }
  if (tracked.host_copy_event != nullptr) {
    GPU_RT_CHECK(gpuEventDestroy(tracked.host_copy_event));
    tracked.host_copy_event = nullptr;
  }
  tracked.host_copy_submitted = false;
  tracked.needs_host_to_device_copy = false;
  // NOTE: bounce memory/MR is request-scoped today. Cleanup always tears down
  // the request-owned bounce resources. A future performance PR can replace
  // this with a longer-lived pool and defer teardown outside request cleanup.
  tracked.bounce_owner.reset();
  if (shm_manager_ && tracked.bounce.valid) {
    (void)mr_manager_.delete_mr(tracked.bounce.ptr);
    shm_manager_->delete_local_shm(tracked.bounce.shm_id);
    tracked.bounce = {};
  }
}

unsigned Communicator::isend(int rank, LocalSlice src,
                             std::optional<RemoteSlice> dst_hint) {
  if (src.mem_id == 0 || src.bytes == 0) {
    throw std::invalid_argument("isend requires non-empty local slice");
  }
  if (!has_peer_path(rank) && !connect(rank)) {
    throw std::runtime_error("transport peer path is not established");
  }
  MR local_mr = get_local_mr(src.mem_id);
  if (src.offset > static_cast<size_t>(local_mr.length) ||
      src.bytes > static_cast<size_t>(local_mr.length) - src.offset) {
    throw std::invalid_argument("isend local slice out of range");
  }

  auto peer_kind = get_peer_transport_kind(rank);
  auto* adapter = get_adapter(peer_kind);
  if (!adapter) {
    throw std::runtime_error("failed to get adapter for peer");
  }
  validate_dst_hint_for_transport(peer_kind, dst_hint, src.bytes);

  void* local_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(local_mr.address) + src.offset);

  bool needs_uccl_registration = (peer_kind == PeerTransportKind::Uccl);
  bool use_shareable = (peer_kind == PeerTransportKind::Ipc);
  bool needs_bounce =
      (peer_kind == PeerTransportKind::Tcp) ||
      (peer_kind == PeerTransportKind::Ipc) ||
      (peer_kind == PeerTransportKind::Uccl &&
       !ensure_uccl_memory_registered(src.mem_id, local_ptr, src.bytes));
  unsigned rid = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  TrackedRequest tracked{};
  tracked.peer_rank = rank;
  tracked.kind = peer_kind;
  std::shared_ptr<SHMItem> bounce_owner;
  if (needs_bounce) {
    // Request-scoped lease for send bounce memory. Current design allocates
    // and frees per request; this is intentionally simple but not optimal for
    // steady-state throughput. Poolization can be added in a follow-up PR.
    bounce_owner = std::shared_ptr<SHMItem>(
        new SHMItem{}, [this](SHMItem* lease) {
          if (lease != nullptr) {
            if (shm_manager_ != nullptr && lease->valid) {
              (void)mr_manager_.delete_mr(lease->ptr);
              shm_manager_->delete_local_shm(lease->shm_id);
            }
            delete lease;
          }
        });
    tracked.bounce_owner = bounce_owner;
  }
  auto bounce_provider = [this, needs_bounce, needs_uccl_registration,
                          use_shareable,
                          bounce_owner](size_t bytes) -> BounceBufferInfo {
    if (!needs_bounce || !bounce_owner) return {};
    if (!bounce_owner->valid) {
      // Lazy allocate on first use within this request.
      *bounce_owner = shm_manager_->create_local_shm(bytes, use_shareable);
    }
    BounceBufferInfo info;
    info.ptr = bounce_owner->ptr;
    if (needs_uccl_registration) {
      // Request-scoped bounce MR id. The MR is deleted in request cleanup.
      auto mr = mr_manager_.create_local_mr(bounce_owner->ptr, bytes);
      info.mr_id = mr.mr.id;
    }
    if (bounce_owner->shareable) {
      info.shm_name = shm_manager_->get_local_shm(bounce_owner->shm_id).shm_name;
    }
    return info;
  };

  unsigned result = adapter->send_async(rank, local_ptr, src.bytes, src.mem_id,
                                        dst_hint, bounce_provider);
  if (result == 0) {
    cleanup_tracked_request(tracked);
    return 0;
  }
  tracked.adapter_request_id = result;

  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_[rid] = std::move(tracked);
  }
  progress_cv_.notify_all();
  return rid;
}

unsigned Communicator::irecv(int rank, LocalSlice dst) {
  if (dst.mem_id == 0 || dst.bytes == 0) {
    throw std::invalid_argument("irecv requires non-empty local slice");
  }
  if (!has_peer_path(rank) && !connect(rank)) {
    throw std::runtime_error("transport peer path is not established");
  }
  MR local_mr = get_local_mr(dst.mem_id);
  if (dst.offset > static_cast<size_t>(local_mr.length) ||
      dst.bytes > static_cast<size_t>(local_mr.length) - dst.offset) {
    throw std::invalid_argument("irecv local slice out of range");
  }

  auto peer_kind = get_peer_transport_kind(rank);
  auto* adapter = get_adapter(peer_kind);
  if (!adapter) {
    throw std::runtime_error("failed to get adapter for peer");
  }

  void* local_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(local_mr.address) + dst.offset);

  bool needs_uccl_registration = (peer_kind == PeerTransportKind::Uccl);
  bool use_shareable = (peer_kind == PeerTransportKind::Ipc);
  unsigned rid = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  TrackedRequest tracked{};
  tracked.peer_rank = rank;
  tracked.kind = peer_kind;

  auto needs_bounce =
      (peer_kind == PeerTransportKind::Tcp) ||
      (peer_kind == PeerTransportKind::Uccl &&
       !ensure_uccl_memory_registered(dst.mem_id, local_ptr, dst.bytes));

  // IPC fast path metadata: let sender resolve remote pointer by dst.mem_id
  // and skip per-request ipc_cache handshake when possible.
  if (peer_kind == PeerTransportKind::Ipc) {
    // Publish MR base mapping (not slice pointer) so sender-side
    // resolve_ipc_buffer_pointer(remote_offset) applies offset exactly once.
    void* ipc_base_ptr =
        reinterpret_cast<void*>(static_cast<uintptr_t>(local_mr.address));
    size_t ipc_bytes = static_cast<size_t>(local_mr.length);
    if (!notify_ipc_buffer(rank, dst.mem_id, ipc_base_ptr, ipc_bytes)) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " failed to publish IPC buffer metadata for rank " << rank
                << ", ipc_id=" << dst.mem_id
                << "; sender may fallback to ipc_cache handshake" << std::endl;
    }
  }

  if (needs_bounce) {
    // Request-scoped recv bounce. It is released when this request completes.
    tracked.bounce = shm_manager_->create_local_shm(dst.bytes, use_shareable);
    tracked.needs_host_to_device_copy = true;
    tracked.completion_buffer = local_ptr;
    tracked.completion_bytes = dst.bytes;
  }

  auto bounce_provider = [this, &tracked, needs_bounce, needs_uccl_registration,
                          use_shareable](size_t bytes) -> BounceBufferInfo {
    if (!needs_bounce) return {};
    if (!tracked.bounce.valid) {
      // Lazy allocate on first use within this request.
      tracked.bounce = shm_manager_->create_local_shm(bytes, use_shareable);
    }
    BounceBufferInfo info;
    info.ptr = tracked.bounce.ptr;
    if (needs_uccl_registration) {
      // Request-scoped bounce MR id. The MR is deleted in request cleanup.
      auto mr = mr_manager_.create_local_mr(tracked.bounce.ptr, bytes);
      info.mr_id = mr.mr.id;
    }
    if (tracked.bounce.shareable) {
      info.shm_name = shm_manager_->get_local_shm(tracked.bounce.shm_id).shm_name;
    }
    return info;
  };

  unsigned result = adapter->recv_async(rank, local_ptr, dst.bytes, dst.mem_id,
                                        bounce_provider);
  if (result == 0) {
    cleanup_tracked_request(tracked);
    return 0;
  }
  tracked.adapter_request_id = result;

  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_[rid] = std::move(tracked);
  }
  progress_cv_.notify_all();
  return rid;
}

bool Communicator::poll_request_completion(unsigned id, bool blocking) {
  TrackedRequest snapshot;
  {
    std::lock_guard<std::mutex> lk(req_mu_);
    auto it = requests_map_.find(id);
    if (it == requests_map_.end()) return true;
    if (it->second.completed) return true;
    snapshot = it->second;
  }

  bool done = false;
  bool failed = false;
  unsigned adapter_id = snapshot.adapter_request_id;
  if (snapshot.kind == PeerTransportKind::Uccl) {
    done = blocking ? uccl_adapter_->wait_completion(adapter_id)
                    : uccl_adapter_->poll_completion(adapter_id);
    failed = done && uccl_adapter_->request_failed(adapter_id);
  } else if (snapshot.kind == PeerTransportKind::Tcp) {
    done = blocking ? tcp_adapter_->wait_completion(adapter_id)
                    : tcp_adapter_->poll_completion(adapter_id);
    failed = done && tcp_adapter_->request_failed(adapter_id);
  } else if (snapshot.kind == PeerTransportKind::Ipc) {
    done = blocking ? ipc_channel_->wait_completion(adapter_id)
                    : ipc_channel_->poll_completion(adapter_id);
    failed = done && ipc_channel_->request_failed(adapter_id);
  }

  if (!done) return false;

  std::lock_guard<std::mutex> lk(req_mu_);
  auto it = requests_map_.find(id);
  if (it == requests_map_.end()) return true;
  if (!failed) {
    bool copy_done = complete_host_bounce_recv(it->second, blocking);
    if (!copy_done) return false;
  }
  it->second.completed = true;
  it->second.failed = failed;
  return true;
}

bool Communicator::wait_finish(std::vector<unsigned> const& reqs) {
  std::unordered_set<unsigned> remaining;
  bool any_failed = false;
  if (reqs.empty()) {
    std::lock_guard<std::mutex> lk(req_mu_);
    for (auto const& [id, _] : requests_map_) {
      remaining.insert(id);
    }
  } else {
    remaining.insert(reqs.begin(), reqs.end());
  }

  while (!remaining.empty()) {
    bool made_progress = false;
    std::vector<unsigned> finished;

    for (auto id : remaining) {
      if (id == 0) return false;

      if (!poll_request_completion(id, false)) {
        continue;
      }

      made_progress = true;
      TrackedRequest to_cleanup{};
      bool needs_cleanup = false;
      bool done = false;
      {
        std::lock_guard<std::mutex> lk(req_mu_);
        auto it = requests_map_.find(id);
        if (it == requests_map_.end()) {
          done = true;
        } else if (it->second.failed || it->second.completed) {
          any_failed = any_failed || it->second.failed;
          to_cleanup = std::move(it->second);
          requests_map_.erase(it);
          needs_cleanup = true;
          done = true;
        }
      }
      if (needs_cleanup) {
        cleanup_tracked_request(to_cleanup);
      }
      if (done) {
        finished.push_back(id);
      }
    }

    for (auto id : finished) remaining.erase(id);
    if (remaining.empty()) return !any_failed;

    if (!made_progress) {
      std::this_thread::yield();
    }
  }

  return !any_failed;
}

bool Communicator::poll(unsigned const req) {
  if (req == 0) return false;
  if (!poll_request_completion(req, false)) return false;

  std::lock_guard<std::mutex> lk(req_mu_);
  auto it = requests_map_.find(req);
  if (it == requests_map_.end()) return true;
  if (it->second.failed) {
    throw std::runtime_error("transport request failed");
  }
  return it->second.completed;
}

void Communicator::release(unsigned const req) {
  TrackedRequest to_cleanup{};
  bool needs_cleanup = false;
  {
    std::lock_guard<std::mutex> lk(req_mu_);
    auto it = requests_map_.find(req);
    if (it == requests_map_.end()) return;
    if (!it->second.completed) {
      throw std::runtime_error("cannot release an in-flight transport request");
    }
    to_cleanup = std::move(it->second);
    requests_map_.erase(it);
    needs_cleanup = true;
  }
  if (needs_cleanup) {
    cleanup_tracked_request(to_cleanup);
  }
}

bool Communicator::wait_finish(unsigned const req) {
  return wait_finish(std::vector<unsigned>{req});
}

MR Communicator::reg_mr(void* local_buf, size_t len) {
  return mr_manager_.create_local_mr(local_buf, len).mr;
}

bool Communicator::dereg_mr(void* local_buf) {
  (void)mr_manager_.delete_mr(local_buf);
  return true;
}

bool Communicator::notify_named_mrs(int remote_rank, uint64_t generation,
                                    NamedMRInfos const& infos) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  std::string key = "named-mr:" + std::to_string(global_rank_) + "->" +
                    std::to_string(remote_rank) + ":" +
                    std::to_string(generation);
  std::lock_guard<std::mutex> lk(mr_exchange_mu_);
  NamedMRInfos payload = infos;
  payload.generation = generation;

  // for (auto const& entry : payload.entries) {
  //   std::cout << "[notify named MR to rank " << remote_rank
  //             << "] generation=" << generation
  //             << " buffer_id=" << entry.buffer_id
  //             << " addr=" << entry.mr.address
  //             << " length=" << entry.mr.length
  //             << " key=" << entry.mr.key << std::endl;
  // }

  return exchanger_client_->publish(key, payload);
}

bool Communicator::wait_named_mrs(int remote_rank, uint64_t generation,
                                  NamedMRInfos& infos) {
  if (!exchanger_client_ || !exchanger_client_->valid()) {
    throw std::runtime_error("Exchanger client is not valid");
  }

  std::string key = "named-mr:" + std::to_string(remote_rank) + "->" +
                    std::to_string(global_rank_) + ":" +
                    std::to_string(generation);
  int timeout_ms = mr_timeout_ms();
  if (!exchanger_client_->wait_and_fetch(
          key, infos, timeout_to_retries(timeout_ms, 1), 1)) {
    std::cerr << "[WARN] Timeout waiting for named MR table from rank "
              << remote_rank << std::endl;
    return false;
  }
  if (infos.generation != generation) {
    std::cerr << "[WARN] Named MR generation mismatch from rank " << remote_rank
              << ": expected=" << generation << " got=" << infos.generation
              << std::endl;
    return false;
  }

  std::vector<MRItem> remote_mrs;
  remote_mrs.reserve(infos.entries.size());
  for (auto const& entry : infos.entries) {
    MRItem item{};
    item.mr = entry.mr;
    item.is_local = false;
    item.rank = remote_rank;
    item.valid = true;
    remote_mrs.push_back(item);
    // std::cout << "[recv named MR from rank " << remote_rank
    //           << "] generation=" << generation
    //           << " buffer_id=" << entry.buffer_id
    //           << " addr=" << entry.mr.address
    //           << " length=" << entry.mr.length
    //           << " key=" << entry.mr.key << std::endl;
  }
  mr_manager_.register_remote_mrs(remote_rank, remote_mrs);
  return true;
}

MR Communicator::get_local_mr(void* local_buf) {
  auto item = mr_manager_.get_mr(local_buf);
  if (!item.valid) throw std::runtime_error("Local MR not found for buffer");
  return item.mr;
}

MR Communicator::get_local_mr(uint32_t mr_id) {
  auto item = mr_manager_.get_mr(mr_id);
  if (!item.valid) throw std::runtime_error("Local MR not found for id");
  return item.mr;
}

MR Communicator::get_remote_mr(int remote_rank, uint32_t mr_id) {
  auto item = mr_manager_.get_mr(remote_rank, mr_id);
  if (!item.valid) throw std::runtime_error("Remote MR not found for id");
  return item.mr;
}

bool Communicator::notify_ipc_buffer(int remote_rank, uint32_t ipc_id,
                                     void* local_buf, size_t len,
                                     uint64_t binding_version) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  IpcBufferInfo info{};
  info.ipc_id = ipc_id;
  if (binding_version == 0) {
    std::lock_guard<std::mutex> lk(ipc_gen_mu_);
    binding_version = ++local_ipc_binding_versions_[remote_rank][ipc_id];
  }
  info.binding_version = binding_version;
  info.valid = false;
  if (local_buf != nullptr && len != 0) {
    int original_device = -1;
    GPU_RT_CHECK(gpuGetDevice(&original_device));
    auto restore =
        uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(original_device)); });
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
    auto exported = ipc_manager_.create_local_ipc(local_buf, len, local_gpu_idx_);
    if (!exported.valid) return false;

    info.handle = exported.handle;
    info.base_offset = reinterpret_cast<uintptr_t>(local_buf) -
                       exported.base_addr;
    info.bytes = len;
    info.device_idx = exported.device_idx;
    info.valid = true;
  }

  auto const latest_key = ipc_buffer_key(global_rank_, remote_rank, ipc_id);
  if (!exchanger_client_->publish(latest_key, info)) return false;
  if (binding_version == 0) return true;
  auto const versioned_key = ipc_buffer_versioned_key(global_rank_, remote_rank,
                                                      ipc_id, binding_version);
  return exchanger_client_->publish(versioned_key, info);
}

bool Communicator::wait_ipc_buffer(int remote_rank, uint32_t ipc_id,
                                   uint64_t expected_binding_version) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;
  IpcBufferInfo info{};
  auto const latest_key = ipc_buffer_key(remote_rank, global_rank_, ipc_id);
  if (expected_binding_version != 0) {
    auto const versioned_key = ipc_buffer_versioned_key(
        remote_rank, global_rank_, ipc_id, expected_binding_version);
    // Prefer deterministic versioned key to avoid stale latest-key reads.
    if (exchanger_client_->wait_and_fetch(
            versioned_key, info,
            timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
            kBootstrapPollDelayMs)) {
      IPCItem state{};
      state.handle = info.handle;
      state.binding_version = info.binding_version;
      state.base_offset = static_cast<uintptr_t>(info.base_offset);
      state.bytes = static_cast<size_t>(info.bytes);
      state.device_idx = info.device_idx;
      state.valid = info.valid;
      state.ipc_id = ipc_id;
      return ipc_manager_.register_remote_ipc(remote_rank, state);
    }
  }
  auto const deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(bootstrap_timeout_ms());
  while (true) {
    if (!exchanger_client_->wait_and_fetch(
            latest_key, info,
            timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
            kBootstrapPollDelayMs)) {
      return false;
    }
    if (expected_binding_version == 0 ||
        info.binding_version == expected_binding_version) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kBootstrapPollDelayMs));
  }

  IPCItem state{};
  state.handle = info.handle;
  state.binding_version = info.binding_version;
  state.base_offset = static_cast<uintptr_t>(info.base_offset);
  state.bytes = static_cast<size_t>(info.bytes);
  state.device_idx = info.device_idx;
  state.valid = info.valid;
  state.ipc_id = ipc_id;
  return ipc_manager_.register_remote_ipc(remote_rank, state);
}

bool Communicator::fetch_ipc_buffer(int remote_rank, uint32_t ipc_id,
                                    uint64_t expected_binding_version) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;
  IpcBufferInfo info{};
  if (expected_binding_version != 0) {
    auto const versioned_key = ipc_buffer_versioned_key(
        remote_rank, global_rank_, ipc_id, expected_binding_version);
    if (exchanger_client_->fetch(versioned_key, info)) {
      IPCItem state{};
      state.handle = info.handle;
      state.binding_version = info.binding_version;
      state.base_offset = static_cast<uintptr_t>(info.base_offset);
      state.bytes = static_cast<size_t>(info.bytes);
      state.device_idx = info.device_idx;
      state.valid = info.valid;
      if (state.valid && info.binding_version != expected_binding_version) {
        invalidate_remote_ipc_buffer(remote_rank, ipc_id);
        return false;
      }
      state.ipc_id = ipc_id;
      return ipc_manager_.register_remote_ipc(remote_rank, state);
    }
  }
  if (!exchanger_client_->fetch(
          ipc_buffer_key(remote_rank, global_rank_, ipc_id), info)) {
    return false;
  }
  IPCItem state{};
  state.handle = info.handle;
  state.binding_version = info.binding_version;
  state.base_offset = static_cast<uintptr_t>(info.base_offset);
  state.bytes = static_cast<size_t>(info.bytes);
  state.device_idx = info.device_idx;
  state.valid = info.valid;
  if (state.valid && expected_binding_version != 0 &&
      info.binding_version != expected_binding_version) {
    invalidate_remote_ipc_buffer(remote_rank, ipc_id);
    return false;
  }
  state.ipc_id = ipc_id;
  return ipc_manager_.register_remote_ipc(remote_rank, state);
}

bool Communicator::has_fresh_remote_ipc_buffer(
    int remote_rank, uint32_t ipc_id, uint64_t expected_binding_version) const {
  auto state = ipc_manager_.get_ipc(remote_rank, ipc_id);
  if (!state.valid) return false;
  if (expected_binding_version != 0 &&
      state.binding_version != expected_binding_version) {
    return false;
  }
  return true;
}

void Communicator::invalidate_remote_ipc_buffer(int remote_rank,
                                                uint32_t ipc_id) {
  ipc_manager_.delete_ipc(remote_rank, ipc_id);
}

bool Communicator::resolve_ipc_buffer_pointer(int remote_rank, uint32_t ipc_id,
                                              size_t offset, size_t bytes,
                                              void** out_ptr,
                                              int* out_device_idx) {
  if (out_ptr == nullptr) return false;

  auto state = ipc_manager_.get_ipc(remote_rank, ipc_id);
  if (!state.valid) return false;
  if (offset > state.bytes || bytes > state.bytes - offset) return false;

  int original_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&original_device));
  auto restore =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(original_device)); });
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  if (state.direct_ptr == nullptr) {
    gpuError_t open_err = gpuIpcOpenMemHandle(&state.direct_ptr, state.handle,
                                              gpuIpcMemLazyEnablePeerAccess);
    if (open_err != gpuSuccess) {
      // Treat IPC open failure as recoverable here so upper layers can
      // fallback to host-bounce relay instead of aborting.
      return false;
    }
    state.ipc_id = ipc_id;
    ipc_manager_.register_remote_ipc(remote_rank, state);
  }

  *out_ptr =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(state.direct_ptr) +
                              state.base_offset + offset);
  if (out_device_idx != nullptr) {
    *out_device_idx = state.device_idx;
  }
  return true;
}

bool Communicator::register_remote_ipc_cache(int remote_rank,
                                             gpuIpcMemHandle_t handle,
                                             IPCItem const& ipc) {
  IPCItem item = ipc;
  item.handle = handle;
  return ipc_manager_.register_remote_ipc(remote_rank, item);
}

IPCItem Communicator::get_remote_ipc_cache(int remote_rank,
                                             gpuIpcMemHandle_t handle) {
  return ipc_manager_.get_ipc(remote_rank, handle);
}

void* Communicator::get_or_open_bounce_shm(std::string const& shm_name) {
  return shm_manager_->open_remote_shm(shm_name).ptr;
}

void Communicator::clear_bounce_shm_cache() {
  if (shm_manager_) {
    shm_manager_->clear_remote_shm_cache();
  }
}

std::shared_ptr<void> Communicator::register_completion_notifier(
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> cb) {
  auto target = std::make_shared<NotifyTarget>();
  target->emit = std::move(cb);

  {
    std::lock_guard<std::mutex> lk(progress_mu_);
    notify_targets_.push_back(target);
  }

  // The progress loop is already running; registering a notifier only changes
  // which callbacks should observe completed requests.
  progress_cv_.notify_all();

  return std::static_pointer_cast<void>(target);
}

void Communicator::progress_loop() {
  while (progress_running_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lk(progress_mu_);
      progress_cv_.wait(lk, [&] {
        if (!progress_running_.load()) return true;

        std::lock_guard<std::mutex> rlk(req_mu_);
        return !requests_map_.empty();
      });
    }

    if (!progress_running_.load()) break;

    bool progress = false;
    std::vector<unsigned> ids;
    {
      std::lock_guard<std::mutex> rlk(req_mu_);
      ids.reserve(requests_map_.size());
      for (auto const& [id, _] : requests_map_) ids.push_back(id);
    }

    auto now = std::chrono::steady_clock::now();
    for (auto id : ids) {
      if (!poll_request_completion(id, false)) continue;

      bool should_emit = false;
      {
        std::lock_guard<std::mutex> rlk(req_mu_);
        auto it = requests_map_.find(id);
        if (it != requests_map_.end() && !it->second.notified) {
          it->second.notified = true;
          should_emit = true;
        }
      }
      if (!should_emit) continue;

      std::vector<std::shared_ptr<NotifyTarget>> targets;
      {
        std::lock_guard<std::mutex> nlk(progress_mu_);
        notify_targets_.erase(
            std::remove_if(notify_targets_.begin(), notify_targets_.end(),
                           [](std::weak_ptr<NotifyTarget> const& target) {
                             return target.expired();
                           }),
            notify_targets_.end());
        targets.reserve(notify_targets_.size());
        for (auto const& weak_target : notify_targets_) {
          if (auto target = weak_target.lock()) {
            targets.push_back(std::move(target));
          }
        }
      }
      for (auto& tgt : targets) {
        if (!tgt) continue;
        tgt->emit(id, now);
      }
      progress = true;
    }

    if (!progress) {
      std::this_thread::yield();
    }
  }
}

}  // namespace Transport
}  // namespace UKernel
