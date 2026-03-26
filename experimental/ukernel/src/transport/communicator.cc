#include "communicator.h"
#include "ipc_channel.h"
#include "utils.h"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
      next_send_match_seq_(world_size, 1),
      next_recv_match_seq_(world_size, 1),
      peer_manager_(std::make_shared<PeerSessionManager>(world_size)),
      config_(config) {
  maybe_configure_uccl_socket_ifname({}, get_local_ip());

  shm_control_ = std::make_shared<ShmRingExchanger>(
      global_rank_, world_size_, generate_host_id(),
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

  host_bounce_pool_ = std::make_unique<HostBouncePool>(
      [this](uint64_t mr_id, void* ptr, size_t len) {
        return ensure_uccl_memory_registered(mr_id, ptr, len);
      },
      [this](uint64_t mr_id) {
        if (uccl_adapter_) uccl_adapter_->deregister_memory(mr_id);
      });

  exchange_peer_metas();
  std::cout << "[INFO] Communicator " << global_rank_
            << " initialized: peer meta exchange success" << std::endl;
}

void Communicator::exchange_peer_metas() {
  CommunicatorMeta local;
  local.host_id = generate_host_id();
  local.local_id = config_->local_id >= 0 ? config_->local_id : global_rank_;
  local.rdma_capable = detect_local_rdma_capable();
  local.is_ready = true;
  local.ip = get_local_ip();
  peer_manager_->get(global_rank_)->set_meta(local);

  std::string meta_key = "meta:" + std::to_string(global_rank_);
  if (!exchanger_client_->publish(meta_key, local)) {
    fprintf(stderr, "[ERROR] Failed to publish local CommunicatorMeta \n");
  }

  CommunicatorMeta remote;
  for (int i = 0; i < world_size_; i++) {
    if (i == global_rank_) continue;
    std::string key = "meta:" + std::to_string(i);
    if (exchanger_client_->wait_and_fetch(
            key, remote,
            timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
            kBootstrapPollDelayMs)) {
      peer_manager_->get(i)->set_meta(remote);
      shm_control_->set_peer_local_id(i, remote.local_id);
    } else {
      fprintf(stderr, "[WARN] Timeout waiting for remote CommunicatorMeta \n");
    }
  }
}

Communicator::~Communicator() {
  if (notifier_started_.load()) {
    notifier_running_.store(false);
    notifier_cv_.notify_all();
    if (notifier_thread_.joinable()) {
      notifier_thread_.join();
    }
  }

  shutdown_ipc_channel();
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
    for (int rank = 0; rank < world_size_; ++rank) {
      if (rank == global_rank_) continue;
      auto* peer = peer_manager_->get(rank);
      if (peer->transport_kind() == PeerTransportKind::Ipc &&
          (peer->send_ready() || peer->recv_ready())) {
        ipc_peers.push_back(rank);
      }
    }
  }

  if (shm_control) {
    for (int rank : ipc_peers) {
      shm_control->close_peer(rank);
    }
  }

  for (auto* p : memory_registry_.local_buffers()) {
    dereg_mr(p);
  }

  memory_registry_.clear_remote_ipc_cache();

  std::cout << "[INFO] Communicator " << global_rank_ << " resources released"
            << std::endl;
}

UcclTransportAdapter& Communicator::ensure_uccl_adapter(
    CommunicatorMeta const& local_meta, CommunicatorMeta const& peer_meta) {
  if (!uccl_adapter_) {
    UcclTransportConfig uccl_config;
    uccl_config.local_ip = local_meta.ip;
    maybe_configure_uccl_socket_ifname(
        get_uccl_remote_hint_ip(config_, peer_meta), local_meta.ip);
    uccl_adapter_ = std::make_unique<UcclTransportAdapter>(
        local_gpu_idx_, world_size_, std::move(uccl_config));
  }
  return *uccl_adapter_;
}

TcpTransportAdapter& Communicator::ensure_tcp_adapter(
    CommunicatorMeta const& local_meta) {
  if (!tcp_adapter_) {
    tcp_adapter_ =
        std::make_unique<TcpTransportAdapter>(local_meta.ip, global_rank_,
                                              world_size_);
  }
  return *tcp_adapter_;
}

bool Communicator::try_fallback_tcp_connect(
    int rank, CommunicatorMeta const& local_meta,
    CommunicatorMeta const& remote_meta) {
  if (config_->preferred_transport != PreferredTransport::Auto) return false;
  auto& tcp_adapter = ensure_tcp_adapter(local_meta);
  if (tcp_adapter.has_send_peer(rank)) {
    cache_peer_session(rank, PeerTransportKind::Tcp, true, false);
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

  bool ret =
      tcp_adapter.connect_to_peer(rank, remote_p2p_info.ip, remote_p2p_info.port);
  if (ret) {
    cache_peer_session(rank, PeerTransportKind::Tcp, true, false);
    std::cout << "[INFO] Communicator " << global_rank_
              << " TCP fallback connect_to succeeded to rank " << rank
              << std::endl;
  }
  return ret;
}

bool Communicator::try_fallback_tcp_accept(int rank,
                                           CommunicatorMeta const& local_meta,
                                           CommunicatorMeta const& remote_meta) {
  if (config_->preferred_transport != PreferredTransport::Auto) return false;
  auto& tcp_adapter = ensure_tcp_adapter(local_meta);
  if (tcp_adapter.has_recv_peer(rank)) {
    cache_peer_session(rank, PeerTransportKind::Tcp, false, true);
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

  bool ret = tcp_adapter.accept_from_peer(rank, remote_meta.ip);
  if (ret) {
    cache_peer_session(rank, PeerTransportKind::Tcp, false, true);
    std::cout << "[INFO] Communicator " << global_rank_
              << " TCP fallback accept_from succeeded from rank " << rank
              << std::endl;
  }
  return ret;
}

bool Communicator::connect_to(int rank) {
  if (!check_ready()) {
    std::cerr << "[WARN] Communicator " << global_rank_
              << " not ready, cannot connect to rank " << rank << std::endl;
    return false;
  }

  if (rank == global_rank_) {
    return true;
  }

  if (rank < 0 || rank >= world_size_) {
    std::cerr << "[ERROR] Communicator " << global_rank_ << " invalid rank "
              << rank << ", world_size=" << world_size_ << std::endl;
    return false;
  }

  auto* local_peer = peer_manager_->get(global_rank_);
  auto* remote_peer = peer_manager_->get(rank);
  if (!local_peer->has_meta() || !remote_peer->has_meta()) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " CommunicatorMeta not found for rank " << rank << std::endl;
    return false;
  }

  PeerTransportKind peer_kind;
  try {
    peer_kind = resolve_peer_transport_kind(*config_, local_peer->meta(),
                                            remote_peer->meta());
  } catch (std::exception const& ex) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " failed to resolve transport for rank " << rank << ": "
              << ex.what() << std::endl;
    return false;
  }
  if (peer_kind == PeerTransportKind::Uccl && has_peer_send_path(rank)) {
    return true;
  }
  if (peer_kind == PeerTransportKind::Ipc && has_peer_send_path(rank)) {
    return true;
  }
  if (peer_kind == PeerTransportKind::Tcp && has_peer_send_path(rank)) {
    return true;
  }

  if (peer_kind == PeerTransportKind::Uccl) {
    CommunicatorMeta const local_meta = local_peer->meta();
    CommunicatorMeta const remote_meta = remote_peer->meta();
    auto& uccl_adapter = ensure_uccl_adapter(local_meta, remote_meta);
    if (uccl_adapter.has_send_peer(rank)) {
      cache_peer_session(rank, PeerTransportKind::Uccl, true, false);
      register_existing_local_mrs_with_uccl();
      return true;
    }

    int dev_idx = uccl_adapter.get_best_dev_idx(local_gpu_idx_);
    uint16_t local_port = uccl_adapter.get_p2p_listen_port(dev_idx);
    std::string local_ip_addr = uccl_adapter.get_p2p_listen_ip(dev_idx);

    UCCLP2PInfo local_p2p_info(local_ip_addr, local_port, dev_idx,
                               local_gpu_idx_);
    std::string p2p_key = uccl_p2p_key(global_rank_, rank);
    std::string peer_p2p_key = uccl_p2p_key(rank, global_rank_);

    if (!exchanger_client_->publish(p2p_key, local_p2p_info)) {
      return false;
    }

    UCCLP2PInfo remote_p2p_info;
    if (!exchanger_client_->wait_and_fetch(
            peer_p2p_key, remote_p2p_info,
            timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
            kBootstrapPollDelayMs)) {
      return false;
    }

    std::cout << "[INFO] Rank " << global_rank_ << " P2P port " << local_port
              << " (GPU " << local_gpu_idx_ << ", dev " << dev_idx << ")"
              << " -> Rank " << rank << " P2P port " << remote_p2p_info.port
              << " (GPU " << remote_p2p_info.gpu_idx << ", dev "
              << remote_p2p_info.dev_idx << ")" << std::endl;

    bool ret = uccl_adapter.connect_to_peer(
        rank, remote_p2p_info.ip, remote_p2p_info.port, dev_idx, local_gpu_idx_,
        remote_p2p_info.dev_idx, remote_p2p_info.gpu_idx);
    if (ret) {
      cache_peer_session(rank, PeerTransportKind::Uccl, true, false);
      register_existing_local_mrs_with_uccl();
      std::cout << "[INFO] Communicator " << global_rank_
                << " UCCL connect_to succeeded to rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " UCCL connect_to failed to rank " << rank << std::endl;
      if (try_fallback_tcp_connect(rank, local_meta, remote_meta)) {
        return true;
      }
    }
    return ret;
  }

  if (peer_kind == PeerTransportKind::Ipc) {
    bool ret = ipc_channel_->connect_to(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << global_rank_
                << " IPC connect_to succeeded to rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC connect_to failed to rank " << rank << std::endl;
      shm_control_->close_peer(rank);
      return false;
    }
    cache_peer_session(rank, PeerTransportKind::Ipc, true, false);
    return true;
  }

  if (peer_kind == PeerTransportKind::Tcp) {
    CommunicatorMeta const local_meta = local_peer->meta();
    CommunicatorMeta const remote_meta = remote_peer->meta();
    auto& tcp_adapter = ensure_tcp_adapter(local_meta);
    if (tcp_adapter.has_send_peer(rank)) {
      cache_peer_session(rank, PeerTransportKind::Tcp, true, false);
      return true;
    }

    TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                              tcp_adapter.get_listen_port());
    std::string p2p_key = tcp_p2p_key(global_rank_, rank);
    std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);
    if (!exchanger_client_->publish(p2p_key, local_p2p_info)) {
      return false;
    }

    TcpP2PInfo remote_p2p_info;
    if (!exchanger_client_->wait_and_fetch(
            peer_p2p_key, remote_p2p_info,
            timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
            kBootstrapPollDelayMs)) {
      return false;
    }

    bool ret = tcp_adapter.connect_to_peer(rank, remote_p2p_info.ip,
                                           remote_p2p_info.port);
    if (ret) {
      cache_peer_session(rank, PeerTransportKind::Tcp, true, false);
      std::cout << "[INFO] Communicator " << global_rank_
                << " TCP connect_to succeeded to rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " TCP connect_to failed to rank " << rank << std::endl;
    }
    return ret;
  }
  return false;
}

bool Communicator::accept_from(int rank) {
  if (!check_ready()) return false;
  if (rank == global_rank_) return true;

  auto* local_peer = peer_manager_->get(global_rank_);
  auto* remote_peer = peer_manager_->get(rank);
  if (!local_peer->has_meta() || !remote_peer->has_meta()) {
    return false;
  }

  PeerTransportKind peer_kind;
  try {
    peer_kind = resolve_peer_transport_kind(*config_, local_peer->meta(),
                                            remote_peer->meta());
  } catch (std::exception const& ex) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " failed to resolve transport for rank " << rank << ": "
              << ex.what() << std::endl;
    return false;
  }
  if (peer_kind == PeerTransportKind::Uccl && has_peer_recv_path(rank)) {
    return true;
  }
  if (peer_kind == PeerTransportKind::Ipc && has_peer_recv_path(rank)) {
    return true;
  }
  if (peer_kind == PeerTransportKind::Tcp && has_peer_recv_path(rank)) {
    return true;
  }

  if (peer_kind == PeerTransportKind::Uccl) {
    CommunicatorMeta const local_meta = local_peer->meta();
    CommunicatorMeta const remote_meta = remote_peer->meta();
    auto& uccl_adapter = ensure_uccl_adapter(local_meta, remote_meta);
    if (uccl_adapter.has_recv_peer(rank)) {
      cache_peer_session(rank, PeerTransportKind::Uccl, false, true);
      register_existing_local_mrs_with_uccl();
      return true;
    }

    int dev_idx = uccl_adapter.get_best_dev_idx(local_gpu_idx_);
    uint16_t local_port = uccl_adapter.get_p2p_listen_port(dev_idx);
    std::string local_ip_addr = uccl_adapter.get_p2p_listen_ip(dev_idx);

    UCCLP2PInfo local_p2p_info(local_ip_addr, local_port, dev_idx,
                               local_gpu_idx_);
    std::string p2p_key = uccl_p2p_key(global_rank_, rank);
    std::string peer_p2p_key = uccl_p2p_key(rank, global_rank_);

    UCCLP2PInfo remote_p2p_info;
    if (!exchanger_client_->wait_and_fetch(
            peer_p2p_key, remote_p2p_info,
            timeout_to_retries(bootstrap_timeout_ms(), kBootstrapPollDelayMs),
            kBootstrapPollDelayMs)) {
      return false;
    }
    if (!exchanger_client_->publish(p2p_key, local_p2p_info)) return false;

    std::cout << "[INFO] Rank " << global_rank_ << " P2P port " << local_port
              << " (GPU " << local_gpu_idx_ << ", dev " << dev_idx << ")"
              << " <- Rank " << rank << " P2P port " << remote_p2p_info.port
              << " (GPU " << remote_p2p_info.gpu_idx << ", dev "
              << remote_p2p_info.dev_idx << ")" << std::endl;

    bool ret = uccl_adapter.accept_from_peer(rank, remote_p2p_info.ip,
                                             remote_p2p_info.dev_idx,
                                             remote_p2p_info.gpu_idx);
    if (ret) {
      cache_peer_session(rank, PeerTransportKind::Uccl, false, true);
      register_existing_local_mrs_with_uccl();
      std::cout << "[INFO] Communicator " << global_rank_
                << " UCCL accept_from succeeded from rank " << rank
                << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " UCCL accept_from failed from rank " << rank << std::endl;
      if (try_fallback_tcp_accept(rank, local_meta, remote_meta)) {
        return true;
      }
    }
    return ret;
  }

  if (peer_kind == PeerTransportKind::Ipc) {
    bool ret = ipc_channel_->accept_from(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << global_rank_
                << " IPC accept_from succeeded from rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC accept_from failed from rank " << rank << std::endl;
      shm_control_->close_peer(rank);
    }
    if (!ret) return false;
    cache_peer_session(rank, PeerTransportKind::Ipc, false, true);
    return true;
  }

  if (peer_kind == PeerTransportKind::Tcp) {
    CommunicatorMeta const local_meta = local_peer->meta();
    CommunicatorMeta const remote_meta = remote_peer->meta();
    auto& tcp_adapter = ensure_tcp_adapter(local_meta);
    if (tcp_adapter.has_recv_peer(rank)) {
      cache_peer_session(rank, PeerTransportKind::Tcp, false, true);
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

    bool ret = tcp_adapter.accept_from_peer(rank, remote_meta.ip);
    if (ret) {
      cache_peer_session(rank, PeerTransportKind::Tcp, false, true);
      std::cout << "[INFO] Communicator " << global_rank_
                << " TCP accept_from succeeded from rank " << rank
                << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " TCP accept_from failed from rank " << rank << std::endl;
    }
    return ret;
  }
  return false;
}

std::shared_ptr<IpcChannel> Communicator::get_ipc_channel_by_rank(int rank) {
  auto* peer = peer_manager_->get(rank);
  if (!peer || peer->transport_kind() != PeerTransportKind::Ipc) return nullptr;
  return ipc_channel_;
}

void Communicator::cache_peer_session(int rank, PeerTransportKind kind,
                                      bool mark_send_ready,
                                      bool mark_recv_ready) {
  auto* peer = peer_manager_->get(rank);
  if (!peer) return;
  peer->set_transport_kind(kind);
  peer->set_send_ready(mark_send_ready || peer->send_ready());
  peer->set_recv_ready(mark_recv_ready || peer->recv_ready());
}

void Communicator::shutdown_ipc_channel() {
  if (ipc_channel_) {
    ipc_channel_->shutdown();
  }
}

bool Communicator::has_peer_send_path(int rank) const {
  return peer_manager_->has_peer_send_path(rank);
}

bool Communicator::has_peer_recv_path(int rank) const {
  return peer_manager_->has_peer_recv_path(rank);
}

PeerTransportKind Communicator::get_peer_transport_kind(int rank) const {
  auto* peer = peer_manager_->get(rank);
  if (!peer) {
    throw std::runtime_error("transport peer session is not established");
  }
  return peer->transport_kind();
}

PeerTransportKind Communicator::peer_transport_kind(int rank) const {
  return get_peer_transport_kind(rank);
}

void Communicator::register_existing_local_mrs_with_uccl() {
  if (!uccl_adapter_ || !uccl_adapter_->is_initialized()) return;
  for (auto* p : memory_registry_.local_buffers()) {
    MR mr = memory_registry_.get_local_mr(p);
    (void)uccl_adapter_->register_memory(mr.id, p, mr.length);
  }
}

bool Communicator::ensure_uccl_memory_registered(uint64_t mr_id, void* ptr,
                                                 size_t len) {
  if (!uccl_adapter_ || !uccl_adapter_->is_initialized()) return false;
  return uccl_adapter_->is_memory_registered(mr_id) ||
         uccl_adapter_->register_memory(mr_id, ptr, len);
}

bool Communicator::complete_host_bounce_recv(TrackedRequest& tracked) {
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
    GPU_RT_CHECK(gpuMemcpyAsync(
        static_cast<char*>(tracked.completion_buffer) +
            tracked.completion_offset,
        tracked.bounce.ptr, tracked.completion_bytes, gpuMemcpyHostToDevice,
        host_copy_stream_));
    GPU_RT_CHECK(gpuEventRecord(tracked.host_copy_event, host_copy_stream_));
    tracked.host_copy_submitted = true;
    return false;
  }

  gpuError_t query = gpuEventQuery(tracked.host_copy_event);
  if (query == gpuErrorNotReady) return false;
  if (query != gpuSuccess) {
    GPU_RT_CHECK(query);
  }
  GPU_RT_CHECK(gpuEventDestroy(tracked.host_copy_event));
  tracked.host_copy_event = nullptr;
  tracked.host_copy_submitted = false;
  tracked.needs_host_to_device_copy = false;
  return true;
}

void Communicator::cleanup_tracked_request(unsigned id, TrackedRequest& tracked) {
  if (tracked.kind == PeerTransportKind::Tcp && tcp_adapter_) {
    tcp_adapter_->release_request(id);
  }
  if (tracked.host_copy_event != nullptr) {
    GPU_RT_CHECK(gpuEventDestroy(tracked.host_copy_event));
    tracked.host_copy_event = nullptr;
  }
  tracked.host_copy_submitted = false;
  tracked.needs_host_to_device_copy = false;
  if (host_bounce_pool_) {
    host_bounce_pool_->release(tracked.bounce);
  }
}

unsigned Communicator::isend(int rank, void* ptr, size_t offset, size_t len,
                             uint32_t local_mr_id, uint32_t remote_mr_id,
                             bool on_gpu) {
  if (!has_peer_send_path(rank)) {
    throw std::runtime_error("transport send path is not established");
  }
  auto peer_kind = get_peer_transport_kind(rank);
  unsigned rid = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  void* actual_ptr = static_cast<char*>(ptr) + offset;

  if (peer_kind == PeerTransportKind::Uccl) {
    bool can_use_direct = !on_gpu || ensure_uccl_memory_registered(
                                         local_mr_id, actual_ptr, len);
    TrackedRequest tracked{
        rank, PeerTransportKind::Uccl, nullptr, false, false, false};
    void* send_ptr = actual_ptr;
    uint64_t send_mr_id = local_mr_id;
    if (!can_use_direct) {
      tracked.bounce = host_bounce_pool_->acquire(len, true);
      GPU_RT_CHECK(gpuMemcpy(tracked.bounce.ptr, actual_ptr, len,
                             gpuMemcpyDeviceToHost));
      send_ptr = tracked.bounce.ptr;
      send_mr_id = tracked.bounce.mr_id;
    }

    int ret = uccl_adapter_->send_async(rank, send_ptr, len, send_mr_id,
                                        remote_mr_id, rid);
    if (ret != 0) {
      cleanup_tracked_request(rid, tracked);
      return 0;
    }
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      requests_map_[rid] = std::move(tracked);
    }
    notifier_cv_.notify_all();
    return rid;
  }

  if (peer_kind == PeerTransportKind::Tcp) {
    TrackedRequest tracked{
        rank, PeerTransportKind::Tcp, nullptr, false, false, false};
    void* send_ptr = actual_ptr;
    if (on_gpu) {
      tracked.bounce = host_bounce_pool_->acquire(len, false);
      GPU_RT_CHECK(gpuMemcpy(tracked.bounce.ptr, actual_ptr, len,
                             gpuMemcpyDeviceToHost));
      send_ptr = tracked.bounce.ptr;
    }

    int ret = tcp_adapter_->send_async(rank, send_ptr, len, rid);
    if (ret != 0) {
      cleanup_tracked_request(rid, tracked);
      return 0;
    }
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      requests_map_[rid] = std::move(tracked);
    }
    notifier_cv_.notify_all();
    return rid;
  }

  auto ipc_channel = get_ipc_channel_by_rank(rank);
  if (!ipc_channel) return 0;

  uint64_t match_seq = next_ipc_match_seq(rank, RequestType::Send);
  auto req =
      std::make_shared<Request>(rid, match_seq, ptr, offset, len, local_mr_id,
                                remote_mr_id, RequestType::Send);

  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_[rid] =
        TrackedRequest{rank, PeerTransportKind::Ipc, req, false, false, false};
  }

  if (!ipc_channel->send_async(rank, req)) {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_.erase(rid);
    return 0;
  }
  notifier_cv_.notify_all();

  return rid;
}

unsigned Communicator::irecv(int rank, void* ptr, size_t offset, size_t len,
                             bool on_gpu) {
  if (!has_peer_recv_path(rank)) {
    throw std::runtime_error("transport recv path is not established");
  }
  auto peer_kind = get_peer_transport_kind(rank);
  unsigned rid = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  void* actual_ptr = static_cast<char*>(ptr) + offset;

  if (peer_kind == PeerTransportKind::Uccl) {
    TrackedRequest tracked{
        rank, PeerTransportKind::Uccl, nullptr, false, false, false};
    void* recv_ptr = actual_ptr;
    uint64_t recv_mr_id = 0;

    if (!on_gpu) {
      auto local_mr = get_local_mr(actual_ptr);
      if (!ensure_uccl_memory_registered(local_mr.id, actual_ptr, len)) {
        return 0;
      }
      recv_mr_id = local_mr.id;
    } else {
      auto local_mr = get_local_mr(actual_ptr);
      if (ensure_uccl_memory_registered(local_mr.id, actual_ptr, len)) {
        recv_mr_id = local_mr.id;
      } else {
        tracked.bounce = host_bounce_pool_->acquire(len, true);
        tracked.needs_host_to_device_copy = true;
        tracked.completion_buffer = ptr;
        tracked.completion_offset = offset;
        tracked.completion_bytes = len;
        recv_ptr = tracked.bounce.ptr;
        recv_mr_id = tracked.bounce.mr_id;
      }
    }

    int ret = uccl_adapter_->recv_async(rank, recv_ptr, len, recv_mr_id, rid);
    if (ret != 0) {
      cleanup_tracked_request(rid, tracked);
      return 0;
    }
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      requests_map_[rid] = std::move(tracked);
    }
    notifier_cv_.notify_all();
    return rid;
  }

  if (peer_kind == PeerTransportKind::Tcp) {
    TrackedRequest tracked{
        rank, PeerTransportKind::Tcp, nullptr, false, false, false};
    void* recv_ptr = actual_ptr;
    if (on_gpu) {
      tracked.bounce = host_bounce_pool_->acquire(len, false);
      tracked.needs_host_to_device_copy = true;
      tracked.completion_buffer = ptr;
      tracked.completion_offset = offset;
      tracked.completion_bytes = len;
      recv_ptr = tracked.bounce.ptr;
    }

    int ret = tcp_adapter_->recv_async(rank, recv_ptr, len, rid);
    if (ret != 0) {
      cleanup_tracked_request(rid, tracked);
      return 0;
    }
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      requests_map_[rid] = std::move(tracked);
    }
    notifier_cv_.notify_all();
    return rid;
  }

  auto ipc_channel = get_ipc_channel_by_rank(rank);
  if (!ipc_channel) return 0;

  uint64_t match_seq = next_ipc_match_seq(rank, RequestType::Recv);
  auto req = std::make_shared<Request>(rid, match_seq, ptr, offset, len, -1, -1,
                                       RequestType::Recv);

  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_[rid] =
        TrackedRequest{rank, PeerTransportKind::Ipc, req, false, false, false};
  }

  if (!ipc_channel->recv_async(rank, req)) {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_.erase(rid);
    return 0;
  }
  notifier_cv_.notify_all();

  return rid;
}

uint64_t Communicator::next_ipc_match_seq(int rank, RequestType type) {
  if (rank < 0 || rank >= world_size_) {
    throw std::out_of_range("invalid rank for IPC match sequence");
  }

  std::lock_guard<std::mutex> lk(ipc_match_seq_mu_);
  auto& next_seq = type == RequestType::Send ? next_send_match_seq_[rank]
                                             : next_recv_match_seq_[rank];
  return next_seq++;
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
  if (snapshot.kind == PeerTransportKind::Uccl) {
    done = blocking ? uccl_adapter_->wait_completion(id)
                    : uccl_adapter_->poll_completion(id);
    if (blocking && !done) failed = true;
  } else if (snapshot.kind == PeerTransportKind::Tcp) {
    done = blocking ? tcp_adapter_->wait_completion(id)
                    : tcp_adapter_->poll_completion(id);
    failed = done && tcp_adapter_->request_failed(id);
  } else if (snapshot.ipc_request) {
    while (true) {
      done = snapshot.ipc_request->is_finished(std::memory_order_acquire);
      if (done || !blocking) break;
      std::this_thread::yield();
    }
    failed =
        done && snapshot.ipc_request->has_failed(std::memory_order_acquire);
  }

  if (!done) return false;

  std::lock_guard<std::mutex> lk(req_mu_);
  auto it = requests_map_.find(id);
  if (it == requests_map_.end()) return true;
  if (!failed) {
    bool copy_done = complete_host_bounce_recv(it->second);
    if (!copy_done) return false;
  }
  it->second.completed = true;
  it->second.failed = failed;
  return true;
}

bool Communicator::wait_finish(std::vector<unsigned> const& reqs) {
  std::unordered_set<unsigned> remaining;
  if (reqs.empty()) {
    std::lock_guard<std::mutex> lk(req_mu_);
    for (auto const& [id, _] : requests_map_) {
      remaining.insert(id);
    }
  } else {
    remaining.insert(reqs.begin(), reqs.end());
  }

  while (true) {
    std::vector<unsigned> finished;
    for (auto id : remaining) {
      if (id == 0) return false;
      if (!poll_request_completion(id, true)) {
        return false;
      }
      std::lock_guard<std::mutex> lk(req_mu_);
      auto it = requests_map_.find(id);
      if (it != requests_map_.end() && it->second.failed) {
        cleanup_tracked_request(id, it->second);
        requests_map_.erase(it);
        return false;
      }
      if (it != requests_map_.end()) {
        cleanup_tracked_request(id, it->second);
        requests_map_.erase(it);
      }
      finished.push_back(id);
    }
    for (auto id : finished) remaining.erase(id);
    if (remaining.empty()) return true;
  }
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
  std::lock_guard<std::mutex> lk(req_mu_);
  auto it = requests_map_.find(req);
  if (it == requests_map_.end()) return;
  if (!it->second.completed) {
    throw std::runtime_error("cannot release an in-flight transport request");
  }
  cleanup_tracked_request(req, it->second);
  requests_map_.erase(it);
}

bool Communicator::wait_finish(unsigned const req) {
  return wait_finish(std::vector<unsigned>{req});
}

bool Communicator::check_ready() const {
  for (int i = 0; i < world_size_; i++) {
    auto* peer = peer_manager_->get(i);
    if (!peer || !peer->has_meta()) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " check_ready: missing CommunicatorMeta for rank " << i
                << std::endl;
      return false;
    }
    if (!peer->meta().is_ready) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " check_ready: CommunicatorMeta for rank " << i
                << " is not ready" << std::endl;
      return false;
    }
  }

  std::cerr << "[INFO] Communicator " << global_rank_ << " is ready"
            << std::endl;
  return true;
}

MR Communicator::reg_mr(void* local_buf, size_t len) {
  MR info = memory_registry_.track_local_buffer(local_buf, len);

  if (uccl_adapter_ && uccl_adapter_->is_initialized()) {
    (void)uccl_adapter_->register_memory(info.id, local_buf, len);
  }

  return info;
}

bool Communicator::dereg_mr(void* local_buf) {
  auto released = memory_registry_.release_local_buffer(local_buf);
  if (uccl_adapter_ && uccl_adapter_->is_initialized() &&
      released.has_local_mr_id) {
    uccl_adapter_->deregister_memory(released.local_mr_id);
  }
  return true;
}

bool Communicator::notify_mr(int remote_rank, MR& mr) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  std::string key =
      "mr:" + std::to_string(global_rank_) + "->" + std::to_string(remote_rank);

  MRInfos wrapper;
  exchanger_client_->fetch(key, wrapper);
  bool replaced = false;
  for (auto& existing : wrapper.mrs) {
    if (existing.id == mr.id) {
      existing = mr;
      replaced = true;
      break;
    }
  }
  if (!replaced) {
    wrapper.mrs.push_back(mr);
  }

  std::cout << "[notify MR to rank " << remote_rank << "] addr=" << mr.address
            << " length=" << mr.length << " key=" << mr.key << std::endl;

  return exchanger_client_->publish(key, wrapper);
}

bool Communicator::wait_mr_notify(int remote_rank, MR& mr) {
  if (!exchanger_client_ || !exchanger_client_->valid()) {
    throw std::runtime_error("Exchanger client is not valid");
  }

  std::string key =
      "mr:" + std::to_string(remote_rank) + "->" + std::to_string(global_rank_);

  if (memory_registry_.take_pending_remote_mr(remote_rank, mr)) return true;

  int timeout_ms = mr_timeout_ms();
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
  while (true) {
    MRInfos wrapper;
    bool fetched = exchanger_client_->fetch(key, wrapper);
    if (fetched && !wrapper.mrs.empty()) {
      memory_registry_.cache_remote_mrs(remote_rank, wrapper.mrs);
      if (memory_registry_.take_pending_remote_mr(remote_rank, mr)) {
        std::cout << "[recv MR from rank " << remote_rank
                  << "] addr=" << mr.address << " length=" << mr.length
                  << " key=" << mr.key << std::endl;
        return true;
      }
    }

    if (timeout_ms >= 0 && std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "[WARN] Timeout waiting for MR from rank " << remote_rank
                << std::endl;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

MR Communicator::get_local_mr(void* local_buf) {
  return memory_registry_.get_local_mr(local_buf);
}

MR Communicator::get_local_mr(uint32_t mr_id) {
  return memory_registry_.get_local_mr(mr_id);
}

MR Communicator::get_remote_mr(int remote_rank, uint32_t mr_id) {
  return memory_registry_.get_remote_mr(remote_rank, mr_id);
}

bool Communicator::register_remote_ipc_cache(
    int remote_rank, gpuIpcMemHandle_t handle,
    IpcCacheManager::IpcCache const& cache) {
  return memory_registry_.register_remote_ipc_cache(remote_rank, handle, cache);
}

IpcCacheManager::IpcCache Communicator::get_remote_ipc_cache(
    int remote_rank, gpuIpcMemHandle_t handle) {
  return memory_registry_.get_remote_ipc_cache(remote_rank, handle);
}

std::shared_ptr<void> Communicator::register_completion_notifier(
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> cb) {
  auto target = std::make_shared<NotifyTarget>();
  target->emit = std::move(cb);

  {
    std::lock_guard<std::mutex> lk(notifier_mu_);
    notify_targets_.push_back(target);
  }

  bool expected = false;
  if (notifier_started_.compare_exchange_strong(expected, true)) {
    notifier_running_.store(true);
    notifier_thread_ =
        std::thread(&Communicator::completion_notifier_loop, this);
  }

  notifier_cv_.notify_all();

  return std::static_pointer_cast<void>(target);
}

void Communicator::completion_notifier_loop() {
  while (notifier_running_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lk(notifier_mu_);
      notifier_cv_.wait(lk, [&] {
        if (!notifier_running_.load()) return true;

        std::lock_guard<std::mutex> rlk(req_mu_);
        return !requests_map_.empty();
      });
    }

    if (!notifier_running_.load()) break;

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
        std::lock_guard<std::mutex> nlk(notifier_mu_);
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
