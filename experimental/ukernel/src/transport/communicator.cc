#include "communicator.h"
#include "adapter/ipc_adapter.h"
#include "adapter/tcp_adapter.h"
#include "adapter/transport_adapter.h"
#include "adapter/uccl_adapter.h"
#include "util/utils.h"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr int kDefaultBootstrapTimeoutMs = 30000;

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

void maybe_configure_uccl_socket_ifname(std::string const& local_hint_ip) {
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

int get_timeout_ms(char const* env_name, int default_ms) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') return default_ms;
  try {
    return std::stoi(value);
  } catch (...) {
    return default_ms;
  }
}

int bootstrap_timeout_ms() {
  return get_timeout_ms("UHM_BOOTSTRAP_TIMEOUT_MS", kDefaultBootstrapTimeoutMs);
}

std::string uccl_p2p_key(int src_rank, int dst_rank) {
  return "uccl_p2p_info_" + std::to_string(src_rank) + "_to_" +
         std::to_string(dst_rank);
}

std::string tcp_p2p_key(int src_rank, int dst_rank) {
  return "tcp_p2p_info_" + std::to_string(src_rank) + "_to_" +
         std::to_string(dst_rank);
}

std::string ipc_global_buffer_key(int owner_rank, uint32_t buffer_id) {
  return "ipc:rank:" + std::to_string(owner_rank) +
         ":buf:" + std::to_string(buffer_id);
}

std::string mr_global_buffer_key(int owner_rank, uint32_t buffer_id) {
  return "mr:rank:" + std::to_string(owner_rank) +
         ":buf:" + std::to_string(buffer_id);
}

std::string oob_scoped_key(std::string const& ns, std::string const& key) {
  if (ns.empty()) return key;
  return ns + "/" + key;
}

template <typename Info>
bool oob_put(Exchanger& ex, std::string const& ns, std::string const& key,
             Info const& value) {
  return ex.put(oob_scoped_key(ns, key), value);
}

template <typename Info>
bool oob_get(Exchanger& ex, std::string const& ns, std::string const& key,
             Info& out, int timeout_ms = 0) {
  std::string const full_key = oob_scoped_key(ns, key);
  if (timeout_ms == 0) return ex.get(full_key, out);
  constexpr int kPollDelayMs = 10;
  int const max_retries =
      timeout_ms < 0
          ? -1
          : std::max(1, (timeout_ms + kPollDelayMs - 1) / kPollDelayMs);
  return ex.wait(full_key, out,
                 Exchanger::WaitOptions(max_retries, kPollDelayMs));
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
  if (!config_) {
    config_ =
        std::make_shared<CommunicatorConfig>(CommunicatorConfig::from_env());
  }
  if (config_->oob_namespace.empty()) {
    config_->oob_namespace = "default";
  }
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
  ipc_adapter_ = std::make_shared<IpcAdapter>(
      this, generate_host_id() + "_p" + std::to_string(config_->exchanger_port),
      config_->local_id >= 0 ? config_->local_id : global_rank_,
      local_gpu_idx_);
  GPU_RT_CHECK(
      gpuStreamCreateWithFlags(&host_copy_stream_, gpuStreamNonBlocking));

  bool is_server = (global_rank_ == 0);
  if (!is_server && config_->exchanger_ip == "0.0.0.0")
    config_->exchanger_ip = "127.0.0.1";
  std::cout << "[INFO] Using hierarchical bootstrap exchanger as "
            << (is_server ? "server" : "client") << " " << config_->exchanger_ip
            << std::endl;
  exchanger_client_ = std::make_shared<HierarchicalExchanger>(
      (global_rank_ == 0), config_->exchanger_ip, config_->exchanger_port,
      /*timeout_ms=*/3000, /*max_line_bytes=*/1 * 1024 * 1024,
      /*local_id=*/config_->local_id);
  if (!exchanger_client_->valid()) {
    fprintf(stderr, "[ERROR] Failed to connect to Exchanger\n");
    return;
  }

  tracker_ = std::make_unique<RequestTracker>(
      uccl_adapter_.get(), tcp_adapter_.get(), ipc_adapter_.get(),
      [this](TrackedRequest& t, bool blocking) {
        return complete_host_bounce_recv(t, blocking);
      },
      [this](TrackedRequest& t) { cleanup_tracked_request(t); });

  tracker_->start_progress();

  exchange_peer_metas();
  std::cout << "[INFO] Communicator " << global_rank_
            << " initialized: peer meta exchange success" << std::endl;
}

void Communicator::set_oob_namespace(std::string ns) {
  if (ns.empty()) ns = "default";
  std::lock_guard<std::mutex> lk(config_mu_);
  config_->oob_namespace = std::move(ns);
}

std::string Communicator::oob_namespace() const {
  std::lock_guard<std::mutex> lk(config_mu_);
  if (config_->oob_namespace.empty()) return "default";
  return config_->oob_namespace;
}

bool Communicator::barrier(std::string const& barrier_namespace,
                           int timeout_ms) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;
  std::string ns = barrier_namespace.empty() ? "default" : barrier_namespace;
  uint64_t const seq = barrier_seq_.fetch_add(1, std::memory_order_relaxed);
  std::string const barrier_prefix =
      oob_namespace() + "/barrier/" + ns + "/seq/" + std::to_string(seq);
  std::string const arrive_key =
      barrier_prefix + "/rank/" + std::to_string(global_rank_);
  if (!exchanger_client_->put(arrive_key, int32_t{1})) return false;

  constexpr int kPollDelayMs = 10;
  int const max_retries =
      timeout_ms < 0
          ? -1
          : std::max(1, (timeout_ms + kPollDelayMs - 1) / kPollDelayMs);
  Exchanger::WaitOptions const wait_opt(max_retries, kPollDelayMs);
  int32_t arrived = 0;
  for (int rank = 0; rank < world_size_; ++rank) {
    std::string const key = barrier_prefix + "/rank/" + std::to_string(rank);
    if (!exchanger_client_->wait(key, arrived, wait_opt)) return false;
  }
  return true;
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
    self.put_ready = true;
    self.wait_ready = true;
    self.gpu_idx = local_gpu_idx_;
  }

  std::string meta_key = "meta:" + std::to_string(global_rank_);
  if (!oob_put(*exchanger_client_, oob_namespace(), meta_key, local)) {
    throw std::runtime_error(
        "failed to publish local communicator meta to exchanger");
  }

  CommunicatorMeta remote;
  std::vector<int> missing_ranks;
  for (int i = 0; i < world_size_; i++) {
    if (i == global_rank_) continue;
    std::string key = "meta:" + std::to_string(i);
    if (oob_get(*exchanger_client_, oob_namespace(), key, remote,
                bootstrap_timeout_ms())) {
      std::lock_guard<std::mutex> lk(peer_mu_);
      auto& peer = peer_states_.at(static_cast<size_t>(i));
      peer.meta = remote;
      peer.has_meta = true;
      ipc_adapter_->set_peer_local_id(i, remote.local_id);
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
  tracker_.reset();

  if (ipc_adapter_) {
    ipc_adapter_->shutdown();
  }

  if (host_copy_stream_ != nullptr) {
    int orig_device = -1;
    GPU_RT_CHECK(gpuGetDevice(&orig_device));
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
    GPU_RT_CHECK(gpuStreamDestroy(host_copy_stream_));
    GPU_RT_CHECK(gpuSetDevice(orig_device));
    host_copy_stream_ = nullptr;
  }

  for (auto const& [buffer_id, item] : mr_manager_.list_local_mrs()) {
    uint64_t const registered_id = buffer_id;
    if (uccl_adapter_ && uccl_adapter_->is_initialized()) {
      std::lock_guard<std::mutex> lk(uccl_reg_mu_);
      if (uccl_registered_mrs_.find(registered_id) !=
          uccl_registered_mrs_.end()) {
        uccl_adapter_->deregister_memory(registered_id);
        uccl_registered_mrs_.erase(registered_id);
      }
      uccl_direct_reg_failed_mrs_.erase(registered_id);
    }
    (void)mr_manager_.delete_mr(static_cast<uint32_t>(buffer_id));
  }

  std::vector<uint32_t> local_ipc_buffer_ids;
  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    local_ipc_buffer_ids.reserve(local_buffer_to_ipc_.size());
    for (auto const& kv : local_buffer_to_ipc_) {
      local_ipc_buffer_ids.push_back(kv.first);
    }
  }
  for (uint32_t buffer_id : local_ipc_buffer_ids) {
    (void)dereg_ipc(buffer_id);
  }

  for (int i = 0; i < world_size_; ++i) {
    if (i == global_rank_) continue;
    ipc_manager_.delete_ipc(i);
  }

  uccl_adapter_.reset();
  tcp_adapter_.reset();
  ipc_adapter_.reset();

  for (gpuEvent_t e : event_pool_) {
    if (e != nullptr) GPU_RT_CHECK(gpuEventDestroy(e));
  }
  event_pool_.clear();

  std::cout << "[INFO] Communicator " << global_rank_ << " resources released"
            << std::endl;
}

UcclTransportAdapter& Communicator::ensure_uccl_adapter(
    CommunicatorMeta const& local_meta) {
  if (!uccl_adapter_) {
    maybe_configure_uccl_socket_ifname(local_meta.ip);
    UcclTransportConfig uccl_cfg;
    uccl_adapter_ = std::make_unique<UcclTransportAdapter>(
        local_gpu_idx_, world_size_, std::move(uccl_cfg));
    tracker_->set_uccl(uccl_adapter_.get());
  }
  return *uccl_adapter_;
}

bool Communicator::exchange_uccl_peer_info(int rank,
                                           UcclTransportAdapter& uccl_adapter,
                                           UCCLP2PInfo* out_remote_p2p_info) {
  if (out_remote_p2p_info == nullptr) return false;

  int dev_idx = uccl_adapter.get_best_dev_idx(local_gpu_idx_);
  if (dev_idx < 0) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " UCCL get_best_dev_idx failed for local gpu "
              << local_gpu_idx_ << std::endl;
    return false;
  }
  uint16_t local_port = uccl_adapter.get_p2p_listen_port(dev_idx);
  if (local_port == 0) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " UCCL local listen port is invalid for dev " << dev_idx
              << std::endl;
    return false;
  }
  std::string local_ip_addr = uccl_adapter.get_p2p_listen_ip(dev_idx);
  if (local_ip_addr.empty()) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " UCCL local listen ip is empty for dev " << dev_idx
              << std::endl;
    return false;
  }

  UCCLP2PInfo local_p2p_info(local_ip_addr, local_port, dev_idx,
                             local_gpu_idx_);
  std::string p2p_key = uccl_p2p_key(global_rank_, rank);
  std::string peer_p2p_key = uccl_p2p_key(rank, global_rank_);

  bool ok = oob_put(*exchanger_client_, oob_namespace(), p2p_key,
                    local_p2p_info) &&
            oob_get(*exchanger_client_, oob_namespace(), peer_p2p_key,
                    *out_remote_p2p_info, bootstrap_timeout_ms());
  if (ok && out_remote_p2p_info->gpu_idx >= 0) {
    std::lock_guard<std::mutex> lk(peer_mu_);
    peer_states_[static_cast<size_t>(rank)].gpu_idx =
        out_remote_p2p_info->gpu_idx;
  }
  return ok;
}

TcpTransportAdapter& Communicator::ensure_tcp_adapter(
    CommunicatorMeta const& local_meta) {
  if (!tcp_adapter_) {
    tcp_adapter_ =
        std::make_unique<TcpTransportAdapter>(local_meta.ip, global_rank_);
    tracker_->set_tcp(tcp_adapter_.get());
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

bool Communicator::try_fallback_tcp_accept(int rank,
                                            CommunicatorMeta const& local_meta) {
  if (config_->preferred_transport != PreferredTransport::Auto) return false;
  auto& tcp = ensure_tcp_adapter(local_meta);

  if (!tcp.has_put_path(rank)) {
    TcpP2PInfo local_p2p(tcp.get_listen_ip(), tcp.get_listen_port());
    std::string key = tcp_p2p_key(global_rank_, rank);
    std::string peer_key = tcp_p2p_key(rank, global_rank_);
    TcpP2PInfo remote;
    if (!oob_put(*exchanger_client_, oob_namespace(), key, local_p2p) ||
        !oob_get(*exchanger_client_, oob_namespace(), peer_key, remote,
                 bootstrap_timeout_ms())) {
      return false;
    }
    PeerConnectSpec spec{};
    spec.peer_rank = rank;
    spec.type = PeerConnectType::Connect;
    spec.detail = TcpPeerConnectSpec{remote.ip, remote.port};
    if (!tcp.ensure_put_path(spec)) return false;
    mark_put_path_ready(rank, PeerTransportKind::Tcp);
  }
  if (!tcp.has_wait_path(rank)) {
    TcpP2PInfo local_p2p(tcp.get_listen_ip(), tcp.get_listen_port());
    std::string key = tcp_p2p_key(global_rank_, rank);
    std::string peer_key = tcp_p2p_key(rank, global_rank_);
    TcpP2PInfo remote;
    if (!oob_put(*exchanger_client_, oob_namespace(), key, local_p2p) ||
        !oob_get(*exchanger_client_, oob_namespace(), peer_key, remote,
                 bootstrap_timeout_ms())) {
      return false;
    }
    PeerConnectSpec spec{};
    spec.peer_rank = rank;
    spec.type = PeerConnectType::Accept;
    spec.detail = TcpPeerConnectSpec{remote.ip, 0};
    if (!tcp.ensure_wait_path(spec)) return false;
    mark_wait_path_ready(rank, PeerTransportKind::Tcp);
  }
  std::cout << "[INFO] Communicator " << global_rank_
            << " TCP fallback succeeded to rank " << rank << std::endl;
  return true;
}

bool Communicator::ensure_path(int rank, bool is_put) {
  if (rank == global_rank_) return true;
  if (rank < 0 || rank >= world_size_) return false;

  if (is_put ? has_put_path(rank) : has_wait_path(rank)) return true;

  ResolvedPeer resolved;
  try {
    resolved = resolve_peer(rank);
  } catch (std::exception const& ex) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " failed to resolve transport for rank " << rank << ": "
              << ex.what() << std::endl;
    return false;
  }

  auto conn_type =
      is_put ? PeerConnectType::Connect : PeerConnectType::Accept;
  char const* dir_label = is_put ? "put" : "wait";

  auto fallback = [&] {
    return try_fallback_tcp_accept(rank, resolved.local_meta);
  };

  if (resolved.kind == PeerTransportKind::Uccl) {
    auto& uccl = ensure_uccl_adapter(resolved.local_meta);
    bool ready = is_put ? uccl.has_put_path(rank) : uccl.has_wait_path(rank);
    if (!ready) {
      int dev = uccl.get_best_dev_idx(local_gpu_idx_);
      if (dev < 0) return fallback();
      UCCLP2PInfo remote;
      if (!exchange_uccl_peer_info(rank, uccl, &remote)) return fallback();
      PeerConnectSpec spec{};
      spec.peer_rank = rank;
      spec.type = conn_type;
      spec.detail = UcclPeerConnectSpec{remote.ip, remote.port, dev,
                                        local_gpu_idx_, remote.dev_idx,
                                        remote.gpu_idx};
      if (!(is_put ? uccl.ensure_put_path(spec)
                   : uccl.ensure_wait_path(spec))) {
        return fallback();
      }
    }
    is_put ? mark_put_path_ready(rank, PeerTransportKind::Uccl)
           : mark_wait_path_ready(rank, PeerTransportKind::Uccl);
    register_existing_local_mrs_with_uccl();
    return true;
  }

  if (resolved.kind == PeerTransportKind::Ipc) {
    PeerConnectSpec spec{};
    spec.peer_rank = rank;
    spec.type = conn_type;
    spec.detail = IpcPeerConnectSpec{};
    if (!(is_put ? ipc_adapter_->ensure_put_path(spec)
                 : ipc_adapter_->ensure_wait_path(spec))) {
      std::cerr << "[ERROR] Communicator " << global_rank_ << " IPC " << dir_label
                << " failed to rank " << rank << std::endl;
      if (is_put) ipc_adapter_->close_peer(rank);
      return false;
    }
    is_put ? mark_put_path_ready(rank, PeerTransportKind::Ipc)
           : mark_wait_path_ready(rank, PeerTransportKind::Ipc);
    return true;
  }

  if (resolved.kind == PeerTransportKind::Tcp) {
    auto& tcp = ensure_tcp_adapter(resolved.local_meta);
    bool ready = is_put ? tcp.has_put_path(rank) : tcp.has_wait_path(rank);
    if (!ready) {
      TcpP2PInfo local_p2p(tcp.get_listen_ip(), tcp.get_listen_port());
      std::string key = tcp_p2p_key(global_rank_, rank);
      std::string peer_key = tcp_p2p_key(rank, global_rank_);
      TcpP2PInfo remote;
      if (!oob_put(*exchanger_client_, oob_namespace(), key, local_p2p) ||
          !oob_get(*exchanger_client_, oob_namespace(), peer_key, remote,
                   bootstrap_timeout_ms())) {
        return false;
      }
      PeerConnectSpec spec{};
      spec.peer_rank = rank;
      spec.type = conn_type;
      spec.detail = TcpPeerConnectSpec{remote.ip,
                                        is_put ? remote.port : uint16_t{0}};
      if (!(is_put ? tcp.ensure_put_path(spec) : tcp.ensure_wait_path(spec))) {
        std::cerr << "[ERROR] Communicator " << global_rank_ << " TCP "
                  << dir_label << " failed to rank " << rank << std::endl;
        return false;
      }
    }
    is_put ? mark_put_path_ready(rank, PeerTransportKind::Tcp)
           : mark_wait_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }
  return false;
}

bool Communicator::connect(int rank) { return ensure_path(rank, true); }

bool Communicator::accept(int rank) { return ensure_path(rank, false); }

bool Communicator::has_put_path(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return false;
  return peer_states_.at(static_cast<size_t>(rank)).put_ready;
}

bool Communicator::has_wait_path(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return false;
  return peer_states_.at(static_cast<size_t>(rank)).wait_ready;
}

void Communicator::mark_put_path_ready(int rank, PeerTransportKind kind) {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto& peer = peer_states_.at(static_cast<size_t>(rank));
  peer.put_kind = kind;
  peer.put_ready = true;
}

void Communicator::mark_wait_path_ready(int rank, PeerTransportKind kind) {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto& peer = peer_states_.at(static_cast<size_t>(rank));
  peer.wait_kind = kind;
  peer.wait_ready = true;
}

PeerTransportKind Communicator::get_put_transport_kind(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (!peer.has_meta || !peer.put_ready || peer.put_kind == PeerTransportKind::Unknown) {
    throw std::runtime_error("transport put path is not established");
  }
  return peer.put_kind;
}

PeerTransportKind Communicator::get_wait_transport_kind(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (!peer.has_meta || !peer.wait_ready || peer.wait_kind == PeerTransportKind::Unknown) {
    throw std::runtime_error("transport wait path is not established");
  }
  return peer.wait_kind;
}

PeerTransportKind Communicator::peer_transport_kind(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (!peer.has_meta) {
    throw std::runtime_error("transport peer session is not established");
  }
  if (!peer.put_ready && !peer.wait_ready) {
    throw std::runtime_error("transport peer path is not established");
  }
  if (peer.put_ready) return peer.put_kind;
  return peer.wait_kind;
}

int Communicator::peer_gpu_idx(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return -1;
  return peer_states_[static_cast<size_t>(rank)].gpu_idx;
}

TransportAdapter* Communicator::get_adapter(PeerTransportKind kind) {
  switch (kind) {
    case PeerTransportKind::Uccl:
      return uccl_adapter_.get();
    case PeerTransportKind::Tcp:
      return tcp_adapter_.get();
    case PeerTransportKind::Ipc:
      return ipc_adapter_.get();
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
  for (auto const& [buffer_id, item] : mr_manager_.list_local_mrs()) {
    void* ptr = reinterpret_cast<void*>(item.mr.address);
    size_t len = static_cast<size_t>(item.mr.length);
    (void)ensure_uccl_memory_registered(buffer_id, ptr, len);
  }
}

bool Communicator::ensure_uccl_memory_registered(uint32_t buffer_id, void* ptr,
                                                 size_t len) {
  if (uccl_adapter_ && uccl_adapter_->is_initialized()) {
    if (uccl_adapter_->is_memory_registered(buffer_id)) return true;

    void* base_ptr = ptr;
    size_t mr_len = len;
    bool is_direct_local_mr = false;

    MRItem item = mr_manager_.get_mr(static_cast<uint32_t>(buffer_id));
    if (item.valid) {
      base_ptr =
          reinterpret_cast<void*>(static_cast<uintptr_t>(item.mr.address));
      mr_len = static_cast<size_t>(item.mr.length);
      is_direct_local_mr = true;
    }

    if (is_direct_local_mr) {
      std::lock_guard<std::mutex> lk(uccl_reg_mu_);
      if (uccl_direct_reg_failed_mrs_.find(buffer_id) !=
          uccl_direct_reg_failed_mrs_.end()) {
        return false;
      }
    }

    if (base_ptr == nullptr || mr_len == 0) {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " has invalid base pointer or length for UCCL registration, "
                << "buffer_id=" << buffer_id << std::endl;
      return false;
    }

    bool ok = uccl_adapter_->register_memory(buffer_id, base_ptr, mr_len);
    if (!ok) {
      if (is_direct_local_mr) {
        std::lock_guard<std::mutex> lk(uccl_reg_mu_);
        uccl_direct_reg_failed_mrs_.insert(buffer_id);
        std::cerr << "[WARN] Communicator " << global_rank_
                  << " failed to register local GPU MR " << buffer_id
                  << " with UCCL, base=" << base_ptr << " len=" << mr_len
                  << "; future requests will fallback to host bounce"
                  << std::endl;
      } else {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " failed to register host bounce MR " << buffer_id
                  << " with UCCL, base=" << base_ptr << " len=" << mr_len
                  << std::endl;
      }
    } else {
      std::lock_guard<std::mutex> lk(uccl_reg_mu_);
      uccl_registered_mrs_.insert(buffer_id);
    }
    return ok;
  }

  return true;
}

unsigned Communicator::isend(int rank, uint32_t src_buf_id, size_t src_off,
                             size_t src_bytes, uint32_t dst_buf_id,
                             size_t dst_off) {
  if (src_buf_id == 0 || src_bytes == 0) {
    throw std::invalid_argument("isend requires non-empty source");
  }
  auto peer_kind = get_put_transport_kind(rank);
  MR local_mr = get_mr(src_buf_id);
  if (src_off > static_cast<size_t>(local_mr.length) ||
      src_bytes > static_cast<size_t>(local_mr.length) - src_off) {
    throw std::invalid_argument("isend local slice out of range");
  }

  auto* adapter = get_adapter(peer_kind);
  if (!adapter) throw std::runtime_error("failed to get adapter for peer");

  void* local_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(local_mr.address) + src_off);

  unsigned rid = 0;
  TrackedRequest* slot = tracker_->allocate(&rid);
  if (slot == nullptr) return 0;

  if (peer_kind == PeerTransportKind::Tcp) {
    void* host_buf;
    GPU_RT_CHECK(gpuHostMalloc(&host_buf, src_bytes));
    GPU_RT_CHECK(gpuMemcpy(host_buf, local_ptr, src_bytes, gpuMemcpyDeviceToHost));
    slot->bounce_ptr = host_buf;
    unsigned result = adapter->put_async(rank, host_buf, 0, nullptr, 0, src_bytes);
    if (result == 0 || !tracker_->activate(rid, result, rank, peer_kind)) {
      cleanup_tracked_request(*slot);
      slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
      return 0;
    }
    return rid;
  }

  if (peer_kind == PeerTransportKind::Uccl) {
    if (!ensure_uccl_memory_registered(src_buf_id, local_ptr, src_bytes)) {
      std::cerr << "[ERROR] UCCL MR not registered for buffer " << src_buf_id
                << std::endl;
      slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
      return 0;
    }
    unsigned result = adapter->put_async(rank, local_ptr, src_buf_id,
                                         nullptr, 0, src_bytes);
    if (result == 0 || !tracker_->activate(rid, result, rank, peer_kind)) {
      cleanup_tracked_request(*slot);
      slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
      return 0;
    }
    return rid;
  }

  // IPC: direct GPU peer copy.
  (void)dst_off;
  if (dst_buf_id == 0) {
    std::cerr << "[ERROR] IPC isend requires non-zero dst_buf_id" << std::endl;
    slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
    return 0;
  }
  void* remote_ptr = nullptr;
  int remote_gpu = -1;
  if (!try_resolve_remote_ipc_pointer(rank, dst_buf_id, dst_off, src_bytes,
                                      &remote_ptr, &remote_gpu)) {
    std::cerr << "[ERROR] IPC failed to resolve remote pointer" << std::endl;
    slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
    return 0;
  }
  unsigned result = adapter->put_async(rank, local_ptr, src_buf_id,
                                       remote_ptr, dst_buf_id, src_bytes);
  if (result == 0 || !tracker_->activate(rid, result, rank, peer_kind)) {
    cleanup_tracked_request(*slot);
    slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
    return 0;
  }
  return rid;
}

unsigned Communicator::irecv(int rank, uint32_t dst_buf_id, size_t dst_off,
                             size_t dst_bytes) {
  if (dst_buf_id == 0 || dst_bytes == 0) {
    throw std::invalid_argument("irecv requires non-empty destination");
  }
  auto peer_kind = get_wait_transport_kind(rank);
  MR local_mr = get_mr(dst_buf_id);
  if (dst_off > static_cast<size_t>(local_mr.length) ||
      dst_bytes > static_cast<size_t>(local_mr.length) - dst_off) {
    throw std::invalid_argument("irecv local slice out of range");
  }

  auto* adapter = get_adapter(peer_kind);
  if (!adapter) throw std::runtime_error("failed to get adapter for peer");

  void* local_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(local_mr.address) + dst_off);

  unsigned rid = 0;
  TrackedRequest* slot = tracker_->allocate(&rid);
  if (slot == nullptr) return 0;

  if (peer_kind == PeerTransportKind::Tcp) {
    void* host_buf;
    GPU_RT_CHECK(gpuHostMalloc(&host_buf, dst_bytes));
    slot->bounce_ptr = host_buf;
    slot->needs_host_to_device_copy = true;
    slot->completion_buffer = local_ptr;
    slot->completion_bytes = dst_bytes;

    TransportAdapter::WaitTarget wt;
    wt.local_ptr = host_buf;
    wt.len = dst_bytes;
    wt.local_buffer_id = 0;
    unsigned result = adapter->wait_async(rank, 0, std::move(wt));
    if (result == 0 || !tracker_->activate(rid, result, rank, peer_kind)) {
      cleanup_tracked_request(*slot);
      slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
      return 0;
    }
    return rid;
  }

  if (peer_kind == PeerTransportKind::Uccl) {
    if (!ensure_uccl_memory_registered(dst_buf_id, local_ptr, dst_bytes)) {
      std::cerr << "[ERROR] UCCL MR not registered for buffer " << dst_buf_id
                << std::endl;
      slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
      return 0;
    }
    TransportAdapter::WaitTarget wt;
    wt.local_ptr = local_ptr;
    wt.len = dst_bytes;
    wt.local_buffer_id = dst_buf_id;
    unsigned result = adapter->wait_async(rank, 0, std::move(wt));
    if (result == 0 || !tracker_->activate(rid, result, rank, peer_kind)) {
      cleanup_tracked_request(*slot);
      slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
      return 0;
    }
    return rid;
  }

  // IPC: SignalWait only. Applications publish/wait IPC buffer metadata
  // explicitly with reg_ipc()/wait_ipc(), mirroring the MR API.
  {
    slot->completion_buffer = local_ptr;
    slot->completion_bytes = dst_bytes;
    uint64_t match_seq = ipc_adapter_->next_recv_match_seq(rank);
    unsigned result = ipc_adapter_->wait_async(rank, match_seq, std::nullopt);
    if (result == 0 || !tracker_->activate(rid, result, rank, peer_kind)) {
      cleanup_tracked_request(*slot);
      slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
      return 0;
    }
    return rid;
  }
}

bool Communicator::reg_mr(uint32_t buffer_id, void* local_buf, size_t len,
                          bool publish) {
  if (buffer_id == 0 || local_buf == nullptr || len == 0) return false;
  MR mr = mr_manager_.create_local_mr(buffer_id, local_buf, len).mr;
  if (mr.address == 0 || mr.length == 0) return false;

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    local_buffer_to_mr_[buffer_id] = mr;
  }

  if (!publish || !exchanger_client_ || !exchanger_client_->valid()) {
    return true;
  }

  NamedMRInfos payload{};
  payload.entries.push_back(NamedMR{buffer_id, mr});
  return oob_put(*exchanger_client_, oob_namespace(),
                 mr_global_buffer_key(global_rank_, buffer_id), payload);
}

bool Communicator::dereg_mr(uint32_t buffer_id) {
  MR local_mr{};
  bool found = false;
  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    auto it = local_buffer_to_mr_.find(buffer_id);
    if (it != local_buffer_to_mr_.end()) {
      local_mr = it->second;
      local_buffer_to_mr_.erase(it);
      found = true;
    }
  }
  uint32_t const registered_id = buffer_id;

  if (registered_id != 0 && uccl_adapter_ && uccl_adapter_->is_initialized()) {
    std::lock_guard<std::mutex> lk(uccl_reg_mu_);
    if (uccl_registered_mrs_.erase(registered_id) > 0) {
      uccl_adapter_->deregister_memory(registered_id);
    }
    uccl_direct_reg_failed_mrs_.erase(registered_id);
  }
  if (found) (void)mr_manager_.delete_mr(buffer_id);
  return true;
}

bool Communicator::wait_mr(int owner_rank, uint32_t buffer_id, int timeout_ms) {
  if (buffer_id == 0) return false;
  if (owner_rank == global_rank_) {
    std::lock_guard<std::mutex> lk(resource_mu_);
    return local_buffer_to_mr_.find(buffer_id) != local_buffer_to_mr_.end();
  }
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  NamedMRInfos payload{};
  if (!oob_get(*exchanger_client_, oob_namespace(),
               mr_global_buffer_key(owner_rank, buffer_id), payload,
               timeout_ms)) {
    return false;
  }
  if (payload.entries.empty()) return false;

  bool found = false;
  MR mr{};
  for (auto const& entry : payload.entries) {
    if (entry.buffer_id != buffer_id || entry.mr.address == 0 ||
        entry.mr.length == 0) {
      continue;
    }
    mr = entry.mr;
    found = true;
    break;
  }
  if (!found) return false;

  MRItem item{};
  item.buffer_id = buffer_id;
  item.mr = mr;
  item.is_local = false;
  item.rank = owner_rank;
  item.valid = true;
  mr_manager_.register_remote_mr(owner_rank, item);

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    remote_buffer_to_mr_[owner_rank][buffer_id] = mr;
  }
  return true;
}

MR Communicator::get_mr(uint32_t buffer_id) const {
  std::lock_guard<std::mutex> lk(resource_mu_);
  auto it = local_buffer_to_mr_.find(buffer_id);
  if (it == local_buffer_to_mr_.end()) {
    throw std::runtime_error("local MR not found for buffer_id");
  }
  return it->second;
}

MR Communicator::get_mr(int owner_rank, uint32_t buffer_id) const {
  if (owner_rank == global_rank_) return get_mr(buffer_id);
  std::lock_guard<std::mutex> lk(resource_mu_);
  auto rank_it = remote_buffer_to_mr_.find(owner_rank);
  if (rank_it == remote_buffer_to_mr_.end()) {
    throw std::runtime_error("remote MR rank cache not found");
  }
  auto id_it = rank_it->second.find(buffer_id);
  if (id_it == rank_it->second.end()) {
    throw std::runtime_error("remote MR not found for buffer_id");
  }
  return id_it->second;
}

bool Communicator::reg_ipc(uint32_t buffer_id, void* local_buf, size_t len,
                           bool publish) {
  if (buffer_id == 0) return false;

  IPCItem local{};
  if (local_buf != nullptr && len != 0) {
    int original_device = -1;
    GPU_RT_CHECK(gpuGetDevice(&original_device));
    auto restore = UKernel::Transport::finally(
        [&]() { GPU_RT_CHECK(gpuSetDevice(original_device)); });
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
    local = ipc_manager_.create_local_ipc(local_buf, len, local_gpu_idx_);
    if (!local.valid) return false;
  } else {
    local.valid = false;
  }

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    local_buffer_to_ipc_[buffer_id] = local;
  }

  if (!publish || !exchanger_client_ || !exchanger_client_->valid()) {
    return true;
  }

  IpcBufferInfo info{};
  info.handle = local.handle;
  info.base_offset = static_cast<uint64_t>(local.base_offset);
  info.bytes = static_cast<uint64_t>(local.bytes);
  info.device_idx = local.device_idx;
  info.valid = local.valid;
  return oob_put(*exchanger_client_, oob_namespace(),
                 ipc_global_buffer_key(global_rank_, buffer_id), info);
}

bool Communicator::dereg_ipc(uint32_t buffer_id) {
  IPCItem local{};
  bool found = false;
  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    auto it = local_buffer_to_ipc_.find(buffer_id);
    if (it != local_buffer_to_ipc_.end()) {
      local = it->second;
      local_buffer_to_ipc_.erase(it);
      found = true;
    }
  }
  if (found && local.base_addr != 0) {
    (void)ipc_manager_.delete_ipc(reinterpret_cast<void*>(local.base_addr));
  }
  return true;
}

bool Communicator::wait_ipc(int owner_rank, uint32_t buffer_id,
                            int timeout_ms) {
  if (buffer_id == 0) return false;
  if (owner_rank == global_rank_) {
    std::lock_guard<std::mutex> lk(resource_mu_);
    return local_buffer_to_ipc_.find(buffer_id) != local_buffer_to_ipc_.end();
  }
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  IpcBufferInfo info{};
  if (!oob_get(*exchanger_client_, oob_namespace(),
               ipc_global_buffer_key(owner_rank, buffer_id), info,
               timeout_ms)) {
    return false;
  }

  IPCItem state{};
  state.handle = info.handle;
  state.base_offset = static_cast<uintptr_t>(info.base_offset);
  state.bytes = static_cast<size_t>(info.bytes);
  state.device_idx = info.device_idx;
  state.valid = info.valid;
  return ipc_manager_.register_remote_ipc(owner_rank, buffer_id, state);
}

std::string Communicator::ipc_open_error_message(int owner_rank,
                                                 uint32_t buffer_id,
                                                 IPCItem const& item,
                                                 gpuError_t err) const {
  std::ostringstream oss;
  oss << "failed to open remote IPC mem handle"
      << " owner_rank=" << owner_rank << " buffer_id=" << buffer_id
      << " local_gpu=" << local_gpu_idx_
      << " remote_device_idx=" << item.device_idx
      << " bytes=" << item.bytes << " base_offset=" << item.base_offset
      << " err=" << static_cast<int>(err) << " (" << gpuGetErrorString(err)
      << ")";
  if (item.device_idx >= 0 && item.device_idx != local_gpu_idx_) {
    int can_access_peer = -1;
    gpuError_t access_err = gpuDeviceCanAccessPeer(
        &can_access_peer, local_gpu_idx_, item.device_idx);
    oss << " peer_access=" << can_access_peer
        << " peer_access_err=" << static_cast<int>(access_err)
        << " (" << gpuGetErrorString(access_err) << ")";
  }
  return oss.str();
}

IPCItem Communicator::get_ipc(uint32_t buffer_id) {
  std::lock_guard<std::mutex> lk(resource_mu_);
  auto it = local_buffer_to_ipc_.find(buffer_id);
  if (it == local_buffer_to_ipc_.end()) {
    throw std::runtime_error("local IPC not found for buffer_id");
  }
  return it->second;
}

IPCItem Communicator::get_ipc(int owner_rank, uint32_t buffer_id) {
  if (owner_rank == global_rank_) return get_ipc(buffer_id);
  IPCItem item = ipc_manager_.get_ipc(owner_rank, buffer_id);
  if (!item.valid) {
    throw std::runtime_error("remote IPC not found for buffer_id");
  }
  if (item.direct_ptr == nullptr) {
    int original_device = -1;
    GPU_RT_CHECK(gpuGetDevice(&original_device));
    auto restore = UKernel::Transport::finally(
        [&]() { GPU_RT_CHECK(gpuSetDevice(original_device)); });
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

    gpuError_t open_err = gpuIpcOpenMemHandle(&item.direct_ptr, item.handle,
                                               gpuIpcMemLazyEnablePeerAccess);
    if (open_err != gpuSuccess) {
      throw std::runtime_error(
          ipc_open_error_message(owner_rank, buffer_id, item, open_err));
    }
    ipc_manager_.register_remote_ipc(owner_rank, buffer_id, item);
  }
  return item;
}

bool Communicator::try_resolve_remote_ipc_pointer(int remote_rank,
                                                  uint32_t remote_buffer_id,
                                                  size_t offset, size_t bytes,
                                                  void** out_ptr,
                                                  int* out_device_idx) {
  if (out_ptr == nullptr || remote_buffer_id == 0) return false;
  *out_ptr = nullptr;

  IPCItem item{};
  if (remote_rank == global_rank_) {
    std::lock_guard<std::mutex> lk(resource_mu_);
    auto it = local_buffer_to_ipc_.find(remote_buffer_id);
    if (it != local_buffer_to_ipc_.end()) {
      item = it->second;
    }
  } else {
    item = ipc_manager_.get_ipc(remote_rank, remote_buffer_id);
  }
  if (!item.valid) return false;

  if (remote_rank == global_rank_) {
    if (item.base_addr == 0) return false;
    if (offset > item.bytes || bytes > item.bytes - offset) return false;
    uintptr_t const resolved = item.base_addr + item.base_offset + offset;
    *out_ptr = reinterpret_cast<void*>(resolved);
    if (out_device_idx != nullptr) {
      *out_device_idx = item.device_idx;
    }
    return true;
  }

  if (item.direct_ptr == nullptr) {
    int original_device = -1;
    if (gpuGetDevice(&original_device) != gpuSuccess) return false;
    if (gpuSetDevice(local_gpu_idx_) != gpuSuccess) return false;
    gpuError_t open_err = gpuIpcOpenMemHandle(&item.direct_ptr, item.handle,
                                               gpuIpcMemLazyEnablePeerAccess);
    gpuError_t restore_err = gpuSetDevice(original_device);
    if (restore_err != gpuSuccess) return false;
    if (open_err != gpuSuccess || item.direct_ptr == nullptr) {
      std::cerr << "[ERROR] "
                << ipc_open_error_message(remote_rank, remote_buffer_id, item,
                                          open_err)
                << std::endl;
      return false;
    }
    if (!ipc_manager_.register_remote_ipc(remote_rank, remote_buffer_id,
                                          item)) {
      return false;
    }
  }

  if (offset > item.bytes || bytes > item.bytes - offset) return false;
  uintptr_t const base = reinterpret_cast<uintptr_t>(item.direct_ptr);
  uintptr_t const resolved = base + item.base_offset + offset;
  *out_ptr = reinterpret_cast<void*>(resolved);
  if (out_device_idx != nullptr) {
    *out_device_idx = item.device_idx;
  }
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
  if (tracked.kind == PeerTransportKind::Ipc && ipc_adapter_) {
    ipc_adapter_->release_request(adapter_id);
  }
  if (tracked.host_copy_event != nullptr) {
    release_event(tracked.host_copy_event);
    tracked.host_copy_event = nullptr;
  }
  tracked.host_copy_submitted = false;
  tracked.needs_host_to_device_copy = false;

  // TCP uses per-transfer host buffer.
  if (tracked.bounce_ptr != nullptr) {
    GPU_RT_CHECK(gpuFreeHost(tracked.bounce_ptr));
    tracked.bounce_ptr = nullptr;
  }
}

bool Communicator::complete_host_bounce_recv(TrackedRequest& tracked,
                                             bool blocking) {
  if (!tracked.needs_host_to_device_copy) return true;
  if (!tracked.bounce_ptr || !tracked.completion_buffer) return false;

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset = finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  if (!tracked.host_copy_submitted) {
    if (tracked.host_copy_event == nullptr) {
      tracked.host_copy_event = acquire_event();
    }
    GPU_RT_CHECK(gpuMemcpyAsync(static_cast<char*>(tracked.completion_buffer) +
                                    tracked.completion_offset,
                                tracked.bounce_ptr, tracked.completion_bytes,
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
    if (query != gpuSuccess) GPU_RT_CHECK(query);
  }
  release_event(tracked.host_copy_event);
  tracked.host_copy_event = nullptr;
  tracked.host_copy_submitted = false;
  tracked.needs_host_to_device_copy = false;
  return true;
}

gpuEvent_t Communicator::acquire_event() {
  std::lock_guard<std::mutex> lk(event_pool_mu_);
  if (!event_pool_.empty()) {
    gpuEvent_t e = event_pool_.back();
    event_pool_.pop_back();
    return e;
  }
  gpuEvent_t e = nullptr;
  GPU_RT_CHECK(gpuEventCreateWithFlags(&e, gpuEventDisableTiming));
  return e;
}

void Communicator::release_event(gpuEvent_t event) {
  if (event == nullptr) return;
  std::lock_guard<std::mutex> lk(event_pool_mu_);
  event_pool_.push_back(event);
}

}  // namespace Transport
}  // namespace UKernel
