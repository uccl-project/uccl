#include "communicator.h"
#include "adapter/ipc_adapter.h"
#include "adapter/rdma_adapter.h"
#include "adapter/tcp_adapter.h"
#include "util/jrqueue.h"
#include "util/uk_debug.h"
#include "util/utils.h"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <sys/socket.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <cerrno>
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

std::string tcp_p2p_key(int src_rank, int dst_rank) {
  return "tcp_p2p_info_" + std::to_string(src_rank) + "_to_" +
         std::to_string(dst_rank);
}

std::string rdma_p2p_key(int src_rank, int dst_rank) {
  return "rdma_p2p_info_" + std::to_string(src_rank) + "_to_" +
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
      local_gpu_idx_);

  bool is_server = (global_rank_ == 0);
  if (!is_server && config_->exchanger_ip == "0.0.0.0")
    config_->exchanger_ip = "127.0.0.1";
  std::cout << "[INFO] Using hierarchical bootstrap exchanger as "
            << (is_server ? "server" : "client") << " "
            << config_->exchanger_ip << ":" << config_->exchanger_port
            << std::endl;
  exchanger_client_ = std::make_shared<HierarchicalExchanger>(
      (global_rank_ == 0), config_->exchanger_ip, config_->exchanger_port,
      /*timeout_ms=*/3000, /*max_line_bytes=*/1 * 1024 * 1024,
      /*local_id=*/config_->local_id);
  if (!exchanger_client_->valid()) {
    // Fail fast: an early return here used to leave a half-initialized
    // Communicator (no peer metas, no completion rings), surfacing
    // later as confusing "transport peer metadata is not established"
    // errors that hide the real cause.
    throw std::runtime_error("Communicator: exchanger init failed at " +
                             config_->exchanger_ip + ":" +
                             std::to_string(config_->exchanger_port) +
                             " (see [oob] lines above for the exact stage)");
  }

  // Completion ring: adapters push CompletionEvent when ops finish.
  size_t ring_sz = jring_get_buf_ring_size(sizeof(CompletionEvent), 2048);
  if (ring_sz != (size_t)-1) {
    put_completion_ring_ = static_cast<jring_t*>(calloc(1, ring_sz));
    if (put_completion_ring_)
      jring_init(put_completion_ring_, 2048, sizeof(CompletionEvent), 0, 0);
  }
  if (put_completion_ring_)
    ipc_adapter_->set_put_completion_ring(put_completion_ring_);

  // Signal send completion ring — separate from data ring
  ring_sz = jring_get_buf_ring_size(sizeof(CompletionEvent), 2048);
  if (ring_sz != (size_t)-1) {
    sig_send_completion_ring_ = static_cast<jring_t*>(calloc(1, ring_sz));
    if (sig_send_completion_ring_)
      jring_init(sig_send_completion_ring_, 2048, sizeof(CompletionEvent), 0,
                 1);
  }
  if (sig_send_completion_ring_)
    ipc_adapter_->set_sig_send_completion_ring(sig_send_completion_ring_);

  // Signal completion ring: on_signal_received pushes here,
  // try_complete_sig_wait dequeues. MP/MC for thread safety.
  // Signal completion ring: on_signal_received pushes here,
  // try_complete_sig_wait dequeues. MP/MC for thread safety. Sized above
  // the executor's in-flight wait cap (UK_CCL_SIG_INFLIGHT_CAP, default
  // 4096) so the overflow deque/mutex is not the common path.
  ring_sz = jring_get_buf_ring_size(sizeof(SignalCompletion), 8192);
  if (ring_sz != (size_t)-1) {
    sig_wait_completion_ring_ = static_cast<jring_t*>(calloc(1, ring_sz));
    if (sig_wait_completion_ring_)
      jring_init(sig_wait_completion_ring_, 8192, sizeof(SignalCompletion), 1,
                 1);
  }

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
    self.paths[PeerTransportKind::Ipc].put_ready = true;
    self.paths[PeerTransportKind::Ipc].wait_ready = true;
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

void Communicator::stop_transports() {
  put_cache_bump();  // teardown invalidates every cached put path
  if (rdma_adapter_) rdma_adapter_->shutdown_workers();
  if (ipc_adapter_) ipc_adapter_->stop();
}

Communicator::~Communicator() {
  stop_transports();

  for (auto const& [buffer_id, item] : mr_manager_.list_local_mrs()) {
    uint64_t const registered_id = buffer_id;
    // RDMA adapter destructor deregisters after QP teardown (flushes WRs).
    // Just clean up bookkeeping here.
    if (rdma_adapter_ && rdma_adapter_->is_initialized()) {
      std::lock_guard<std::mutex> lk(rdma_reg_mu_);
      rdma_registered_mrs_.erase(registered_id);
      rdma_direct_reg_failed_mrs_.erase(registered_id);
    }
    (void)mr_manager_.delete_mr(static_cast<uint32_t>(buffer_id));
  }

  tcp_adapter_.reset();
  rdma_adapter_.reset();

  // FIXME: gpuIpcCloseMemHandle hangs when GPU and NIC share PCIe topology.
  // IPC dereg must happen after RDMA QP/PD destruction (all ibv_mr references
  // released), but the CUDA driver still blocks on internal state.  Resources
  // are reclaimed by the OS on process exit.
  if (false) {
    int orig_dev = -1;
    gpuGetDevice(&orig_dev);
    gpuSetDevice(local_gpu_idx_);

    std::vector<uint32_t> local_ipc_buffer_ids;
    {
      std::lock_guard<std::mutex> lk(resource_mu_);
      local_ipc_buffer_ids.reserve(local_buffer_to_ipc_.size());
      for (auto const& kv : local_buffer_to_ipc_)
        local_ipc_buffer_ids.push_back(kv.first);
    }
    for (uint32_t buffer_id : local_ipc_buffer_ids) (void)dereg_ipc(buffer_id);

    for (int i = 0; i < world_size_; ++i) {
      if (i == global_rank_) continue;
      ipc_manager_.delete_ipc(i);
    }

    gpuSetDevice(orig_dev);
  }
  ipc_adapter_.reset();

  if (sig_wait_completion_ring_) {
    free(sig_wait_completion_ring_);
    sig_wait_completion_ring_ = nullptr;
  }
  if (sig_send_completion_ring_) {
    free(sig_send_completion_ring_);
    sig_send_completion_ring_ = nullptr;
  }
  if (put_completion_ring_) {
    free(put_completion_ring_);
    put_completion_ring_ = nullptr;
  }

  std::cout << "[INFO] Communicator " << global_rank_ << " resources released"
            << std::endl;
}

RdmaTransportAdapter& Communicator::ensure_rdma_adapter(
    CommunicatorMeta const& local_meta) {
  (void)local_meta;
  if (!rdma_adapter_) {
    RdmaTransportConfig rdma_cfg;
    rdma_adapter_ = std::make_unique<RdmaTransportAdapter>(local_gpu_idx_,
                                                           std::move(rdma_cfg));
    if (put_completion_ring_)
      rdma_adapter_->set_put_completion_ring(put_completion_ring_);
    if (sig_send_completion_ring_)
      rdma_adapter_->set_sig_send_completion_ring(sig_send_completion_ring_);
    rdma_adapter_->set_communicator(this);
  }
  return *rdma_adapter_;
}

bool Communicator::exchange_rdma_peer_info(int rank,
                                           RdmaTransportAdapter& rdma_adapter,
                                           RdmaP2PInfo* out_remote_p2p_info) {
  if (out_remote_p2p_info == nullptr) return false;

  auto init = rdma_adapter.get_connect_init(rank);
  RdmaP2PInfo local_p2p_info;
  local_p2p_info.data_qpn0 = init.remote_data_qpns[0];
  local_p2p_info.data_qpn1 = init.remote_data_qpns[1];
  local_p2p_info.data_qpn2 = init.remote_data_qpns[2];
  local_p2p_info.data_qpn3 = init.remote_data_qpns[3];
  local_p2p_info.signal_qpn = init.remote_signal_qpn;
  local_p2p_info.num_qps = init.num_qps;
  local_p2p_info.lid = init.remote_lid;
  memcpy(&local_p2p_info.gid_prefix, init.remote_gid_raw.data(), 8);
  memcpy(&local_p2p_info.gid_iface, init.remote_gid_raw.data() + 8, 8);
  local_p2p_info.dev_idx = init.local_dev_idx;
  local_p2p_info.gpu_idx = init.local_gpu_idx;

  std::string key = rdma_p2p_key(global_rank_, rank);
  std::string peer_key = rdma_p2p_key(rank, global_rank_);

  bool ok = oob_put(*exchanger_client_, oob_namespace(), key, local_p2p_info) &&
            oob_get(*exchanger_client_, oob_namespace(), peer_key,
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
    tcp_adapter_ = std::make_unique<TcpTransportAdapter>(
        local_meta.ip, global_rank_, local_gpu_idx_);
    if (put_completion_ring_)
      tcp_adapter_->set_put_completion_ring(put_completion_ring_);
    if (sig_send_completion_ring_)
      tcp_adapter_->set_sig_send_completion_ring(sig_send_completion_ring_);
  }
  return *tcp_adapter_;
}

Communicator::ResolvedPeer Communicator::resolve_peer(
    int rank, PeerTransportKind transport) const {
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

  if (transport != PeerTransportKind::Unknown) {
    // Explicit transport: validate compatibility
    if (transport == PeerTransportKind::Ipc ||
        transport == PeerTransportKind::Tcp ||
        transport == PeerTransportKind::Rdma) {
      bool same_host_val =
          (resolved.local_meta.host_id == resolved.remote_meta.host_id);
      bool rdma_capable_val = (resolved.local_meta.rdma_capable &&
                               resolved.remote_meta.rdma_capable);
      if (transport == PeerTransportKind::Ipc && !same_host_val) {
        throw std::invalid_argument("IPC transport requires same-host peer");
      }
      if (transport == PeerTransportKind::Rdma && !rdma_capable_val) {
        throw std::invalid_argument(
            "Rdma transport requires RDMA-capable peers");
      }
    }
    resolved.kind = transport;
  } else {
    resolved.kind = resolve_peer_transport_kind(*config_, resolved.local_meta,
                                                resolved.remote_meta);
  }
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

bool Communicator::ensure_path(int rank, bool is_put,
                               PeerTransportKind transport) {
  if (rank == global_rank_) return true;
  if (rank < 0 || rank >= world_size_) return false;

  if (is_put ? has_put_path(rank, transport) : has_wait_path(rank, transport))
    return true;

  ResolvedPeer resolved;
  try {
    resolved = resolve_peer(rank, transport);
  } catch (std::exception const& ex) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " failed to resolve transport for rank " << rank << ": "
              << ex.what() << std::endl;
    return false;
  }

  auto conn_type = is_put ? PeerConnectType::Connect : PeerConnectType::Accept;
  char const* dir_label = is_put ? "put" : "wait";

  auto fallback = [&] {
    if (transport != PeerTransportKind::Unknown) return false;
    return try_fallback_tcp_accept(rank, resolved.local_meta);
  };

  // Fall through to RDMA.
  if (resolved.kind == PeerTransportKind::Rdma) {
    UK_DBG(UK_DBG_LVL_TPT, "[ensure_path r%d] RDMA ensure_rdma_adapter ...",
           global_rank_);
    auto& rdma = ensure_rdma_adapter(resolved.local_meta);
    UK_DBG(UK_DBG_LVL_TPT, "[ensure_path r%d] RDMA ensure_rdma_adapter done",
           global_rank_);
    bool ready = is_put ? rdma.has_put_path(rank) : rdma.has_wait_path(rank);
    if (!ready) {
      RdmaP2PInfo remote;
      UK_DBG(UK_DBG_LVL_TPT,
             "[ensure_path r%d] RDMA exchange_rdma_peer_info ...",
             global_rank_);
      if (!exchange_rdma_peer_info(rank, rdma, &remote)) return fallback();
      UK_DBG(UK_DBG_LVL_TPT,
             "[ensure_path r%d] RDMA exchange_rdma_peer_info done",
             global_rank_);

      RdmaPeerConnectSpec rspec;
      rspec.num_qps = remote.num_qps;
      rspec.remote_lid = remote.lid;
      memcpy(&rspec.remote_gid_raw[0], &remote.gid_prefix, 8);
      memcpy(&rspec.remote_gid_raw[8], &remote.gid_iface, 8);
      rspec.remote_data_qpns[0] = remote.data_qpn0;
      rspec.remote_data_qpns[1] = remote.data_qpn1;
      rspec.remote_data_qpns[2] = remote.data_qpn2;
      rspec.remote_data_qpns[3] = remote.data_qpn3;
      rspec.remote_signal_qpn = remote.signal_qpn;
      rspec.local_dev_idx = remote.dev_idx;
      rspec.local_gpu_idx = local_gpu_idx_;
      rspec.remote_dev_idx = remote.dev_idx;
      rspec.remote_gpu_idx = remote.gpu_idx;

      PeerConnectSpec spec{};
      spec.peer_rank = rank;
      spec.type = conn_type;
      spec.detail = std::move(rspec);
      if (!(is_put ? rdma.ensure_put_path(spec)
                   : rdma.ensure_wait_path(spec))) {
        return fallback();
      }
    }
    is_put ? mark_put_path_ready(rank, PeerTransportKind::Rdma)
           : mark_wait_path_ready(rank, PeerTransportKind::Rdma);
    register_existing_local_mrs_with_rdma();
    return true;
  }

  if (resolved.kind == PeerTransportKind::Ipc) {
    PeerConnectSpec spec{};
    spec.peer_rank = rank;
    spec.type = conn_type;
    spec.detail = IpcPeerConnectSpec{};
    if (!(is_put ? ipc_adapter_->ensure_put_path(spec)
                 : ipc_adapter_->ensure_wait_path(spec))) {
      std::cerr << "[ERROR] Communicator " << global_rank_ << " IPC "
                << dir_label << " failed to rank " << rank << std::endl;
      if (is_put) {
        put_cache_bump();  // the peer's put path is being torn down
        ipc_adapter_->close_comp(rank);
      }
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
      spec.detail =
          TcpPeerConnectSpec{remote.ip, is_put ? remote.port : uint16_t{0}};
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

bool Communicator::connect(int rank, PeerTransportKind transport) {
  return ensure_path(rank, true, transport);
}

bool Communicator::accept(int rank, PeerTransportKind transport) {
  return ensure_path(rank, false, transport);
}

bool Communicator::has_put_path(int rank, PeerTransportKind transport) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return false;
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (transport != PeerTransportKind::Unknown) {
    auto it = peer.paths.find(transport);
    return it != peer.paths.end() && it->second.put_ready;
  }
  // Unknown: check if ANY path has put_ready
  for (auto const& kv : peer.paths) {
    if (kv.second.put_ready) return true;
  }
  return false;
}

bool Communicator::has_wait_path(int rank, PeerTransportKind transport) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return false;
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (transport != PeerTransportKind::Unknown) {
    auto it = peer.paths.find(transport);
    return it != peer.paths.end() && it->second.wait_ready;
  }
  // Unknown: check if ANY path has wait_ready
  for (auto const& kv : peer.paths) {
    if (kv.second.wait_ready) return true;
  }
  return false;
}

void Communicator::mark_put_path_ready(int rank, PeerTransportKind kind) {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto& peer = peer_states_.at(static_cast<size_t>(rank));
  peer.paths[kind].put_ready = true;
  if (peer.resolved_kind == PeerTransportKind::Unknown) {
    peer.resolved_kind = kind;
  }
  put_cache_bump();  // a new path may resolve to a different adapter
}

void Communicator::mark_wait_path_ready(int rank, PeerTransportKind kind) {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto& peer = peer_states_.at(static_cast<size_t>(rank));
  peer.paths[kind].wait_ready = true;
}

PeerTransportKind Communicator::get_put_transport_kind(
    int rank, PeerTransportKind transport) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (!peer.has_meta) {
    throw std::runtime_error("transport peer session is not established");
  }
  if (transport != PeerTransportKind::Unknown) {
    auto it = peer.paths.find(transport);
    if (it != peer.paths.end() && it->second.put_ready) return transport;
  }
  if (peer.resolved_kind != PeerTransportKind::Unknown) {
    auto it = peer.paths.find(peer.resolved_kind);
    if (it != peer.paths.end() && it->second.put_ready)
      return peer.resolved_kind;
  }
  // Fallback: return first available put path
  for (auto const& kv : peer.paths) {
    if (kv.second.put_ready) return kv.first;
  }
  throw std::runtime_error("transport put path is not established");
}

PeerTransportKind Communicator::get_wait_transport_kind(
    int rank, PeerTransportKind transport) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (!peer.has_meta) {
    throw std::runtime_error("transport peer session is not established");
  }
  if (transport != PeerTransportKind::Unknown) {
    auto it = peer.paths.find(transport);
    if (it != peer.paths.end() && it->second.wait_ready) return transport;
  }
  if (peer.resolved_kind != PeerTransportKind::Unknown) {
    auto it = peer.paths.find(peer.resolved_kind);
    if (it != peer.paths.end() && it->second.wait_ready)
      return peer.resolved_kind;
  }
  // Fallback: return first available wait path
  for (auto const& kv : peer.paths) {
    if (kv.second.wait_ready) return kv.first;
  }
  throw std::runtime_error("transport wait path is not established");
}

PeerTransportKind Communicator::peer_transport_kind(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (!peer.has_meta) {
    throw std::runtime_error("transport peer session is not established");
  }
  if (peer.resolved_kind != PeerTransportKind::Unknown)
    return peer.resolved_kind;
  // Fallback: return first available kind from paths
  for (auto const& kv : peer.paths) {
    if (kv.second.put_ready || kv.second.wait_ready) return kv.first;
  }
  throw std::runtime_error("transport peer path is not established");
}

int Communicator::peer_gpu_idx(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return -1;
  return peer_states_[static_cast<size_t>(rank)].gpu_idx;
}

TransportAdapter* Communicator::get_adapter(PeerTransportKind kind) {
  switch (kind) {
    case PeerTransportKind::Tcp:
      return tcp_adapter_.get();
    case PeerTransportKind::Ipc:
      return ipc_adapter_.get();
    case PeerTransportKind::Rdma:
      return rdma_adapter_.get();
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

void Communicator::register_existing_local_mrs_with_rdma() {
  if (!rdma_adapter_ || !rdma_adapter_->is_initialized()) return;
  for (auto const& [buffer_id, item] : mr_manager_.list_local_mrs()) {
    void* ptr = reinterpret_cast<void*>(item.mr.address);
    size_t len = static_cast<size_t>(item.mr.length);
    if (!ensure_rdma_memory_registered(buffer_id, ptr, len)) continue;

    // Re-publish with correct rkey now that RDMA adapter is ready.
    // Buffers registered before the RDMA adapter was created have
    // rkey=0 in the published OOB entry.  Update it so the peer
    // can use RDMA writes to this buffer.
    if (rdma_adapter_->is_memory_registered(buffer_id)) {
      uint32_t rkey = rdma_adapter_->get_memory_rkey(buffer_id);
      if (rkey != 0) {
        MR mr = get_mr(buffer_id);
        mr.key = rkey;
        {
          std::lock_guard<std::mutex> lk(resource_mu_);
          local_buffer_to_mr_[buffer_id].key = rkey;
        }
        NamedMRInfos payload{};
        payload.generation =
            mr_generation_.fetch_add(1, std::memory_order_relaxed);
        payload.entries.push_back(NamedMR{buffer_id, mr});
        oob_put(*exchanger_client_, oob_namespace(),
                mr_global_buffer_key(global_rank_, buffer_id), payload);
      }
    }
  }
}

bool Communicator::ensure_rdma_memory_registered(uint32_t buffer_id, void* ptr,
                                                 size_t len) {
  if (!rdma_adapter_ || !rdma_adapter_->is_initialized()) return false;

  if (rdma_adapter_->is_memory_registered(buffer_id)) return true;

  void* base_ptr = ptr;
  size_t mr_len = len;
  bool is_direct_local_mr = false;

  MRItem item = mr_manager_.get_mr(static_cast<uint32_t>(buffer_id));
  if (item.valid) {
    base_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(item.mr.address));
    mr_len = static_cast<size_t>(item.mr.length);
    is_direct_local_mr = true;
  }

  if (is_direct_local_mr) {
    std::lock_guard<std::mutex> lk(rdma_reg_mu_);
    if (rdma_direct_reg_failed_mrs_.find(buffer_id) !=
        rdma_direct_reg_failed_mrs_.end()) {
      return false;
    }
  }

  if (base_ptr == nullptr || mr_len == 0) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " invalid pointer or length for RDMA reg, buffer_id="
              << buffer_id << std::endl;
    return false;
  }

  bool ok = rdma_adapter_->register_memory(buffer_id, base_ptr, mr_len);
  if (!ok) {
    if (is_direct_local_mr) {
      std::lock_guard<std::mutex> lk(rdma_reg_mu_);
      rdma_direct_reg_failed_mrs_.insert(buffer_id);
      std::cerr << "[WARN] Communicator " << global_rank_
                << " failed to register local GPU MR " << buffer_id
                << " with RDMA, base=" << base_ptr << " len=" << mr_len
                << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " failed to register host MR " << buffer_id << " with RDMA"
                << std::endl;
    }
  } else {
    std::lock_guard<std::mutex> lk(rdma_reg_mu_);
    rdma_registered_mrs_.insert(buffer_id);
  }
  return ok;
}

unsigned Communicator::send_put_async(int peer, uint32_t src_buf,
                                      size_t src_off, uint32_t dst_buf,
                                      size_t dst_off, size_t bytes,
                                      PeerTransportKind transport) {
  unsigned rid = alloc_rid();
  record_user_ctx(rid, 0);
  if (!send_put_async_with_rid(peer, src_buf, src_off, dst_buf, dst_off, bytes,
                               transport, rid)) {
    consume_user_ctx(rid);
    return 0;
  }
  return rid;
}

bool Communicator::put_cache_hit(int peer, uint32_t src_buf, uint32_t dst_buf,
                                 size_t src_off, size_t dst_off, size_t bytes,
                                 PeerTransportKind transport,
                                 void** local_ptr, void** remote_ptr) {
  if (bytes == 0) return false;
  uint64_t gen = put_cache_gen_.load(std::memory_order_relaxed);
  PutFastKey key{peer, src_buf, dst_buf};
  size_t const start = PutFastKeyHash{}(key) & (kPutCacheSlots - 1);
  PutFastEntry e;
  bool found = false;
  for (;;) {
    uint64_t const s0 = put_cache_seq_.load(std::memory_order_acquire);
    if (s0 & 1) {  // writer in flight — retry
      std::this_thread::yield();
      continue;
    }
    found = false;
    for (size_t i = 0; i < kPutCacheSlots; ++i) {
      auto const& slot =
          put_cache_slots_[(start + i) & (kPutCacheSlots - 1)];
      if (!slot.valid) break;  // no deletes: empty terminates the probe
      if (slot.key == key) {
        e = slot.entry;
        found = true;
        break;
      }
    }
    if (!found) return false;
    if (put_cache_seq_.load(std::memory_order_acquire) == s0) break;
  }
  if (e.gen != gen) return false;
  // The cache only holds IPC entries, and only matches a request for the
  // same kind — a later explicit RDMA/TCP request for the same
  // (peer, src, dst) must not be silently served over IPC.
  if (e.kind != PeerTransportKind::Ipc || e.adapter == nullptr ||
      e.remote_base == nullptr)
    return false;
  if (transport != PeerTransportKind::Unknown && e.kind != transport)
    return false;
  if (src_off > e.local_len || bytes > e.local_len - src_off) return false;
  *local_ptr = static_cast<char*>(e.local_base) + src_off;
  *remote_ptr = static_cast<char*>(e.remote_base) + dst_off;
  return true;
}

void Communicator::put_cache_fill(int peer, uint32_t src_buf, uint32_t dst_buf,
                                  PeerTransportKind kind,
                                  TransportAdapter* adapter, void* local_base,
                                  size_t local_len, void* remote_base) {
  PutFastKey key{peer, src_buf, dst_buf};
  size_t const start = PutFastKeyHash{}(key) & (kPutCacheSlots - 1);
  PutFastEntry entry{put_cache_gen_.load(std::memory_order_relaxed), kind,
                     adapter, local_base, local_len, remote_base};
  std::lock_guard<std::mutex> lk(put_cache_write_mu_);
  put_cache_seq_.fetch_add(1, std::memory_order_release);  // odd: write
  for (size_t i = 0; i < kPutCacheSlots; ++i) {
    auto& slot = put_cache_slots_[(start + i) & (kPutCacheSlots - 1)];
    if (!slot.valid || slot.key == key) {
      slot.key = key;
      slot.entry = entry;
      slot.valid = true;
      break;
    }
    // Table full: drop the insertion (a miss just takes the slow path).
  }
  put_cache_seq_.fetch_add(1, std::memory_order_release);  // even: stable
}

bool Communicator::send_put_async_with_rid(int peer, uint32_t src_buf,
                                           size_t src_off, uint32_t dst_buf,
                                           size_t dst_off, size_t bytes,
                                           PeerTransportKind transport,
                                           unsigned rid, uint32_t qp_affinity) {
  // Fast path: resolved (peer, src, dst) IPC entry — no path/MR/IPC
  // locks. Only safe for kind==Ipc (the dominant same-host path); other
  // kinds always fall through.
  void* fast_local = nullptr;
  void* fast_remote = nullptr;
  if (put_cache_hit(peer, src_buf, dst_buf, src_off, dst_off, bytes,
                    transport, &fast_local, &fast_remote)) {
    return ipc_adapter_->send_put_async(peer, fast_local, src_buf, fast_remote,
                                        dst_buf, bytes, rid);
  }

  if (!ensure_path(peer, /*is_put=*/true, transport)) return false;
  PeerTransportKind kind = get_put_transport_kind(peer, transport);
  auto* adapter = get_adapter(kind);
  if (!adapter) return false;

  MR local_mr = get_mr(src_buf);
  if (src_off > local_mr.length || bytes > local_mr.length - src_off) {
    UK_DBG(UK_DBG_LVL_TPT,
           "[comm-send r%d] REJECT local bounds: src_buf=%u src_off=%zu "
           "bytes=%zu mr.length=%llu",
           global_rank_, src_buf, src_off, bytes,
           (unsigned long long)local_mr.length);
    return false;
  }
  void* local_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(local_mr.address) + src_off);

  if (kind == PeerTransportKind::Tcp) {
    return adapter->send_put_async(peer, local_ptr, 0, nullptr, 0, bytes, rid);
  }
  {
    static int dbg_cnt = 0;
    if (dbg_cnt++ < 5)
      UK_DBG(UK_DBG_LVL_TPT, "[comm-send r%d] peer=%d kind=%d IPC-or-RDMA",
             global_rank_, peer, (int)kind);
  }
  if (kind == PeerTransportKind::Rdma) {
    if (!rdma_adapter_ || !rdma_adapter_->is_initialized()) {
      static int once = 0;
      if (!once++)
        std::cerr << "[WARN] RDMA adapter not initialized" << std::endl;
      return false;
    }
    if (!ensure_rdma_memory_registered(src_buf, local_ptr, bytes)) return false;
    uint32_t remote_id = dst_buf != 0 ? dst_buf : src_buf;
    MR remote_mr = get_mr(peer, remote_id);
    if (remote_mr.length == 0 || remote_mr.key == 0) return false;
    void* remote_ptr = reinterpret_cast<void*>(
        static_cast<uint64_t>(remote_mr.address) + dst_off);
    rdma_adapter_->register_remote_buffer(
        peer, remote_id, reinterpret_cast<uint64_t>(remote_ptr), remote_mr.key);
    return rdma_adapter_->send_put_async(peer, local_ptr, src_buf, remote_ptr,
                                         remote_id, bytes, qp_affinity,
                                         rid) != 0;
  }
  void* remote_ptr = nullptr;
  int remote_gpu = -1;
  if (dst_buf != 0) {
    if (!try_resolve_remote_ipc_pointer(peer, dst_buf, dst_off, bytes,
                                        &remote_ptr, &remote_gpu)) {
      UK_DBG(UK_DBG_LVL_TPT,
             "[comm-send r%d] REJECT remote resolve: peer=%d dst_buf=%u "
             "dst_off=%zu bytes=%zu",
             global_rank_, peer, dst_buf, dst_off, bytes);
      return false;
    }
  }
  bool ok = adapter->send_put_async(peer, local_ptr, src_buf, remote_ptr,
                                    dst_buf, bytes, rid);
  if (ok && kind == PeerTransportKind::Ipc && remote_ptr)
    put_cache_fill(peer, src_buf, dst_buf, kind, adapter,
                   reinterpret_cast<void*>(local_mr.address), local_mr.length,
                   static_cast<char*>(remote_ptr) - dst_off);
  return ok;
}

bool Communicator::send_put_signal_async_with_rid(
    int peer, uint32_t src_buf, size_t src_off, uint32_t dst_buf,
    size_t dst_off, size_t bytes, PeerTransportKind transport, uint64_t tag,
    unsigned rid, uint32_t qp_affinity) {
  void* fast_local = nullptr;
  void* fast_remote = nullptr;
  if (put_cache_hit(peer, src_buf, dst_buf, src_off, dst_off, bytes,
                    transport, &fast_local, &fast_remote)) {
    return ipc_adapter_->send_put_signal_async(
        peer, fast_local, src_buf, fast_remote, dst_buf, bytes, tag, rid);
  }

  if (!ensure_path(peer, /*is_put=*/true, transport)) return false;
  PeerTransportKind kind = get_put_transport_kind(peer, transport);
  auto* adapter = get_adapter(kind);
  if (!adapter || !adapter->supports_put_signal()) return false;

  MR local_mr = get_mr(src_buf);
  if (src_off > local_mr.length || bytes > local_mr.length - src_off)
    return false;
  void* local_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(local_mr.address) + src_off);

  if (kind == PeerTransportKind::Rdma) {
    if (!rdma_adapter_ || !rdma_adapter_->is_initialized()) return false;
    if (!ensure_rdma_memory_registered(src_buf, local_ptr, bytes)) return false;
    uint32_t remote_id = dst_buf != 0 ? dst_buf : src_buf;
    MR remote_mr = get_mr(peer, remote_id);
    if (remote_mr.length == 0 || remote_mr.key == 0) return false;
    void* remote_ptr = reinterpret_cast<void*>(
        static_cast<uint64_t>(remote_mr.address) + dst_off);
    rdma_adapter_->register_remote_buffer(
        peer, remote_id, reinterpret_cast<uint64_t>(remote_ptr), remote_mr.key);
    return rdma_adapter_->send_put_signal_async(peer, local_ptr, src_buf,
                                                remote_ptr, remote_id, bytes,
                                                tag, qp_affinity, rid) != 0;
  }

  // IPC: resolve the peer's buffer pointer and let the adapter's send
  // worker write the signal ring after the copy completes.
  void* remote_ptr = nullptr;
  int remote_gpu = -1;
  if (dst_buf != 0) {
    if (!try_resolve_remote_ipc_pointer(peer, dst_buf, dst_off, bytes,
                                        &remote_ptr, &remote_gpu)) {
      UK_DBG(UK_DBG_LVL_TPT,
             "[comm-send r%d] REJECT fused remote resolve: peer=%d dst_buf=%u "
             "dst_off=%zu bytes=%zu",
             global_rank_, peer, dst_buf, dst_off, bytes);
      return false;
    }
  }
  bool ok = adapter->send_put_signal_async(peer, local_ptr, src_buf,
                                           remote_ptr, dst_buf, bytes, tag,
                                           rid) != 0;
  if (ok && kind == PeerTransportKind::Ipc && remote_ptr)
    put_cache_fill(peer, src_buf, dst_buf, kind, adapter,
                   reinterpret_cast<void*>(local_mr.address), local_mr.length,
                   static_cast<char*>(remote_ptr) - dst_off);
  return ok;
}

bool Communicator::can_fuse_put_signal(int peer, PeerTransportKind transport) {
  PeerTransportKind kind = get_put_transport_kind(peer, transport);
  auto* adapter = get_adapter(kind);
  return adapter && adapter->supports_put_signal();
}

void* Communicator::ipc_signal_ring_device_ptr(int peer) const {
  return ipc_adapter_ ? ipc_adapter_->peer_signal_ring_device_ptr(peer)
                      : nullptr;
}

void* Communicator::ipc_device_flag_ptr(int peer) const {
  return ipc_adapter_ ? ipc_adapter_->peer_device_flag_ptr(peer) : nullptr;
}

unsigned Communicator::wait_signal_async(int peer, uint64_t tag,
                                         PeerTransportKind transport) {
  unsigned rid = alloc_rid();
  record_user_ctx(rid, 0);
  if (!wait_signal_async_with_rid(peer, tag, transport, rid)) {
    consume_user_ctx(rid);
    return 0;
  }
  return rid;
}

bool Communicator::wait_flag_async_with_rid(int peer, uint32_t slot,
                                            uint64_t tag, unsigned rid,
                                            uint32_t count) {
  if (!ensure_path(peer, /*is_put=*/false, PeerTransportKind::Ipc))
    return false;
  if (!ipc_adapter_) return false;
  uint64_t* slots = ipc_adapter_->local_device_flag_slots(peer);
  if (!slots) return false;
  if (count == 0) count = 1;
  if (static_cast<uint64_t>(slot) + count > Transport::kDeviceFlagSlots)
    return false;
  uint64_t* s = slots + slot;
  SignalCompletion ev{rid, tag, peer, false};
  uint32_t matched = 0;
  {
    std::lock_guard<std::mutex> lk(flag_waits_mu_);
    // Count slots that already hold their tag (the device may have
    // completed before the wait registered).
    for (uint32_t i = 0; i < count; ++i)
      if (s[i] == tag + i) ++matched;
    if (matched < count) {
      pending_flag_waits_.push_back({rid, peer, s, tag, count, matched});
      pending_waits_count_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (matched == count) {
    if (jring_mp_enqueue_bulk(sig_wait_completion_ring_, &ev, 1, nullptr) !=
        1) {
      std::lock_guard<std::mutex> lk(sig_wait_overflow_mu_);
      sig_wait_overflow_.push_back(ev);
    }
  }
  return true;
}

bool Communicator::wait_signal_async_with_rid(int peer, uint64_t tag,
                                              PeerTransportKind transport,
                                              unsigned rid, uint32_t count,
                                              bool force_imm) {
  if (!ensure_path(peer, /*is_put=*/false, transport)) return false;

  PeerTransportKind kind = get_wait_transport_kind(peer, transport);

  if (kind == PeerTransportKind::Tcp) {
    // Counted waits are only produced by fused RDMA groups; a TCP peer
    // never fuses, so count is always 1 here.
    if (count > 1) return false;
    auto* adapter = get_adapter(kind);
    if (!adapter || !adapter->wait_signal_async(peer, tag, std::nullopt, rid))
      return false;
    {
      std::lock_guard<std::mutex> lk(tcp_sig_mu_);
      tcp_signal_rids_[rid] = {peer, tag};
      pending_waits_count_.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
  }

  // Fused RDMA PutSignal waits match write-with-imm arrivals, not the
  // 64-bit tag map. The ONLY reliable signal is force_imm (the executor's
  // mirror of a fused group): a small tag value does NOT imply the tag
  // travelled as a 32-bit immediate — a plain 64-bit signal QP SEND can
  // carry any tag, including 0. Matching is per-peer FIFO in arrival
  // order.
  bool const imm = force_imm;
  if (imm) {
    uint32_t const low32 = static_cast<uint32_t>(tag);
    SignalCompletion ev{};
    bool matched = false;
    {
      std::lock_guard<std::mutex> lk(sig_maps_mu_);
      auto& buf = buffered_imms_[peer];
      uint32_t remaining = count;
      // Drain buffered arrivals, but only while the OLDEST buffered imm
      // matches: an earlier arrival belongs to an earlier wait.
      while (remaining > 0 && !buf.empty() && buf.front() == low32) {
        buf.pop_front();
        --remaining;
      }
      if (buf.empty()) buffered_imms_.erase(peer);
      if (remaining == 0) {
        ev.rid = rid;
        ev.tag = tag;
        ev.peer = peer;
        ev.failed = false;
        matched = true;
      } else {
        pending_imm_waits_[peer].push_back({rid, remaining, tag, low32});
        pending_waits_count_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (matched) {
      jrpush(sig_wait_completion_ring_, ev);
    }
    return true;
  }

  SignalCompletion ev{};
  bool matched = false;
  {
    std::lock_guard<std::mutex> lk(sig_maps_mu_);
    auto sig_it = pending_signals_.find(peer);
    static int dbg = 0;
    if (dbg++ < 5 && uk_dbg_lvl() >= UK_DBG_LVL_TPT) {
      std::string tags;
      if (sig_it != pending_signals_.end())
        for (auto& t : sig_it->second) {
          tags += std::to_string((unsigned long)t);
          tags += ',';
        }
      UK_DBG(
          UK_DBG_LVL_TPT,
          "[wsig-wait r%d] peer=%d tag=%lu pending_sigs_has_peer=%d tags=[%s]",
          global_rank_, peer, (unsigned long)tag,
          sig_it != pending_signals_.end() ? 1 : 0, tags.c_str());
    }
    // Drain up to `count` buffered arrivals of this tag first.
    uint32_t remaining = count;
    if (sig_it != pending_signals_.end()) {
      auto& sigs = sig_it->second;
      auto sit = sigs.begin();
      while (remaining > 0 && sit != sigs.end()) {
        if (*sit == tag) {
          sit = sigs.erase(sit);
          --remaining;
        } else {
          ++sit;
        }
      }
      if (sigs.empty()) pending_signals_.erase(sig_it);
    }
    if (remaining == 0) {
      ev.rid = rid;
      ev.tag = tag;
      ev.peer = peer;
      ev.failed = false;
      matched = true;
    } else {
      if (uk_dbg_lvl() >= UK_DBG_LVL_TPT)
        std::fprintf(stderr, "[sig-wait r%d] peer=%d tag=%#lx count=%u\n",
                     global_rank_, peer, (unsigned long)tag, remaining);
      pending_signal_waits_[peer][tag].emplace_back(rid, remaining);
      pending_waits_count_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (matched) {
    jrpush(sig_wait_completion_ring_, ev);
  }
  return true;
}

unsigned Communicator::wait_signal_async(int peer, uint64_t tag,
                                         uint32_t recv_buf, size_t off,
                                         size_t len,
                                         PeerTransportKind transport) {
  if (!ensure_path(peer, /*is_put=*/false, transport)) return 0;
  PeerTransportKind kind = get_wait_transport_kind(peer, transport);
  auto* adapter = get_adapter(kind);
  if (!adapter) return 0;

  unsigned rid = next_rid_.fetch_add(1, std::memory_order_relaxed);

  // IPC DataWait: pass a non-null target so adapter uses the DataWait
  // path (next_recv_match_seq + last_completed counter). The
  // local_ptr/len inside the target are not consumed by the IPC
  // recv_one; the send_worker already performed the GPU copy.
  if (kind == PeerTransportKind::Ipc) {
    TransportAdapter::WaitTarget target;
    target.local_ptr = nullptr;
    target.len = 0;
    if (!adapter->wait_signal_async(peer, tag, std::move(target), rid))
      return 0;
    return rid;
  }

  // Resolve recv buffer to get local GPU pointer
  MR local_mr = get_mr(recv_buf);
  if (off > local_mr.length || len > local_mr.length - off) return 0;
  void* local_ptr =
      reinterpret_cast<void*>(static_cast<uintptr_t>(local_mr.address) + off);

  TransportAdapter::WaitTarget target;
  target.local_ptr = local_ptr;
  target.len = len;
  target.local_buffer_id = recv_buf;

  if (!adapter->wait_signal_async(peer, tag, std::move(target), rid)) return 0;
  return rid;
}

unsigned Communicator::send_signal_async(int peer, uint64_t tag,
                                         PeerTransportKind transport) {
  unsigned rid = alloc_rid();
  record_user_ctx(rid, 0);
  if (!send_signal_async_with_rid(peer, tag, transport, rid)) {
    consume_user_ctx(rid);
    return 0;
  }
  return rid;
}

bool Communicator::send_signal_async_with_rid(int peer, uint64_t tag,
                                              PeerTransportKind transport,
                                              unsigned rid) {
  if (!ensure_path(peer, /*is_put=*/true, transport)) return false;
  PeerTransportKind kind = get_put_transport_kind(peer, transport);
  auto* adapter = get_adapter(kind);
  if (!adapter) return false;
  return adapter->send_signal_async(peer, tag, rid);
}

size_t Communicator::try_complete_put(CompletionResult* results, size_t max) {
  if (!put_completion_ring_) return 0;
  CompletionEvent ev;
  size_t count = 0;
  while (count < max &&
         jring_mc_dequeue_bulk(put_completion_ring_, &ev, 1, nullptr) == 1) {
    results[count].rid = ev.rid;
    results[count].failed = (ev.failed != 0);
    results[count].user_ctx = consume_user_ctx(ev.rid);
    count++;
  }
  return count;
}

size_t Communicator::try_complete_sig_send(CompletionResult* results,
                                           size_t max) {
  if (!sig_send_completion_ring_) return 0;
  CompletionEvent ev;
  size_t count = 0;
  while (count < max && jring_sc_dequeue_bulk(sig_send_completion_ring_, &ev, 1,
                                              nullptr) == 1) {
    results[count].rid = ev.rid;
    results[count].failed = (ev.failed != 0);
    results[count].user_ctx = consume_user_ctx(ev.rid);
    count++;
  }
  return count;
}

size_t Communicator::try_complete_sig_wait(SignalCompletion* events,
                                           size_t max) {
  // Dequeue from the completion ring FIRST to free capacity, then drain
  // IPC signals — avoiding a deadlock where drain_ipc_signals blocks on
  // jrpush because the ring is full while the only consumer (this thread)
  // is stuck trying to push.
  size_t count = 0;
  if (sig_wait_completion_ring_) {
    count = jring_sc_dequeue_burst(sig_wait_completion_ring_, events,
                                   std::min(max, (size_t)256), nullptr);
    if (count > 0) {
      UK_DBG(UK_DBG_LVL_TPT, "[sig-wait-ring r%d] drained=%zu", global_rank_,
             count);
    } else {
      static int dbg = 0;
      if (uk_dbg_lvl() >= UK_DBG_LVL_ALL && ++dbg % 50000 == 0)
        UK_DBG(UK_DBG_LVL_ALL, "[sig-wait-ring r%d] drained=0", global_rank_);
    }
    for (size_t i = 0; i < count; ++i)
      events[i].user_ctx = consume_user_ctx(events[i].rid);
  }

  // Drain overflow: completions that couldn't fit the ring.
  {
    std::lock_guard<std::mutex> lk(sig_wait_overflow_mu_);
    size_t to_drain = std::min(static_cast<size_t>(max) - count,
                               sig_wait_overflow_.size());
    for (size_t i = 0; i < to_drain; ++i) {
      events[count] = sig_wait_overflow_.front();
      events[count].user_ctx = consume_user_ctx(events[count].rid);
      sig_wait_overflow_.pop_front();
      ++count;
    }
  }

  drain_ipc_signals();

  // Device-flag waits: poll the per-slot flags written by the peer's
  // device tasks (plain stores + fence). Single writer and single
  // consumer per slot, so no claim/clear — the epoch-salted tag
  // invalidates stale values across runs.
  if (count < max && !pending_flag_waits_.empty()) {
    std::lock_guard<std::mutex> lk(flag_waits_mu_);
    for (size_t i = 0; i < pending_flag_waits_.size() && count < max;) {
      auto& fw = pending_flag_waits_[i];
      while (fw.matched < fw.count &&
             fw.base[fw.matched] == fw.expected + fw.matched)
        ++fw.matched;
      if (fw.matched == fw.count) {
        events[count].rid = fw.rid;
        events[count].tag = fw.expected;
        events[count].peer = fw.peer;
        events[count].failed = false;
        events[count].user_ctx = consume_user_ctx(fw.rid);
        ++count;
        pending_flag_waits_[i] = pending_flag_waits_.back();
        pending_flag_waits_.pop_back();
        pending_waits_count_.fetch_sub(1, std::memory_order_relaxed);
      } else {
        ++i;
      }
    }
  }

  // Drain data completion ring for TCP signal completions.
  if (put_completion_ring_ && count < max) {
    CompletionEvent ce;
    {
      std::lock_guard<std::mutex> lk(tcp_sig_mu_);
      while (count < max && jring_mc_dequeue_bulk(put_completion_ring_, &ce, 1,
                                                  nullptr) == 1) {
        auto it = tcp_signal_rids_.find(ce.rid);
        if (it != tcp_signal_rids_.end()) {
          events[count].rid = ce.rid;
          events[count].tag = it->second.second;
          events[count].peer = it->second.first;
          events[count].failed = (ce.failed != 0);
          events[count].user_ctx = consume_user_ctx(ce.rid);
          tcp_signal_rids_.erase(it);
          pending_waits_count_.fetch_sub(1, std::memory_order_relaxed);
          ++count;
        } else {
          jring_mp_enqueue_bulk(put_completion_ring_, &ce, 1, nullptr);
          break;
        }
      }
    }
  }

  return count;
}

void Communicator::record_user_ctx(unsigned rid, uint32_t user_ctx) {
  std::lock_guard<std::mutex> lk(user_ctx_mu_);
  rid_to_user_ctx_[rid] = user_ctx;
}

uint32_t Communicator::consume_user_ctx(unsigned rid) {
  // Backend-tagged rid: the low 30 bits are the be_idx itself; no map
  // entry was ever recorded for it.
  if (rid & kRidTagMask) return rid & kRidBeIdxMask;
  std::lock_guard<std::mutex> lk(user_ctx_mu_);
  auto it = rid_to_user_ctx_.find(rid);
  if (it == rid_to_user_ctx_.end()) return 0;
  uint32_t ctx = it->second;
  rid_to_user_ctx_.erase(it);
  return ctx;
}

void Communicator::drain_ipc_signals() {
  if (!ipc_adapter_) return;
  for (int peer = 0; peer < world_size_; ++peer) {
    if (peer == global_rank_) continue;
    // O(1) arrival hint: two relaxed atomic loads instead of walking the
    // peer's ring (read_idx/write_idx + slot ready flags) on every poll.
    if (!ipc_adapter_->has_signal_arrivals(peer)) continue;
    uint64_t tags[64];
    size_t n = ipc_adapter_->drain_signal_tags(peer, tags, 64);
    if (n > 0) on_signals_received(peer, tags, n);
  }
}

bool Communicator::has_pending_signal_waits() const {
  return pending_waits_count_.load(std::memory_order_relaxed) > 0;
}

void Communicator::dump_signal_state() const {
  for (int peer = 0; peer < world_size_; ++peer) {
    if (peer == global_rank_) continue;
    std::lock_guard<std::mutex> lk(sig_maps_mu_);
    auto s_it = pending_signals_.find(peer);
    if (s_it != pending_signals_.end()) {
      std::string sample;
      int n = 0;
      for (uint64_t t : s_it->second) {
        if (n++ >= 10) {
          sample += "...";
          break;
        }
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%#lx ", (unsigned long)t);
        sample += buf;
      }
      std::fprintf(stderr,
                   "[sig-dump r%d] peer %d: %zu buffered arrivals: %s\n",
                   global_rank_, peer, s_it->second.size(), sample.c_str());
    }
    auto w_it = pending_signal_waits_.find(peer);
    if (w_it != pending_signal_waits_.end()) {
      size_t waits = 0;
      std::string sample;
      for (auto const& [tag, dq] : w_it->second) {
        waits += dq.size();
        if (sample.size() < 120) {
          sample += std::to_string((unsigned long)tag);
          sample += "x";
          sample += std::to_string(dq.size());
          sample += " ";
        }
      }
      std::fprintf(
          stderr,
          "[sig-dump r%d] peer %d: %zu posted waits across %zu tags: %s\n",
          global_rank_, peer, waits, w_it->second.size(), sample.c_str());
    }
  }
  {
    std::lock_guard<std::mutex> lk(flag_waits_mu_);
    if (!pending_flag_waits_.empty())
      std::fprintf(stderr, "[sig-dump r%d] %zu device-flag waits parked\n",
                   global_rank_, pending_flag_waits_.size());
  }
}

size_t Communicator::poll(unsigned* rids, size_t count) {
  drain_ipc_signals();

  size_t completed = 0;
  // O(count) lookup instead of the previous linear scan per completion
  // (O(count^2) worst case).
  std::unordered_set<unsigned> want(rids, rids + count);

  // Check data completion ring
  if (put_completion_ring_ && completed < count) {
    CompletionEvent ce;
    std::vector<CompletionEvent> stash;
    while (completed < count &&
           jring_mc_dequeue_bulk(put_completion_ring_, &ce, 1, nullptr) == 1) {
      if (want.erase(ce.rid)) {
        rids[completed++] = ce.rid;
      } else {
        stash.push_back(ce);
      }
    }
    for (auto& ev : stash)
      jring_mp_enqueue_bulk(put_completion_ring_, &ev, 1, nullptr);
  }

  // Check signal completion ring
  if (sig_wait_completion_ring_ && completed < count) {
    SignalCompletion sc;
    std::vector<SignalCompletion> stash;
    while (completed < count && jring_sc_dequeue_bulk(sig_wait_completion_ring_,
                                                      &sc, 1, nullptr) == 1) {
      if (want.erase(sc.rid)) {
        rids[completed++] = sc.rid;
      } else {
        stash.push_back(sc);
      }
    }
    for (auto& ev : stash)
      jring_mp_enqueue_bulk(sig_wait_completion_ring_, &ev, 1, nullptr);
  }

  return completed;
}

void Communicator::on_signal_received(int peer, uint64_t tag) {
  on_signals_received(peer, &tag, 1);
}

void Communicator::on_signals_received(int peer, uint64_t const* tags,
                                       size_t n) {
  // drain_ipc_signals never passes more than 64 tags per call.
  SignalCompletion done[64];
  size_t ndone = 0;
  {
    std::lock_guard<std::mutex> lk(sig_maps_mu_);
    for (size_t t = 0; t < n; ++t) {
      uint64_t tag = tags[t];
      SignalCompletion ev{};
      bool matched = false;
      bool consumed = false;
      auto it = pending_signal_waits_.find(peer);
      if (it != pending_signal_waits_.end()) {
        auto it2 = it->second.find(tag);
        if (it2 != it->second.end()) {
          auto& front = it2->second.front();
          consumed = true;
          if (front.second > 1) {
            // Counted wait (fused signal group): one arrival per tile.
            --front.second;
          } else {
            ev.rid = front.first;
            ev.tag = tag;
            ev.peer = peer;
            ev.failed = false;
            it2->second.erase(it2->second.begin());
            if (it2->second.empty()) it->second.erase(it2);
            matched = true;
          }
        }
      }
      if (!consumed) pending_signals_[peer].push_back(tag);
      if (uk_dbg_lvl() >= UK_DBG_LVL_TPT)
        std::fprintf(stderr, "[sig-recv r%d] peer=%d tag=%#lx %s\n",
                     global_rank_, peer, (unsigned long)tag,
                     matched ? "matched" : (consumed ? "counted" : "buffered"));
      if (matched) done[ndone++] = ev;
    }
  }
  if (ndone > 0)
    pending_waits_count_.fetch_sub(ndone, std::memory_order_relaxed);
  for (size_t i = 0; i < ndone; ++i) {
    if (jring_mp_enqueue_bulk(sig_wait_completion_ring_, &done[i], 1,
                              nullptr) != 1) {
      std::lock_guard<std::mutex> lk(sig_wait_overflow_mu_);
      sig_wait_overflow_.push_back(done[i]);
    }
  }
}

void Communicator::on_imm_received(int peer, uint32_t low32) {
  std::vector<SignalCompletion> done;
  {
    std::lock_guard<std::mutex> lk(sig_maps_mu_);
    auto& buf = buffered_imms_[peer];
    // The arrival either matches the head wait or joins the buffer. Never
    // drop it: a wait registered late (or a group continuing) must still
    // be able to consume the arrival. Push to the FRONT when it matches
    // the head so the drain below consumes it in order.
    auto it = pending_imm_waits_.find(peer);
    if (it != pending_imm_waits_.end() && !it->second.empty() &&
        it->second.front().low32 == low32) {
      buf.push_front(low32);
    } else {
      buf.push_back(low32);
    }
    // Drain buffered arrivals into the head wait while they match. This
    // also catches up waits that were parked behind a head mismatch.
    while (true) {
      auto wit = pending_imm_waits_.find(peer);
      if (wit == pending_imm_waits_.end() || wit->second.empty()) break;
      if (buf.empty() || buf.front() != wit->second.front().low32) break;
      buf.pop_front();
      ImmWait& h = wit->second.front();
      if (--h.remaining == 0) {
        SignalCompletion ev{h.rid, h.tag, peer, false, 0};
        done.push_back(ev);
        wit->second.pop_front();
        if (wit->second.empty()) pending_imm_waits_.erase(wit);
      }
    }
    if (buf.empty()) buffered_imms_.erase(peer);
  }
  if (!done.empty())
    pending_waits_count_.fetch_sub(done.size(), std::memory_order_relaxed);
  for (auto const& ev : done) {
    if (jring_mp_enqueue_bulk(sig_wait_completion_ring_, &ev, 1, nullptr) !=
        1) {
      std::lock_guard<std::mutex> lk(sig_wait_overflow_mu_);
      sig_wait_overflow_.push_back(ev);
    }
  }
}

bool Communicator::reg_mr(uint32_t buffer_id, void* local_buf, size_t len,
                          bool publish) {
  if (buffer_id == 0 || local_buf == nullptr || len == 0) return false;

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    auto it = local_buffer_to_mr_.find(buffer_id);
    if (it != local_buffer_to_mr_.end()) {
      // Idempotent re-registration of the same buffer is harmless;
      // reject only when the pointer or size changed, which would
      // race the remote side's OOB-based generation tracking.
      if (it->second.address == reinterpret_cast<uintptr_t>(local_buf) &&
          it->second.length == len) {
        return true;  // same buffer, no-op
      }
      std::cerr << "[WARN] reg_mr: buffer_id " << buffer_id
                << " already registered with different pointer/size;"
                << " call dereg_mr first\n";
      return false;
    }
  }

  MR mr = mr_manager_.create_local_mr(buffer_id, local_buf, len).mr;
  if (mr.address == 0 || mr.length == 0) return false;

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    local_buffer_to_mr_[buffer_id] = mr;
  }
  put_cache_bump();  // local MR pointer may feed cached puts

  if (rdma_adapter_ && rdma_adapter_->is_initialized()) {
    if (!ensure_rdma_memory_registered(buffer_id, local_buf, len)) {
      // RDMA registration failure is NOT fatal: the buffer can still be
      // used over IPC (same-host puts never need an rkey). Publish the MR
      // entry with rkey=0 — the peer's wait_mr sees a valid entry, IPC
      // resolve works, and a cross-node RDMA put rejects cleanly at send
      // time (send_put_async_with_rid requires a nonzero rkey) instead of
      // silently publishing an empty entry that breaks IPC too.
      std::fprintf(stderr,
                   "[reg_mr r%d] WARN RDMA reg buf=%u ptr=%p len=%zu failed; "
                   "continuing with rkey=0 (IPC-only for this buffer)\n",
                   global_rank_, buffer_id, local_buf, len);
    } else {
      uint32_t rkey = rdma_adapter_->get_memory_rkey(buffer_id);
      if (rkey != 0) {
        mr.key = rkey;
        std::lock_guard<std::mutex> lk(resource_mu_);
        local_buffer_to_mr_[buffer_id].key = rkey;
      }
    }
  }

  if (!publish || !exchanger_client_ || !exchanger_client_->valid()) {
    return true;
  }

  NamedMRInfos payload{};
  payload.generation = mr_generation_.fetch_add(1, std::memory_order_relaxed);
  payload.entries.push_back(NamedMR{buffer_id, mr});
  bool ok = oob_put(*exchanger_client_, oob_namespace(),
                    mr_global_buffer_key(global_rank_, buffer_id), payload);
  if (!ok)
    std::fprintf(stderr,
                 "[reg_mr r%d] ERROR: failed to publish mr:rank:%d:buf:%u "
                 "(exchanger store full?)\n",
                 global_rank_, global_rank_, buffer_id);
  return ok;
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

  if (registered_id != 0 && rdma_adapter_ && rdma_adapter_->is_initialized()) {
    std::lock_guard<std::mutex> lk(rdma_reg_mu_);
    if (rdma_registered_mrs_.erase(registered_id) > 0) {
      rdma_adapter_->deregister_memory(registered_id);
    }
    rdma_direct_reg_failed_mrs_.erase(registered_id);
  }
  if (found) (void)mr_manager_.delete_mr(buffer_id);
  if (found) put_cache_bump();  // cached local base is now stale

  if (exchanger_client_ && exchanger_client_->valid()) {
    NamedMRInfos empty{};
    empty.generation = mr_generation_.fetch_add(1, std::memory_order_relaxed);
    oob_put(*exchanger_client_, oob_namespace(),
            mr_global_buffer_key(global_rank_, buffer_id), empty);
  }

  return true;
}

bool Communicator::wait_mr(int owner_rank, uint32_t buffer_id, int timeout_ms) {
  if (buffer_id == 0) return false;
  if (owner_rank == global_rank_) {
    std::lock_guard<std::mutex> lk(resource_mu_);
    return local_buffer_to_mr_.find(buffer_id) != local_buffer_to_mr_.end();
  }
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  uint64_t last_gen = 0;
  bool have_last_gen = false;
  {
    std::lock_guard<std::mutex> lk(mr_gen_mu_);
    auto it =
        last_mr_generation_.find((uint64_t(owner_rank) << 32) | buffer_id);
    if (it != last_mr_generation_.end()) {
      last_gen = it->second;
      have_last_gen = true;
    }
  }

  constexpr int kPollMs = 10;
  int elapsed = 0;
  NamedMRInfos payload{};
  auto fail = [&](char const* why) {
    std::fprintf(stderr,
                 "[wait_mr r%d] FAIL key=mr:rank:%d:buf:%u (%s)\n",
                 global_rank_, owner_rank, buffer_id, why);
    return false;
  };
  while (true) {
    int poll_to =
        (timeout_ms < 0) ? kPollMs : std::min(kPollMs, timeout_ms - elapsed);
    if (!oob_get(*exchanger_client_, oob_namespace(),
                 mr_global_buffer_key(owner_rank, buffer_id), payload,
                 poll_to)) {
      if (timeout_ms >= 0) {
        elapsed += kPollMs;
        if (elapsed >= timeout_ms) return fail("key not found");
      }
      continue;
    }

    if (payload.entries.empty()) {
      // An EMPTY entry is an explicit failure signal (the owner failed
      // to register the buffer and published a placeholder for
      // fail-fast). Never retry it — the entry is not going to change.
      return fail("empty entries");
    }

    // First-ever resolve must be accepted unconditionally: the initial publish
    // generation (0) equals the default last_gen (0), so a pure generation
    // comparison would loop forever. De-duplicate by generation only after
    // that.
    if (!have_last_gen || payload.generation != last_gen) break;

    // Same generation, check if we already have it cached (CCL repeat calls)
    {
      std::lock_guard<std::mutex> lk(resource_mu_);
      auto it = remote_buffer_to_mr_.find(owner_rank);
      if (it != remote_buffer_to_mr_.end()) {
        auto jt = it->second.find(buffer_id);
        if (jt != it->second.end() && jt->second.key != 0) return true;
      }
    }

    if (timeout_ms >= 0) {
      elapsed += kPollMs;
      if (elapsed >= timeout_ms) return fail("stale generation, no cached rkey");
    }
  }

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
  {
    std::lock_guard<std::mutex> lk(mr_gen_mu_);
    last_mr_generation_[(uint64_t(owner_rank) << 32) | buffer_id] =
        payload.generation;
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

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    auto it = local_buffer_to_ipc_.find(buffer_id);
    if (it != local_buffer_to_ipc_.end()) {
      if (it->second.bytes == len && it->second.is_local) {
        // Same size, still local — idempotent re-registration.
        return true;
      }
      std::cerr << "[WARN] reg_ipc: buffer_id " << buffer_id
                << " already registered; call dereg_ipc first\n";
      return false;
    }
  }

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
  put_cache_bump();  // cached remote base (for dst_buf) may be stale

  if (!publish || !exchanger_client_ || !exchanger_client_->valid()) {
    return true;
  }

  IpcBufferInfo info{};
  info.generation = ipc_generation_.fetch_add(1, std::memory_order_relaxed);
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
  if (found) put_cache_bump();  // cached remote base is now stale

  if (exchanger_client_ && exchanger_client_->valid()) {
    IpcBufferInfo empty{};
    empty.generation = ipc_generation_.fetch_add(1, std::memory_order_relaxed);
    empty.valid = false;
    oob_put(*exchanger_client_, oob_namespace(),
            ipc_global_buffer_key(global_rank_, buffer_id), empty);
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

  uint64_t last_gen = 0;
  bool have_last_gen = false;
  {
    std::lock_guard<std::mutex> lk(mr_gen_mu_);
    auto it =
        last_ipc_generation_.find((uint64_t(owner_rank) << 32) | buffer_id);
    if (it != last_ipc_generation_.end()) {
      last_gen = it->second;
      have_last_gen = true;
    }
  }

  constexpr int kPollMs = 10;
  int elapsed = 0;
  IpcBufferInfo info{};
  while (true) {
    int poll_to =
        (timeout_ms < 0) ? kPollMs : std::min(kPollMs, timeout_ms - elapsed);
    if (!oob_get(*exchanger_client_, oob_namespace(),
                 ipc_global_buffer_key(owner_rank, buffer_id), info, poll_to)) {
      if (timeout_ms >= 0) {
        elapsed += kPollMs;
        if (elapsed >= timeout_ms) return false;
      }
      continue;
    }

    // First-ever resolve, or a newer publish. The initial publish generation
    // (0) equals the default last_gen (0); without the have_last_gen guard a
    // re-resolve of an already-resolved buffer (generation unchanged) would
    // spin forever. A peer may deliberately publish an INVALID entry
    // (reg_ipc with nullptr/0) as a marker; on a first-ever wait that is a
    // resolution — get_ipc() reports the "published but unmappable" state.
    // Only a NEW-generation invalid entry after a prior resolution is a
    // deregister tombstone: keep waiting, and never record its generation
    // (recording it would let the reuse branch below short-circuit a later
    // re-publish with the stale cached item).
    if (!have_last_gen || info.generation != last_gen) {
      if (!info.valid) {
        if (have_last_gen) {
          if (timeout_ms >= 0) {
            elapsed += kPollMs;
            if (elapsed >= timeout_ms) return false;
          }
          continue;  // deregister tombstone — wait for a valid re-publish
        }
        put_cache_bump();  // remote IPC meta may have changed
        break;
      }
      put_cache_bump();  // remote IPC meta may have changed
      break;
    }

    // Same generation: reuse only with a cached VALID item (mirrors
    // wait_mr's cached-rkey check).
    if (ipc_manager_.get_ipc(owner_rank, buffer_id).valid) return true;
    if (timeout_ms >= 0) {
      elapsed += kPollMs;
      if (elapsed >= timeout_ms) return false;
    }
  }

  IPCItem state{};
  state.handle = info.handle;
  state.base_offset = static_cast<uintptr_t>(info.base_offset);
  state.bytes = static_cast<size_t>(info.bytes);
  state.device_idx = info.device_idx;
  state.valid = info.valid;
  bool ok = ipc_manager_.register_remote_ipc(owner_rank, buffer_id, state);
  if (ok) {
    std::lock_guard<std::mutex> lk(mr_gen_mu_);
    last_ipc_generation_[(uint64_t(owner_rank) << 32) | buffer_id] =
        info.generation;
  }
  return ok;
}

std::string Communicator::ipc_open_error_message(int owner_rank,
                                                 uint32_t buffer_id,
                                                 IPCItem const& item,
                                                 gpuError_t err) const {
  std::ostringstream oss;
  oss << "failed to open remote IPC mem handle"
      << " owner_rank=" << owner_rank << " buffer_id=" << buffer_id
      << " local_gpu=" << local_gpu_idx_
      << " remote_device_idx=" << item.device_idx << " bytes=" << item.bytes
      << " base_offset=" << item.base_offset << " err=" << static_cast<int>(err)
      << " (" << gpuGetErrorString(err) << ")";
  if (item.device_idx >= 0 && item.device_idx != local_gpu_idx_) {
    int can_access_peer = -1;
    gpuError_t access_err = gpuDeviceCanAccessPeer(
        &can_access_peer, local_gpu_idx_, item.device_idx);
    oss << " peer_access=" << can_access_peer
        << " peer_access_err=" << static_cast<int>(access_err) << " ("
        << gpuGetErrorString(access_err) << ")";
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

bool Communicator::open_remote_ipc_mapping(int owner_rank, uint32_t buffer_id,
                                           IPCItem& item) {
  if (owner_rank == global_rank_ || item.direct_ptr != nullptr) return true;

  // Serialize every remote IPC mapping open across processes on this
  // host. Concurrent bidirectional cudaIpcOpenMemHandle calls race on
  // A40-class dual-GPU systems: one direction can return a mapping that
  // accepts writes but points at the wrong physical memory, while the
  // other direction fails with cudaErrorInvalidResourceHandle. flock
  // auto-releases if a peer dies mid-open, so a crashed rank cannot
  // deadlock the next run.
  char lock_path[128];
  std::snprintf(lock_path, sizeof(lock_path), "/tmp/uk_ccl_ipc_open_%d.lock",
                static_cast<int>(::getuid()));
  int lfd = ::open(lock_path, O_CREAT | O_RDWR, 0666);
  if (lfd < 0) {
    std::cerr << "[ipc-open r" << global_rank_ << "] lock open failed: "
              << std::strerror(errno) << std::endl;
    return false;
  }
  struct flock fl {};
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_SET;
  if (::fcntl(lfd, F_SETLKW, &fl) != 0) {
    std::cerr << "[ipc-open r" << global_rank_ << "] lock wait failed: "
              << std::strerror(errno) << std::endl;
    ::close(lfd);
    return false;
  }

  bool ok = true;
  int original_device = -1;
  if (gpuGetDevice(&original_device) != gpuSuccess) {
    ok = false;
  } else if (gpuSetDevice(local_gpu_idx_) != gpuSuccess) {
    ok = false;
  } else {
    // Explicitly enable peer access before opening: lazy enablement is
    // asynchronous and a mapping used before it completes can accept
    // writes that never reach the owner's pages (observed on A40 pairs).
    // gpuDeviceSynchronize after open would also work but deadlocks when
    // the persistent device worker is already running on a non-blocking
    // stream; enable + open is stable and needs no device-wide sync.
    // The flock above serializes both directions, so the two peers never
    // race their enable calls against each other.
    if (item.device_idx >= 0) {
      gpuError_t pa = gpuDeviceEnablePeerAccess(item.device_idx, 0);
      // Consume the sticky error state: when the access is already
      // enabled, the runtime records cudaErrorPeerAccessAlreadyEnabled
      // and a later cudaGetLastError (e.g. the device worker's post-launch
      // check) would abort on it.
      if (pa != gpuSuccess) (void)gpuGetLastError();
      if (pa != gpuSuccess && pa != gpuErrorPeerAccessAlreadyEnabled) {
        std::cerr << "[ipc-open r" << global_rank_
                  << "] peer enable dev" << item.device_idx << " failed: "
                  << gpuGetErrorString(pa) << std::endl;
        ok = false;
      }
    }
    if (ok) {
      gpuError_t open_err = gpuIpcOpenMemHandle(
          &item.direct_ptr, item.handle, gpuIpcMemLazyEnablePeerAccess);
      UK_DBG(UK_DBG_LVL_TPT,
             "[ipc-open r%d] owner=%d buf=%u handle=%02x%02x%02x%02x "
             "direct=%p dev=%d err=%d",
             global_rank_, owner_rank, buffer_id,
             (unsigned char)item.handle.reserved[0],
             (unsigned char)item.handle.reserved[1],
             (unsigned char)item.handle.reserved[2],
             (unsigned char)item.handle.reserved[3], item.direct_ptr,
             item.device_idx, static_cast<int>(open_err));
      if (open_err != gpuSuccess) {
        std::cerr << "[ERROR] "
                  << ipc_open_error_message(owner_rank, buffer_id, item,
                                            open_err)
                  << std::endl;
        ok = false;
      }
    }
    if (gpuSetDevice(original_device) != gpuSuccess) ok = false;
  }

  fl.l_type = F_UNLCK;
  if (::fcntl(lfd, F_SETLK, &fl) != 0) {
    std::cerr << "[ipc-open r" << global_rank_ << "] unlock failed: "
              << std::strerror(errno) << std::endl;
  }
  ::close(lfd);
  return ok && item.direct_ptr != nullptr;
}

IPCItem Communicator::get_ipc(int owner_rank, uint32_t buffer_id) {
  if (owner_rank == global_rank_) return get_ipc(buffer_id);
  IPCItem item = ipc_manager_.get_ipc(owner_rank, buffer_id);
  if (!item.valid) {
    throw std::runtime_error("remote IPC not found for buffer_id");
  }
  if (item.direct_ptr == nullptr) {
    if (!open_remote_ipc_mapping(owner_rank, buffer_id, item)) {
      throw std::runtime_error(
          "remote IPC open failed for buffer_id " + std::to_string(buffer_id));
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
    if (!open_remote_ipc_mapping(remote_rank, remote_buffer_id, item))
      return false;
    if (!ipc_manager_.register_remote_ipc(remote_rank, remote_buffer_id,
                                          item)) {
      return false;
    }
  }

  if (offset > item.bytes || bytes > item.bytes - offset) return false;
  uintptr_t const base = reinterpret_cast<uintptr_t>(item.direct_ptr);
  uintptr_t const resolved = base + item.base_offset + offset;
  *out_ptr = reinterpret_cast<void*>(resolved);
  static int dbg_cnt = 0;
  if (dbg_cnt++ < 12)
    UK_DBG(UK_DBG_LVL_TPT,
           "[resolve r%d] peer=%d buf=%u off=%zu base_off=%llu base=%#lx "
           "resolved=%#lx bytes=%zu",
           global_rank_, remote_rank, remote_buffer_id, offset,
           (unsigned long long)item.base_offset, (unsigned long)base,
           (unsigned long)resolved, bytes);
  if (out_device_idx != nullptr) {
    *out_device_idx = item.device_idx;
  }
  return true;
}

bool Communicator::register_buffer(uint32_t buffer_id, void* ptr, size_t len) {
  bool ok = reg_mr(buffer_id, ptr, len, true);

  // Only exchange IPC handles if there is at least one IPC-connected peer.
  bool has_ipc_peer = false;
  for (int r = 0; r < world_size_; ++r) {
    if (r != global_rank_ && has_put_path(r, PeerTransportKind::Ipc)) {
      has_ipc_peer = true;
      break;
    }
  }
  if (has_ipc_peer) ok = reg_ipc(buffer_id, ptr, len, true) && ok;

  return ok;
}

bool Communicator::resolve_remote_buffer(int peer_rank, uint32_t buffer_id,
                                         int timeout_ms) {
  // Same-host buffers are normally moved over IPC, which needs no rkey.
  // Resolve the RDMA MR best-effort as well: a forced RDMA put
  // (UK_CCL_PUT_PATH=rdma) still needs remote_buffer_to_mr_ populated,
  // otherwise get_mr() throws "remote MR rank cache not found". A failed
  // MR resolve is tolerated — small allocations can register with rkey=0,
  // and the RDMA send path then rejects the buffer cleanly instead.
  if (same_host(peer_rank)) {
    if (!wait_ipc(peer_rank, buffer_id, timeout_ms)) {
      std::fprintf(stderr, "[resolve r%d] TIMEOUT peer=%d buf=%u (ipc)\n",
                   global_rank_, peer_rank, buffer_id);
      return false;
    }
    (void)wait_mr(peer_rank, buffer_id, timeout_ms);
    return true;
  }
  bool ok_mr = wait_mr(peer_rank, buffer_id, timeout_ms);
  bool ok_ipc = wait_ipc(peer_rank, buffer_id, timeout_ms);
  if (!ok_mr || !ok_ipc) {
    std::fprintf(stderr, "[resolve r%d] TIMEOUT peer=%d buf=%u (mr=%d ipc=%d)\n",
                 global_rank_, peer_rank, buffer_id, (int)ok_mr, (int)ok_ipc);
    return false;
  }
  // Cross-node: RDMA is required, so wait for a nonzero rkey.
  for (int retry = 0; retry < 10; ++retry) {
    MR mr = get_mr(peer_rank, buffer_id);
    if (mr.key != 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!wait_mr(peer_rank, buffer_id, timeout_ms >= 0 ? timeout_ms : 30000))
      break;
    {
      std::lock_guard<std::mutex> lk(mr_gen_mu_);
      last_mr_generation_.erase((uint64_t(peer_rank) << 32) | buffer_id);
    }
  }
  return true;
}

}  // namespace Transport
}  // namespace UKernel
