#include "transport_engine.h"
#include "util/net.h"
#include <arpa/inet.h>
#include <cstdlib>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <stdexcept>

namespace UKernel {
namespace Transport {

namespace {

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

std::string find_ifname_for_remote_subnet(std::string const& ip) {
  if (is_unspecified_ip(ip)) return {};

  uccl::socketAddress remote_addr{};
  uccl::socketAddress local_addr{};
  char if_name[MAX_IF_NAME_SIZE + 1] = {};
  std::string ip_port = ip + ":1";

  if (!uccl::get_socket_addr_from_string(&remote_addr, ip_port.c_str())) {
    return {};
  }

  int found = uccl::find_interface_match_subnet(
      if_name, &local_addr, &remote_addr, MAX_IF_NAME_SIZE, 1);
  if (found != 1) return {};

  return if_name;
}

void maybe_configure_uccl_socket_ifname(std::string const& remote_hint_ip,
                                        std::string const& local_hint_ip) {
  if (has_env_value("UCCL_SOCKET_IFNAME") ||
      has_env_value("NCCL_SOCKET_IFNAME")) {
    return;
  }

  std::string ifname = find_ifname_for_remote_subnet(remote_hint_ip);
  if (ifname.empty()) {
    ifname = find_ifname_for_local_ip(local_hint_ip);
  }
  if (ifname.empty()) return;

  ::setenv("UCCL_SOCKET_IFNAME", ifname.c_str(), 0);
  std::cout << "[INFO] Auto-selected UCCL_SOCKET_IFNAME=" << ifname;
  if (!is_unspecified_ip(remote_hint_ip)) {
    std::cout << " using remote hint " << remote_hint_ip;
  } else if (!is_unspecified_ip(local_hint_ip)) {
    std::cout << " using local hint " << local_hint_ip;
  }
  std::cout << std::endl;
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

}  // namespace

char const* peer_transport_kind_name(PeerTransportKind kind) {
  switch (kind) {
    case PeerTransportKind::Uccl:
      return "uccl";
    case PeerTransportKind::Ipc:
      return "ipc";
  }
  return "unknown";
}

PeerTransportKind resolve_peer_transport_kind(
    CommunicatorConfig const& config, CommunicatorMeta const& local_meta,
    CommunicatorMeta const& peer_meta) {
  if (config.preferred_transport == PreferredTransport::Ipc) {
    if (local_meta.host_id != peer_meta.host_id) {
      throw std::invalid_argument(
          "preferred IPC transport requires same-host peer");
    }
    return PeerTransportKind::Ipc;
  }
  if (config.preferred_transport == PreferredTransport::Uccl) {
    return PeerTransportKind::Uccl;
  }
  return local_meta.host_id == peer_meta.host_id ? PeerTransportKind::Ipc
                                                 : PeerTransportKind::Uccl;
}

UcclTransportEngine::UcclTransportEngine(int local_gpu_idx, int world_size)
    : local_gpu_idx_(local_gpu_idx), world_size_(world_size) {}

bool UcclTransportEngine::ensure_adapter(
    std::shared_ptr<CommunicatorConfig> const& config,
    CommunicatorMeta const& local_meta, CommunicatorMeta const& peer_meta) {
  if (adapter_) return true;

  UcclTransportConfig uccl_config;
  uccl_config.local_ip = local_meta.ip;
  maybe_configure_uccl_socket_ifname(get_uccl_remote_hint_ip(config, peer_meta),
                                     local_meta.ip);
  adapter_ = std::make_unique<UcclTransportAdapter>(local_gpu_idx_, world_size_,
                                                    uccl_config);
  return true;
}

bool UcclTransportEngine::connect_to_peer(
    int global_rank, int peer_rank,
    std::shared_ptr<CommunicatorConfig> const& config,
    std::shared_ptr<Exchanger> const& exchanger,
    CommunicatorMeta const& local_meta, CommunicatorMeta const& peer_meta) {
  if (!ensure_adapter(config, local_meta, peer_meta)) return false;
  if (adapter_->has_send_peer(peer_rank)) return true;

  int dev_idx = adapter_->get_best_dev_idx(local_gpu_idx_);
  int gpu_idx = local_gpu_idx_;
  uint16_t local_port = adapter_->get_p2p_listen_port(dev_idx);
  std::string local_ip_addr = adapter_->get_p2p_listen_ip(dev_idx);

  UCCLP2PInfo local_p2p_info(local_ip_addr, local_port, dev_idx, gpu_idx);
  std::string p2p_key = "uccl_p2p_info_" + std::to_string(global_rank);
  std::string peer_p2p_key = "uccl_p2p_info_" + std::to_string(peer_rank);

  if (!exchanger->publish(p2p_key, local_p2p_info)) return false;

  UCCLP2PInfo remote_p2p_info;
  if (!exchanger->wait_and_fetch(peer_p2p_key, remote_p2p_info, -1)) {
    return false;
  }

  std::cout << "[INFO] Rank " << global_rank << " P2P port " << local_port
            << " (GPU " << gpu_idx << ", dev " << dev_idx << ")"
            << " -> Rank " << peer_rank << " P2P port "
            << remote_p2p_info.port << " (GPU " << remote_p2p_info.gpu_idx
            << ", dev " << remote_p2p_info.dev_idx << ")" << std::endl;

  return adapter_->connect_to_peer(peer_rank, remote_p2p_info.ip,
                                   remote_p2p_info.port, dev_idx, gpu_idx,
                                   remote_p2p_info.dev_idx,
                                   remote_p2p_info.gpu_idx);
}

bool UcclTransportEngine::accept_from_peer(
    int global_rank, int peer_rank,
    std::shared_ptr<CommunicatorConfig> const& config,
    std::shared_ptr<Exchanger> const& exchanger,
    CommunicatorMeta const& local_meta, CommunicatorMeta const& peer_meta) {
  if (!ensure_adapter(config, local_meta, peer_meta)) return false;
  if (adapter_->has_recv_peer(peer_rank)) return true;

  int dev_idx = adapter_->get_best_dev_idx(local_gpu_idx_);
  int gpu_idx = local_gpu_idx_;
  uint16_t local_port = adapter_->get_p2p_listen_port(dev_idx);
  std::string local_ip_addr = adapter_->get_p2p_listen_ip(dev_idx);

  UCCLP2PInfo local_p2p_info(local_ip_addr, local_port, dev_idx, gpu_idx);
  std::string p2p_key = "uccl_p2p_info_" + std::to_string(global_rank);
  std::string peer_p2p_key = "uccl_p2p_info_" + std::to_string(peer_rank);

  UCCLP2PInfo remote_p2p_info;
  if (!exchanger->wait_and_fetch(peer_p2p_key, remote_p2p_info, -1)) {
    return false;
  }
  if (!exchanger->publish(p2p_key, local_p2p_info)) return false;

  std::cout << "[INFO] Rank " << global_rank << " P2P port " << local_port
            << " (GPU " << gpu_idx << ", dev " << dev_idx << ")"
            << " <- Rank " << peer_rank << " P2P port "
            << remote_p2p_info.port << " (GPU " << remote_p2p_info.gpu_idx
            << ", dev " << remote_p2p_info.dev_idx << ")" << std::endl;

  return adapter_->accept_from_peer(peer_rank);
}

bool UcclTransportEngine::register_memory(uint64_t mr_id, void* ptr,
                                          size_t len) {
  return adapter_ && adapter_->register_memory(mr_id, ptr, len);
}

void UcclTransportEngine::deregister_memory(uint64_t mr_id) {
  if (adapter_) adapter_->deregister_memory(mr_id);
}

int UcclTransportEngine::send_async(int peer_rank, void* local_ptr, size_t len,
                                    uint64_t local_mr_id,
                                    uint64_t remote_mr_id,
                                    uint64_t request_id) {
  return adapter_ ? adapter_->send_async(peer_rank, local_ptr, len, local_mr_id,
                                         remote_mr_id, request_id)
                  : -1;
}

int UcclTransportEngine::recv_async(int peer_rank, void* local_ptr, size_t len,
                                    uint64_t local_mr_id,
                                    uint64_t request_id) {
  return adapter_ ? adapter_->recv_async(peer_rank, local_ptr, len, local_mr_id,
                                         request_id)
                  : -1;
}

bool UcclTransportEngine::poll_completion(uint64_t request_id) {
  return adapter_ && adapter_->poll_completion(request_id);
}

bool UcclTransportEngine::wait_completion(uint64_t request_id) {
  return adapter_ && adapter_->wait_completion(request_id);
}

}  // namespace Transport
}  // namespace UKernel
