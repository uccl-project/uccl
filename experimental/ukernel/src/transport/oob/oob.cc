#include "oob.h"

namespace UKernel {
namespace Transport {

bool Exchanger::wait_raw(std::string_view key, std::string& value,
                         WaitOptions const& options) {
  if (options.max_retries == 0) {
    return get_raw(key, value);
  }

  int const delay_ms = options.delay_ms > 0 ? options.delay_ms : 1;
  if (options.max_retries > 0) {
    for (int i = 0; i < options.max_retries; ++i) {
      if (!valid()) return false;
      if (get_raw(key, value)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    return false;
  }

  while (valid()) {
    if (get_raw(key, value)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }
  return false;
}

char const* peer_transport_kind_name(PeerTransportKind kind) {
  switch (kind) {
    case PeerTransportKind::Unknown:
      return "unknown";
    case PeerTransportKind::Uccl:
      return "uccl";
    case PeerTransportKind::Ipc:
      return "ipc";
    case PeerTransportKind::Tcp:
      return "tcp";
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
    if (!local_meta.rdma_capable || !peer_meta.rdma_capable) {
      throw std::invalid_argument(
          "preferred UCCL transport requires RDMA-capable peers");
    }
    return PeerTransportKind::Uccl;
  }
  if (config.preferred_transport == PreferredTransport::Tcp) {
    return PeerTransportKind::Tcp;
  }
  if (local_meta.host_id == peer_meta.host_id) {
    return PeerTransportKind::Ipc;
  }
  return (local_meta.rdma_capable && peer_meta.rdma_capable)
             ? PeerTransportKind::Uccl
             : PeerTransportKind::Tcp;
}

}  // namespace Transport
}  // namespace UKernel
