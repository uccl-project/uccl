#include "oob.h"

namespace UKernel {
namespace Transport {

char const* peer_transport_kind_name(PeerTransportKind kind) {
  switch (kind) {
    case PeerTransportKind::Unknown:
      return "unknown";
    case PeerTransportKind::Uccl:
      return "uccl";
    case PeerTransportKind::Rdma:
      return "rdma";
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
  if (config.preferred_transport == PreferredTransport::Rdma) {
    if (!local_meta.rdma_capable || !peer_meta.rdma_capable) {
      throw std::invalid_argument(
          "preferred RDMA transport requires RDMA-capable peers");
    }
    return PeerTransportKind::Rdma;
  }
  if (config.preferred_transport == PreferredTransport::Tcp) {
    return PeerTransportKind::Tcp;
  }
  if (local_meta.host_id == peer_meta.host_id) {
    return PeerTransportKind::Ipc;
  }
  return (local_meta.rdma_capable && peer_meta.rdma_capable)
             ? PeerTransportKind::Rdma
             : PeerTransportKind::Tcp;
}

}  // namespace Transport
}  // namespace UKernel
