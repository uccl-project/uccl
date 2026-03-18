#include "oob.h"

namespace UKernel {
namespace Transport {

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

}  // namespace Transport
}  // namespace UKernel
