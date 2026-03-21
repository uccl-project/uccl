#include "selector.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace UKernel {
namespace CCL {

namespace {

std::string to_lower(char const* value) {
  std::string out = value ? value : "";
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

PeerRuntimeCapabilities peer_caps_for(RuntimeCapabilities const& caps,
                                      int peer_rank) {
  if (peer_rank < 0 ||
      static_cast<size_t>(peer_rank) >= caps.peers.size()) {
    return PeerRuntimeCapabilities{};
  }
  return caps.peers[static_cast<size_t>(peer_rank)];
}

}  // namespace

BackendKind resolve_backend_kind(BackendKind requested,
                                 ExecutionOp const& op,
                                 RuntimeCapabilities const& caps,
                                 BackendSelectorConfig const& cfg) {
  if (char const* env = std::getenv("UKERNEL_CCL_BACKEND")) {
    std::string override_value = to_lower(env);
    if (override_value == "transport") return BackendKind::Transport;
    if (override_value == "device") return BackendKind::Device;
  }

  if (requested != BackendKind::Auto) return requested;

  switch (op.kind) {
    case ExecutionOpKind::Send:
    case ExecutionOpKind::Recv:
      return BackendKind::Transport;
    case ExecutionOpKind::Reduce:
      return BackendKind::Device;
    case ExecutionOpKind::Copy: {
      if (op.src.slot == MemorySlot::RecvStaging ||
          op.dst.slot == MemorySlot::RecvStaging) {
        return BackendKind::Device;
      }

      int remote_rank = -1;
      if (op.src.slot == MemorySlot::SymmetricTensor && op.src.rank >= 0) {
        remote_rank = op.src.rank;
      } else if (op.dst.slot == MemorySlot::SymmetricTensor && op.dst.rank >= 0) {
        remote_rank = op.dst.rank;
      }

      PeerRuntimeCapabilities peer = peer_caps_for(caps, remote_rank);
      if (peer.peer_accessible) {
        if (!cfg.prefer_transport_for_large_same_node_copy ||
            op.chunk.size_bytes < cfg.transport_copy_threshold_bytes) {
          return BackendKind::Device;
        }
      }
      return remote_rank >= 0 ? BackendKind::Transport : BackendKind::Device;
    }
    case ExecutionOpKind::EventWait:
    case ExecutionOpKind::Barrier:
      return BackendKind::Device;
  }
  return BackendKind::Device;
}

}  // namespace CCL
}  // namespace UKernel
