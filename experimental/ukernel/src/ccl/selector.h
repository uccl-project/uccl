#pragma once

#include "plan.h"
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

enum class BackendKind : uint32_t { Auto, Device, Transport };

struct PeerRuntimeCapabilities {
  bool same_node = false;
  bool peer_accessible = false;
  bool has_nvlink = false;
  bool has_copy_engine_path = false;
  bool supports_rdma = false;
};

struct RuntimeCapabilities {
  std::vector<PeerRuntimeCapabilities> peers;
};

struct BackendSelectorConfig {
  bool prefer_transport_for_large_same_node_copy = true;
  uint64_t transport_copy_threshold_bytes = 256 * 1024;
};

BackendKind resolve_backend_kind(BackendKind requested,
                                 ExecutionOp const& op,
                                 RuntimeCapabilities const& caps,
                                 BackendSelectorConfig const& cfg);

}  // namespace CCL
}  // namespace UKernel
