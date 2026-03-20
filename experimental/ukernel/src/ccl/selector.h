#pragma once

#include <cstdint>

namespace UKernel {
namespace CCL {

enum class BackendKind : uint32_t { Auto, Rdma, Cpu };

struct RuntimeCapabilities {
  bool is_same_node = true;
  bool has_peer_access = false;
  bool has_nvlink = false;
  bool has_copy_engine_path = false;
  bool supports_rdma = false;
};

struct BackendSelectorConfig {
  bool prefer_ce_for_large_copy = true;
  uint64_t copy_engine_threshold_bytes = 256 * 1024;
};

BackendKind resolve_backend_kind(BackendKind requested, bool is_copy,
                                 uint64_t bytes,
                                 RuntimeCapabilities const& caps,
                                 BackendSelectorConfig const& cfg);

}  // namespace CCL
}  // namespace UKernel
