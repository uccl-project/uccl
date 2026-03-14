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

}  // namespace

BackendKind resolve_backend_kind(BackendKind requested, bool is_copy,
                                 uint64_t bytes,
                                 RuntimeCapabilities const& caps,
                                 BackendSelectorConfig const& cfg) {
  if (char const* env = std::getenv("UKERNEL_CCL_CPU_BACKEND")) {
    std::string override_value = to_lower(env);
    if (override_value == "rdma") return BackendKind::Rdma;
    if (override_value == "ce") return BackendKind::Ce;
    if (override_value == "pk") return BackendKind::Pk;
  }

  if (requested != BackendKind::Auto) return requested;
  if (caps.supports_rdma && !caps.is_same_node) return BackendKind::Rdma;
  if (is_copy && cfg.prefer_ce_for_large_copy && caps.has_copy_engine_path &&
      bytes >= cfg.copy_engine_threshold_bytes) {
    return BackendKind::Ce;
  }
  return BackendKind::Pk;
}

}  // namespace CCL
}  // namespace UKernel
