#include "task.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace UKernel {
namespace Compute {

namespace {

std::string to_lower(char const* value) {
  std::string out = value ? value : "";
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

}  // namespace

CpuBackendKind resolve_cpu_backend_kind(CpuBackendKind requested, bool is_copy,
                                        uint64_t bytes,
                                        DeviceCapabilities const& caps,
                                        CpuBackendSelectorConfig const& cfg) {
  if (char const* env = std::getenv("UKERNEL_CCL_CPU_BACKEND")) {
    std::string override_value = to_lower(env);
    if (override_value == "rdma") return CpuBackendKind::Rdma;
    if (override_value == "ce") return CpuBackendKind::Ce;
    if (override_value == "pk") return CpuBackendKind::Pk;
  }

  if (requested != CpuBackendKind::Auto) return requested;
  if (caps.supports_rdma && !caps.is_same_node) return CpuBackendKind::Rdma;
  if (is_copy && cfg.prefer_ce_for_large_copy && caps.has_copy_engine_path &&
      bytes >= cfg.copy_engine_threshold_bytes) {
    return CpuBackendKind::Ce;
  }
  return CpuBackendKind::Pk;
}

TransferPath resolve_pk_transfer_path(TransferPath requested, uint64_t bytes,
                                      DeviceCapabilities const& caps,
                                      PkSelectorConfig const& cfg) {
  if (char const* env = std::getenv("UKERNEL_CCL_PK_BACKEND")) {
    std::string override_value = to_lower(env);
    if (override_value == "reg" || override_value == "register" ||
        override_value == "registerop") {
      return TransferPath::RegisterOp;
    }
    if (override_value == "tma" && caps.has_tma &&
        bytes >= cfg.tma_threshold_bytes) {
      return TransferPath::TmaOp;
    }
  }

  if (requested == TransferPath::RegisterOp) return TransferPath::RegisterOp;
  if (requested == TransferPath::TmaOp) {
    return caps.has_tma ? TransferPath::TmaOp : TransferPath::RegisterOp;
  }

  if (!cfg.enable_auto_transport) return TransferPath::RegisterOp;
  if (caps.has_tma && bytes >= cfg.tma_threshold_bytes) {
    return TransferPath::TmaOp;
  }
  return TransferPath::RegisterOp;
}

}  // namespace Compute
}  // namespace UKernel
