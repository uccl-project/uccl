#pragma once

#include "util/device/defaults.h"

namespace uccl::device {
struct Backend : BackendDefaults {
  static constexpr char const* name = "cnrt";
  static constexpr char const* runtime_prefix = "cnrt";
  static constexpr char const* runtime_library = "libcnrt.so";
  static constexpr bool ipc_owner_context = true;
  static constexpr bool ipc_async_owner_context = true;
  static constexpr bool ipc_export_owner = true;
  static constexpr bool cache_ipc_exports = true;
  static constexpr bool staged_network = true;
  static constexpr bool visible_nvidia_fallback = true;
  static constexpr bool filter_pci_vendor = true;
  static constexpr RuntimeDiscovery runtime_discovery = RuntimeDiscovery::First;
};
}  // namespace uccl::device

#define UCCL_DEVICE_RUNTIME_HEADER "util/device/cnrt/runtime.h"
#define UCCL_DEVICE_HAS_DMABUF 1
#define UCCL_DEVICE_RDMA_ONLY 0

#define UCCL_DEVICE_STAGING_HEADER "device/cnrt/staging.h"
