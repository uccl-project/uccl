#pragma once

#include "util/device/defaults.h"

namespace uccl::device {
struct Backend : BackendDefaults {
  static constexpr char const* name = "musa";
  static constexpr char const* runtime_prefix = "musa";
  static constexpr char const* runtime_library = "libmusart.so";
  static constexpr bool ipc_async_owner_context = true;
  static constexpr bool ipc_export_owner = true;
  static constexpr RuntimeDiscovery runtime_discovery =
      RuntimeDiscovery::Supplement;
};
}  // namespace uccl::device

#define UCCL_DEVICE_RUNTIME_HEADER "util/device/musa/runtime.h"
#define UCCL_DEVICE_HAS_DMABUF 0
#define UCCL_DEVICE_RDMA_ONLY 1

#define UCCL_DEVICE_STAGING_HEADER "device/no_staging.h"
