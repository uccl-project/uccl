#pragma once

#include "util/device/defaults.h"

namespace uccl::device {
struct Backend : BackendDefaults {
  static constexpr char const* runtime_init = "hipInit";
  static constexpr char const* name = "hip";
  static constexpr char const* runtime_prefix = "hip";
  static constexpr char const* runtime_library = "libamdhip64.so";
  static constexpr bool filter_pci_vendor = true;
};
}  // namespace uccl::device

#define UCCL_DEVICE_RUNTIME_HEADER "util/device/hip/runtime.h"
#ifdef __UCCL_DTK__
#define UCCL_DEVICE_HAS_DMABUF 0
#else
#define UCCL_DEVICE_HAS_DMABUF 1
#endif
#define UCCL_DEVICE_RDMA_ONLY 0

#define UCCL_DEVICE_STAGING_HEADER "device/no_staging.h"
