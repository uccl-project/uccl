#pragma once

#include "util/device/defaults.h"

namespace uccl::device {
struct Backend : BackendDefaults {
  static constexpr bool scan_nvidia_proc = true;
  static constexpr bool visible_nvidia_fallback = true;
  static constexpr bool filter_pci_vendor = true;
  static constexpr char const* name = "cuda";
  static constexpr char const* runtime_prefix = "cuda";
  static constexpr char const* runtime_library = "libcudart.so";
};
}  // namespace uccl::device

#define UCCL_DEVICE_RUNTIME_HEADER "util/device/cuda/runtime.h"
#define UCCL_DEVICE_HAS_DMABUF 1
#define UCCL_DEVICE_RDMA_ONLY 0

#define UCCL_DEVICE_STAGING_HEADER "device/no_staging.h"
