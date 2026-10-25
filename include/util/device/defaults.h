#pragma once
#include <cstddef>

namespace uccl::device {
// Policies preserve the behavior of the existing runtime adapters. Hardware
// support (for example DMA-BUF availability) must still be checked at runtime.
enum class RuntimeDiscovery { None, First, Supplement };

struct BackendDefaults {
  static constexpr char const* runtime_init = "";
  static constexpr bool ipc_owner_context = false;
  static constexpr bool ipc_async_owner_context = false;
  static constexpr bool cache_ipc_exports = false;
  static constexpr bool ipc_export_owner = false;
  static constexpr bool staged_network = false;
  static constexpr size_t dmabuf_granularity = 2ULL << 20;
  static constexpr bool scan_nvidia_proc = false;
  static constexpr bool visible_nvidia_fallback = false;
  static constexpr RuntimeDiscovery runtime_discovery = RuntimeDiscovery::None;
  static constexpr bool filter_pci_vendor = false;
};
}  // namespace uccl::device
