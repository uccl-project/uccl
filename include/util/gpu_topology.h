#pragma once

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace uccl {

// Runtime enumeration defines device ordinals, including visibility remapping.
// Filtering PCI vendors here would change those ordinals for CUDA-compatible
// accelerators and may leave callers indexing an empty or shortened vector.
inline std::vector<std::filesystem::path> gpu_paths_from_runtime_bdfs(
    std::vector<std::string> const& bdfs,
    std::filesystem::path const& sysfs_root = "/sys/bus/pci/devices") {
  std::vector<std::filesystem::path> paths;
  paths.reserve(bdfs.size());
  for (auto const& bdf : bdfs) {
    auto link = sysfs_root / bdf;
    std::error_code error;
    auto canonical = std::filesystem::canonical(link, error);
    // Restricted sysfs should not remove an ordinal. Distance calculation can
    // still use the unresolved PCI path, as sysfs_pci_path_from_bdf did before.
    paths.push_back(error ? link : canonical);
  }
  return paths;
}

}  // namespace uccl
