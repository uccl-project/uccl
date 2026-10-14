#include "gpu_topology.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  auto root =
      fs::temp_directory_path() /
      ("uccl-gpu-topology-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(root / "sysfs");
  try {
    std::vector<std::string> bdfs = {"0000:e3:00.0", "0000:23:00.0",
                                     "0000:63:00.0", "0000:a3:00.0"};
    std::vector<std::string> vendors = {"0x1d22", "0x10de", "0x1002", "0xcabc"};
    for (size_t i = 0; i < bdfs.size(); ++i) {
      auto device = root / "devices" / bdfs[i];
      fs::create_directories(device);
      std::ofstream(device / "vendor") << vendors[i];
      fs::create_directory_symlink(device, root / "sysfs" / bdfs[i]);
    }
    auto paths = uccl::gpu_paths_from_runtime_bdfs(bdfs, root / "sysfs");
    require(paths.size() == bdfs.size(),
            "a runtime-visible vendor was dropped");
    for (size_t i = 0; i < bdfs.size(); ++i)
      require(paths[i] == root / "devices" / bdfs[i],
              "runtime ordinal changed");
    auto subset =
        uccl::gpu_paths_from_runtime_bdfs({bdfs[2], bdfs[0]}, root / "sysfs");
    require(subset == std::vector<fs::path>{paths[2], paths[0]},
            "visibility remapping was replaced by PCI order");
    auto missing = uccl::gpu_paths_from_runtime_bdfs(
        {bdfs[0], "0000:ff:00.0", bdfs[1]}, root / "sysfs");
    require(missing.size() == 3 && missing[0] == paths[0] &&
                missing[1] == root / "sysfs/0000:ff:00.0" &&
                missing[2] == paths[1],
            "restricted sysfs changed an ordinal");
    require(uccl::gpu_paths_from_runtime_bdfs({}, root / "sysfs").empty(),
            "empty enumeration must stay empty");
    fs::remove_all(root);
    std::cout << "runtime GPU topology: PASS\n";
  } catch (...) {
    fs::remove_all(root);
    throw;
  }
}
