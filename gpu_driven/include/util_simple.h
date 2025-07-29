#pragma once

#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
#include <fstream>
#include <iostream>

namespace uccl {

namespace fs = std::filesystem;

static bool is_bdf(std::string const& s) {
  // Match full PCI BDF allowing hexadecimal digits
  static const std::regex re(
      R"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])");
  return std::regex_match(s, re);
}

static int cal_pcie_distance(fs::path const& devA, fs::path const& devB) {
  auto devA_parent = devA.parent_path();
  auto devB_parent = devB.parent_path();

  auto build_chain = [](fs::path const& dev) {
    std::vector<std::string> chain;
    for (fs::path p = fs::canonical(dev);; p = p.parent_path()) {
      std::string leaf = p.filename();
      if (is_bdf(leaf)) chain.push_back(leaf);  // collect BDF components
      if (p == p.root_path()) break;            // reached filesystem root
    }
    return chain; /* self → root */
  };

  auto chainA = build_chain(devA_parent);
  auto chainB = build_chain(devB_parent);

  // Walk back from root until paths diverge
  size_t i = chainA.size();
  size_t j = chainB.size();
  while (i > 0 && j > 0 && chainA[i - 1] == chainB[j - 1]) {
    --i;
    --j;
  }
  // Distance = remaining unique hops in each chain
  return static_cast<int>(i + j);
}

static std::vector<fs::path> get_gpu_cards() {
  // Discover GPU BDF using /sys/class/drm/cardX/device symlinks
  std::vector<fs::path> gpu_cards;
  const fs::path drm_class{"/sys/class/drm"};
  const std::regex card_re(R"(card(\d+))");

  if (fs::exists(drm_class)) {
    for (auto const& entry : fs::directory_iterator(drm_class)) {
      const std::string name = entry.path().filename();
      std::smatch m;
      if (!std::regex_match(name, m, card_re)) continue;

      fs::path dev_path = fs::canonical(entry.path() / "device");

      // check vendor id
      std::ifstream vf(dev_path / "vendor");
      std::string vs;
      if (!(vf >> vs)) continue;
      uint32_t vendor = std::stoul(vs, nullptr, 0);  // handles "0x10de"

      if (vendor != 0x10de && vendor != 0x1002) continue;  // NVIDIA or AMD

      gpu_cards.push_back(dev_path);
    }
  }

  const fs::path nvidia_gpus{"/proc/driver/nvidia/gpus"};
  if (gpu_cards.empty() && fs::exists(nvidia_gpus)) {
    for (auto const& entry : fs::directory_iterator(nvidia_gpus)) {
      gpu_cards.push_back(entry.path());
    }
  }

  std::sort(gpu_cards.begin(), gpu_cards.end(),
            [](fs::path const& a, fs::path const& b) {
              return a.filename() < b.filename();
            });

  return gpu_cards;
}

static std::vector<std::pair<std::string, fs::path>> get_rdma_nics() {
  // Discover RDMA NICs under /sys/class/infiniband
  std::vector<std::pair<std::string, fs::path>> ib_nics;
  const fs::path ib_class{"/sys/class/infiniband"};
  if (!fs::exists(ib_class)) {
    std::cerr << "No /sys/class/infiniband directory found. Are RDMA drivers "
                 "loaded?\n";
    return ib_nics;
  }

  for (auto const& ib_entry : fs::directory_iterator(ib_class)) {
    std::string ibdev = ib_entry.path().filename();
    fs::path ib_device_path = fs::canonical(ib_entry.path() / "device");

    // Collect interface names under RDMA device
    fs::path netdir = ib_device_path / "net";
    if (fs::exists(netdir) && fs::is_directory(netdir)) {
      ib_nics.push_back(std::make_pair(ibdev, ib_device_path));
    }
  }
  std::sort(ib_nics.begin(), ib_nics.end(),
            [](std::pair<std::string, fs::path> const& a,
               std::pair<std::string, fs::path> const& b) {
              return a.first < b.first;
            });
  return ib_nics;
}

} // namespace uccl