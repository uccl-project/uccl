#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <infiniband/verbs.h>
#include "gpu_rt.h"

namespace UKernel {
namespace Transport {

namespace fs = std::filesystem;

inline bool is_bdf(std::string const& s) {
  // Match full PCI BDF allowing hexadecimal digits
  static std::regex const re(
      R"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])");
  return std::regex_match(s, re);
}

inline int cal_pcie_distance(fs::path const& devA, fs::path const& devB) {
  auto devA_parent = devA.parent_path();
  auto devB_parent = devB.parent_path();

  auto build_chain = [](fs::path const& dev) {
    std::vector<std::string> chain;
    fs::path p = fs::canonical(dev);
    for (;; p = p.parent_path()) {
      std::string leaf = p.filename();
      if (is_bdf(leaf)) {
        chain.push_back(leaf);  // collect BDF components
      }
      if (p == p.root_path()) break;  // reached filesystem root
    }
    return chain;  // self -> root
  };

  // thread-safe cache
  static std::mutex cache_mutex;
  static std::unordered_map<fs::path, std::vector<std::string>>
      dev_to_chain_cache;

  std::vector<std::string> chainA, chainB;
  {
    std::lock_guard<std::mutex> g(cache_mutex);

    auto itA = dev_to_chain_cache.find(devA_parent);
    if (itA == dev_to_chain_cache.end()) {
      itA = dev_to_chain_cache.emplace(devA_parent, build_chain(devA_parent))
                .first;
    }
    chainA = itA->second;

    auto itB = dev_to_chain_cache.find(devB_parent);
    if (itB == dev_to_chain_cache.end()) {
      itB = dev_to_chain_cache.emplace(devB_parent, build_chain(devB_parent))
                .first;
    }
    chainB = itB->second;
  }

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

inline uint32_t safe_pcie_distance(fs::path const& gpu, fs::path const& nic) {
  try {
    return static_cast<uint32_t>(cal_pcie_distance(gpu, nic));
  } catch (std::exception const& e) {
    fprintf(stderr, "[WARN] safe_pcie_distance failed: %s\n", e.what());
    return UINT32_MAX / 2;  // Treat as "very far"
  } catch (...) {
    fprintf(stderr, "[WARN] safe_pcie_distance unknown failure\n");
    return UINT32_MAX / 2;
  }
}

inline std::string normalize_pci_bus_id(std::string const& pci_bus_id) {
  std::string normalized = pci_bus_id;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 ::tolower);
  return normalized;
}

inline fs::path sysfs_pci_path_from_bdf(std::string const& bdf_lower) {
  fs::path link = fs::path("/sys/bus/pci/devices") / bdf_lower;
  try {
    return fs::canonical(link);  // -> /sys/devices/.../pci.../<bdf>
  } catch (...) {
    return link;  // still valid for accessing sysfs files via the symlink
  }
}

inline std::vector<fs::path> get_gpu_cards() {
  // 1) Collect visible GPUs (ranked) and their normalized BDFs
  int num_gpus = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&num_gpus));
  std::vector<std::string> gpu_bdfs_ranked;
  gpu_bdfs_ranked.reserve(num_gpus);

  char bdf[64];
  for (int i = 0; i < num_gpus; ++i) {
    GPU_RT_CHECK(gpuDeviceGetPCIBusId(bdf, sizeof(bdf), i));  // full BDF
    gpu_bdfs_ranked.emplace_back(normalize_pci_bus_id(bdf));
  }

  // 2) Map each BDF -> canonical sysfs PCI path
  std::vector<fs::path> gpu_cards;
  gpu_cards.reserve(num_gpus);
  std::unordered_map<fs::path, int> rank_map;

  for (int rank = 0; rank < (int)gpu_bdfs_ranked.size(); ++rank) {
    std::string const& bdf_lower = gpu_bdfs_ranked[rank];
    fs::path pci_path = sysfs_pci_path_from_bdf(bdf_lower);

    // Optional sanity check: ensure it's actually a GPU (NVIDIA=0x10de,
    // AMD=0x1002)
    bool ok = true;
    try {
      std::ifstream vf(pci_path / "vendor");
      std::string vs;
      if (vf >> vs) {
        // vs may be "0x10de" or "0x1002"
        uint32_t vendor = std::stoul(vs, nullptr, 0);
        if (!(vendor == 0x10de || vendor == 0x1002)) ok = false;
      }
    } catch (...) {
      // If vendor check fails due to restricted sysfs, keep going.
    }
    if (!ok) continue;

    rank_map[pci_path] = rank;
    gpu_cards.push_back(pci_path);
  }

  // 3) Fallbacks (only if needed): DRM class (cardX) or
  // /proc/driver/nvidia/gpus
  if (gpu_cards.empty()) {
    fs::path const drm_class{"/sys/class/drm"};
    std::regex const card_re(R"(card(\d+))");
    if (fs::exists(drm_class)) {
      for (auto const& entry : fs::directory_iterator(drm_class)) {
        std::string const name = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_match(name, m, card_re)) continue;
        fs::path dev_path;
        try {
          dev_path = fs::canonical(entry.path() / "device");
        } catch (...) {
          continue;
        }
        // Extract BDF and try to match ranks
        std::string bdf_name = dev_path.filename().string();
        std::string nbdf = normalize_pci_bus_id(bdf_name);
        auto it =
            std::find(gpu_bdfs_ranked.begin(), gpu_bdfs_ranked.end(), nbdf);
        if (it == gpu_bdfs_ranked.end()) continue;
        rank_map[dev_path] = int(std::distance(gpu_bdfs_ranked.begin(), it));
        gpu_cards.push_back(dev_path);
      }
    }
#ifndef __HIP_PLATFORM_AMD__
    if (gpu_cards.empty()) {
      fs::path const nvidia_gpus{"/proc/driver/nvidia/gpus"};
      if (fs::exists(nvidia_gpus)) {
        for (auto const& entry : fs::directory_iterator(nvidia_gpus)) {
          fs::path dev_path;
          try {
            dev_path = fs::canonical(entry.path());
          } catch (...) {
            continue;
          }
          std::string bdf_name = dev_path.filename().string();  // already BDF
          std::string nbdf = normalize_pci_bus_id(bdf_name);
          auto it =
              std::find(gpu_bdfs_ranked.begin(), gpu_bdfs_ranked.end(), nbdf);
          if (it == gpu_bdfs_ranked.end()) continue;
          rank_map[dev_path] = int(std::distance(gpu_bdfs_ranked.begin(), it));
          gpu_cards.push_back(dev_path);
        }
      }
    }
#endif
  }

  // 4) Sort by original runtime rank
  std::sort(gpu_cards.begin(), gpu_cards.end(),
            [&rank_map](fs::path const& a, fs::path const& b) {
              return rank_map[a] < rank_map[b];
            });

  return gpu_cards;
}

inline std::vector<std::pair<std::string, fs::path>> get_rdma_nics() {
  std::vector<std::pair<std::string, fs::path>> rdma_nics;

  int num_devices = 0;
  ibv_device** dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list) {
    std::cerr << "Failed to get RDMA device list (is rdma-core installed?).\n";
    return rdma_nics;
  }

  for (int i = 0; i < num_devices; ++i) {
    ibv_device* dev = dev_list[i];
    char const* dev_name = ibv_get_device_name(dev);  // e.g. "mlx5_0", "efa_0"

    // dev->ibdev_path is something like: /sys/class/infiniband/mlx5_0
    fs::path dev_sysfs = fs::path(dev->ibdev_path) / "device";

    // Resolve symlink to get PCI path
    char link_target[PATH_MAX];
    ssize_t len =
        readlink(dev_sysfs.c_str(), link_target, sizeof(link_target) - 1);
    if (len < 0) {
      perror("readlink");
      continue;
    }
    link_target[len] = '\0';

    fs::path pci_path;
    if (link_target[0] == '/') {
      pci_path = fs::path(link_target);
    } else {
      // Relative symlink - resolve relative to parent directory
      pci_path = fs::canonical(dev_sysfs.parent_path() / link_target);
    }

    // The last component of pci_path is the full BDF: 0000:3b:00.0
    std::string pci_bdf = pci_path.filename().string();

    // Store as { device_name, full_pci_path }
    rdma_nics.emplace_back(dev_name, pci_path);
  }

  ibv_free_device_list(dev_list);

  std::sort(rdma_nics.begin(), rdma_nics.end(),
            [](auto const& a, auto const& b) { return a.first < b.first; });

  return rdma_nics;
}

}  // namespace Transport
}  // namespace UKernel
