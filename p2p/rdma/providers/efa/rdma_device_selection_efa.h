#pragma once
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Forward declaration
class RDMADeviceSelectionStrategy;

// EFA device selection strategy
class EFADeviceSelectionStrategy : public RDMADeviceSelectionStrategy {
 public:
  std::vector<std::string> selectNICs(
      std::vector<std::pair<std::string, uint32_t>> const& dist,
      int gpu_idx) override {
    auto is_nic_usable = [](std::string const& nic_name) -> bool {
      if (strncmp(nic_name.c_str(), "rdmap", 5) != 0) {
        return false;
      }
      int dev_count = 0;
      ibv_device** dev_list = ibv_get_device_list(&dev_count);
      if (!dev_list) {
        return false;
      }
      bool usable = false;
      for (int i = 0; i < dev_count; ++i) {
        if (std::strcmp(ibv_get_device_name(dev_list[i]), nic_name.c_str()) != 0) {
          continue;
        }
        ibv_context* ctx = ibv_open_device(dev_list[i]);
        if (!ctx) break;
        ibv_port_attr port_attr{};
        if (ibv_query_port(ctx, kPortNum, &port_attr) == 0) {
          if (port_attr.state == IBV_PORT_ACTIVE &&
              (port_attr.link_layer == IBV_LINK_LAYER_UNSPECIFIED ||
               port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
               port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND)) {
            usable = true;
          }
        }
        ibv_close_device(ctx);
        break;
      }
      ibv_free_device_list(dev_list);
      return usable;
    };

    std::vector<std::pair<std::string, uint32_t>> usable_dist;
    usable_dist.reserve(dist.size());
    for (auto const& p : dist) {
      if (is_nic_usable(p.first)) {
        usable_dist.push_back(p);
      }
    }
    if (usable_dist.empty()) {
      return {};
    }

    // Count total EFA NICs and find min distance among them.
    int num_efas = 0;
    uint32_t min_d = UINT32_MAX;
    for (auto const& p : usable_dist) {
      if (strncmp(p.first.c_str(), "rdmap", 5) == 0) {
        num_efas++;
        if (p.second < min_d) min_d = p.second;
      }
    }

    // Collect EFA NICs with equal minimum distance.
    std::vector<std::string> candidates;
    for (auto const& p : usable_dist) {
      if (strncmp(p.first.c_str(), "rdmap", 5) == 0 && p.second == min_d) {
        candidates.push_back(p.first);
      }
    }

    std::vector<std::string> selected;
    if (num_efas == 32) {
      // On p5, there are 4 NICs with the same distance.
      // All 4 candidates are available to each GPU.
      assert(candidates.size() == 4);
      selected = candidates;
    } else if (num_efas == 16) {
      // On p5e/p5en, there are 4 NICs with the same distance.
      // GPU0 uses candidates[0/1], GPU1 uses candidates[2/3], etc.
      assert(candidates.size() == 4);
      int half = (gpu_idx % 2) * 2;
      selected.push_back(candidates[half]);
      selected.push_back(candidates[half + 1]);
    } else {
      // On p6-b200, there are 2 NICs with the same distance.
      assert(num_efas == 8);
      assert(candidates.size() == 2);
      int half = (gpu_idx % 2) * 1;
      selected.push_back(candidates[half]);
    }
    return selected;
  }
};
