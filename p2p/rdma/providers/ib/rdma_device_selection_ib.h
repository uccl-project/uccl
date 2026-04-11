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

// IB device selection strategy
class IBDeviceSelectionStrategy : public RDMADeviceSelectionStrategy {
 public:
  std::vector<std::string> selectNICs(
      std::vector<std::pair<std::string, uint32_t>> const& dist,
      int gpu_idx) override {
    (void)gpu_idx;

    auto is_nic_usable = [](std::string const& nic_name) -> bool {
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
              (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
               port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) &&
              (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET ||
               port_attr.gid_tbl_len > 0)) {
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
      UCCL_LOG(WARN) << "NIC: " << p.first << ", distance: " << p.second;
      if (is_nic_usable(p.first)) {
        usable_dist.push_back(p);
      } else {
        UCCL_LOG(WARN) << "NIC filtered as unusable: " << p.first;
      }
    }

    if (usable_dist.empty()) {
      return {};
    }

    // Find the minimum distance
    auto min_it = std::min_element(
        usable_dist.begin(), usable_dist.end(),
        [](auto const& a, auto const& b) { return a.second < b.second; });
    auto min_d = min_it->second;

    // Collect all NICs with equal minimum distance
    std::vector<std::string> candidates;
    for (auto const& p : usable_dist) {
      if (p.second == min_d) candidates.push_back(p.first);
    }

    return candidates;
  }
};
