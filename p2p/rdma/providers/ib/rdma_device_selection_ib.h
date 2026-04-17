#pragma once
#include "../util.h"
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

    std::vector<std::pair<std::string, uint32_t>> usable_dist;
    usable_dist.reserve(dist.size());
    for (auto const& p : dist) {
      UCCL_LOG(WARN) << "NIC: " << p.first << ", distance: " << p.second;
      if (is_nic_usable(p.first, NicMode::IB)) {
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
