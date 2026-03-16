#pragma once
#include <algorithm>
#include <cassert>
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

    for (auto const& p : dist) {
      UCCL_LOG(WARN) << "NIC: " << p.first << ", distance: " << p.second;
    }

    // Find the minimum distance
    auto min_it = std::min_element(
        dist.begin(), dist.end(),
        [](auto const& a, auto const& b) { return a.second < b.second; });
    auto min_d = min_it->second;

    // Collect all NICs with equal minimum distance
    std::vector<std::string> candidates;
    for (auto const& p : dist) {
      if (p.second == min_d) candidates.push_back(p.first);
    }

    return candidates;
  }
};
