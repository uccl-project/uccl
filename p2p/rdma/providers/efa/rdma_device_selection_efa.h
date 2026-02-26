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
    // Count total EFA NICs and find min distance among them.
    int num_efas = 0;
    uint32_t min_d = UINT32_MAX;
    for (auto const& p : dist) {
      if (strncmp(p.first.c_str(), "rdmap", 5) == 0) {
        num_efas++;
        if (p.second < min_d) min_d = p.second;
      }
    }

    // Collect EFA NICs with equal minimum distance.
    std::vector<std::string> candidates;
    for (auto const& p : dist) {
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
