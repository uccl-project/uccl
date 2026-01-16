#pragma once
#include <cassert>
#include <memory>
#include <string>
#include <vector>

// Forward declaration
class RDMADeviceSelectionStrategy;

// EFA device selection strategy
class EFADeviceSelectionStrategy : public RDMADeviceSelectionStrategy {
 public:
  std::vector<std::string> selectNICs(
      std::vector<std::string> const& candidates, int gpu_idx) override {
    // NOTE(xzhiying): This is a temporary hack.
    // On p5/p5en/p6-b200, there are 8/4/2 NICs with the same distance.
    // E.g., on p5en, GPU0 uses candidates[0/1], GPU1 uses candidates[2/3], etc.
    assert(candidates.size() == 8 || candidates.size() == 4 ||
           candidates.size() == 2);
    int half_size = candidates.size() / 2;
    int start_idx = (gpu_idx % 2 == 0) ? 0 : half_size;
    int end_idx = start_idx + half_size;
    std::vector<std::string> selected;
    for (int i = start_idx; i < end_idx; i++) {
      selected.push_back(candidates[i]);
    }
    return selected;
  }
};
