#pragma once
#include <cassert>
#include <memory>
#include <string>
#include <vector>

// Forward declaration
class RDMADeviceSelectionStrategy;

// IB device selection strategy
class IBDeviceSelectionStrategy : public RDMADeviceSelectionStrategy {
 public:
  std::vector<std::string> selectNICs(
      std::vector<std::string> const& candidates, int gpu_idx) override {
    (void)gpu_idx;
    std::vector<std::string> selected;
    if (!candidates.empty()) {
      selected.push_back(candidates.front());
    }
    return selected;
  }
};
