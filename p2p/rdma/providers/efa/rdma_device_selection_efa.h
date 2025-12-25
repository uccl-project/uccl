#pragma once
#include <cassert>
#include <memory>
#include <string>
#include <vector>

// Base class for device selection strategy
class RDMADeviceSelectionStrategy {
 public:
  virtual ~RDMADeviceSelectionStrategy() = default;

  // Select NIC names from candidates based on GPU index
  virtual std::vector<std::string> selectNICs(
      std::vector<std::string> const& candidates, int gpu_idx) = 0;
};

// EFA device selection strategy
class EFADeviceSelectionStrategy : public RDMADeviceSelectionStrategy {
 public:
  std::vector<std::string> selectNICs(
      std::vector<std::string> const& candidates, int gpu_idx) override {
    // NOTE(xzhiying): This is a temporary hack.
    // On p5en, there are 4 NICs with the same distance.
    // GPU0 uses candidates[0/1], GPU1 uses candidates[2/3], etc.
    assert(candidates.size() == 4);
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

// Factory function (inline)
inline std::unique_ptr<RDMADeviceSelectionStrategy>
createDeviceSelectionStrategy() {
  return std::make_unique<EFADeviceSelectionStrategy>();
}
