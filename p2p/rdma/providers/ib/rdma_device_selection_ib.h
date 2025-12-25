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

// Factory function (inline)
inline std::unique_ptr<RDMADeviceSelectionStrategy>
createDeviceSelectionStrategy() {
  return std::make_unique<IBDeviceSelectionStrategy>();
}
