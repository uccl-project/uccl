#pragma once
#include "common.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Base class for device selection strategy
class RDMADeviceSelectionStrategy {
 public:
  virtual ~RDMADeviceSelectionStrategy() = default;

  // Select NIC names from dist based on GPU index
  // dist: all NICs with their PCIe distances
  virtual std::vector<std::string> select_nics(
      std::vector<std::pair<std::string, uint32_t>> const& dist,
      int gpu_idx) = 0;
};

#include "providers/efa_device_selection.h"
#include "providers/ib_device_selection.h"
#include "transport_type.h"

// Factory: select IB or EFA device strategy at runtime.
inline std::unique_ptr<RDMADeviceSelectionStrategy>
create_device_selection_strategy() {
  if (is_efa_transport())
    return std::make_unique<EFADeviceSelectionStrategy>();
  else
    return std::make_unique<IBDeviceSelectionStrategy>();
}

class RdmaDevice {
 public:
  RdmaDevice(struct ibv_device* dev);

  std::string const& name() const;

  struct ibv_device* get() const;

  std::shared_ptr<struct ibv_context> open();

 private:
  struct ibv_device* dev_;
  std::string name_;
};

class RdmaDeviceManager {
 public:
  // Thread-safe singleton with C++11 static initialization
  // Automatically initializes on first call using std::call_once
  static RdmaDeviceManager& instance();

  // Delete copy and move constructors/operators
  RdmaDeviceManager(RdmaDeviceManager const&) = delete;
  RdmaDeviceManager& operator=(RdmaDeviceManager const&) = delete;
  RdmaDeviceManager(RdmaDeviceManager&&) = delete;
  RdmaDeviceManager& operator=(RdmaDeviceManager&&) = delete;

  std::shared_ptr<RdmaDevice> get_device(size_t id);

  size_t device_count() const;

  std::vector<size_t> get_best_dev_idx(int gpu_idx);

  int get_numa_node(size_t id);

 private:
  RdmaDeviceManager();
  ~RdmaDeviceManager();

  void initialize();

  std::once_flag init_flag_;  // Ensures initialize() is called only once
  std::vector<std::shared_ptr<RdmaDevice>> devices_;
};
