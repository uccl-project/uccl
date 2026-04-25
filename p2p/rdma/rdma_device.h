#pragma once
#include "define.h"
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
  virtual std::vector<std::string> selectNICs(
      std::vector<std::pair<std::string, uint32_t>> const& dist,
      int gpu_idx) = 0;
};

#include "include/transport_type.h"
#include "providers/efa/rdma_device_selection_efa.h"
#include "providers/ib/rdma_device_selection_ib.h"

// Factory: select IB or EFA device strategy at runtime.
inline std::unique_ptr<RDMADeviceSelectionStrategy>
createDeviceSelectionStrategy() {
  if (uccl::is_efa_transport())
    return std::make_unique<EFADeviceSelectionStrategy>();
  else
    return std::make_unique<IBDeviceSelectionStrategy>();
}

class RdmaDevice {
 public:
  RdmaDevice(struct ibv_device* dev)
      : dev_(dev), name_(ibv_get_device_name(dev)) {}

  std::string const& name() const { return name_; }

  struct ibv_device* get() const {
    return dev_;
  }

  std::shared_ptr<struct ibv_context> open() {
    struct ibv_context* ctx = ibv_open_device(dev_);
    if (!ctx) {
      perror("ibv_open_device failed");
      return nullptr;
    }
    return std::shared_ptr<struct ibv_context>(
        ctx, [](ibv_context* c) { ibv_close_device(c); });
  }

 private:
  struct ibv_device* dev_;
  std::string name_;
};

class RdmaDeviceManager {
 public:
  // Thread-safe singleton with C++11 static initialization
  // Automatically initializes on first call using std::call_once
  static RdmaDeviceManager& instance() {
    static RdmaDeviceManager inst;
    std::call_once(inst.init_flag_, &RdmaDeviceManager::initialize, &inst);
    return inst;
  }

  // Delete copy and move constructors/operators
  RdmaDeviceManager(RdmaDeviceManager const&) = delete;
  RdmaDeviceManager& operator=(RdmaDeviceManager const&) = delete;
  RdmaDeviceManager(RdmaDeviceManager&&) = delete;
  RdmaDeviceManager& operator=(RdmaDeviceManager&&) = delete;

  std::shared_ptr<RdmaDevice> getDevice(size_t id) {
    if (id >= devices_.size()) return nullptr;
    return devices_[id];
  }

  size_t deviceCount() const { return devices_.size(); }

  std::vector<size_t> get_best_dev_idx(int gpu_idx) {
    // Allow user to override NIC selection via environment variable.
    // UCCL_P2P_RDMA_DEV can be a comma-separated list of device names
    // (e.g. "irdma-mkp0,irdma-mkp1") and/or numeric indices (e.g. "4,5")
    // into the `ibv_get_device_list` order.
    char const* env_dev = getenv("UCCL_P2P_RDMA_DEV");
    if (env_dev) {
      std::string env_str(env_dev);

      auto add_unique_idx = [&](size_t idx, std::vector<size_t>& out) {
        if (std::find(out.begin(), out.end(), idx) == out.end())
          out.push_back(idx);
      };

      std::vector<size_t> selected_dev_indices;
      selected_dev_indices.reserve(devices_.size());

      std::stringstream ss(env_str);
      std::string token;
      while (std::getline(ss, token, ',')) {
        // Trim ASCII whitespace around the token.
        auto start = token.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        auto end = token.find_last_not_of(" \t\n\r");
        token = token.substr(start, end - start + 1);
        if (token.empty()) continue;

        bool found = false;

        // Try numeric index token first (must match the whole token).
        try {
          size_t pos = 0;
          size_t idx = std::stoul(token, &pos);
          if (pos == token.size()) {
            if (idx < devices_.size()) {
              add_unique_idx(idx, selected_dev_indices);
              UCCL_LOG(INFO, UCCL_RDMA)
                  << "UCCL_P2P_RDMA_DEV override: token '" << token
                  << "' -> using device index " << idx << " ("
                  << devices_[idx]->name() << ")";
              found = true;
            }
          }
        } catch (...) {
        }

        // If numeric parse didn't match / was out of range, try matching by
        // name.
        if (!found) {
          for (size_t i = 0; i < devices_.size(); i++) {
            if (devices_[i]->name() == token) {
              add_unique_idx(i, selected_dev_indices);
              UCCL_LOG(INFO, UCCL_RDMA)
                  << "UCCL_P2P_RDMA_DEV override: token '" << token
                  << "' -> using device " << token << " (index " << i << ")";
              found = true;
              break;
            }
          }
        }

        if (!found) {
          UCCL_LOG(WARN) << "UCCL_P2P_RDMA_DEV token '" << token
                         << "' not found (neither index nor device name)";
        }
      }

      if (!selected_dev_indices.empty()) {
        std::stringstream summary;
        summary << "UCCL_P2P_RDMA_DEV override: using device indices [";
        for (size_t i = 0; i < selected_dev_indices.size(); i++) {
          if (i != 0) summary << ", ";
          auto idx = selected_dev_indices[i];
          summary << idx;
        }
        summary << "]";
        UCCL_LOG(INFO, UCCL_RDMA) << summary.str();
        return selected_dev_indices;
      }

      UCCL_LOG(WARN) << "UCCL_P2P_RDMA_DEV=" << env_str
                     << " not usable, falling back to auto-selection";
    }

    // Ranked by GPU idx
    auto gpu_cards = uccl::get_gpu_cards();
    // Ranked by RDMA NIC name (not the ibv_get_device_list order)
    auto ib_nics = uccl::get_rdma_nics();
    // Get GPU pcie path
    auto gpu_device_path = gpu_cards[gpu_idx];
    // Find the RDMA NIC that is closest to the GPU.

    std::vector<std::pair<std::string, uint32_t>> dist;
    dist.reserve(ib_nics.size());

    std::vector<std::string> selected_nic_names;
    for (auto& nic : ib_nics) {
      uint32_t d = uccl::safe_pcie_distance(gpu_device_path, nic.second);
      dist.emplace_back(nic.first, d);
    }

    if (dist.empty()) {
      UCCL_LOG(WARN) << "no NIC found, defaulting to empty";
    } else {
      auto strategy = createDeviceSelectionStrategy();
      auto selected = strategy->selectNICs(dist, gpu_idx);
      if (selected.empty()) {
        UCCL_LOG(WARN) << "no candidate NIC found, defaulting to first";
        selected_nic_names.push_back(dist.front().first);
      } else {
        selected_nic_names.insert(selected_nic_names.end(), selected.begin(),
                                  selected.end());
      }
    }

    std::vector<size_t> selected_dev_indices;
    for (auto const& nic_name : selected_nic_names) {
      int dev_idx = -1;
      for (size_t i = 0; i < devices_.size(); i++) {
        if (devices_[i]->name() == nic_name) {
          dev_idx = i;
          break;
        }
      }
      if (dev_idx < 0) {
        UCCL_LOG(FATAL) << "Selected RDMA NIC '" << nic_name
                        << "' not found in verbs device list";
      }
      selected_dev_indices.push_back(dev_idx);
    }

    std::stringstream ss;
    ss << "[RDMA] GPU " << gpu_idx << " selected NICs: ";
    for (size_t i = 0; i < selected_nic_names.size(); i++) {
      ss << selected_nic_names[i] << " (device idx " << selected_dev_indices[i]
         << ")";
      if (i < selected_nic_names.size() - 1) ss << ", ";
    }
    UCCL_LOG(INFO, UCCL_RDMA) << ss.str();

    return selected_dev_indices;
  }

  int get_numa_node(size_t id) {
    if (id >= devices_.size()) {
      UCCL_LOG(WARN) << "Invalid device id: " << id;
      return -1;
    }
    std::string device_name = devices_[id]->name();
    return uccl::get_dev_numa_node(device_name.c_str());
  }

 private:
  RdmaDeviceManager() = default;
  ~RdmaDeviceManager() = default;

  void initialize() {
    int num = 0;
    ibv_device** dev_list = ibv_get_device_list(&num);
    if (!dev_list) {
      perror("ibv_get_device_list failed");
      return;
    }
    std::cout << "RdmaDeviceManager: Found " << num << " RDMA device(s)"
              << std::endl;
    for (int i = 0; i < num; ++i) {
      auto dev = std::make_shared<RdmaDevice>(dev_list[i]);
      std::cout << "  [" << i << "] " << dev->name() << std::endl;
      devices_.push_back(dev);
    }
    ibv_free_device_list(dev_list);
    std::cout << "RdmaDeviceManager: Initialization complete" << std::endl;
  }

  std::once_flag init_flag_;  // Ensures initialize() is called only once
  std::vector<std::shared_ptr<RdmaDevice>> devices_;
};
