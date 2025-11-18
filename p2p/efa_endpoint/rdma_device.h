// rdma_device.h
#pragma once
#include "define.h"

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
