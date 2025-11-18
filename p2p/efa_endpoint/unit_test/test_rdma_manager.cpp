#include "rdma_context.h"

int main() {
  auto& mgr = RdmaDeviceManager::instance();

  auto dev = mgr.getDevice(0);
  if (!dev) {
    std::cerr << "Failed to get device 0" << std::endl;
    return 1;
  }

  auto ctx = std::make_shared<RdmaContext>(dev);

  // Query and display GID
  union ibv_gid gid = ctx->queryGid(0);
  (void)gid;  // Mark as intentionally unused

  std::cout << "Context and PD initialized successfully." << std::endl;
  std::cout << "Device count: " << mgr.deviceCount() << std::endl;
  return 0;
}
