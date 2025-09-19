#include "util/util.h"
#include <iostream>

int main() {
  std::cout << "=== Testing GPU Cards Detection ===" << std::endl;

  // Test get_gpu_cards()
  auto gpu_cards = uccl::get_gpu_cards();
  std::cout << "Found " << gpu_cards.size() << " GPU cards:" << std::endl;
  for (size_t i = 0; i < gpu_cards.size(); ++i) {
    std::cout << "  GPU " << i << ": " << gpu_cards[i] << std::endl;
  }

  std::cout << "\n=== Testing RDMA NICs Detection ===" << std::endl;

  // Test get_rdma_nics()
  auto rdma_nics = uccl::get_rdma_nics();
  std::cout << "Found " << rdma_nics.size() << " RDMA NICs:" << std::endl;
  for (size_t i = 0; i < rdma_nics.size(); ++i) {
    std::cout << "  NIC " << i << ": " << rdma_nics[i].first << " -> "
              << rdma_nics[i].second << std::endl;
  }

  std::cout << "\n=== Testing GPU to Device Mapping ===" << std::endl;

  // Convert rdma_nics to the format expected by map_gpu_to_dev
  std::vector<std::tuple<std::string, std::filesystem::path, int>> ib_nics;
  for (size_t i = 0; i < rdma_nics.size(); ++i) {
    ib_nics.emplace_back(rdma_nics[i].first, rdma_nics[i].second,
                         static_cast<int>(i));
  }

  // Test map_gpu_to_dev()
  if (!gpu_cards.empty() && !ib_nics.empty()) {
    auto gpu_to_dev_map = uccl::map_gpu_to_dev(gpu_cards, ib_nics);
    std::cout << "GPU to Device mapping:" << std::endl;
    for (auto const& [gpu_idx, dev_idx] : gpu_to_dev_map) {
      std::cout << "  GPU " << gpu_idx << " -> NIC " << dev_idx;
      if (dev_idx < static_cast<int>(ib_nics.size())) {
        std::cout << " (" << std::get<0>(ib_nics[dev_idx]) << ")";
      }
      std::cout << std::endl;
    }
  } else {
    std::cout << "Cannot test mapping - ";
    if (gpu_cards.empty()) std::cout << "no GPUs found ";
    if (ib_nics.empty()) std::cout << "no RDMA NICs found";
    std::cout << std::endl;
  }

  std::cout << "\n=== Test Summary ===" << std::endl;
  std::cout << "GPUs detected: " << gpu_cards.size() << std::endl;
  std::cout << "RDMA NICs detected: " << rdma_nics.size() << std::endl;
  std::cout << "Mapping successful: "
            << (!gpu_cards.empty() && !ib_nics.empty() ? "Yes" : "No")
            << std::endl;

  return 0;
}
