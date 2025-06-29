#include "util/util.h"
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <GPU_RANK>\n";
    return 1;
  }

  int gpu_rank = std::stoi(argv[1]);
  auto gpu_cards = uccl::get_gpu_cards();

  std::cout << "All GPU cards:\n";
  for (auto const& gpu_card : gpu_cards) {
    std::cout << "  • " << gpu_card.filename() << '\n';
  }
  if (gpu_rank < 0 || gpu_rank >= static_cast<int>(gpu_cards.size())) {
    std::cerr << "GPU rank " << gpu_rank << " out of range (found "
              << gpu_cards.size() << " card(s))\n";
    return 1;
  }

  auto gpu_device_path = gpu_cards[gpu_rank];
  std::cout << "Selecting GPU " << gpu_device_path.filename() << '\n';

  auto ib_nics = uccl::get_rdma_nics();
  std::cout << "All RDMA NICs and their distance to the selected GPU:\n";
  for (auto const& ib_nic : ib_nics) {
    int dist = uccl::cal_pcie_distance(gpu_device_path, ib_nic.second);
    std::cout << "  • " << ib_nic.first << " distance=" << dist << '\n';
  }

  return 0;
}
