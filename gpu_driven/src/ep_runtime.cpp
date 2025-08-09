#include "ep_runtime.cuh"
#include <iostream>

namespace internode {

int init(std::vector<uint8_t> const& root_unique_id_val, int rank,
         int num_ranks, bool low_latency_mode) {
  std::cout << "[internode::init] dummy init invoked" << std::endl;
  return 0;
}

void* alloc(std::size_t size, std::size_t alignment) {
  std::cout << "[internode::alloc] dummy alloc invoked" << std::endl;
  return nullptr;
}

void finalize() {
  std::cout << "[internode::finalize] dummy finalize invoked" << std::endl;
}

void barrier() {
  std::cout << "[internode::barrier] dummy barrier invoked" << std::endl;
}

void free(void* ptr) {
  std::cout << "[internode::free] dummy free invoked" << std::endl;
}

std::vector<uint8_t> get_unique_id() { return std::vector<uint8_t>(64, 0); }

}  // namespace internode

namespace intranode {

// No device definition here; just the host wrapper stub.
void barrier(int** barrier_signal_ptrs, int rank, int num_ranks,
             cudaStream_t stream) {
  std::cout << "[intranode::barrier] dummy intranode barrier invoked"
            << std::endl;
}

}  // namespace intranode