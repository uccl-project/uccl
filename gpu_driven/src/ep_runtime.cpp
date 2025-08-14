#include "ep_runtime.cuh"
#include <iostream>

namespace internode {

int init(std::vector<uint8_t> const& root_unique_id_val, int rank,
         int num_ranks, bool low_latency_mode) {
  std::cout << "[internode::init] dummy init invoked" << std::endl;
  return 0;
}

void* alloc(std::size_t size, std::size_t alignment) {
  // NOTE(MaoZiming): alignment is ignored here since cudaMalloc already aligns
  // to at least 256 bytes
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    std::cerr << "[internode::alloc] cudaMalloc failed: "
              << cudaGetErrorString(err) << std::endl;
    return nullptr;
  }
  std::cout << "[internode::alloc] allocated " << size << " bytes at " << ptr
            << std::endl;
  return ptr;
}

void finalize() {
  std::cout << "[internode::finalize] dummy finalize invoked" << std::endl;
}

void barrier() {
  std::cout << "[internode::barrier] dummy barrier invoked" << std::endl;
}

void free(void* ptr) {
  std::cout << "[internode::free] dummy free invoked" << std::endl;
  // free
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    std::cerr << "[internode::free] cudaFree failed: "
              << cudaGetErrorString(err) << std::endl;
  } else {
    std::cout << "[internode::free] freed memory at " << ptr << std::endl;
  }
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