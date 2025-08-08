#pragma once
#include <iostream>
#include <vector>

// TODO(MaoZiming): This corresponds to DeepEP/csrc/kernels/runtime.cu
// They use nvshmem, but we don't have that in our environment.
int init(std::vector<uint8_t> const& root_unique_id_val, int rank,
         int num_ranks, bool low_latency_mode) {
  // TODO(MaoZiming): Fix.
  std::cout << "[uccl::internode_ll::init] dummy init invoked" << std::endl;
  return 0;  // Return success
}

void* alloc(size_t size, size_t alignment) {
  // TODO(MaoZiming): Fix.
  std::cout << "[uccl::internode_ll::alloc] dummy alloc invoked" << std::endl;
  return nullptr;
}

void barrier() {
  // TODO(MaoZiming): Fix.
  std::cout << "[uccl::internode_ll::barrier] dummy barrier invoked"
            << std::endl;
  return;
}

std::vector<uint8_t> get_unique_id() {
  // TODO(MaoZiming): Fix.
  return std::vector<uint8_t>(64, 0);  // Dummy unique ID
}