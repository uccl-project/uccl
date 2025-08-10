#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime_api.h>

namespace internode {
int init(std::vector<uint8_t> const& root_unique_id_val, int rank,
         int num_ranks, bool low_latency_mode);
void* alloc(std::size_t size, std::size_t alignment);
void finalize();
void barrier();
void free(void* ptr);
std::vector<uint8_t> get_unique_id();

}  // namespace internode

namespace intranode {
template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank);

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks,
             cudaStream_t stream);
}  // namespace intranode