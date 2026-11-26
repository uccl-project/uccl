#include "rank_permutation.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>

namespace uccl_lite::runtime {
namespace {
struct Mapping {
  int ranks, order[32];
};
__global__ void permute(uint32_t const* input, uint32_t* output, size_t count,
                        Mapping map, bool toExecution) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count * map.ranks; i += size_t(gridDim.x) * blockDim.x) {
    size_t logical = size_t(map.order[i / count]) * count + i % count;
    if (toExecution)
      output[i] = input[logical];
    else
      output[logical] = input[i];
  }
}
}  // namespace
void permuteRankShards(void const* input, void* output, size_t count,
                       int const* order, int ranks, bool toExecution,
                       cudaStream_t stream) {
  if (!input || !output || !order || ranks < 1 || ranks > 32 || !count ||
      count > std::numeric_limits<size_t>::max() / ranks / sizeof(uint32_t))
    throw std::invalid_argument("invalid rank permutation extent");
  Mapping map{};
  map.ranks = ranks;
  uint32_t used = 0;
  for (int i = 0; i < ranks; ++i) {
    if (order[i] < 0 || order[i] >= ranks || (used & (1u << order[i])))
      throw std::invalid_argument("rank permutation is not bijective");
    used |= 1u << order[i];
    map.order[i] = order[i];
  }
  int blocks = int(std::min<size_t>((count * ranks + 255) / 256, 256));
  permute<<<blocks, 256, 0, stream>>>(static_cast<uint32_t const*>(input),
                                      static_cast<uint32_t*>(output), count,
                                      map, toExecution);
  auto status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace uccl_lite::runtime
