#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>

namespace uccl_lite::runtime {
// Bitwise 32-bit shard permutation between disjoint allocations. The map is
// execution-rank -> logical-rank, constructed during collective preparation.
void permuteRankShards(void const* input, void* output, size_t count,
                       int const* order, int ranks, bool toExecution,
                       cudaStream_t stream);
}  // namespace uccl_lite::runtime
