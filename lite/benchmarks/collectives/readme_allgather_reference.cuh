#pragma once
// Unchanged kernel bodies from the recorded four-L4 FP32 reference:
// benchmarks/data/small_medium_4l4_20261113/peer_before_mini_graph.cu.
// This benchmark-only extraction restores a fresh small/medium control; it is
// not a new public-library implementation. Kernel geometry is preserved.
#include <cooperative_groups.h>
namespace uccl_lite_reference {
constexpr int kRanks = 4;
__global__ void pushSerialGather(float const* input, float* out0, float* out1,
                                 float* out2, float* out3, size_t shard,
                                 int source) {
  float* targets[4] = {out0, out1, out2, out3};
  for (int phase = 0; phase < kRanks; ++phase) {
    int dest = (source + phase) % kRanks;
    float* output = targets[dest] + size_t(source) * shard;
    if (shard % 4 == 0) {
      auto const* in = reinterpret_cast<uint4 const*>(input);
      auto* out = reinterpret_cast<uint4*>(output);
      for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x)
        out[i] = in[i];
    } else
      for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
        output[i] = input[i];
    __syncthreads();
  }
}

__device__ void releaseSystemFlag(uint32_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value)
               : "memory");
}
__device__ uint32_t acquireSystemFlag(uint32_t const* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__global__ void miniHostGather(float const* input, float* output, float* h0,
                               float* h1, float* h2, float* h3, uint32_t* flags,
                               size_t shard, int rank, uint32_t generation) {
  auto grid = cooperative_groups::this_grid();
  float* host[4] = {h0, h1, h2, h3};
  if (grid.thread_rank() == 0)
    for (int r = 0; r < kRanks; ++r)
      while (acquireSystemFlag(flags + (4 + r) * 32) != generation - 1)
        __nanosleep(64);
  grid.sync();
  if (shard % 4 == 0) {
    auto* to = reinterpret_cast<uint4*>(host[rank]);
    auto const* from = reinterpret_cast<uint4 const*>(input);
    for (size_t i = grid.thread_rank(); i < shard / 4; i += grid.size())
      to[i] = from[i];
  } else
    for (size_t i = grid.thread_rank(); i < shard; i += grid.size())
      host[rank][i] = input[i];
  __threadfence_system();
  grid.sync();
  if (grid.thread_rank() == 0) releaseSystemFlag(flags + rank * 32, generation);
  int source = blockIdx.x % kRanks;
  if (threadIdx.x == 0 && source != rank)
    while (acquireSystemFlag(flags + source * 32) != generation)
      __nanosleep(64);
  __syncthreads();
  size_t tid = (blockIdx.x / kRanks) * blockDim.x + threadIdx.x;
  size_t stride = (gridDim.x / kRanks) * blockDim.x;
  float const* from = source == rank ? input : host[source];
  float* to = output + size_t(source) * shard;
  if (shard % 4 == 0) {
    for (size_t i = tid; i < shard / 4; i += stride)
      reinterpret_cast<uint4*>(to)[i] = reinterpret_cast<uint4 const*>(from)[i];
  } else
    for (size_t i = tid; i < shard; i += stride) to[i] = from[i];
  grid.sync();
  if (grid.thread_rank() == 0)
    releaseSystemFlag(flags + (4 + rank) * 32, generation);
}

}  // namespace uccl_lite_reference
