#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Bitwise FP32 payloads vary with rank, position and invocation.
__host__ __device__ inline uint32_t cePattern(size_t i, int rank, int epoch) {
  return uint32_t(0x3f800000 + ((i * 13 + rank * 37 + epoch * 19) % 127));
}
__global__ void ceFill(uint32_t* input, size_t count, int rank, int epoch) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    input[i] = cePattern(i, rank, epoch);
}
__global__ void ceCheck(uint32_t const* output, size_t count, int epoch,
                        unsigned* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count * 4;
       i += size_t(gridDim.x) * blockDim.x)
    if (output[i] != cePattern(i % count, int(i / count), epoch))
      atomicAdd(errors, 1u);
}
