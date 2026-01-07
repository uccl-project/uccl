/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StaticUtils.h"

#include <cub/cub.cuh>

namespace dietgpu {

template <typename T>
struct ReduceXor {
  __host__ __device__ __forceinline__ T
  operator()(const T& a, const T& b) const {
    return a ^ b;
  }
};

template <int Threads>
__device__ void checksumSingle(
    const uint8_t* __restrict__ in,
    uint32_t size,
    uint32_t* __restrict__ out) {
  // FIXME: general presumption in dietgpu that input data for ANS is only byte
  // aligned, while float data is only float word aligned, whereas ideally we
  // would like a 32 bit checksum. Since there is ultimately no guarantee of
  // anything but byte alignment and we wish to compute the same checksum
  // regardless of memory placement, the only checksum that makes sense to
  // produce is uint8.
  // We can fix this to compute a full 32-bit checksum by keeping track of
  // initial alignment and shuffling data around I think.
  uint32_t checksum32 = 0;

  // If the size of batch is smaller than the increment for alignment, we only
  // handle the batch
  auto roundUp4 = min(size, getAlignmentRoundUp<sizeof(uint4)>(in));

  // The size of data that remains after alignment
  auto remaining = size - roundUp4;

  // The size of data (in uint4 words) that we can process with alignment
  uint32_t numU4 = divDown(remaining, sizeof(uint4));

  auto inAligned = in + roundUp4;
  auto inAligned4 = (const uint4*)inAligned;

  // Handle the non-aligned portion that we have to load as single bytes, if any
  if (blockIdx.x == 0 && threadIdx.x < roundUp4) {
    static_assert(sizeof(uint4) <= Threads, "");
    checksum32 ^= in[threadIdx.x];
  }

  // Handle the portion that is aligned and uint4 vectorizable
  // 37.60 us / 80.76% gmem / 51.29% smem for uint4 on A100
  for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < numU4;
       i += gridDim.x * Threads) {
    uint4 v = inAligned4[i];

    checksum32 ^= v.x;
    checksum32 ^= v.y;
    checksum32 ^= v.z;
    checksum32 ^= v.w;
  }

  if (blockIdx.x == 0) {
    // Handle the remainder portion that doesn't comprise full words
    int i = numU4 * sizeof(uint4) + threadIdx.x;
    if (i < remaining) {
      checksum32 ^= inAligned[i];
    }
  }

  // Fold the bytes of checksum32
  checksum32 = (checksum32 & 0xffU) ^ ((checksum32 >> 8) & 0xffU) ^
      ((checksum32 >> 16) & 0xffU) ^ ((checksum32 >> 24) & 0xffU);

  // Reduce within a warp
  using BlockReduce = cub::BlockReduce<uint32_t, Threads>;
  __shared__ typename BlockReduce::TempStorage smem;

  checksum32 = BlockReduce(smem).Reduce(checksum32, ReduceXor<uint32_t>());

  if (threadIdx.x == 0) {
    atomicXor(out, checksum32);
  }
}

template <typename InProvider, int Threads>
__global__ void checksumBatch(InProvider in, uint32_t* out) {
  auto batch = blockIdx.y;
  out += batch;

  checksumSingle<Threads>(
      (const uint8_t*)in.getBatchStart(batch), in.getBatchSize(batch), out);
}

template <typename InProvider>
void checksumBatch(
    uint32_t numInBatch,
    InProvider inProvider,
    // size numInBatch
    uint32_t* checksum_dev,
    cudaStream_t stream) {
  // zero out checksum before proceeding, as we aggregate with atomic xor
  CUDA_VERIFY(
      cudaMemsetAsync(checksum_dev, 0, sizeof(uint32_t) * numInBatch, stream));

  constexpr uint32_t kThreads = 256;

  // We unfortunately don't know the per-batch element sizes in advance
  // What is the maximum number of blocks to saturate the GPU just in case some
  // per-batch members are big?
  int maxBlocks = 0;
  CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks, checksumBatch<InProvider, kThreads>, kThreads, 0));
  maxBlocks *= getCurrentDeviceProperties().multiProcessorCount;

  // The y block dimension will be for each batch element
  uint32_t xBlocks = divUp(maxBlocks, numInBatch);
  auto grid = dim3(xBlocks, numInBatch);

  checksumBatch<InProvider, kThreads>
      <<<grid, kThreads, 0, stream>>>(inProvider, checksum_dev);

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
