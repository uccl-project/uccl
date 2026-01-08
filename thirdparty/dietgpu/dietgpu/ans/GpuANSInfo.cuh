/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

template <typename InProvider>
__global__ void ansGetCompressedInfoKernel(
    InProvider inProvider,
    uint32_t numInBatch,
    uint32_t* outSizes,
    uint32_t* outChecksum) {
  auto batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    auto header = (const ANSCoalescedHeader*)inProvider.getBatchStart(batch);
    // Make sure it is valid
    header->checkMagicAndVersion();

    if (outSizes) {
      outSizes[batch] = header->getTotalUncompressedWords();
    }

    if (outChecksum) {
      assert(header->getUseChecksum());
      outChecksum[batch] = header->getChecksum();
    }
  }
}

template <typename InProvider>
void ansGetCompressedInfo(
    InProvider& inProvider,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outChecksum_dev,
    cudaStream_t stream) {
  if (!outSizes_dev && !outChecksum_dev) {
    return;
  }

  auto block = 128;
  auto grid = divUp(numInBatch, block);

  ansGetCompressedInfoKernel<<<grid, block, 0, stream>>>(
      inProvider, numInBatch, outSizes_dev, outChecksum_dev);

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
