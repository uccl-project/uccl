/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <assert.h>
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

template <typename InProvider>
__global__ void floatGetCompressedInfoKernel(
    InProvider inProvider,
    uint32_t numInBatch,
    uint32_t* outSizes,
    uint32_t* outTypes,
    uint32_t* outChecksum) {
  auto batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    auto header = (const GpuFloatHeader*)inProvider.getBatchStart(batch);
    header->checkMagicAndVersion();

    if (outSizes) {
      outSizes[batch] = header->size;
    }
    if (outTypes) {
      outTypes[batch] = uint32_t(header->getFloatType());
    }
    if (outChecksum) {
      assert(header->getUseChecksum());
      outChecksum[batch] = header->getChecksum();
    }
  }
}

template <typename InProvider>
void floatGetCompressedInfo(
    InProvider& inProvider,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outTypes_dev,
    uint32_t* outChecksum_dev,
    cudaStream_t stream) {
  if (!outSizes_dev && !outTypes_dev && !outTypes_dev) {
    return;
  }

  auto block = 128;
  auto grid = divUp(numInBatch, block);

  floatGetCompressedInfoKernel<<<grid, block, 0, stream>>>(
      inProvider, numInBatch, outSizes_dev, outTypes_dev, outChecksum_dev);

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
