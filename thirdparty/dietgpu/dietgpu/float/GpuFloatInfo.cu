/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatInfo.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

void floatGetCompressedInfo(
    StackDeviceMemory& res,
    const void** in,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outTypes_dev,
    uint32_t* outChecksum_dev,
    cudaStream_t stream) {
  if (!outSizes_dev && !outTypes_dev && !outChecksum_dev) {
    return;
  }

  auto in_dev = res.copyAlloc<const void*>(stream, in, numInBatch);

  floatGetCompressedInfoDevice(
      res,
      in_dev.data(),
      numInBatch,
      outSizes_dev,
      outTypes_dev,
      outChecksum_dev,
      stream);

  CUDA_TEST_ERROR();
}

void floatGetCompressedInfoDevice(
    StackDeviceMemory& res,
    const void** in_dev,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outTypes_dev,
    uint32_t* outChecksum_dev,
    cudaStream_t stream) {
  if (!outSizes_dev && !outTypes_dev && !outChecksum_dev) {
    return;
  }

  auto inProvider = BatchProviderPointer((void**)in_dev);

  floatGetCompressedInfo(
      inProvider,
      numInBatch,
      outSizes_dev,
      outTypes_dev,
      outChecksum_dev,
      stream);
}

} // namespace dietgpu
