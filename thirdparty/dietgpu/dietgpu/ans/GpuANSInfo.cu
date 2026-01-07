/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSInfo.cuh"

namespace dietgpu {

void ansGetCompressedInfo(
    StackDeviceMemory& res,
    const void** in,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outChecksum_dev,
    cudaStream_t stream) {
  if (!outSizes_dev && !outChecksum_dev) {
    return;
  }

  auto in_dev = res.copyAlloc<void*>(stream, (void**)in, numInBatch);
  ansGetCompressedInfoDevice(
      res,
      (const void**)in_dev.data(),
      numInBatch,
      outSizes_dev,
      outChecksum_dev,
      stream);
}

void ansGetCompressedInfoDevice(
    StackDeviceMemory& res,
    const void** in_dev,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outChecksum_dev,
    cudaStream_t stream) {
  if (!outSizes_dev && !outChecksum_dev) {
    return;
  }

  auto inProvider = BatchProviderPointer((void**)in_dev);
  ansGetCompressedInfo(
      inProvider, numInBatch, outSizes_dev, outChecksum_dev, stream);
}

} // namespace dietgpu
