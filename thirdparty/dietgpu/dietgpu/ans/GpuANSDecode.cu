/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSDecode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"

#include <glog/logging.h>
#include <cmath>
#include <memory>
#include <vector>

namespace dietgpu {

ANSDecodeStatus ansDecodeBatchStride(
    StackDeviceMemory& res,
    const ANSCodecConfig& config,
    uint32_t numInBatch,
    const void* in_dev,
    uint32_t inPerBatchStride,
    void* out_dev,
    uint32_t outPerBatchStride,
    uint32_t outPerBatchCapacity,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  auto inProvider = BatchProviderStride((void*)in_dev, inPerBatchStride);
  auto outProvider =
      BatchProviderStride(out_dev, outPerBatchStride, outPerBatchCapacity);

  return ansDecodeBatch(
      res,
      config,
      numInBatch,
      inProvider,
      outProvider,
      outSuccess_dev,
      outSize_dev,
      stream);
}

ANSDecodeStatus ansDecodeBatchPointer(
    StackDeviceMemory& res,
    const ANSCodecConfig& config,
    uint32_t numInBatch,
    const void** in,
    void** out,
    const uint32_t* outCapacity,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  // If the batch size is <= kBSLimit, we avoid cudaMemcpy and send all data at
  // kernel launch
  constexpr int kBSLimit = 128;

  if (numInBatch <= kBSLimit) {
    auto inProvider =
        BatchProviderInlinePointer<kBSLimit>(numInBatch, (void**)in);
    auto outProvider = BatchProviderInlinePointerCapacity<kBSLimit>(
        numInBatch, out, outCapacity);

    return ansDecodeBatch(
        res,
        config,
        numInBatch,
        inProvider,
        outProvider,
        outSuccess_dev,
        outSize_dev,
        stream);
  }

  // Otherwise, we have to perform h2d copies
  auto in_dev = res.alloc<void*>(stream, numInBatch);

  CUDA_VERIFY(cudaMemcpyAsync(
      in_dev.data(),
      in,
      numInBatch * sizeof(void*),
      cudaMemcpyHostToDevice,
      stream));

  auto out_dev = res.alloc<void*>(stream, numInBatch);

  CUDA_VERIFY(cudaMemcpyAsync(
      out_dev.data(),
      out,
      numInBatch * sizeof(void*),
      cudaMemcpyHostToDevice,
      stream));

  auto outCapacity_dev = res.alloc<uint32_t>(stream, numInBatch);

  CUDA_VERIFY(cudaMemcpyAsync(
      outCapacity_dev.data(),
      outCapacity,
      numInBatch * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream));

  // Data is now on the device
  auto inProvider = BatchProviderPointer(in_dev.data());
  auto outProvider =
      BatchProviderPointer(out_dev.data(), outCapacity_dev.data());

  return ansDecodeBatch(
      res,
      config,
      numInBatch,
      inProvider,
      outProvider,
      outSuccess_dev,
      outSize_dev,
      stream);
}

ANSDecodeStatus ansDecodeBatchSplitSize(
    StackDeviceMemory& res,
    const ANSCodecConfig& config,
    uint32_t numInBatch,
    const void** in,
    void* out_dev,
    const uint32_t* outSplitSizes,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  auto splitSizeHost = std::vector<uint32_t>(numInBatch * 2);
  auto splitSize = splitSizeHost.data();
  auto splitSizePrefix = splitSizeHost.data() + numInBatch;
  uint32_t maxSplitSize = 0;

  // check alignment
  CHECK_EQ(uintptr_t(out_dev) % kANSRequiredAlignment, 0);

  for (uint32_t i = 0; i < numInBatch; ++i) {
    auto size = outSplitSizes[i];

    if (i != (numInBatch - 1)) {
      // check alignment (internal splits affect alignment of things after it)
      CHECK_EQ(size % kANSRequiredAlignment, 0);
    }

    splitSize[i] = size;
    if (i > 0) {
      splitSizePrefix[i] = splitSizePrefix[i - 1] + splitSize[i - 1];
    }

    maxSplitSize = std::max(size, maxSplitSize);
  }

  // Concatenate splitSize and splitSizePrefix together for a single h2d copy
  auto sizes_dev = res.alloc<uint32_t>(stream, splitSizeHost.size());

  CUDA_VERIFY(cudaMemcpyAsync(
      sizes_dev.data(),
      splitSizeHost.data(),
      splitSizeHost.size() * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream));

  // FIXME: combine with above for a single h2d copy
  auto in_dev = res.alloc<void*>(stream, numInBatch);

  CUDA_VERIFY(cudaMemcpyAsync(
      in_dev.data(),
      in,
      numInBatch * sizeof(void*),
      cudaMemcpyHostToDevice,
      stream));

  auto inProvider = BatchProviderPointer(in_dev.data());

  auto outProvider = BatchProviderSplitSize(
      out_dev,
      sizes_dev.data(),
      sizes_dev.data() + numInBatch,
      sizeof(uint8_t));

  return ansDecodeBatch(
      res,
      config,
      numInBatch,
      inProvider,
      outProvider,
      outSuccess_dev,
      outSize_dev,
      stream);
}

} // namespace dietgpu
