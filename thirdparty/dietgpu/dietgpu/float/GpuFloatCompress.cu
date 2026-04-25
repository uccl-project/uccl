/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatCompress.cuh"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/GreenContext.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include "util/debug.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace dietgpu {

uint32_t getMaxFloatCompressedSize(FloatType floatType, uint32_t size) {
  // FP8: two fp8 values pack into one bf16-sized word; size is in pairs
  if (isFloat8Type(floatType)) {
    assert(size % 2 == 0);
    size = size / 2;
  }

  // kNotCompressed bytes per float are simply stored uncompressed
  // rounded up to 16 bytes to ensure alignment of the following ANS data
  // portion
  uint32_t baseSize = sizeof(GpuFloatHeader) + getMaxCompressedSize(size);

  switch (floatType) {
    case FloatType::kFloat16:
      baseSize += FloatTypeInfo<FloatType::kFloat16>::getUncompDataSize(size);
      break;
    case FloatType::kBFloat16:
      baseSize += FloatTypeInfo<FloatType::kBFloat16>::getUncompDataSize(size);
      break;
    case FloatType::kFloat32:
      baseSize += FloatTypeInfo<FloatType::kFloat32>::getUncompDataSize(size);
      break;
    case FloatType::kFloat8E4M3FN:
      baseSize += FloatTypeInfo<FloatType::kFloat8E4M3FN>::getUncompDataSize(size);
      break;
    case FloatType::kFloat8E5M2:
      baseSize += FloatTypeInfo<FloatType::kFloat8E5M2>::getUncompDataSize(size);
      break;
    default:
      UCCL_CHECK(false);
      break;
  }

  return baseSize;
}

void floatCompress(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    const void** in,
    const uint32_t* inSize,
    void** out,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  initGreenContextIfNeeded(getCurrentDevice());
  // FP8: sizes are in fp8 elements; the fused pipeline works in pairs (uint16),
  // so halve all sizes before sending to the device function.
  std::vector<uint32_t> halvedSizes;
  const uint32_t* effectiveInSize = inSize;
  if (isFloat8Type(config.floatType)) {
    halvedSizes.resize(numInBatch);
    for (uint32_t i = 0; i < numInBatch; ++i) {
      assert(inSize[i] % 2 == 0);
      halvedSizes[i] = inSize[i] / 2;
    }
    effectiveInSize = halvedSizes.data();
  }

  // Get the total and maximum input size
  uint32_t maxSize = 0;

  for (uint32_t i = 0; i < numInBatch; ++i) {
    maxSize = std::max(maxSize, effectiveInSize[i]);
  }

  // Copy data to device
  // To reduce latency, we prefer to coalesce all data together and copy as one
  // contiguous chunk
  static_assert(sizeof(void*) == sizeof(uintptr_t), "");
  static_assert(sizeof(uint32_t) <= sizeof(uintptr_t), "");

  // in, inSize, out
  auto params_dev = res.alloc<uintptr_t>(stream, numInBatch * 3);
  auto params_host =
      std::unique_ptr<uintptr_t[]>(new uintptr_t[3 * numInBatch]);

  std::memcpy(&params_host[0], in, numInBatch * sizeof(void*));
  std::memcpy(&params_host[numInBatch], effectiveInSize, numInBatch * sizeof(uint32_t));
  std::memcpy(&params_host[2 * numInBatch], out, numInBatch * sizeof(void*));

  CUDA_VERIFY(cudaMemcpyAsync(
      params_dev.data(),
      params_host.get(),
      3 * numInBatch * sizeof(uintptr_t),
      cudaMemcpyHostToDevice,
      stream));

  auto in_dev = (const void**)params_dev.data();
  auto inSize_dev = (const uint32_t*)(params_dev.data() + numInBatch);
  auto out_dev = (void**)(params_dev.data() + 2 * numInBatch);

  auto inProvider = BatchProviderPointer((void**)in_dev, inSize_dev);
  auto outProvider = BatchProviderPointer(out_dev);

  floatCompressDevice(
      res,
      config,
      numInBatch,
      inProvider,
      maxSize,
      outProvider,
      outSize_dev,
      stream);
}

void floatCompressSplitSize(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    const void* in_dev,
    const uint32_t* inSplitSizes,
    void* out_dev,
    uint32_t outStride,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  // FP8: the fused pipeline reads fp8 pairs as uint16, so use sizeof(uint16_t)
  // as word size and halve the split sizes.
  std::vector<uint32_t> halvedSplitSizes;
  const uint32_t* effectiveSplitSizes = inSplitSizes;
  size_t floatWordSize;
  if (isFloat8Type(config.floatType)) {
    floatWordSize = sizeof(uint16_t); // WordT for fp8 pairs
    halvedSplitSizes.resize(numInBatch);
    for (uint32_t i = 0; i < numInBatch; ++i) {
      assert(inSplitSizes[i] % 2 == 0);
      halvedSplitSizes[i] = inSplitSizes[i] / 2;
    }
    effectiveSplitSizes = halvedSplitSizes.data();
  } else {
    floatWordSize = getWordSizeFromFloatType(config.floatType);
  }

  auto splitSizeHost = std::vector<uint32_t>(numInBatch * 2);
  auto splitSize = splitSizeHost.data();
  auto splitSizePrefix = splitSizeHost.data() + numInBatch;
  uint32_t maxSplitSize = 0;

  for (uint32_t i = 0; i < numInBatch; ++i) {
    auto size = effectiveSplitSizes[i];

    splitSize[i] = size;
    if (i > 0) {
      splitSizePrefix[i] = splitSizePrefix[i - 1] + splitSize[i - 1];
    }

    maxSplitSize = std::max(size, maxSplitSize);
  }

  // Copy data to device
  // splitSize, splitSizePrefix
  auto sizes_dev = res.alloc<uint32_t>(stream, splitSizeHost.size());

  CUDA_VERIFY(cudaMemcpyAsync(
      sizes_dev.data(),
      splitSizeHost.data(),
      splitSizeHost.size() * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream));

  auto inProvider = BatchProviderSplitSize(
      (void*)in_dev,
      sizes_dev.data(),
      sizes_dev.data() + numInBatch,
      floatWordSize);

  auto outProvider = BatchProviderStride(out_dev, outStride);

  floatCompressDevice(
      res,
      config,
      numInBatch,
      inProvider,
      maxSplitSize,
      outProvider,
      outSize_dev,
      stream);
}


void floatCompressSplitOneBatch(dietgpu::StackDeviceMemory& res,
                                dietgpu::FloatCompressConfig const& config,
                                cudaStream_t stream,
                                FloatCompressSplitContext& ctx) {
  initGreenContextIfNeeded(getCurrentDevice());
  assert(ctx.params_dev.data() != nullptr);
  assert(ctx.maxSize > 0);

  // ---- unpack params ----
  auto in_dev = reinterpret_cast<void const**>(ctx.params_dev.data() + 0);
  auto inSize_dev =
      reinterpret_cast<uint32_t const*>(ctx.params_dev.data() + 1);
  auto out_dev = reinterpret_cast<void**>(ctx.params_dev.data() + 2);

  auto inProvider = BatchProviderPointer((void**)in_dev, inSize_dev);
  auto outProvider = BatchProviderPointer(out_dev);

  // ---- allocate split temporaries (stored as RAII in ctx) ----
  // FP8 types pack two values into one uint16, so the effective element
  // count (and therefore the compressed-row width) is half the raw byte size.
  uint32_t effectiveMaxSize = ctx.maxSize;
  if (isFloat8Type(config.floatType)) {
    assert(effectiveMaxSize % 2 == 0);
    effectiveMaxSize /= 2;
  }
  ctx.compRowStride = roundUp(effectiveMaxSize, sizeof(uint4));

  ctx.toComp_dev = res.alloc<uint8_t>(stream, ctx.compRowStride);

  ctx.histogram_dev = res.alloc<uint32_t>(stream, kNumSymbols);

  CUDA_VERIFY(cudaMemsetAsync(ctx.histogram_dev.data(), 0,
                              sizeof(uint32_t) * kNumSymbols, stream));

  // ---- run split kernel ----
  runSplitFloat<BatchProviderPointer, BatchProviderPointer>(
      config.floatType,
      1,  // numInBatch == 1
      inProvider, config.useChecksum,
      nullptr,  // checksum not allowed in float mode
      ctx.toComp_dev.data(), ctx.compRowStride, outProvider,
      ctx.histogram_dev.data(), stream);

  // ---- explicit sync: split is complete on return ----
  CUDA_VERIFY(cudaStreamSynchronize(stream));
}

void floatCompressEncodeOneBatch(dietgpu::StackDeviceMemory& res,
                                 dietgpu::FloatCompressConfig const& config,
                                 FloatCompressSplitContext& ctx,
                                 uint32_t* outSize_dev, cudaStream_t stream) {
  initGreenContextIfNeeded(getCurrentDevice());
  assert(ctx.params_dev.data() != nullptr);
  assert(ctx.toComp_dev.data() != nullptr);
  assert(ctx.histogram_dev.data() != nullptr);
  assert(ctx.maxSize > 0);

  // ---- unpack params ----
  auto in_dev = reinterpret_cast<void const**>(ctx.params_dev.data() + 0);
  auto inSize_dev =
      reinterpret_cast<uint32_t const*>(ctx.params_dev.data() + 1);
  auto out_dev = reinterpret_cast<void**>(ctx.params_dev.data() + 2);

  auto inProvider = BatchProviderPointer((void**)in_dev, inSize_dev);
  auto outProvider = BatchProviderPointer(out_dev);

  auto inProviderANS = FloatANSInProvider<BatchProviderPointer>(
      ctx.toComp_dev.data(), ctx.compRowStride, inProvider);

  // ---- ANS encode ----
  switch (config.floatType) {
    case FloatType::kFloat16: {
      auto outProviderANS =
          FloatANSOutProvider<FloatType::kFloat16, BatchProviderPointer,
                              BatchProviderPointer>(outProvider, inProvider);

      runANSEncodeForType<FloatType::kFloat16>(
          res, config.ansConfig, 1, inProvider, inProviderANS,
          ctx.histogram_dev.data(), ctx.maxSize, outProviderANS, outSize_dev,
          stream);
      break;
    }

    case FloatType::kBFloat16: {
      auto outProviderANS =
          FloatANSOutProvider<FloatType::kBFloat16, BatchProviderPointer,
                              BatchProviderPointer>(outProvider, inProvider);

      runANSEncodeForType<FloatType::kBFloat16>(
          res, config.ansConfig, 1, inProvider, inProviderANS,
          ctx.histogram_dev.data(), ctx.maxSize, outProviderANS, outSize_dev,
          stream);
      break;
    }

    case FloatType::kFloat32: {
      auto outProviderANS =
          FloatANSOutProvider<FloatType::kFloat32, BatchProviderPointer,
                              BatchProviderPointer>(outProvider, inProvider);

      runANSEncodeForType<FloatType::kFloat32>(
          res, config.ansConfig, 1, inProvider, inProviderANS,
          ctx.histogram_dev.data(), ctx.maxSize, outProviderANS, outSize_dev,
          stream);
      break;
    }

    case FloatType::kFloat8E4M3FN: {
      auto outProviderANS =
          FloatANSOutProvider<FloatType::kFloat8E4M3FN, BatchProviderPointer,
                              BatchProviderPointer>(outProvider, inProvider);

      runANSEncodeForType<FloatType::kFloat8E4M3FN>(
          res, config.ansConfig, 1, inProvider, inProviderANS,
          ctx.histogram_dev.data(), ctx.maxSize, outProviderANS, outSize_dev,
          stream);
      break;
    }

    case FloatType::kFloat8E5M2: {
      auto outProviderANS =
          FloatANSOutProvider<FloatType::kFloat8E5M2, BatchProviderPointer,
                              BatchProviderPointer>(outProvider, inProvider);

      runANSEncodeForType<FloatType::kFloat8E5M2>(
          res, config.ansConfig, 1, inProvider, inProviderANS,
          ctx.histogram_dev.data(), ctx.maxSize, outProviderANS, outSize_dev,
          stream);
      break;
    }

    default:
      assert(false);
  }
  CUDA_VERIFY(cudaStreamSynchronize(stream));
}

inline size_t getElementCountFromBytes(FloatType ft, size_t bytes) {
  const size_t wordSize = getWordSizeFromFloatType(ft);

  UCCL_CHECK(bytes % wordSize == 0)
      << "Bytes (" << bytes << ") not aligned with FloatType word size ("
      << wordSize << ")";

  return bytes / wordSize;
}

uint32_t getUncompDataSizeFromByteSize(FloatType floatType, uint32_t datasize) {
  if (floatType == FloatType::kFloat8E4M3FN) {
    assert(datasize % 2 == 0);
    uint32_t numPairs = datasize / 2;
    return FloatTypeInfo<FloatType::kFloat8E4M3FN>::getUncompDataSize(numPairs);
  }
  if (floatType == FloatType::kFloat8E5M2) {
    assert(datasize % 2 == 0);
    uint32_t numPairs = datasize / 2;
    return FloatTypeInfo<FloatType::kFloat8E5M2>::getUncompDataSize(numPairs);
  }
  uint32_t numFloats =
      static_cast<uint32_t>(getElementCountFromBytes(floatType, datasize));
  switch (floatType) {
    case FloatType::kFloat16: {
      return FloatTypeInfo<FloatType::kFloat16>::getUncompDataSize(numFloats);
    }
    case FloatType::kBFloat16: {
      return FloatTypeInfo<FloatType::kBFloat16>::getUncompDataSize(numFloats);
    }
    case FloatType::kFloat32: {
      return FloatTypeInfo<FloatType::kFloat32>::getUncompDataSize(numFloats);
    }
  }
  return 0;
}
}  // namespace dietgpu
