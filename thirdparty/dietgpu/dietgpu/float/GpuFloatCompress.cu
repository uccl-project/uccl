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
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"
#include <glog/logging.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>
namespace dietgpu {

uint32_t getMaxFloatCompressedSize(FloatType floatType, uint32_t size) {
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
    default:
      CHECK(false);
      break;
  }

  return baseSize;
}

template <typename InProvider, typename OutProvider>
void floatCompressDevice(StackDeviceMemory& res,
                         FloatCompressConfig const& config, uint32_t numInBatch,
                         InProvider& inProvider, uint32_t maxSize,
                         OutProvider& outProvider, uint32_t* outSize_dev,
                         cudaStream_t stream,
                         std::shared_ptr<CompressMarker> marker) {
  auto maxUncompressedWords = maxSize / sizeof(ANSDecodedT);
  uint32_t maxNumCompressedBlocks =
      divUp(maxUncompressedWords, kDefaultBlockSize);

  // Compute checksum on input data (optional)
  auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);

  // not allowed in float mode
  assert(!config.ansConfig.useChecksum);

  if (config.useChecksum) {
    checksumBatch(numInBatch, inProvider, checksum_dev.data(), stream);
  }

  // Temporary space for the extracted exponents; all rows must be 16 byte
  // aligned
  uint32_t compRowStride = roundUp(maxSize, sizeof(uint4));
  auto toComp_dev = res.alloc<uint8_t>(stream, numInBatch * compRowStride);

  // We calculate a histogram of the symbols to be compressed as part of
  // extracting the compressible symbol from the float
  auto histogram_dev = res.alloc<uint32_t>(stream, numInBatch * kNumSymbols);

  // zero out buckets before proceeding, as we aggregate with atomic adds
  CUDA_VERIFY(cudaMemsetAsync(histogram_dev.data(), 0,
                              sizeof(uint32_t) * numInBatch * kNumSymbols,
                              stream));
  auto t0 = std::chrono::steady_clock::now();
  if (marker) {
    marker->create();
  }
  runSplitFloat<InProvider, OutProvider>(
      config.floatType, numInBatch, inProvider, config.useChecksum,
      checksum_dev.data(), toComp_dev.data(), compRowStride, outProvider,
      histogram_dev.data(), stream);

  // Record split completion event
  if (marker) {
    marker->recordSplit(stream);
  }

  // outSize as reported by ansEncode is just the ANS-encoded portion of the
  // data.
  // We need to increment the sizes by the uncompressed portion (header plus
  // uncompressed float data) with incOutputSizes
  //
  // We have written the non-compressed portions of the floats into the output,
  // along with a header that indicates how many floats there are.
  // For compression, we need to increment the address in which the compressed
  // outputs are written.
  auto inProviderANS = FloatANSInProvider<InProvider>(
      toComp_dev.data(), compRowStride, inProvider);
  if (marker) {
    marker->waitSplit();
  }
  auto tSplit = std::chrono::steady_clock::now();
  switch (config.floatType) {
    case FloatType::kFloat16: {
      auto outProviderANS =
          FloatANSOutProvider<FloatType::kFloat16, OutProvider, InProvider>(
              outProvider, inProvider);
      if (marker) {
        marker->setUncompDataSize(outProviderANS.getUncompDataSize(maxSize));
      }
      runANSEncodeForType<FloatType::kFloat16>(
          res, config.ansConfig, numInBatch, inProvider, inProviderANS,
          histogram_dev.data(), maxSize, outProviderANS, outSize_dev, stream);
      break;
    }
    case FloatType::kBFloat16: {
      auto outProviderANS =
          FloatANSOutProvider<FloatType::kBFloat16, OutProvider, InProvider>(
              outProvider, inProvider);
      if (marker) {
        marker->setUncompDataSize(outProviderANS.getUncompDataSize(maxSize));
      }
      runANSEncodeForType<FloatType::kBFloat16>(
          res, config.ansConfig, numInBatch, inProvider, inProviderANS,
          histogram_dev.data(), maxSize, outProviderANS, outSize_dev, stream);
      break;
    }
    case FloatType::kFloat32: {
      auto outProviderANS =
          FloatANSOutProvider<FloatType::kFloat32, OutProvider, InProvider>(
              outProvider, inProvider);
      if (marker) {
        marker->setUncompDataSize(outProviderANS.getUncompDataSize(maxSize));
      }
      runANSEncodeForType<FloatType::kFloat32>(
          res, config.ansConfig, numInBatch, inProvider, inProviderANS,
          histogram_dev.data(), maxSize, outProviderANS, outSize_dev, stream);
      break;
    }
    default:
      assert(false);
      break;
  }
  auto tCompress = std::chrono::steady_clock::now();
  auto splitUs =
      std::chrono::duration_cast<std::chrono::microseconds>(tSplit - t0)
          .count();
  auto compressUs =
      std::chrono::duration_cast<std::chrono::microseconds>(tCompress - t0)
          .count();

  auto deltaUs = compressUs - splitUs;
  // std::cout << "???Time submit to split done:    " << splitUs << " us" <<
  // std::endl; std::cout << "Time submit to compress done: " << compressUs << "
  // us" << std::endl; std::cout << "Delta submit (compress - split): " <<
  // deltaUs << " us" << std::endl; Record compress completion event
  if (marker) {
    // std::cout<< "recordCompress(stream);"<<std::endl;
    marker->recordCompress(stream);
  }

  CUDA_TEST_ERROR();
}

void floatCompress(StackDeviceMemory& res, FloatCompressConfig const& config,
                   uint32_t numInBatch, void const** in, uint32_t const* inSize,
                   void** out, uint32_t* outSize_dev, cudaStream_t stream,
                   std::shared_ptr<CompressMarker> marker) {
  // Get the total and maximum input size
  uint32_t maxSize = 0;

  for (uint32_t i = 0; i < numInBatch; ++i) {
    maxSize = std::max(maxSize, inSize[i]);
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
  cudaEvent_t preStart, preEnd;
  cudaEventCreate(&preStart);
  cudaEventCreate(&preEnd);

  cudaEventRecord(preStart, stream);
  std::memcpy(&params_host[0], in, numInBatch * sizeof(void*));
  std::memcpy(&params_host[numInBatch], inSize, numInBatch * sizeof(uint32_t));
  std::memcpy(&params_host[2 * numInBatch], out, numInBatch * sizeof(void*));

  CUDA_VERIFY(cudaMemcpyAsync(params_dev.data(), params_host.get(),
                              3 * numInBatch * sizeof(uintptr_t),
                              cudaMemcpyHostToDevice, stream));
  float pre_ms;

  cudaEventRecord(preEnd, stream);
  cudaEventSynchronize(preEnd);
  cudaEventElapsedTime(&pre_ms, preStart, preEnd);
  // std::cout<<"pre_ms: "<<pre_ms*1000<<std::endl;

  auto in_dev = (void const**)params_dev.data();
  auto inSize_dev = (uint32_t const*)(params_dev.data() + numInBatch);
  auto out_dev = (void**)(params_dev.data() + 2 * numInBatch);

  auto inProvider = BatchProviderPointer((void**)in_dev, inSize_dev);
  auto outProvider = BatchProviderPointer(out_dev);
  auto t0 = std::chrono::steady_clock::now();
  floatCompressDevice(res, config, numInBatch, inProvider, maxSize, outProvider,
                      outSize_dev, stream, marker);
  auto tCompress = std::chrono::steady_clock::now();
  auto compressUs =
      std::chrono::duration_cast<std::chrono::microseconds>(tCompress - t0)
          .count();
  // std::cout << "Time to compress submit: " << compressUs << " us" <<
  // std::endl;
}

void floatCompressSplitSize(StackDeviceMemory& res,
                            FloatCompressConfig const& config,
                            uint32_t numInBatch, void const* in_dev,
                            uint32_t const* inSplitSizes, void* out_dev,
                            uint32_t outStride, uint32_t* outSize_dev,
                            cudaStream_t stream) {
  auto floatWordSize = getWordSizeFromFloatType(config.floatType);

  auto splitSizeHost = std::vector<uint32_t>(numInBatch * 2);
  auto splitSize = splitSizeHost.data();
  auto splitSizePrefix = splitSizeHost.data() + numInBatch;
  uint32_t maxSplitSize = 0;

  for (uint32_t i = 0; i < numInBatch; ++i) {
    auto size = inSplitSizes[i];

    splitSize[i] = size;
    if (i > 0) {
      splitSizePrefix[i] = splitSizePrefix[i - 1] + splitSize[i - 1];
    }

    maxSplitSize = std::max(size, maxSplitSize);
  }

  // Copy data to device
  // splitSize, splitSizePrefix
  auto sizes_dev = res.alloc<uint32_t>(stream, splitSizeHost.size());

  CUDA_VERIFY(cudaMemcpyAsync(sizes_dev.data(), splitSizeHost.data(),
                              splitSizeHost.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream));

  auto inProvider =
      BatchProviderSplitSize((void*)in_dev, sizes_dev.data(),
                             sizes_dev.data() + numInBatch, floatWordSize);

  auto outProvider = BatchProviderStride(out_dev, outStride);

  auto t0 = std::chrono::steady_clock::now();
  floatCompressDevice(res, config, numInBatch, inProvider, maxSplitSize,
                      outProvider, outSize_dev, stream);
  auto tCompress = std::chrono::steady_clock::now();
  auto compressUs =
      std::chrono::duration_cast<std::chrono::microseconds>(tCompress - t0)
          .count();
  std::cout << "Time to compress submit: " << compressUs << " us" << std::endl;
}

void floatCompressOneBatch(dietgpu::StackDeviceMemory& res,
                           dietgpu::FloatCompressConfig const& config,
                           uint32_t numInBatch, uintptr_t* params_dev,
                           uint32_t maxSize, uint32_t* outSize_dev,
                           cudaStream_t stream,
                           std::shared_ptr<dietgpu::CompressMarker> marker) {
  assert(numInBatch == 1);
  assert(params_dev != nullptr);

  // params_dev layout:
  // [0] in ptr
  // [1] inSize
  // [2] out ptr
  auto in_dev = reinterpret_cast<void const**>(params_dev + 0);
  auto inSize_dev = reinterpret_cast<uint32_t const*>(params_dev + 1);
  auto out_dev = reinterpret_cast<void**>(params_dev + 2);

  auto inProvider = BatchProviderPointer((void**)in_dev, inSize_dev);
  auto outProvider = BatchProviderPointer(out_dev);

  floatCompressDevice(res, config, 1, inProvider, maxSize, outProvider,
                      outSize_dev, stream, marker);
}

void floatCompressSplitOneBatch(dietgpu::StackDeviceMemory& res,
                                dietgpu::FloatCompressConfig const& config,
                                cudaStream_t stream,
                                FloatCompressSplitContext& ctx) {
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
  ctx.compRowStride = roundUp(ctx.maxSize, sizeof(uint4));

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

    default:
      assert(false);
  }
  CUDA_VERIFY(cudaStreamSynchronize(stream));
}

uint32_t getUncompDataSizeFromByteSize(FloatType floatType, uint32_t datasize) {
  uint32_t numFloats =
      static_cast<uint32_t>(getElementCountFromBytes(floatType, datasize));
  switch (floatType) {
    case FloatType::kFloat16: {
      return FloatTypeInfo<FloatType::kFloat16>::getUncompDataSize(numFloats);
      break;
    }
    case FloatType::kBFloat16: {
      return FloatTypeInfo<FloatType::kBFloat16>::getUncompDataSize(numFloats);
      break;
    }
    case FloatType::kFloat32: {
      return FloatTypeInfo<FloatType::kFloat32>::getUncompDataSize(numFloats);
      break;
    }
  }
  return 0;
}
}  // namespace dietgpu
