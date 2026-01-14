/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSEncode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/ans/GpuChecksum.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <memory>
#include <vector>

namespace dietgpu {

template <FloatType FT, int Threads>
struct SplitFloatNonAligned {
  static __device__ void split(
      const typename FloatTypeInfo<FT>::WordT* in,
      uint32_t size,
      typename FloatTypeInfo<FT>::CompT* compOut,
      typename FloatTypeInfo<FT>::NonCompT* nonCompOut,
      uint32_t* warpHistogram) {
    using FTI = FloatTypeInfo<FT>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
      CompT comp;
      NonCompT nonComp;
      FTI::split(in[i], comp, nonComp);

      atomicAdd(&warpHistogram[comp], 1);

      compOut[i] = comp;
      nonCompOut[i] = nonComp;
    }
  }
};

template <int Threads>
struct SplitFloatNonAligned<FloatType::kFloat32, Threads> {
  static __device__ void split(
      const typename FloatTypeInfo<FloatType::kFloat32>::WordT* in,
      uint32_t size,
      typename FloatTypeInfo<FloatType::kFloat32>::CompT* compOut,
      typename FloatTypeInfo<FloatType::kFloat32>::NonCompT* nonCompOut,
      uint32_t* warpHistogram) {
    using FTI = FloatTypeInfo<FloatType::kFloat32>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;

    // Where the low order 2 bytes are written
    uint16_t* nonComp2Out = (uint16_t*)nonCompOut;

    // Where the high order byte is written
    uint8_t* nonComp1Out = (uint8_t*)(nonComp2Out + roundUp(size, 8));

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
      CompT comp;
      NonCompT nonComp;
      FTI::split(in[i], comp, nonComp);

      nonComp2Out[i] = nonComp & 0xffffU;
      nonComp1Out[i] = nonComp >> 16;
      compOut[i] = comp;

      atomicAdd(&warpHistogram[comp], 1);
    }
  }
};

template <FloatType FT, int Threads>
struct SplitFloatAligned16 {
  static __device__ void split(
      const typename FloatTypeInfo<FT>::WordT* __restrict__ in,
      uint32_t size,
      typename FloatTypeInfo<FT>::CompT* __restrict__ compOut,
      typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompOut,
      uint32_t* warpHistogram) {
    using FTI = FloatTypeInfo<FT>;

    using WordT = typename FTI::WordT;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using VecT = typename FTI::VecT;
    using CompVecT = typename FTI::CompVecT;
    using NonCompVecT = typename FTI::NonCompVecT;

    constexpr int kOuterUnroll = 2;
    constexpr int kInnerUnroll = sizeof(VecT) / sizeof(WordT);

    const VecT* inV = (const VecT*)in;
    CompVecT* compOutV = (CompVecT*)compOut;
    NonCompVecT* nonCompOutV = (NonCompVecT*)nonCompOut;

    // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs
    // at a time, or Threads * kOuterUnroll 16-byte words at a time

    constexpr int kWordsPerBlock = Threads * kOuterUnroll;
    constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
    uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

    // Handle by block
    uint32_t startBlock = blockIdx.x * kWordsPerBlock;
    inV += startBlock + threadIdx.x;
    compOutV += startBlock + threadIdx.x;
    nonCompOutV += startBlock + threadIdx.x;

    for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                  inV += gridDim.x * kWordsPerBlock,
                  compOutV += gridDim.x * kWordsPerBlock,
                  nonCompOutV += gridDim.x * kWordsPerBlock) {
      VecT v[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        v[i] = inV[i * Threads];
      }

      CompVecT compV[kOuterUnroll];
      NonCompVecT nonCompV[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          CompT comp;
          NonCompT nonComp;
          FTI::split(v[i].x[j], comp, nonComp);

          atomicAdd(&warpHistogram[comp], 1);

          compV[i].x[j] = comp;
          nonCompV[i].x[j] = nonComp;
        }
      }

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        compOutV[i * Threads] = compV[i];
        nonCompOutV[i * Threads] = nonCompV[i];
      }
    }

    // Handle last (partial) block
    for (uint32_t i =
             fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
         i < size;
         i += gridDim.x * Threads) {
      CompT comp;
      NonCompT nonComp;
      FTI::split(in[i], comp, nonComp);

      atomicAdd(&warpHistogram[comp], 1);

      compOut[i] = comp;
      nonCompOut[i] = nonComp;
    }
  }
};

// float32 specialization
template <int Threads>
struct SplitFloatAligned16<FloatType::kFloat32, Threads> {
  static __device__ void split(
      const typename FloatTypeInfo<FloatType::kFloat32>::WordT* __restrict__ in,
      uint32_t size,
      typename FloatTypeInfo<FloatType::kFloat32>::CompT* __restrict__ compOut,
      typename FloatTypeInfo<
          FloatType::kFloat32>::NonCompT* __restrict__ nonCompOut,
      uint32_t* warpHistogram) {
    using FTI = FloatTypeInfo<FloatType::kFloat32>;

    using WordT = typename FTI::WordT;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;

    constexpr int kOuterUnroll = 1;
    constexpr int kInnerUnroll = sizeof(uint32x4) / sizeof(uint32_t);

    auto inV = (const uint32x4*)in;
    auto compOutV = (uint8x4*)compOut;

    auto nonCompOut2 = (uint16_t*)nonCompOut;
    auto nonCompOut1 = (uint8_t*)(nonCompOut2 + roundUp(size, 8));

    auto nonCompOutV2 = (uint16x4*)nonCompOut2;
    auto nonCompOutV1 = (uint8x4*)nonCompOut1;

    // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs
    // at a time, or Threads * kOuterUnroll 16-byte words at a time
    constexpr int kWordsPerBlock = Threads * kOuterUnroll;
    constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
    uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

    // Handle by block
    uint32_t startBlock = blockIdx.x * kWordsPerBlock;
    inV += startBlock + threadIdx.x;
    compOutV += startBlock + threadIdx.x;
    nonCompOutV2 += startBlock + threadIdx.x;
    nonCompOutV1 += startBlock + threadIdx.x;

    for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                  inV += gridDim.x * kWordsPerBlock,
                  compOutV += gridDim.x * kWordsPerBlock,
                  nonCompOutV2 += gridDim.x * kWordsPerBlock,
                  nonCompOutV1 += gridDim.x * kWordsPerBlock) {
      uint32x4 v[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        v[i] = inV[i * Threads];
      }

      uint8x4 compV[kOuterUnroll];
      uint32x4 nonCompV[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          CompT comp;
          NonCompT nonComp;
          FTI::split(v[i].x[j], comp, nonComp);

          atomicAdd(&warpHistogram[comp], 1);

          compV[i].x[j] = comp;
          nonCompV[i].x[j] = nonComp;
        }
      }

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        compOutV[i * Threads] = compV[i];

        uint16x4 nonCompV2;
        uint8x4 nonCompV1;
        for (int j = 0; j < kInnerUnroll; ++j) {
          nonCompV2.x[j] = nonCompV[i].x[j] & 0xffffU;
          nonCompV1.x[j] = nonCompV[i].x[j] >> 16;
        }

        nonCompOutV2[i * Threads] = nonCompV2;
        nonCompOutV1[i * Threads] = nonCompV1;
      }
    }

    // Handle last (partial) block
    for (uint32_t i =
             fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
         i < size;
         i += gridDim.x * Threads) {
      CompT comp;
      NonCompT nonComp;
      FTI::split(in[i], comp, nonComp);

      atomicAdd(&warpHistogram[comp], 1);

      compOut[i] = comp;
      nonCompOut2[i] = nonComp & 0xffffU;
      nonCompOut1[i] = nonComp >> 16;
    }
  }
};

template <
    typename InProvider,
    typename NonCompProvider,
    FloatType FT,
    int Threads>
__global__ void splitFloat(
    InProvider inProvider,
    bool useChecksum,
    const uint32_t* __restrict__ checksum,
    void* __restrict__ compOut,
    uint32_t compOutStride,
    NonCompProvider nonCompProvider,
    uint32_t* __restrict__ histogramOut) {
  using WordT = typename FloatTypeInfo<FT>::WordT;
  using CompT = typename FloatTypeInfo<FT>::CompT;
  using NonCompT = typename FloatTypeInfo<FT>::NonCompT;

  constexpr int kWarps = Threads / kWarpSize;
  static_assert(Threads == kNumSymbols, "");

  auto batch = blockIdx.y;
  auto warpId = threadIdx.x / kWarpSize;

  histogramOut += batch * kNumSymbols;
  checksum += batch;

  // +1 in order to force very common symbols that could overlap into different
  // banks between different warps
  __shared__ uint32_t histogram[kWarps][kNumSymbols + 1];

#pragma unroll
  for (int i = 0; i < kWarps; ++i) {
    histogram[i][threadIdx.x] = 0;
  }

  __syncthreads();

  uint32_t* warpHistogram = histogram[warpId];

  auto curIn = (const WordT*)inProvider.getBatchStart(batch);
  auto headerOut = (GpuFloatHeader*)nonCompProvider.getBatchStart(batch);
  auto curCompOut = (CompT*)compOut + compOutStride * batch;
  auto curSize = inProvider.getBatchSize(batch);

  // Write size as a header
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    GpuFloatHeader h;
    h.setMagicAndVersion();
    h.size = curSize;
    h.setFloatType(FT);
    h.setUseChecksum(useChecksum);

    if (useChecksum) {
      h.setChecksum(*checksum);
    }

    *headerOut = h;
  }

  auto curNonCompOut = (NonCompT*)(headerOut + 1);

  // How many bytes are before the point where we are 16 byte aligned?
  auto nonAlignedBytes = getAlignmentRoundUp<sizeof(uint4)>(curIn);

  if (nonAlignedBytes > 0) {
    SplitFloatNonAligned<FT, Threads>::split(
        curIn, curSize, curCompOut, curNonCompOut, warpHistogram);
  } else {
    SplitFloatAligned16<FT, Threads>::split(
        curIn, curSize, curCompOut, curNonCompOut, warpHistogram);
  }

  // Accumulate warp histogram data and write into the gmem histogram
  __syncthreads();

  uint32_t sum = histogram[0][threadIdx.x];
#pragma unroll
  for (int j = 1; j < kWarps; ++j) {
    sum += histogram[j][threadIdx.x];
  }

  // The count for the thread's bucket could be 0
  if (sum) {
    atomicAdd(&histogramOut[threadIdx.x], sum);
  }
}

// Update the final byte counts for the batch to take into account the
// uncompressed and compressed portions
template <FloatType FT, typename InProvider>
__global__ void
incOutputSizes(InProvider inProvider, uint32_t* outSize, uint32_t numInBatch) {
  uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    outSize[batch] += sizeof(GpuFloatHeader) +
        FloatTypeInfo<FT>::getUncompDataSize(inProvider.getBatchSize(batch));
  }
}

// Provides the input data to ANS compression
template <typename SizeProvider>
struct FloatANSInProvider {
  using Writer = BatchWriter;

  __host__
  FloatANSInProvider(void* ptr_dev, uint32_t stride, SizeProvider& sizeProvider)
      : ptr_dev_(ptr_dev), stride_(stride), sizeProvider_(sizeProvider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    return (uint8_t*)ptr_dev_ + batch * stride_;
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    return (uint8_t*)ptr_dev_ + batch * stride_;
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return sizeProvider_.getBatchSize(batch);
  }

  void* ptr_dev_;
  uint32_t stride_;
  SizeProvider sizeProvider_;
};

// Provides the output data to ANS compression
template <FloatType FT, typename OutProvider, typename SizeProvider>
struct FloatANSOutProvider {
  using Writer = BatchWriter;
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatANSOutProvider(
      OutProvider& outProvider,
      SizeProvider& sizeProvider)
      : outProvider_(outProvider), sizeProvider_(sizeProvider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)outProvider_.getBatchStart(batch);

    // Increment the pointer to past the floating point data
    ((GpuFloatHeader*)p)->checkMagicAndVersion();
    return p + sizeof(GpuFloatHeader) +
        FTI::getUncompDataSize(sizeProvider_.getBatchSize(batch));
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)outProvider_.getBatchStart(batch);

    // Increment the pointer to past the floating point data
    ((GpuFloatHeader*)p)->checkMagicAndVersion();
    return p + sizeof(GpuFloatHeader) +
        FTI::getUncompDataSize(sizeProvider_.getBatchSize(batch));
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  OutProvider outProvider_;
  SizeProvider sizeProvider_;
};

template <typename InProvider, typename OutProvider>
void floatCompressDevice(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    InProvider& inProvider,
    uint32_t maxSize,
    OutProvider& outProvider,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
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
  CUDA_VERIFY(cudaMemsetAsync(
      histogram_dev.data(),
      0,
      sizeof(uint32_t) * numInBatch * kNumSymbols,
      stream));

#define RUN_SPLIT(FLOAT_TYPE)                                      \
  do {                                                             \
    constexpr int kBlock = 256;                                    \
    auto& props = getCurrentDeviceProperties();                    \
    int maxBlocksPerSM = 0;                                        \
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
        &maxBlocksPerSM,                                           \
        splitFloat<InProvider, OutProvider, FLOAT_TYPE, kBlock>,   \
        kBlock,                                                    \
        0));                                                       \
    uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
    uint32_t perBatchGrid = 4 * divUp(maxGrid, numInBatch);        \
    auto grid = dim3(perBatchGrid, numInBatch);                    \
                                                                   \
    splitFloat<InProvider, OutProvider, FLOAT_TYPE, kBlock>        \
        <<<grid, kBlock, 0, stream>>>(                             \
            inProvider,                                            \
            config.useChecksum,                                    \
            checksum_dev.data(),                                   \
            toComp_dev.data(),                                     \
            compRowStride,                                         \
            outProvider,                                           \
            histogram_dev.data());                                 \
  } while (false)

  switch (config.floatType) {
    case FloatType::kFloat16:
      RUN_SPLIT(FloatType::kFloat16);
      break;
    case FloatType::kBFloat16:
      RUN_SPLIT(FloatType::kBFloat16);
      break;
    case FloatType::kFloat32:
      RUN_SPLIT(FloatType::kFloat32);
      break;
    default:
      assert(false);
      break;
  }

#undef RUN_SPLIT

    // outSize as reported by ansEncode is just the ANS-encoded portion of the
    // data.
    // We need to increment the sizes by the uncompressed portion (header plus
    // uncompressed float data) with incOutputSizes
#define RUN_ANS(FT)                                                         \
  do {                                                                      \
    auto inProviderANS = FloatANSInProvider<InProvider>(                    \
        toComp_dev.data(), compRowStride, inProvider);                      \
                                                                            \
    auto outProviderANS = FloatANSOutProvider<FT, OutProvider, InProvider>( \
        outProvider, inProvider);                                           \
                                                                            \
    ansEncodeBatchDevice(                                                   \
        res,                                                                \
        config.ansConfig,                                                   \
        numInBatch,                                                         \
        inProviderANS,                                                      \
        histogram_dev.data(),                                               \
        maxSize,                                                            \
        outProviderANS,                                                     \
        outSize_dev,                                                        \
        stream);                                                            \
                                                                            \
    incOutputSizes<FT><<<divUp(numInBatch, 128), 128, 0, stream>>>(         \
        inProvider, outSize_dev, numInBatch);                               \
                                                                            \
  } while (false)

  // We have written the non-compressed portions of the floats into the output,
  // along with a header that indicates how many floats there are.
  // For compression, we need to increment the address in which the compressed
  // outputs are written.

  switch (config.floatType) {
    case FloatType::kFloat16:
      RUN_ANS(FloatType::kFloat16);
      break;
    case FloatType::kBFloat16:
      RUN_ANS(FloatType::kBFloat16);
      break;
    case FloatType::kFloat32:
      RUN_ANS(FloatType::kFloat32);
      break;
    default:
      assert(false);
      break;
  }

#undef RUN_ANS

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
