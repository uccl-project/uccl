/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSDecode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatInfo.cuh"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <sstream>
#include <vector>

namespace dietgpu {

template <FloatType FT, int Threads>
struct JoinFloatNonAligned {
  static __device__ void join(
      const typename FloatTypeInfo<FT>::CompT* __restrict__ compIn,
      const typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompIn,
      uint32_t size,
      typename FloatTypeInfo<FT>::WordT* __restrict__ out) {
    for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < size;
         i += gridDim.x * Threads) {
      out[i] = FloatTypeInfo<FT>::join(compIn[i], nonCompIn[i]);
    }
  }
};

template <int Threads>
struct JoinFloatNonAligned<FloatType::kFloat32, Threads> {
  static __device__ void join(
      const typename FloatTypeInfo<
          FloatType::kFloat32>::CompT* __restrict__ compIn,
      const typename FloatTypeInfo<
          FloatType::kFloat32>::NonCompT* __restrict__ nonCompIn,
      uint32_t size,
      typename FloatTypeInfo<FloatType::kFloat32>::WordT* __restrict__ out) {
    using FTI = FloatTypeInfo<FloatType::kFloat32>;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;

    // Where the low order 2 bytes are read
    uint16_t* nonComp2In = (uint16_t*)nonCompIn;

    // Where the high order byte is read
    uint8_t* nonComp1In = (uint8_t*)(nonComp2In + roundUp(size, 8));

    for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < size;
         i += gridDim.x * Threads) {
      uint32_t nc =
          (uint32_t(nonComp1In[i]) * 65536U) + uint32_t(nonComp2In[i]);

      out[i] = FTI::join(compIn[i], nc);
    }
  }
};

template <FloatType FT, int Threads>
struct JoinFloatAligned16 {
  static __device__ void join(
      const typename FloatTypeInfo<FT>::CompT* __restrict__ compIn,
      const typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompIn,
      uint32_t size,
      typename FloatTypeInfo<FT>::WordT* __restrict__ out) {
    using FTI = FloatTypeInfo<FT>;

    using WordT = typename FTI::WordT;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;
    using VecT = typename FTI::VecT;
    using CompVecT = typename FTI::CompVecT;
    using NonCompVecT = typename FTI::NonCompVecT;

    constexpr int kOuterUnroll = 2;
    constexpr int kInnerUnroll = sizeof(VecT) / sizeof(WordT);

    const CompVecT* compInV = (const CompVecT*)compIn;
    const NonCompVecT* nonCompInV = (const NonCompVecT*)nonCompIn;
    VecT* outV = (VecT*)out;

    // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs
    // at a time, or Threads * kOuterUnroll 16-byte words at a time

    constexpr int kWordsPerBlock = Threads * kOuterUnroll;
    constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
    uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

    // Handle by block
    uint32_t startBlock = blockIdx.x * kWordsPerBlock;
    compInV += startBlock + threadIdx.x;
    nonCompInV += startBlock + threadIdx.x;
    outV += startBlock + threadIdx.x;

    for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                  compInV += gridDim.x * kWordsPerBlock,
                  nonCompInV += gridDim.x * kWordsPerBlock,
                  outV += gridDim.x * kWordsPerBlock) {
      CompVecT comp[kOuterUnroll];
      NonCompVecT nonComp[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        comp[i] = compInV[i * Threads];
        nonComp[i] = nonCompInV[i * Threads];
      }

      VecT v[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          v[i].x[j] = FTI::join(comp[i].x[j], nonComp[i].x[j]);
        }
      }

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        outV[i * Threads] = v[i];
      }
    }

    // Handle last (partial) block
    for (uint32_t i =
             fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
         i < size;
         i += blockDim.x) {
      out[i] = FTI::join(compIn[i], nonCompIn[i]);
    }
  }
};

// float32 specialization
template <int Threads>
struct JoinFloatAligned16<FloatType::kFloat32, Threads> {
  static __device__ void join(
      const typename FloatTypeInfo<
          FloatType::kFloat32>::CompT* __restrict__ compIn,
      const typename FloatTypeInfo<
          FloatType::kFloat32>::NonCompT* __restrict__ nonCompIn,
      uint32_t size,
      typename FloatTypeInfo<FloatType::kFloat32>::WordT* __restrict__ out) {
    using FTI = FloatTypeInfo<FloatType::kFloat32>;

    using WordT = typename FTI::WordT;
    using CompT = typename FTI::CompT;
    using NonCompT = typename FTI::NonCompT;

    constexpr int kOuterUnroll = 1;
    constexpr int kInnerUnroll = sizeof(uint32x4) / sizeof(uint32_t);

    auto compInV = (const uint8x4*)compIn;
    auto nonCompIn2 = (const uint16_t*)nonCompIn;
    auto nonCompIn1 = (const uint8_t*)(nonCompIn2 + roundUp(size, 8));

    auto nonCompInV2 = (uint16x4*)nonCompIn2;
    auto nonCompInV1 = (uint8x4*)nonCompIn1;

    auto outV = (uint32x4*)out;

    // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs
    // at a time, or Threads * kOuterUnroll 16-byte words at a time
    constexpr int kWordsPerBlock = Threads * kOuterUnroll;
    constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
    uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

    // Handle by block
    uint32_t startBlock = blockIdx.x * kWordsPerBlock;
    compInV += startBlock + threadIdx.x;
    nonCompInV2 += startBlock + threadIdx.x;
    nonCompInV1 += startBlock + threadIdx.x;
    outV += startBlock + threadIdx.x;

    for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                  compInV += gridDim.x * kWordsPerBlock,
                  nonCompInV2 += gridDim.x * kWordsPerBlock,
                  nonCompInV1 += gridDim.x * kWordsPerBlock,
                  outV += gridDim.x * kWordsPerBlock) {
      uint8x4 comp[kOuterUnroll];
      uint16x4 nonComp2[kOuterUnroll];
      uint8x4 nonComp1[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        comp[i] = compInV[i * Threads];
        nonComp2[i] = nonCompInV2[i * Threads];
        nonComp1[i] = nonCompInV1[i * Threads];
      }

      uint32x4 nonComp[kOuterUnroll];
#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          nonComp[i].x[j] = nonComp1[i].x[j] * 65536U + nonComp2[i].x[j];
        }
      }

      uint32x4 v[kOuterUnroll];

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
        for (int j = 0; j < kInnerUnroll; ++j) {
          v[i].x[j] = FTI::join(comp[i].x[j], nonComp[i].x[j]);
        }
      }

#pragma unroll
      for (uint32_t i = 0; i < kOuterUnroll; ++i) {
        outV[i * Threads] = v[i];
      }
    }

    // Handle last (partial) block
    for (uint32_t i =
             fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
         i < size;
         i += blockDim.x) {
      uint32_t nc2 = nonCompIn2[i];
      uint32_t nc1 = nonCompIn1[i];
      uint32_t nc = nc1 * 65536U + nc2;

      out[i] = FTI::join(compIn[i], nc);
    }
  }
};

template <FloatType FT, int Threads>
struct JoinFloatImpl {
  static __device__ void join(
      const typename FloatTypeInfo<FT>::CompT* compIn,
      const typename FloatTypeInfo<FT>::NonCompT* nonCompIn,
      uint32_t size,
      typename FloatTypeInfo<FT>::WordT* out) {
    // compIn should always be aligned, as we decompress into temporary memory
    auto compUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(compIn);
    auto nonCompUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(nonCompIn);
    auto outUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(out);

    if (compUnalignedBytes || nonCompUnalignedBytes || outUnalignedBytes) {
      JoinFloatNonAligned<FT, Threads>::join(compIn, nonCompIn, size, out);
    } else {
      JoinFloatAligned16<FT, Threads>::join(compIn, nonCompIn, size, out);
    }
  }
};

template <int Threads>
struct JoinFloatImpl<FloatType::kFloat32, Threads> {
  static __device__ void join(
      const typename FloatTypeInfo<FloatType::kFloat32>::CompT* compIn,
      const typename FloatTypeInfo<FloatType::kFloat32>::NonCompT* nonCompIn,
      uint32_t size,
      typename FloatTypeInfo<FloatType::kFloat32>::WordT* out) {
    // FIXME: implement vectorization
    JoinFloatNonAligned<FloatType::kFloat32, Threads>::join(
        compIn, nonCompIn, size, out);
  }
};

template <
    typename InProviderComp,
    typename InProviderNonComp,
    typename OutProvider,
    FloatType FT,
    int Threads>
__global__ void joinFloat(
    InProviderComp inProviderComp,
    InProviderNonComp inProviderNonComp,
    OutProvider outProvider,
    uint8_t* __restrict__ outSuccess,
    uint32_t* __restrict__ outSize) {
  using FTI = FloatTypeInfo<FT>;
  using WordT = typename FTI::WordT;
  using CompT = typename FTI::CompT;
  using NonCompT = typename FTI::NonCompT;

  auto batch = blockIdx.y;

  auto curCompIn = (const CompT*)inProviderComp.getBatchStart(batch);
  auto curHeaderIn =
      (const GpuFloatHeader*)inProviderNonComp.getBatchStart(batch);
  auto curOut = (WordT*)outProvider.getBatchStart(batch);

  // FIXME: test out capacity

  if (outSuccess && !outSuccess[batch]) {
    // ANS decompression failed, so nothing for us to do
    return;
  }

  // Get size as a header
  GpuFloatHeader h = *curHeaderIn;
  h.checkMagicAndVersion();

  auto curSize = h.size;

  if (outSize && (curSize != outSize[batch])) {
    // Reported size mismatch between ANS decompression and fp unpacking
    assert(false);
    return;
  }

  auto curNonCompIn = (const NonCompT*)(curHeaderIn + 1);

  JoinFloatImpl<FT, Threads>::join(curCompIn, curNonCompIn, curSize, curOut);
}

template <FloatType FT, typename InProvider>
struct FloatANSProvider {
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatANSProvider(InProvider& provider) : inProvider_(provider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

    // This is the first place that touches the header
    GpuFloatHeader h = *((GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) + FTI::getUncompDataSize(h.size);
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)inProvider_.getBatchStart(batch);

    // This is the first place that touches the header
    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) + FTI::getUncompDataSize(h.size);
  }

  InProvider inProvider_;
};

template <FloatType FT, int N>
struct FloatANSProviderInline {
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatANSProviderInline(int num, const void** in) {
    CHECK_LE(num, N);
    for (int i = 0; i < num; ++i) {
      in_[i] = in[i];
    }
  }

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)in_[batch];

    // This is the first place that touches the header
    GpuFloatHeader h = *((GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) + FTI::getUncompDataSize(h.size);
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)in_[batch];

    // This is the first place that touches the header
    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    h.checkMagicAndVersion();
    assert(FT == h.getFloatType());

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) + FTI::getUncompDataSize(h.size);
  }

  const void* in_[N];
};

template <FloatType FT, uint32_t BlockSize>
struct JoinFloatWriter {
  using FTI = FloatTypeInfo<FT>;

  __host__ __device__ JoinFloatWriter(
      uint32_t size,
      typename FTI::WordT* out,
      const typename FTI::NonCompT* nonComp)
      : out_(out),
        nonComp_(nonComp),
        outBlock_(nullptr),
        nonCompBlock_(nullptr) {}

  __host__ __device__ void setBlock(uint32_t block) {
    outBlock_ = out_ + block * BlockSize;
    nonCompBlock_ = nonComp_ + block * BlockSize;
  }

  __device__ void write(uint32_t offset, uint8_t sym) {
    auto nonComp = nonCompBlock_[offset];
    outBlock_[offset] = FTI::join(sym, nonComp);
  }

  // // The preload is an offset of a NonCompVec4
  // __device__ void preload(uint32_t offset) {
  //   // We can preload this before decompressing all of the ANS compressed
  //   data
  //   // to hide memory latency
  //   preload_ = ((typename FTI::NonCompVec4*)nonCompBlock_)[offset];
  // }

  //   __device__ void writeVec(uint32_t offset, ANSDecodedTx4 symV) {
  //     typename FTI::Vec4 outV;
  // #pragma unroll
  //     // We always receive 4 decoded values each iteration
  //     // FIXME: this is hacky
  //     for (int i = 0; i < 4; ++i) {
  //       outV.x[i] = JoinFloat<FT>::join(symV.x[i], preload_.x[i]);
  //     }

  //     ((typename FTI::Vec4*)outBlock_)[offset] = outV;
  //   }

  // typename FTI::NonCompVec4 preload_;
  typename FTI::WordT* out_;
  const typename FTI::NonCompT* nonComp_;
  typename FTI::WordT* outBlock_;
  const typename FTI::NonCompT* nonCompBlock_;
};

template <uint32_t BlockSize>
struct JoinFloatWriter<FloatType::kFloat32, BlockSize> {
  static constexpr bool kVectorize = false;
  using FTI = FloatTypeInfo<FloatType::kFloat32>;

  __host__ __device__ JoinFloatWriter(
      uint32_t size,
      typename FTI::WordT* out,
      const typename FTI::NonCompT* nonComp)
      : size_(size),
        out_(out),
        nonComp_(nonComp),
        outBlock_(nullptr),
        nonCompBlock2_(nullptr),
        nonCompBlock1_(nullptr) {}

  __host__ __device__ void setBlock(uint32_t block) {
    nonCompBlock2_ = (const uint16_t*)nonComp_ + block * BlockSize;
    nonCompBlock1_ =
        (const uint8_t*)((const uint16_t*)nonComp_ + roundUp(size_, 8U)) +
        block * BlockSize;
    outBlock_ = out_ + block * BlockSize;
  }

  __device__ void write(uint32_t offset, uint8_t sym) {
    uint32_t nc = uint32_t(nonCompBlock1_[offset]) * 65536U +
        uint32_t(nonCompBlock2_[offset]);

    outBlock_[offset] = FTI::join(sym, nc);
  }

  // // This implementation does not preload
  // __device__ void preload(uint32_t offset) {
  // }

  // // This implementation does not vectorize
  // __device__ void writeVec(uint32_t offset, ANSDecodedTx4 symV) {
  // }

  uint32_t size_;
  typename FTI::WordT* out_;
  const typename FTI::NonCompT* nonComp_;
  typename FTI::WordT* outBlock_;
  const uint16_t* nonCompBlock2_;
  const uint8_t* nonCompBlock1_;
};

template <
    typename InProvider,
    typename OutProvider,
    FloatType FT,
    uint32_t BlockSize>
struct FloatOutProvider {
  using Writer = JoinFloatWriter<FT, BlockSize>;
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatOutProvider(InProvider& inProvider, OutProvider& outProvider)
      : inProvider_(inProvider), outProvider_(outProvider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    return inProvider_.getBatchStart(batch);
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return outProvider_.getBatchSize(batch);
  }

  __device__ Writer getWriter(uint32_t batch) {
    // Get float header
    auto h = (const GpuFloatHeader*)getBatchStart(batch);

    return Writer(
        h->size,
        (typename FTI::WordT*)outProvider_.getBatchStart(batch),
        // advance past the header
        (const typename FTI::NonCompT*)(h + 1));
  }

  InProvider inProvider_;
  OutProvider outProvider_;
};

template <int N, FloatType FT, uint32_t BlockSize>
struct FloatOutProviderInline {
  using FTI = FloatTypeInfo<FT>;
  using Writer = JoinFloatWriter<FT, BlockSize>;

  __host__ FloatOutProviderInline(
      int num,
      const void** in,
      void** out,
      const uint32_t* outCapacity) {
    CHECK_LE(num, N);
    for (int i = 0; i < num; ++i) {
      in_[i] = in[i];
      out_[i] = out[i];
      outCapacity_[i] = outCapacity[i];
    }
  }

  __device__ void* getBatchStart(uint32_t batch) {
    return in_[batch];
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return outCapacity_[batch];
  }

  __device__ Writer getWriter(uint32_t batch) {
    // Get float header
    auto h = (const GpuFloatHeader*)getBatchStart(batch);

    return Writer(
        h->size,
        (typename FTI::WordT*)out_[batch],
        // advance past the header
        (const typename FTI::NonCompT*)(h + 1));
  }

  const void* in_[N];
  void* out_[N];
  uint32_t outCapacity_[N];
};

template <typename InProvider, typename OutProvider>
FloatDecompressStatus floatDecompressDevice(
    StackDeviceMemory& res,
    const FloatDecompressConfig& config,
    uint32_t numInBatch,
    InProvider& inProvider,
    OutProvider& outProvider,
    uint32_t maxCapacity,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  // not allowed in float mode
  assert(!config.ansConfig.useChecksum);

  // We can perform decoding in a single pass if all input data is 16 byte
  // aligned
  if (config.is16ByteAligned) {
    //
    // Fused kernel: perform decompression in a single pass
    //

#define RUN_FUSED(FT)                                                     \
  do {                                                                    \
    auto inProviderANS = FloatANSProvider<FT, InProvider>(inProvider);    \
    auto outProviderANS =                                                 \
        FloatOutProvider<InProvider, OutProvider, FT, kDefaultBlockSize>( \
            inProvider, outProvider);                                     \
                                                                          \
    ansDecodeBatch(                                                       \
        res,                                                              \
        config.ansConfig,                                                 \
        numInBatch,                                                       \
        inProviderANS,                                                    \
        outProviderANS,                                                   \
        outSuccess_dev,                                                   \
        outSize_dev,                                                      \
        stream);                                                          \
  } while (false)

    switch (config.floatType) {
      case FloatType::kFloat16:
        RUN_FUSED(FloatType::kFloat16);
        break;
      case FloatType::kBFloat16:
        RUN_FUSED(FloatType::kBFloat16);
        break;
      case FloatType::kFloat32:
        RUN_FUSED(FloatType::kFloat32);
        break;
      default:
        CHECK(false);
        break;
    }

#undef RUN_FUSED
  }

  else {
    //
    // Two pass kernel: decompress the ANS compressed data, then rejoin with
    // uncompressed data
    //

    // Temporary space for the decompressed exponents
    // We need to ensure 16 byte alignment for the decompressed data due to
    // vectorization
    uint32_t maxCapacityAligned = roundUp(maxCapacity, sizeof(uint4));

    auto exp_dev = res.alloc<uint8_t>(stream, numInBatch * maxCapacityAligned);

#define RUN_DECODE(FT)                                                    \
  do {                                                                    \
    using InProviderANS = FloatANSProvider<FT, InProvider>;               \
    auto inProviderANS = InProviderANS(inProvider);                       \
                                                                          \
    using OutProviderANS = BatchProviderStride;                           \
    auto outProviderANS = OutProviderANS(                                 \
        exp_dev.data(), maxCapacityAligned, maxCapacityAligned);          \
                                                                          \
    ansDecodeBatch(                                                       \
        res,                                                              \
        config.ansConfig,                                                 \
        numInBatch,                                                       \
        inProviderANS,                                                    \
        outProviderANS,                                                   \
        outSuccess_dev,                                                   \
        outSize_dev,                                                      \
        stream);                                                          \
                                                                          \
    constexpr int kThreads = 256;                                         \
    auto& props = getCurrentDeviceProperties();                           \
    int maxBlocksPerSM = 0;                                               \
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(            \
        &maxBlocksPerSM,                                                  \
        joinFloat<OutProviderANS, InProvider, OutProvider, FT, kThreads>, \
        kThreads,                                                         \
        0));                                                              \
    uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount;        \
    uint32_t perBatchGrid = divUp(maxGrid, numInBatch);                   \
    if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {      \
      perBatchGrid -= 1;                                                  \
    }                                                                     \
    auto grid = dim3(perBatchGrid, numInBatch);                           \
                                                                          \
    joinFloat<OutProviderANS, InProvider, OutProvider, FT, kThreads>      \
        <<<grid, kThreads, 0, stream>>>(                                  \
            outProviderANS,                                               \
            inProvider,                                                   \
            outProvider,                                                  \
            outSuccess_dev,                                               \
            outSize_dev);                                                 \
  } while (false)

    switch (config.floatType) {
      case FloatType::kFloat16:
        RUN_DECODE(FloatType::kFloat16);
        break;
      case FloatType::kBFloat16:
        RUN_DECODE(FloatType::kBFloat16);
        break;
      case FloatType::kFloat32:
        RUN_DECODE(FloatType::kFloat32);
        break;
      default:
        CHECK(false);
        break;
    }

#undef RUN_DECODE
  }

  FloatDecompressStatus status;

  // Perform optional checksum, if desired
  if (config.useChecksum) {
    auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);
    auto sizes_dev = res.alloc<uint32_t>(stream, numInBatch);
    auto archiveChecksum_dev = res.alloc<uint32_t>(stream, numInBatch);

    // Checksum the output data
    checksumBatch(numInBatch, outProvider, checksum_dev.data(), stream);

    // Get prior checksum from the float headers
    floatGetCompressedInfo(
        inProvider,
        numInBatch,
        sizes_dev.data(),
        nullptr,
        archiveChecksum_dev.data(),
        stream);

    // Compare against previously seen checksums on the host
    auto sizes = sizes_dev.copyToHost(stream);
    auto newChecksums = checksum_dev.copyToHost(stream);
    auto oldChecksums = archiveChecksum_dev.copyToHost(stream);

    std::stringstream errStr;

    for (int i = 0; i < numInBatch; ++i) {
      if (oldChecksums[i] != newChecksums[i]) {
        status.error = FloatDecompressError::ChecksumMismatch;

        errStr << "Checksum mismatch in batch member " << i
               << ": expected checksum " << std::hex << oldChecksums[i]
               << " got " << newChecksums[i] << "\n";
        status.errorInfo.push_back(std::make_pair(i, errStr.str()));
      }
    }
  }

  CUDA_TEST_ERROR();

  return status;
}

} // namespace dietgpu
