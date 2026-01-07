/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "dietgpu/ans/BatchPrefixSum.cuh"
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSStatistics.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/ans/GpuChecksum.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace dietgpu {

// maximum raw compressed data block size in bytes
constexpr __host__ __device__ uint32_t
getRawCompBlockMaxSize(uint32_t uncompressedBlockBytes) {
  // (an estimate from zstd)
  return roundUp(
      uncompressedBlockBytes + (uncompressedBlockBytes / 4), kBlockAlignment);
}

inline uint32_t getMaxBlockSizeUnCoalesced(uint32_t uncompressedBlockBytes) {
  // uncoalesced data has a warp state header
  return sizeof(ANSWarpState) + getRawCompBlockMaxSize(uncompressedBlockBytes);
}

inline uint32_t getMaxBlockSizeCoalesced(uint32_t uncompressedBlockBytes) {
  return getRawCompBlockMaxSize(uncompressedBlockBytes);
}

// Returns number of values written to the compressed output
// Assumes all lanes in the warp are presented valid input symbols
template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOneWarp(
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    ANSEncodedT* __restrict__ out,
    const uint4* __restrict__ smemLookup) {
  auto lookup = smemLookup[sym];

  uint32_t pdf = lookup.x;
  uint32_t cdf = lookup.y;
  uint32_t div_m1 = lookup.z;
  uint32_t div_shift = lookup.w;

  constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

  ANSStateT maxStateCheck = pdf * kStateCheckMul;
  bool write = (state >= maxStateCheck);

  auto vote = __ballot_sync(0xffffffff, write);
  auto prefix = __popc(vote & getLaneMaskLt());

  // Some lanes wish to write out their data
  if (write) {
    out[outOffset + prefix] = state & kANSEncodedMask;
    state >>= kANSEncodedBits;
  }

  constexpr uint32_t kProbBitsMul = 1 << ProbBits;

  uint32_t t = __umulhi(state, div_m1);
  // We prevent addition overflow here by restricting `state` to < 2^31
  // (kANSStateBits)
  uint32_t div = (t + state) >> div_shift;
  auto mod = state - (div * pdf);

  // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
  state = div * kProbBitsMul + mod + cdf;

  // how many values we actually write to the compressed output
  return __popc(vote);
}

// Returns number of values written to the compressed output
// Assumes only some lanes in the warp are presented valid input symbols
template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOnePartialWarp(
    // true for the lanes in the warp for which data read is valid
    bool valid,
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    ANSEncodedT* __restrict__ out,
    const uint4* __restrict__ smemLookup) {
  auto lookup = smemLookup[sym];

  uint32_t pdf = lookup.x;
  uint32_t cdf = lookup.y;
  uint32_t div_m1 = lookup.z;
  uint32_t div_shift = lookup.w;

  constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

  ANSStateT maxStateCheck = pdf * kStateCheckMul;
  bool write = valid && (state >= maxStateCheck);

  auto vote = __ballot_sync(0xffffffff, write);
  auto prefix = __popc(vote & getLaneMaskLt());

  // Some lanes wish to write out their data
  if (write) {
    out[outOffset + prefix] = state & kANSEncodedMask;
    state >>= kANSEncodedBits;
  }

  uint32_t t = __umulhi(state, div_m1);
  // We prevent addition overflow here by restricting `state` to < 2^31
  // (kANSStateBits)
  uint32_t div = (t + state) >> div_shift;
  auto mod = state - (div * pdf);

  // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
  constexpr uint32_t kProbBitsMul = 1 << ProbBits;
  state = valid ? div * kProbBitsMul + mod + cdf : state;

  // how many values we actually write to the compressed output
  return __popc(vote);
}

// Fully encode a single block of data, along with the state for that block as
// the initial header.
// Returns the number of compressed words (ANSEncodedT) written
template <int ProbBits>
__device__ uint32_t ansEncodeWarpBlock(
    // Current lane ID in the warp
    uint32_t laneId,
    // Input for this block
    const ANSDecodedT* __restrict__ in,
    // Number of ANSDecodedT words in this block
    uint32_t inWords,
    // encoded table in smem
    const uint4* __restrict__ table,
    // Output for this block
    ANSWarpState* __restrict__ out) {
  // where we write the compressed words
  ANSEncodedT* outWords = (ANSEncodedT*)(out + 1);

  // Start state value for this warp
  ANSStateT state = kANSStartState;

  uint32_t inOffset = laneId;
  uint32_t outOffset = 0;

  constexpr int kUnroll = 8;

  // Unrolled iterations
  uint32_t limit = roundDown(inWords, kWarpSize * kUnroll);
  {
    ANSDecodedT sym[kUnroll];

    for (; inOffset < limit; inOffset += kWarpSize * kUnroll) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        sym[j] = in[inOffset + j * kWarpSize];
      }

#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        outOffset +=
            encodeOneWarp<ProbBits>(state, sym[j], outOffset, outWords, table);
      }
    }
  }

  if (limit != inWords) {
    // Remainder iterations
    limit = roundDown(inWords, kWarpSize);

    // Whole warp iterations
    for (; inOffset < limit; inOffset += kWarpSize) {
      ANSDecodedT sym = in[inOffset];

      outOffset +=
          encodeOneWarp<ProbBits>(state, sym, outOffset, outWords, table);
    }

    // Partial warp iteration
    if (limit != inWords) {
      // Last iteration may not be a full warp
      bool valid = inOffset < inWords;
      ANSDecodedT sym = valid ? in[inOffset] : ANSDecodedT(0);

      outOffset += encodeOnePartialWarp<ProbBits>(
          valid, state, sym, outOffset, outWords, table);
    }
  }

  // Write final state at the beginning (aligned addresses)
  out->warpState[laneId] = state;

  // Number of compressed words written
  return outOffset;
}

// Fully encode a single, full sized block of data, along with the state for
// that block as the initial header.
// Returns the number of compressed words (ANSEncodedT) written
// UseVec4 means that each lane in the warp loads 4 bytes of the input at a
// time, with each byte compressed by the warp in an interleaved fashion.
// Such vectorization must match with decom
template <int ProbBits, int BlockSize, bool UseVec4>
struct ANSEncodeWarpFullBlock;

// template <int ProbBits, int BlockSize>
// struct ANSEncodeWarpFullBlock<ProbBits, BlockSize, true> {
//   static __device__ uint32_t encode(
//     // Current lane ID in the warp
//     uint32_t laneId,
//     // Input for this block
//     const ANSDecodedT* __restrict__ in,
//     // encoded table in smem
//     const uint4* __restrict__ table,
//     // Output for this block
//     ANSWarpState* __restrict__ out) {
//     // where we write the compressed words
//     ANSEncodedT* outWords = (ANSEncodedT*)(out + 1);

//     // Start state value for this warp
//     ANSStateT state = kANSStartState;

//     uint32_t outOffset = 0;

//     using VecT = uint32_t;

//     auto inV = (const VecT*)in;
//     inV += laneId;

//     // kUnroll 4, unroll 2 164.93 us
//     // kUnroll 8, unroll 0 161.86 us
//     constexpr int kUnroll = 16;

//     static_assert(
//       isEvenDivisor((int)BlockSize, (int)(kUnroll * kWarpSize *
//       sizeof(VecT))),
//       "");

//     for (int i = 0; i < BlockSize / (kWarpSize * sizeof(VecT));
//          i += kUnroll, inV += kUnroll * kWarpSize) {
//       VecT symV[kUnroll];

// #pragma unroll
//       for (int j = 0; j < kUnroll; ++j) {
//         symV[j] = inV[j * kWarpSize];
//       }

// #pragma unroll
//       for (int j = 0; j < kUnroll; ++j) {
//         asm volatile("prefetch.global.L2 [%0];" : : "l"(inV + 128 + j * 32));
// #pragma unroll
//         for (int k = 0; k < 4; ++k) {
//           outOffset += encodeOneWarp<ProbBits>(
//             state, symV[j] & 0xff, outOffset, outWords, table);

//           symV[j] >>= 8;
//         }
//       }
//     }

//     // Write final state at the beginning (aligned addresses)
//     out->warpState[laneId] = state;

//     // Number of compressed words written
//     return outOffset;
//   }
// };

template <int ProbBits, int BlockSize>
struct ANSEncodeWarpFullBlock<ProbBits, BlockSize, false> {
  static __device__ uint32_t encode(
      // Current lane ID in the warp
      uint32_t laneId,
      // Input for this block
      const ANSDecodedT* __restrict__ in,
      // encoded table in smem
      const uint4* __restrict__ table,
      // Output for this block
      ANSWarpState* __restrict__ out) {
    // Just use the normal implementation
    return ansEncodeWarpBlock<ProbBits>(laneId, in, BlockSize, table, out);
  }
};

template <int ProbBits, int BlockSize>
__device__ void ansEncodeBlocksFull(
    // input data for all blocks
    const ANSDecodedT* __restrict__ in,
    // length in ANSDecodedT words
    uint32_t uncompressedWords,
    // number of blocks that different warps will process
    uint32_t numBlocks,
    // the stride of each encoded output block
    uint32_t outBlockStride,
    // address of the output for all blocks
    uint8_t* __restrict__ out,
    // output array of per-block sizes of number of ANSEncodedT words per block
    uint32_t* __restrict__ compressedWords,
    // the encoding table that we will load into smem
    const uint4* __restrict__ table) {
  // grid-wide warp id
  auto tid = threadIdx.x;
  // so we know the block is warp uniform
  int block =
      __shfl_sync(0xffffffff, (blockIdx.x * blockDim.x + tid) / kWarpSize, 0);
  int laneId = getLaneId();

  __shared__ uint4 smemLookup[kNumSymbols];

  // we always have at least 256 threads
  if (tid < kNumSymbols) {
    smemLookup[tid] = table[tid];
  }

  __syncthreads();

  // How big is this block?
  uint32_t start = block * BlockSize;
  uint32_t end = min(start + BlockSize, uncompressedWords);

  auto blockSize = end - start;

  // Either the warp is an excess one, or the last block is not a full block and
  // needs to be processed using the partial kernel
  if (block >= numBlocks || blockSize != BlockSize) {
    return;
  }

  auto inBlock = in + start;
  auto outBlock = (ANSWarpState*)(out + block * outBlockStride);

  // all input blocks must meet alignment requirements
  assert(isPointerAligned(inBlock, kANSRequiredAlignment));

  auto outWords = ANSEncodeWarpFullBlock<ProbBits, BlockSize, false>::encode(
      laneId, inBlock, smemLookup, outBlock);

  if (laneId == 0) {
    // If the bound on max compressed size is not correct, this assert will go
    // off. This block of data was then somewhat adversarial in terms of
    // incompressibility. In this case, the getRawCompBlockMaxSize max estimate
    // needs to increase.
    assert(outWords <= getRawCompBlockMaxSize(BlockSize) / sizeof(ANSEncodedT));
    compressedWords[block] = outWords;
  }
}

template <int ProbBits, int BlockSize>
__device__ void ansEncodeBlocksPartial(
    // input data for all blocks
    const ANSDecodedT* __restrict__ in,
    // length in ANSDecodedT words
    uint32_t uncompressedWords,
    // number of blocks that different warps will process
    uint32_t numBlocks,
    // the stride of each encoded output block
    uint32_t outBlockStride,
    // address of the output for all blocks
    uint8_t* __restrict__ out,
    // output array of per-block sizes of number of ANSEncodedT words per block
    uint32_t* __restrict__ compressedWords,
    // the encoding table that we will load into smem
    const uint4* __restrict__ table) {
  int block = numBlocks - 1;
  uint32_t tid = threadIdx.x;
  int laneId = getLaneId();

  __shared__ uint4 smemLookup[kNumSymbols];

  // we always have at least 256 threads
  if (tid < kNumSymbols) {
    smemLookup[tid] = table[tid];
  }

  __syncthreads();

  // We only have a single partial block to handle
  if (tid >= kWarpSize) {
    return;
  }

  // How big is this block?
  uint32_t start = block * BlockSize;
  uint32_t end = min(start + BlockSize, uncompressedWords);

  auto blockSize = end - start;

  // If the end block is a full block, it would have been handled by the full
  // block kernel
  if (blockSize == BlockSize) {
    return;
  }

  auto inBlock = in + start;
  auto outBlock = (ANSWarpState*)(out + block * outBlockStride);

  // all input blocks must meet required alignment
  assert(isPointerAligned(inBlock, kANSRequiredAlignment));

  auto outWords = ansEncodeWarpBlock<ProbBits>(
      laneId, inBlock, blockSize, smemLookup, outBlock);

  if (laneId == 0) {
    // If the bound on max compressed size is not correct, this assert will go
    // off. This block of data was then somewhat adversarial in terms of
    // incompressibility. In this case, the getRawCompBlockMaxSize max estimate
    // needs to increase.
    assert(outWords <= getRawCompBlockMaxSize(BlockSize) / sizeof(ANSEncodedT));
    compressedWords[block] = outWords;
  }
}

template <typename InProvider, int ProbBits, int BlockSize>
__global__ void ansEncodeBatchFull(
    // Input data for all blocks
    InProvider inProvider,
    // maximum number of blocks across all the batch
    uint32_t maxNumCompressedBlocks,
    // maximum size of a compressed block
    uint32_t maxCompressedBlockSize,
    // address of the output for all blocks
    uint8_t* __restrict__ out,
    // output array of per-block sizes of number of ANSEncodedT words per block
    // per batch
    // [batch][numBlocks]
    uint32_t* __restrict__ compressedWords,
    // the encoding table that we will load into smem
    // [batch][kNumSymbols]
    const uint4* __restrict__ table) {
  // which batch element we are processing
  auto batch = blockIdx.y;

  // Number of blocks for the current problem
  uint32_t curSize = inProvider.getBatchSize(batch);
  uint32_t numBlocks = divUp(curSize, BlockSize);

  ansEncodeBlocksFull<ProbBits, BlockSize>(
      (const ANSDecodedT*)inProvider.getBatchStart(batch),
      curSize,
      numBlocks,
      maxCompressedBlockSize,
      out + batch * maxNumCompressedBlocks * maxCompressedBlockSize,
      compressedWords + batch * maxNumCompressedBlocks,
      table + batch * kNumSymbols);
}

template <typename InProvider, int ProbBits, int BlockSize>
__global__ void ansEncodeBatchPartial(
    // input data for all blocks
    InProvider inProvider,
    // maximum number of blocks across all the batch
    uint32_t maxNumCompressedBlocks,
    // maximum size of a compressed block
    uint32_t maxCompressedBlockSize,
    // address of the output for all blocks
    uint8_t* __restrict__ out,
    // output array of per-block sizes of number of ANSEncodedT words per block
    // per batch
    // [batch][numBlocks]
    uint32_t* __restrict__ compressedWords,
    // the encoding table that we will load into smem
    // [batch][kNumSymbols]
    const uint4* __restrict__ table) {
  // which batch element we are processing
  auto batch = blockIdx.y;

  // Number of blocks for the current problem
  uint32_t curSize = inProvider.getBatchSize(batch);
  uint32_t numBlocks = divUp(curSize, BlockSize);

  ansEncodeBlocksPartial<ProbBits, BlockSize>(
      (const ANSDecodedT*)inProvider.getBatchStart(batch),
      inProvider.getBatchSize(batch),
      numBlocks,
      maxCompressedBlockSize,
      out + batch * maxNumCompressedBlocks * maxCompressedBlockSize,
      compressedWords + batch * maxNumCompressedBlocks,
      table + batch * kNumSymbols);
}

template <typename A, int B>
struct Align {
  typedef uint32_t argument_type;
  typedef uint32_t result_type;

  __thrust_exec_check_disable__ template <typename T>
  __host__ __device__ uint32_t operator()(T x) const {
    constexpr int kDiv = B / sizeof(A);
    constexpr int kSize = kDiv < 1 ? 1 : kDiv;

    return roundUp(x, T(kSize));
  }
};

template <int Threads>
__device__ void ansEncodeCoalesce(
    const uint8_t* __restrict__ inUncoalescedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords,
    const uint32_t* __restrict__ compressedWordsPrefix,
    const uint32_t* __restrict__ checksum,
    const uint4* __restrict__ table,
    uint32_t probBits,
    bool useChecksum,
    uint32_t numBlocks,
    uint32_t uncompressedWords,
    uint8_t* __restrict__ out,
    uint32_t* __restrict__ compressedBytes) {
  auto block = blockIdx.x;
  auto tid = threadIdx.x;

  ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)out;

  // The first block will be responsible for the coalesced header
  if (block == 0) {
    if (tid == 0) {
      uint32_t totalCompressedWords = 0;

      // Could be a header for a zero sized array
      if (numBlocks > 0) {
        totalCompressedWords =
            // total number of compressed words in all blocks
            // this is already a multiple of kBlockAlignment /
            // sizeof(ANSEncodedT)
            compressedWordsPrefix[numBlocks - 1] +
            // this is not yet a multiple of kBlockAlignment /
            // sizeof(ANSEncodedT), but needs to be
            roundUp(
                compressedWords[numBlocks - 1],
                kBlockAlignment / sizeof(ANSEncodedT));
      }

      ANSCoalescedHeader header;
      header.setMagicAndVersion();
      header.setNumBlocks(numBlocks);
      header.setTotalUncompressedWords(uncompressedWords);
      header.setTotalCompressedWords(totalCompressedWords);
      header.setProbBits(probBits);
      header.setUseChecksum(useChecksum);

      if (useChecksum) {
        header.setChecksum(*checksum);
      }

      if (compressedBytes) {
        *compressedBytes = header.getTotalCompressedSize();
      }

      *headerOut = header;
    }

    auto probsOut = headerOut->getSymbolProbs();

    // Write out pdf
    for (int i = tid; i < kNumSymbols; i += Threads) {
      probsOut[i] = table[i].x;
    }
  }

  if (block >= numBlocks) {
    return;
  }

  // where our per-warp data lies
  auto uncoalescedBlock = inUncoalescedBlocks + block * uncoalescedBlockStride;

  // Write per-block warp state
  if (tid < kWarpSize) {
    auto warpStateIn = (ANSWarpState*)uncoalescedBlock;

    headerOut->getWarpStates()[block].warpState[tid] =
        warpStateIn->warpState[tid];
  }

  auto blockWordsOut = headerOut->getBlockWords(numBlocks);

  // Write out per-block word length
  for (auto i = blockIdx.x * Threads + tid; i < numBlocks;
       i += gridDim.x * Threads) {
    uint32_t lastBlockWords = uncompressedWords % kDefaultBlockSize;
    lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

    uint32_t blockWords =
        (i == numBlocks - 1) ? lastBlockWords : kDefaultBlockSize;

    blockWordsOut[i] = uint2{
        (blockWords << 16) | compressedWords[i], compressedWordsPrefix[i]};
  }

  // Number of compressed words in this block
  uint32_t numWords = compressedWords[block];

  // We always have a valid multiple of kBlockAlignment bytes on both
  // uncoalesced src and coalesced dest, even though numWords (actual encoded
  // words) may be less than that
  using LoadT = uint4;
  static_assert(sizeof(LoadT) == kBlockAlignment, "");

  uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

  auto inT = (const LoadT*)(uncoalescedBlock + sizeof(ANSWarpState));
  auto outT = (LoadT*)(headerOut->getBlockDataStart(numBlocks) +
                       compressedWordsPrefix[block]);

  for (uint32_t i = tid; i < limitEnd; i += Threads) {
    outT[i] = inT[i];
  }
}

template <typename SizeProvider, typename OutProvider, int Threads>
__global__ void ansEncodeCoalesceBatch(
    const uint8_t* __restrict__ inUncoalescedBlocks,
    SizeProvider sizeProvider,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords,
    const uint32_t* __restrict__ compressedWordsPrefix,
    const uint32_t* __restrict__ checksum,
    const uint4* __restrict__ table,
    uint32_t probBits,
    bool useChecksum,
    OutProvider outProvider,
    uint32_t* __restrict__ compressedBytes) {
  auto batch = blockIdx.y;
  auto uncompressedWords = sizeProvider.getBatchSize(batch);

  // Number of compressed blocks in this batch element
  auto numBlocks = divUp(uncompressedWords, kDefaultBlockSize);

  // Advance all pointers to handle our specific batch member
  inUncoalescedBlocks +=
      batch * uncoalescedBlockStride * maxNumCompressedBlocks;
  compressedWords += batch * maxNumCompressedBlocks;
  compressedWordsPrefix += batch * maxNumCompressedBlocks;
  compressedBytes += batch;
  checksum += batch;
  table += batch * kNumSymbols;

  ansEncodeCoalesce<Threads>(
      inUncoalescedBlocks,
      uncoalescedBlockStride,
      compressedWords,
      compressedWordsPrefix,
      checksum,
      table,
      probBits,
      useChecksum,
      numBlocks,
      uncompressedWords,
      (uint8_t*)outProvider.getBatchStart(batch),
      compressedBytes);
}

template <typename InProvider, typename OutProvider>
void ansEncodeBatchDevice(
    StackDeviceMemory& res,
    const ANSCodecConfig& config,
    uint32_t numInBatch,
    InProvider inProvider,
    const uint32_t* histogram_dev,
    uint32_t maxSize,
    OutProvider outProvider,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  auto maxUncompressedWords = maxSize / sizeof(ANSDecodedT);
  uint32_t maxNumCompressedBlocks =
      divUp(maxUncompressedWords, kDefaultBlockSize);

  // 1. Compute symbol statistics
  auto table_dev = res.alloc<uint4>(stream, numInBatch * kNumSymbols);

  if (histogram_dev) {
    // use pre-calculated histogram
    ansCalcWeights(
        numInBatch,
        config.probBits,
        inProvider,
        histogram_dev,
        table_dev.data(),
        stream);
  } else {
    auto tempHistogram_dev =
        res.alloc<uint32_t>(stream, numInBatch * kNumSymbols);

    // need to calculate a histogram
    ansHistogramBatch(numInBatch, inProvider, tempHistogram_dev.data(), stream);

    ansCalcWeights(
        numInBatch,
        config.probBits,
        inProvider,
        tempHistogram_dev.data(),
        table_dev.data(),
        stream);
  }

  // 2. Compute checksum on input data (optional)
  auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);
  if (config.useChecksum) {
    checksumBatch(numInBatch, inProvider, checksum_dev.data(), stream);
  }

  // 3. Allocate memory for the per-warp results
  // How much space in bytes we need to reserve for each warp's output
  uint32_t uncoalescedBlockStride =
      getMaxBlockSizeUnCoalesced(kDefaultBlockSize);

  auto compressedBlocks_dev = res.alloc<uint8_t>(
      stream, numInBatch * maxNumCompressedBlocks * uncoalescedBlockStride);

  // +1 in order to get the final sum as well
  auto compressedWords_dev =
      res.alloc<uint32_t>(stream, numInBatch * maxNumCompressedBlocks);

  // Exclusive prefix sum of the compressed sizes (so we know where to write in
  // the contiguous output). The offsets are aligned to a multple of 4
  auto compressedWordsPrefix_dev =
      res.alloc<uint32_t>(stream, numInBatch * maxNumCompressedBlocks);

  // Run per-warp encoding
  // (only if we have blocks to compress)
  if (maxNumCompressedBlocks > 0) {
    constexpr int kThreads = 256;

    // The grid for the full block kernel
    auto gridFull = dim3(
        divUp((int)maxNumCompressedBlocks, kThreads / kWarpSize), numInBatch);

    // The grid for the partial block kernel; at most 1 partial block per input
    // in the batch
    auto gridPartial = dim3(1, numInBatch);

#define RUN_ENCODE(BITS)                                       \
  do {                                                         \
    ansEncodeBatchFull<InProvider, BITS, kDefaultBlockSize>    \
        <<<gridFull, kThreads, 0, stream>>>(                   \
            inProvider,                                        \
            maxNumCompressedBlocks,                            \
            uncoalescedBlockStride,                            \
            compressedBlocks_dev.data(),                       \
            compressedWords_dev.data(),                        \
            table_dev.data());                                 \
                                                               \
    ansEncodeBatchPartial<InProvider, BITS, kDefaultBlockSize> \
        <<<gridPartial, kThreads, 0, stream>>>(                \
            inProvider,                                        \
            maxNumCompressedBlocks,                            \
            uncoalescedBlockStride,                            \
            compressedBlocks_dev.data(),                       \
            compressedWords_dev.data(),                        \
            table_dev.data());                                 \
  } while (false)

    switch (config.probBits) {
      case 9:
        RUN_ENCODE(9);
        break;
      case 10:
        RUN_ENCODE(10);
        break;
      case 11:
        RUN_ENCODE(11);
        break;
      default:
        CHECK(false) << "unhandled pdf precision " << config.probBits;
    }

#undef RUN_ENCODE
  }

  // Perform exclusive prefix sum of the number of compressed words per block,
  // so we know where to write the output. We align the blocks so that we can
  // write state values at 4 byte alignment at the beginning.
  // FIXME: probably some way to do this via thrust::exclusive_scan_by_key with
  // transform iterators and what not
  if (maxNumCompressedBlocks > 0) {
    auto sizeRequired =
        getBatchExclusivePrefixSumTempSize(numInBatch, maxNumCompressedBlocks);

    // FIXME: we can run a more minimal segmented prefix sum instead of using
    // maxNumCompressedBlocks
    if (sizeRequired == 0) {
      batchExclusivePrefixSum<uint32_t, Align<ANSEncodedT, kBlockAlignment>>(
          compressedWords_dev.data(),
          compressedWordsPrefix_dev.data(),
          nullptr,
          numInBatch,
          maxNumCompressedBlocks,
          Align<ANSEncodedT, kBlockAlignment>(),
          stream);
    } else {
      auto tempPrefixSum_dev = res.alloc<uint8_t>(stream, sizeRequired);

      batchExclusivePrefixSum<uint32_t, Align<ANSEncodedT, kBlockAlignment>>(
          compressedWords_dev.data(),
          compressedWordsPrefix_dev.data(),
          tempPrefixSum_dev.data(),
          numInBatch,
          maxNumCompressedBlocks,
          Align<ANSEncodedT, kBlockAlignment>(),
          stream);
    }
  }

  // Coalesce the data into one contiguous buffer
  // Even if there is nothing to compress, we still need to create a compression
  // header
  {
    constexpr int kThreads = 64;
    auto grid = dim3(std::max(maxNumCompressedBlocks, 1U), numInBatch);

    ansEncodeCoalesceBatch<InProvider, OutProvider, kThreads>
        <<<grid, kThreads, 0, stream>>>(
            compressedBlocks_dev.data(),
            inProvider,
            maxNumCompressedBlocks,
            uncoalescedBlockStride,
            compressedWords_dev.data(),
            compressedWordsPrefix_dev.data(),
            checksum_dev.data(),
            table_dev.data(),
            config.probBits,
            config.useChecksum,
            outProvider,
            outSize_dev);
  }

  CUDA_TEST_ERROR();
}

} // namespace dietgpu

#undef RUN_ENCODE_ALL
