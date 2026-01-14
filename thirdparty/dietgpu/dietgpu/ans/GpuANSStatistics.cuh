/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StaticUtils.h"

#include <cmath>
#include <cub/cub.cuh>
#include <memory>

namespace dietgpu {

template <int Threads>
__device__ void histogramSingle(
    const ANSDecodedT* __restrict__ in,
    uint32_t size,
    uint32_t* __restrict__ out) {
  constexpr int kWarps = Threads / kWarpSize;
  static_assert(Threads == kNumSymbols, "");

  // +1 in order to force very common symbols that could overlap into different
  // banks?
  __shared__ uint32_t buckets[kWarps][kNumSymbols + 1];

  auto warpId = threadIdx.x / kWarpSize;

#pragma unroll
  for (int i = 0; i < kWarps; ++i) {
    buckets[i][threadIdx.x] = 0;
  }

  __syncthreads();

  uint32_t* warpBucket = buckets[warpId];

  // If the size of batch is smaller than the increment for alignment, we only
  // handle the batch
  auto roundUp4 = min(size, getAlignmentRoundUp<sizeof(uint4)>(in));

  // The size of data that remains after alignment
  auto remaining = size - roundUp4;

  // The size of data (in uint4 words) that we can process with alignment
  uint32_t numU4 = divDown(remaining, sizeof(uint4));

  auto inAligned = in + roundUp4;
  auto inAligned4 = (const uint4*)inAligned;

  // Handle the non-aligned portion that we have to load as single words, if any
  if (blockIdx.x == 0 && threadIdx.x < roundUp4) {
    static_assert(sizeof(uint4) <= Threads, "");
    atomicAdd(&warpBucket[in[threadIdx.x]], 1);
  }

  // Handle the portion that is aligned and uint4 vectorizable
  // 37.60 us / 80.76% gmem / 51.29% smem for uint4 on A100
  for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < numU4;
       i += gridDim.x * Threads) {
    uint4 v = inAligned4[i];

    {
      uint32_t x = v.x;
      atomicAdd(&warpBucket[x & 0xff], 1);
      x >>= 8;
      atomicAdd(&warpBucket[x & 0xff], 1);
      x >>= 8;
      atomicAdd(&warpBucket[x & 0xff], 1);
      x >>= 8;
      atomicAdd(&warpBucket[x], 1);
    }

    {
      uint32_t y = v.y;
      atomicAdd(&warpBucket[y & 0xff], 1);
      y >>= 8;
      atomicAdd(&warpBucket[y & 0xff], 1);
      y >>= 8;
      atomicAdd(&warpBucket[y & 0xff], 1);
      y >>= 8;
      atomicAdd(&warpBucket[y], 1);
    }

    {
      uint32_t z = v.z;
      atomicAdd(&warpBucket[z & 0xff], 1);
      z >>= 8;
      atomicAdd(&warpBucket[z & 0xff], 1);
      z >>= 8;
      atomicAdd(&warpBucket[z & 0xff], 1);
      z >>= 8;
      atomicAdd(&warpBucket[z], 1);
    }

    {
      uint32_t w = v.w;
      atomicAdd(&warpBucket[w & 0xff], 1);
      w >>= 8;
      atomicAdd(&warpBucket[w & 0xff], 1);
      w >>= 8;
      atomicAdd(&warpBucket[w & 0xff], 1);
      w >>= 8;
      atomicAdd(&warpBucket[w], 1);
    }
  }

  if (blockIdx.x == 0) {
    // Handle the remainder portion that doesn't comprise full words
    int i = numU4 * sizeof(uint4) + threadIdx.x;
    if (i < remaining) {
      atomicAdd(&warpBucket[inAligned[i]], 1);
    }
  }

  __syncthreads();

  uint32_t sum = buckets[0][threadIdx.x];
#pragma unroll
  for (int j = 1; j < kWarps; ++j) {
    sum += buckets[j][threadIdx.x];
  }

  // The count for the thread's bucket could be 0
  if (sum) {
    atomicAdd(&out[threadIdx.x], sum);
  }
}

template <typename InProvider, int Threads>
__global__ void histogramBatch(InProvider in, uint32_t* out) {
  auto batch = blockIdx.y;
  out += batch * kNumSymbols;

  histogramSingle<Threads>(
      (const ANSDecodedT*)in.getBatchStart(batch), in.getBatchSize(batch), out);
}

// sum that allows passing in smem for usage, so as to avoid a trailing
// syncthreads and associated latency
template <int Threads>
__device__ inline int
blockSum(int warpId, int laneId, int valForSum, int* smem) {
  static_assert(isEvenDivisor(Threads, kWarpSize), "");
  constexpr int kWarps = Threads / kWarpSize;

  auto allSum = warpReduceAllSum(valForSum);

  if (laneId == 0) {
    smem[warpId] = allSum;
  }
  __syncthreads();

  if (warpId == 0) {
    int v = laneId < kWarps ? smem[laneId] : 0;
    v = warpReduceAllSum(v);

    if (laneId == 0) {
      smem[0] = v;
    }
  }

  __syncthreads();

  // trailing syncthreads is elsewhere
  return smem[0];
}

// Function that allows normalization of symbol probabilities with a varying
// (statically known) number of threads, to allow for kernel fusion as needed
// Stand-alone normalization will use Threads == kNumSymbols (256)
template <int Threads>
__device__ void normalizeProbabilitiesFromHistogram(
    // Size 256 histogram in gmem
    const uint32_t* __restrict__ counts,
    uint32_t totalNum,
    int probBits,
    uint4* __restrict__ table) {
  static_assert(
      kNumSymbols == Threads || isEvenDivisor(kNumSymbols, uint32_t(Threads)),
      "");

  constexpr int kNumSymPerThread =
      kNumSymbols == Threads ? 1 : (kNumSymbols / Threads);

  // There's nothing to do if the input array in the batch was of zero size
  if (totalNum == 0) {
    return;
  }

  constexpr int kWarps = Threads / kWarpSize;
  uint32_t kProbWeight = 1 << probBits;
  auto tid = threadIdx.x;
  int warpId = tid / kWarpSize;
  int laneId = getLaneId();

  // Load the current count and compute the min/max non-zero values, then
  // perform an approximate quantization
  uint32_t qProb[kNumSymPerThread];

  int qProbSum = 0;

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    int curSym = i * Threads + tid;
    uint32_t count = counts[curSym];

    // Rough initial quantization
    qProb[i] = kProbWeight * ((float)count / (float)totalNum);

    // All weights for symbols present must be > 0
    qProb[i] = (count > 0 && qProb[i] == 0) ? 1 : qProb[i];

    qProbSum += qProb[i];
  }

  // Sum qProbSym across all threads
  __shared__ int smemSum[kWarps];
  qProbSum = blockSum<Threads>(warpId, laneId, qProbSum, smemSum);

  // In order to use radix sorting, and also in order to only sort a single
  // word, pack both the weight and index into a single integer
  uint32_t sortedPair[kNumSymPerThread];

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    int curSym = i * Threads + tid;
    sortedPair[i] = (qProb[i] << 16) | curSym;
  }

  // The sort assumes a blocked arrangement as input, which we don't have, but
  // this doesn't matter as we only care about the arrangement post-sort
  using Sort = cub::BlockRadixSort<uint32_t, Threads, kNumSymPerThread>;
  __shared__ typename Sort::TempStorage smemSort;
  Sort(smemSort).SortDescending(sortedPair);

  // The (prob, symbol) pair that each thread is considered to
  // hold is the following:
  uint32_t tidSymbol[kNumSymPerThread];

  // Recover the values
#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    tidSymbol[i] = sortedPair[i] & 0xffffU;
    qProb[i] = sortedPair[i] >> 16;
  }

  // How far below (positive) or above (negative) our current first-pass
  // quantization is from our target sum 2^probBits
  int diff = (int)kProbWeight - (int)qProbSum;

  if (diff > 0) {
    // We are below our total sum target; add 1 to largest values
    // FIXME: use div/mod to avoid iterations
    while (diff > 0) {
      int iterToApply = diff < kNumSymbols ? diff : kNumSymbols;

#pragma unroll
      for (int i = 0; i < kNumSymPerThread; ++i) {
        int curSym = tidSymbol[i];
        if (curSym < iterToApply) {
          qProb[i] += 1;
        }
      }

      diff -= iterToApply;
    }
  } else if (diff < 0) {
    // We are above our total sum target; subtract 1 from the smallest values
    // that are > 1 (all symbols with a weight of 1 cannot go to zero as they
    // are assumed present in the input)
    diff = -diff;

    while (diff > 0) {
      // Need to determine the number of
      int qNumGt1s = 0;

#pragma unroll
      for (int i = 0; i < kNumSymPerThread; ++i) {
        qNumGt1s += (int)(qProb[i] > 1);
      }

      // We need to determine the remaining number of >1 values
      // We reuse smemSum but there is a syncthreads in the sort above, and at
      // the end of the loop below
      qNumGt1s = blockSum<Threads>(warpId, laneId, qNumGt1s, smemSum);
      __syncthreads(); // FIXME: not needed?

      // subtract from smallest >1 values
      // This should be the index of the first 1 value
      // FIXME: use div/mod to avoid iterations
      int iterToApply = diff < qNumGt1s ? diff : qNumGt1s;
      assert(iterToApply > 0);
      int startIndex = qNumGt1s - iterToApply;

#pragma unroll
      for (int i = 0; i < kNumSymPerThread; ++i) {
        // Post-sort, the data is in a blocked arrangement
        int curSym = tid * kNumSymPerThread + i;
        if (curSym >= startIndex && curSym < qNumGt1s) {
          qProb[i] -= 1;
        }
      }

      diff -= iterToApply;

      __syncthreads();
    }
  }

  // Recover the pre-sort order
  __shared__ uint32_t smemPdf[kNumSymbols];

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    smemPdf[tidSymbol[i]] = qProb[i];
  }

  __syncthreads();

  // NOTE: we need to have a contiguous blocked arrangement for cub::BlockScan
  // when kNumSymPerThread > 1, so the order is now tid * kNumSymPerThread + reg
  uint32_t symPdf[kNumSymPerThread];
#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    int curSym = tid * kNumSymPerThread + i;
    symPdf[i] = smemPdf[curSym];
  }

  using Scan = cub::BlockScan<uint32_t, Threads>;
  __shared__ typename Scan::TempStorage smemScan;

  // FIXME: initialize to 0?
  uint32_t symCdf[kNumSymPerThread];
  Scan(smemScan).ExclusiveSum(symPdf, symCdf);

  // Compute divisor information (constant division via integer
  // multiplication + shift)
  uint32_t shift[kNumSymPerThread];
  uint32_t magic[kNumSymPerThread];

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    shift[i] = 32 - __clz(symPdf[i] - 1);

    constexpr uint64_t one = 1;
    uint64_t magic64 =
        ((one << 32) * ((one << shift[i]) - symPdf[i])) / symPdf[i] + 1;

    // should not overflow
    magic[i] = (uint32_t)magic64;
  }

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    // Same blocked contiguous ordering as before
    // Note that this is no longer a coalesced write
    int curSym = tid * kNumSymPerThread + i;
    table[curSym] = uint4{symPdf[i], symCdf[i], magic[i], shift[i]};
  }
}

template <typename SizeProvider, int Threads>
__global__ void quantizeWeights(
    const uint32_t* __restrict__ counts,
    SizeProvider sizeProvider,
    int probBits,
    uint4* __restrict__ table) {
  auto batch = blockIdx.x;

  normalizeProbabilitiesFromHistogram<Threads>(
      counts + batch * kNumSymbols,
      sizeProvider.getBatchSize(batch),
      probBits,
      table + batch * kNumSymbols);
}

template <typename InProvider>
void ansHistogramBatch(
    uint32_t numInBatch,
    InProvider inProvider,
    // size numInBatch * kNumSymbols
    uint32_t* histogram_dev,
    cudaStream_t stream) {
  // 1. Compute symbol histogram
  // zero out buckets before proceeding, as we aggregate with atomic adds
  CUDA_VERIFY(cudaMemsetAsync(
      histogram_dev, 0, sizeof(uint32_t) * kNumSymbols * numInBatch, stream));

  {
    constexpr uint32_t kThreads = kNumSymbols;

    // What is the maximum number of blocks to saturate the GPU?
    int maxBlocks = 0;
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, histogramBatch<InProvider, kThreads>, kThreads, 0));
    maxBlocks *= getCurrentDeviceProperties().multiProcessorCount;

    // The y block dimension will be for each batch element
    uint32_t xBlocks = divUp(maxBlocks, numInBatch);
    auto grid = dim3(xBlocks, numInBatch);

    histogramBatch<InProvider, kThreads>
        <<<grid, kThreads, 0, stream>>>(inProvider, histogram_dev);
  }
}

template <typename SizeProvider>
inline void ansCalcWeights(
    uint32_t numInBatch,
    int probBits,
    // we only use this for sizes (of each input batch member)
    SizeProvider sizeProvider,
    // size numInBatch * kNumSymbols
    const uint32_t* histogram_dev,
    // size numInBatch * kNumSymbols
    uint4* table_dev,
    cudaStream_t stream) {
  // Quantize weights and determine integer ANS division factors
  constexpr int kThreads = kNumSymbols;

  quantizeWeights<SizeProvider, kThreads><<<numInBatch, kThreads, 0, stream>>>(
      histogram_dev, sizeProvider, probBits, table_dev);
}

} // namespace dietgpu
