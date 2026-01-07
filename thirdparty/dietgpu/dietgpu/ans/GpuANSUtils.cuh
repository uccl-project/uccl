/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <cuda.h>
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

using ANSStateT = uint32_t;
using ANSEncodedT = uint16_t;
using ANSDecodedT = uint8_t;

struct __align__(16) ANSDecodedTx16 {
  ANSDecodedT x[16];
};

struct __align__(8) ANSDecodedTx8 {
  ANSDecodedT x[8];
};

struct __align__(4) ANSDecodedTx4 {
  ANSDecodedT x[4];
};

constexpr uint32_t kNumSymbols = 1 << (sizeof(ANSDecodedT) * 8);
static_assert(kNumSymbols > 1, "");

// Default block size for compression (in bytes)
constexpr uint32_t kDefaultBlockSize = 4096;

// limit state to 2^31 - 1, so as to prevent addition overflow in the integer
// division via mul and shift by constants
constexpr int kANSStateBits = sizeof(ANSStateT) * 8 - 1;
constexpr int kANSEncodedBits = sizeof(ANSEncodedT) * 8; // out bits
constexpr ANSStateT kANSEncodedMask =
    (ANSStateT(1) << kANSEncodedBits) - ANSStateT(1);

constexpr ANSStateT kANSStartState = ANSStateT(1)
    << (kANSStateBits - kANSEncodedBits);
constexpr ANSStateT kANSMinState = ANSStateT(1)
    << (kANSStateBits - kANSEncodedBits);

// magic number to verify archive integrity
constexpr uint32_t kANSMagic = 0xd00d;

// current DietGPU version number
constexpr uint32_t kANSVersion = 0x0001;

// Each block of compressed data (either coalesced or uncoalesced) is aligned to
// this number of bytes and has a valid (if not all used) segment with this
// multiple of bytes
constexpr uint32_t kBlockAlignment = 16;

struct ANSWarpState {
  // The ANS state data for this warp
  ANSStateT warpState[kWarpSize];
};

struct __align__(32) ANSCoalescedHeader {
  static __host__ __device__ uint32_t getCompressedOverhead(
      uint32_t numBlocks) {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1
        : kBlockAlignment / sizeof(uint2);

    return sizeof(ANSCoalescedHeader) +
        // probs
        sizeof(uint16_t) * kNumSymbols +
        // states
        sizeof(ANSWarpState) * numBlocks +
        // block words
        sizeof(uint2) * roundUp(numBlocks, kAlignment);
  }

  __host__ __device__ uint32_t getTotalCompressedSize() const {
    return getCompressedOverhead() +
        getTotalCompressedWords() * sizeof(ANSEncodedT);
  }

  __host__ __device__ uint32_t getCompressedOverhead() const {
    return getCompressedOverhead(getNumBlocks());
  }

  __host__ __device__ float getCompressionRatio() const {
    return (float)getTotalCompressedSize() /
        (float)getTotalUncompressedWords() * sizeof(ANSDecodedT);
  }

  __host__ __device__ uint32_t getNumBlocks() const {
    return numBlocks;
  }

  __host__ __device__ void setNumBlocks(uint32_t nb) {
    numBlocks = nb;
  }

  __host__ __device__ void setMagicAndVersion() {
    magicAndVersion = (kANSMagic << 16) | kANSVersion;
  }

  __host__ __device__ void checkMagicAndVersion() const {
    assert((magicAndVersion >> 16) == kANSMagic);
    assert((magicAndVersion & 0xffffU) == kANSVersion);
  }

  __host__ __device__ uint32_t getTotalUncompressedWords() const {
    return totalUncompressedWords;
  }

  __host__ __device__ void setTotalUncompressedWords(uint32_t words) {
    totalUncompressedWords = words;
  }

  __host__ __device__ uint32_t getTotalCompressedWords() const {
    return totalCompressedWords;
  }

  __host__ __device__ void setTotalCompressedWords(uint32_t words) {
    totalCompressedWords = words;
  }

  __host__ __device__ uint32_t getProbBits() const {
    return options & 0xf;
  }

  __host__ __device__ void setProbBits(uint32_t bits) {
    assert(bits <= 0xf);
    options = (options & 0xfffffff0U) | bits;
  }

  __host__ __device__ bool getUseChecksum() const {
    return options & 0x10;
  }

  __host__ __device__ void setUseChecksum(bool uc) {
    options = (options & 0xffffffef) | (uint32_t(uc) << 4);
  }

  __host__ __device__ uint32_t getChecksum() const {
    return checksum;
  }

  __host__ __device__ void setChecksum(uint32_t c) {
    checksum = c;
  }

  __device__ uint16_t* getSymbolProbs() {
    return (uint16_t*)(this + 1);
  }

  __device__ const uint16_t* getSymbolProbs() const {
    return (const uint16_t*)(this + 1);
  }

  __device__ ANSWarpState* getWarpStates() {
    return (ANSWarpState*)(getSymbolProbs() + kNumSymbols);
  }

  __device__ const ANSWarpState* getWarpStates() const {
    return (const ANSWarpState*)(getSymbolProbs() + kNumSymbols);
  }

  __device__ uint2* getBlockWords(uint32_t numBlocks) {
    // All of the ANSWarpStates are already kBlockAlignment aligned
    return (uint2*)(getWarpStates() + numBlocks);
  }

  __device__ const uint2* getBlockWords(uint32_t numBlocks) const {
    // All of the ANSWarpStates are already kBlockAlignment aligned
    return (const uint2*)(getWarpStates() + numBlocks);
  }

  __device__ ANSEncodedT* getBlockDataStart(uint32_t numBlocks) {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1
        : kBlockAlignment / sizeof(uint2);

    return (ANSEncodedT*)(getBlockWords(numBlocks) +
                          roundUp(numBlocks, kAlignment));
  }

  __device__ const ANSEncodedT* getBlockDataStart(uint32_t numBlocks) const {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1
        : kBlockAlignment / sizeof(uint2);

    return (const ANSEncodedT*)(getBlockWords(numBlocks) +
                                roundUp(numBlocks, kAlignment));
  }

  // (16: magic)(16: version)
  uint32_t magicAndVersion;
  uint32_t numBlocks;
  uint32_t totalUncompressedWords;
  uint32_t totalCompressedWords;

  // (27: unused)(1: use checksum)(4: probBits)
  uint32_t options;
  uint32_t checksum;
  uint32_t unused0;
  uint32_t unused1;

  // Data that follows after the header (some of which is variable length):

  // Fixed length array
  // uint16_t probs[kNumSymbols];

  // Variable length array:
  // ANSWarpState states[numBlocks];

  // Per-block information:
  // (uint16: uncompressedWords, uint16: compressedWords)
  // uint32: blockCompressedWordStart
  //
  // Variable length array:
  // uint2 blockWords[roundUp(numBlocks, kBlockAlignment / sizeof(uint2))];

  // Then follows the compressed per-warp/block data for each segment
};

static_assert(sizeof(ANSCoalescedHeader) == 32, "");

static_assert(isEvenDivisor(sizeof(ANSCoalescedHeader), sizeof(uint4)), "");

} // namespace dietgpu
