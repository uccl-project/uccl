/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

uint16_t float32ToBFloat16(float f) {
  // FIXME: does not round to nearest even
  static_assert(sizeof(float) == sizeof(uint32_t), "");
  uint32_t x;
  std::memcpy(&x, &f, sizeof(float));

  x >>= 16;
  return x;
}

uint16_t float32ToFloat16(float f) {
  static_assert(sizeof(float) == sizeof(uint32_t), "");
  uint32_t x;
  std::memcpy(&x, &f, sizeof(float));

  uint32_t u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  uint32_t sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000U) {
    return 0x7fffU;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefffU) {
    return sign | 0x7c00U;
  }
  if (u < 0x33000001U) {
    return (sign | 0x0000);
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  return (sign | (exponent << 10) | mantissa);
}

template <FloatType FT>
struct GenerateFloat;

template <>
struct GenerateFloat<FloatType::kFloat16> {
  static FloatTypeInfo<FloatType::kFloat16>::WordT gen(float v) {
    return float32ToFloat16(v);
  }
};

template <>
struct GenerateFloat<FloatType::kBFloat16> {
  static FloatTypeInfo<FloatType::kBFloat16>::WordT gen(float v) {
    return float32ToBFloat16(v);
  }
};

template <>
struct GenerateFloat<FloatType::kFloat32> {
  static FloatTypeInfo<FloatType::kFloat32>::WordT gen(float v) {
    FloatTypeInfo<FloatType::kFloat32>::WordT out;
    std::memcpy(&out, &v, sizeof(float));
    return out;
  }
};

template <FloatType FT>
std::vector<typename FloatTypeInfo<FT>::WordT> generateFloats(int num) {
  std::mt19937 gen(10 + num);
  std::normal_distribution<float> dist;

  auto out = std::vector<typename FloatTypeInfo<FT>::WordT>(num);
  for (auto& v : out) {
    v = GenerateFloat<FT>::gen(dist(gen));
  }

  return out;
}

template <FloatType FT>
void runBatchPointerTest(
    StackDeviceMemory& res,
    int probBits,
    const std::vector<uint32_t>& batchSizes) {
  using FTI = FloatTypeInfo<FT>;

  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

  int numInBatch = batchSizes.size();
  uint32_t totalSize = 0;
  uint32_t maxSize = 0;
  for (auto v : batchSizes) {
    totalSize += v;
    maxSize = std::max(maxSize, v);
  }

  auto maxCompressedSize = getMaxFloatCompressedSize(FT, maxSize);

  auto orig = generateFloats<FT>(totalSize);
  auto orig_dev = res.copyAlloc(stream, orig);

  auto inPtrs = std::vector<const void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      inPtrs[i] = (const typename FTI::WordT*)orig_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

  auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * maxCompressedSize);

  auto encPtrs = std::vector<void*>(batchSizes.size());
  {
    for (int i = 0; i < inPtrs.size(); ++i) {
      encPtrs[i] = (uint8_t*)enc_dev.data() + i * maxCompressedSize;
    }
  }

  auto outBatchSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  auto compConfig =
      FloatCompressConfig(FT, ANSCodecConfig(probBits), false, true);

  floatCompress(
      res,
      compConfig,
      numInBatch,
      inPtrs.data(),
      batchSizes.data(),
      encPtrs.data(),
      outBatchSize_dev.data(),
      stream);

  // Decode data
  auto dec_dev = res.alloc<typename FTI::WordT>(stream, totalSize);

  auto decPtrs = std::vector<void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      decPtrs[i] = (typename FTI::WordT*)dec_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

  auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  auto decompConfig =
      FloatDecompressConfig(FT, ANSCodecConfig(probBits), false, true);

  floatDecompress(
      res,
      decompConfig,
      numInBatch,
      (const void**)encPtrs.data(),
      decPtrs.data(),
      batchSizes.data(),
      outSuccess_dev.data(),
      outSize_dev.data(),
      stream);

  auto outSuccess = outSuccess_dev.copyToHost(stream);
  auto outSize = outSize_dev.copyToHost(stream);

  for (int i = 0; i < outSuccess.size(); ++i) {
    EXPECT_TRUE(outSuccess[i]);
    EXPECT_EQ(outSize[i], batchSizes[i]);
  }

  auto dec = dec_dev.copyToHost(stream);

  for (int i = 0; i < orig.size(); ++i) {
    if (orig[i] != dec[i]) {
      printf(
          "mismatch at %d / %d: 0x%08X 0x%08X\n",
          i,
          (int)orig.size(),
          orig[i],
          dec[i]);
      break;
    }
  }

  EXPECT_EQ(orig, dec);
}

void runBatchPointerTest(
    StackDeviceMemory& res,
    FloatType ft,
    int probBits,
    const std::vector<uint32_t>& batchSizes) {
  switch (ft) {
    case FloatType::kFloat16:
      runBatchPointerTest<FloatType::kFloat16>(res, probBits, batchSizes);
      break;
    case FloatType::kBFloat16:
      runBatchPointerTest<FloatType::kBFloat16>(res, probBits, batchSizes);
      break;
    case FloatType::kFloat32:
      runBatchPointerTest<FloatType::kFloat32>(res, probBits, batchSizes);
      break;
    default:
      CHECK(false);
      break;
  }
}

void runBatchPointerTest(
    StackDeviceMemory& res,
    FloatType ft,
    int probBits,
    int numInBatch,
    uint32_t multipleOf = 1) {
  std::mt19937 gen(10 + numInBatch);
  std::uniform_int_distribution<uint32_t> dist(1, 10000);

  auto batchSizes = std::vector<uint32_t>(numInBatch);
  for (auto& v : batchSizes) {
    v = roundUp(dist(gen), multipleOf);
  }

  runBatchPointerTest(res, ft, probBits, batchSizes);
}

TEST(FloatTest, Batch) {
  auto res = makeStackMemory();

  for (auto ft :
       {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32}) {
    for (auto probBits : {9, 10}) {
      for (auto numInBatch : {1, 3, 16, 23}) {
        runBatchPointerTest(res, ft, probBits, numInBatch);
        // Also test the case where there is uniform 16 byte alignment across
        // all batches
        runBatchPointerTest(res, ft, probBits, numInBatch, 16);
      }
    }
  }
}

TEST(FloatTest, LargeBatch) {
  auto res = makeStackMemory();

  auto batchSizes = std::vector<uint32_t>(256);
  for (auto& v : batchSizes) {
    v = 512 * 1024;
  }

  for (auto ft :
       {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32}) {
    runBatchPointerTest(res, ft, 10, batchSizes);
  }
}

TEST(FloatTest, BatchSize1) {
  auto res = makeStackMemory();

  for (auto ft :
       {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32}) {
    for (auto probBits : {9, 10}) {
      runBatchPointerTest(res, ft, probBits, {1});
      runBatchPointerTest(res, ft, probBits, {13, 1});
      runBatchPointerTest(res, ft, probBits, {12345, 1, 8083, 1, 17});
    }
  }
}
