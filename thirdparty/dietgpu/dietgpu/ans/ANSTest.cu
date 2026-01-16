/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

std::vector<uint8_t> generateSymbols(int num, float lambda = 20.0f) {
  std::random_device rd;
  std::mt19937 gen(10);
  std::exponential_distribution<float> dist(lambda);

  auto out = std::vector<uint8_t>(num);
  for (auto& v : out) {
    auto sample = std::min(dist(gen), 1.0f);

    v = sample * 256.0;
  }

  return out;
}

std::vector<GpuMemoryReservation<uint8_t>> toDevice(
    StackDeviceMemory& res,
    const std::vector<std::vector<uint8_t>>& vs,
    cudaStream_t stream) {
  auto out = std::vector<GpuMemoryReservation<uint8_t>>();

  for (auto& v : vs) {
    out.emplace_back(res.copyAlloc(stream, v, AllocType::Permanent));
  }

  return out;
}

std::vector<std::vector<uint8_t>> toHost(
    StackDeviceMemory& res,
    const std::vector<GpuMemoryReservation<uint8_t>>& vs,
    cudaStream_t stream) {
  auto out = std::vector<std::vector<uint8_t>>();

  for (auto& v : vs) {
    out.emplace_back(v.copyToHost(stream));
  }

  return out;
}

std::vector<GpuMemoryReservation<uint8_t>> buffersToDevice(
    StackDeviceMemory& res,
    const std::vector<uint32_t>& sizes,
    cudaStream_t stream) {
  auto out = std::vector<GpuMemoryReservation<uint8_t>>();

  for (auto& s : sizes) {
    out.emplace_back(res.alloc<uint8_t>(stream, s, AllocType::Permanent));
  }

  return out;
}

std::vector<std::vector<uint8_t>> genBatch(
    const std::vector<uint32_t>& sizes,
    double lambda) {
  auto out = std::vector<std::vector<uint8_t>>();

  for (auto s : sizes) {
    out.push_back(generateSymbols(s, lambda));
  }

  return out;
}

void runBatchPointer(
    StackDeviceMemory& res,
    int prec,
    const std::vector<uint32_t>& batchSizes,
    double lambda = 100.0) {
  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

  int numInBatch = batchSizes.size();
  uint32_t maxSize = 0;
  for (auto v : batchSizes) {
    maxSize = std::max(maxSize, v);
  }

  auto outBatchStride = getMaxCompressedSize(maxSize);

  auto batch_host = genBatch(batchSizes, lambda);
  auto batch_dev = toDevice(res, batch_host, stream);

  auto inPtrs = std::vector<const void*>(batchSizes.size());
  {
    for (int i = 0; i < inPtrs.size(); ++i) {
      inPtrs[i] = batch_dev[i].data();
    }
  }

  auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

  auto encPtrs = std::vector<void*>(batchSizes.size());
  for (int i = 0; i < inPtrs.size(); ++i) {
    encPtrs[i] = (uint8_t*)enc_dev.data() + i * outBatchStride;
  }

  auto outCompressedSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  ansEncodeBatchPointer(
      res,
      ANSCodecConfig(prec, true),
      numInBatch,
      inPtrs.data(),
      batchSizes.data(),
      nullptr,
      encPtrs.data(),
      outCompressedSize_dev.data(),
      stream);

  auto encSize = outCompressedSize_dev.copyToHost(stream);
  for (auto v : encSize) {
    // Reported compressed sizes in bytes should be a multiple of 16 for aligned
    // packing
    EXPECT_EQ(v % 16, 0);
  }

  // Decode data
  auto dec_dev = buffersToDevice(res, batchSizes, stream);

  auto decPtrs = std::vector<void*>(batchSizes.size());
  for (int i = 0; i < inPtrs.size(); ++i) {
    decPtrs[i] = dec_dev[i].data();
  }

  auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  ansDecodeBatchPointer(
      res,
      ANSCodecConfig(prec, true),
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

  auto dec_host = toHost(res, dec_dev, stream);
  EXPECT_EQ(batch_host, dec_host);
}

void runBatchStride(
    StackDeviceMemory& res,
    int prec,
    int numInBatch,
    int inBatchSize,
    double lambda = 100.0) {
  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

  auto orig = generateSymbols(numInBatch * inBatchSize, lambda);
  auto orig_dev = res.copyAlloc(stream, orig);

  int outBatchStride = getMaxCompressedSize(inBatchSize);

  auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

  auto outCompressedSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  ansEncodeBatchStride(
      res,
      ANSCodecConfig(prec, true),
      numInBatch,
      orig_dev.data(),
      inBatchSize,
      inBatchSize,
      nullptr,
      enc_dev.data(),
      outBatchStride,
      outCompressedSize_dev.data(),
      stream);

  auto encSize = outCompressedSize_dev.copyToHost(stream);
  for (auto v : encSize) {
    // Reported compressed sizes in bytes should be a multiple of 16 for aligned
    // packing
    EXPECT_EQ(v % 16, 0);
  }

  auto dec_dev = res.alloc<uint8_t>(stream, numInBatch * inBatchSize);
  auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  // FIXME: Copy the compressed data to the host and truncate it to make
  // sure the compressed size is accurate
  ansDecodeBatchStride(
      res,
      ANSCodecConfig(prec, true),
      numInBatch,
      enc_dev.data(),
      outBatchStride,
      dec_dev.data(),
      inBatchSize,
      inBatchSize,
      outSuccess_dev.data(),
      outSize_dev.data(),
      stream);

  auto outSuccess = outSuccess_dev.copyToHost(stream);
  auto outSize = outSize_dev.copyToHost(stream);

  for (auto s : outSuccess) {
    EXPECT_TRUE(s);
  }

  for (auto s : outSize) {
    EXPECT_EQ(s, inBatchSize);
  }

  auto dec = dec_dev.copyToHost(stream);
  EXPECT_EQ(orig, dec);
}

TEST(ANSTest, ZeroSized) {
  auto res = makeStackMemory();
  runBatchPointer(res, 10, {0}, 10.0);
}

TEST(ANSTest, BatchPointer) {
  auto res = makeStackMemory();

  for (auto prec : {9, 10, 11}) {
    for (auto lambda : {1.0, 10.0, 100.0, 1000.0}) {
      runBatchPointer(res, prec, {1}, lambda);
      runBatchPointer(res, prec, {1, 1}, lambda);
      runBatchPointer(res, prec, {4096, 4095, 4096}, lambda);
      runBatchPointer(res, prec, {1234, 2345, 3456}, lambda);
      runBatchPointer(res, prec, {10000, 10013, 10000}, lambda);
    }
  }
}

TEST(ANSTest, BatchPointerLarge) {
  auto res = makeStackMemory();

  std::random_device rd;
  std::mt19937 gen(10);
  std::uniform_int_distribution<uint32_t> dist(100, 10000);

  std::vector<uint32_t> sizes;
  for (int i = 0; i < 100; ++i) {
    sizes.push_back(dist(gen));
  }

  runBatchPointer(res, 10, sizes);
}

TEST(ANSTest, BatchStride) {
  auto res = makeStackMemory();

  // FIXME: 16 byte alignment required
  runBatchStride(res, 10, 13, 8192 + 16);
}
