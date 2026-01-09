/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "dietgpu/ans/BatchPrefixSum.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

std::vector<uint32_t>
makeSequence(uint32_t numInBatch, uint32_t batchSize, int seed = 10) {
  auto gen = std::mt19937(10);
  auto dist = std::uniform_int_distribution<uint32_t>(0, 20);

  auto out = std::vector<uint32_t>(numInBatch * batchSize);

  for (auto& v : out) {
    v = dist(gen);
  }

  return out;
}

std::vector<uint32_t> exclusivePrefixSum(
    const std::vector<uint32_t>& in,
    uint32_t numInBatch,
    uint32_t batchSize) {
  auto out = std::vector<uint32_t>(numInBatch * batchSize);

  for (uint32_t b = 0; b < numInBatch; ++b) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < batchSize; ++i) {
      auto v = in[b * batchSize + i];
      out[b * batchSize + i] = sum;
      sum += v;
    }
  }

  return out;
}

TEST(BatchPrefixSum, OneLevel) {
  auto res = makeStackMemory();
  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

  auto gen = std::mt19937(10);
  auto nbDist = std::uniform_int_distribution<uint32_t>(1, 20);

  for (auto batchSize : {1, 10, 32, 33, 64, 65, 128, 129, 256, 257, 512}) {
    auto numInBatch = nbDist(gen);

    auto data = makeSequence(numInBatch, batchSize, nbDist(gen));
    auto dataPrefix = exclusivePrefixSum(data, numInBatch, batchSize);

    auto data_dev = res.copyAlloc(stream, data);

    auto tempSize = getBatchExclusivePrefixSumTempSize(numInBatch, batchSize);
    EXPECT_EQ(tempSize, 0);

    auto prefix_dev = res.alloc<uint32_t>(stream, numInBatch * batchSize);

    batchExclusivePrefixSum<uint32_t, NoTransform<uint32_t>>(
        data_dev.data(),
        prefix_dev.data(),
        nullptr,
        numInBatch,
        batchSize,
        NoTransform<uint32_t>(),
        stream);

    auto gpuDataPrefix = prefix_dev.copyToHost(stream);
    EXPECT_EQ(dataPrefix, gpuDataPrefix);
  }
}

TEST(BatchPrefixSum, TwoLevel) {
  auto res = makeStackMemory();
  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

  auto batchSizes = std::vector<int>{
      513, 1024, 2047, 2048, 4096, 4097, 10000, 100000, 512 * 512};

  auto gen = std::mt19937(10);
  auto nbDist = std::uniform_int_distribution<uint32_t>(1, 20);
  auto bsDist = std::uniform_int_distribution<uint32_t>(513, 512 * 512);

  for (int i = 0; i < 10; ++i) {
    batchSizes.push_back(bsDist(gen));
  }

  for (auto batchSize : batchSizes) {
    auto numInBatch = nbDist(gen);

    auto data = makeSequence(numInBatch, batchSize, nbDist(gen));
    auto dataPrefix = exclusivePrefixSum(data, numInBatch, batchSize);

    auto data_dev = res.copyAlloc(stream, data);

    auto tempSize = getBatchExclusivePrefixSumTempSize(numInBatch, batchSize);
    EXPECT_GT(tempSize, 0);

    auto prefix_dev = res.alloc<uint32_t>(stream, numInBatch * batchSize);
    auto temp_dev = res.alloc<uint32_t>(stream, tempSize);

    batchExclusivePrefixSum<uint32_t, NoTransform<uint32_t>>(
        data_dev.data(),
        prefix_dev.data(),
        temp_dev.data(),
        numInBatch,
        batchSize,
        NoTransform<uint32_t>(),
        stream);

    auto gpuDataPrefix = prefix_dev.copyToHost(stream);
    EXPECT_EQ(dataPrefix, gpuDataPrefix);
  }
}

// TEST(BatchPrefixSum, Perf) {
//   StandardGpuResources res;
//   auto stream = res.getDefaultStreamCurrentDevice();

//   int numInBatch = 128;
//   int batchSize = 4000;

//   auto data = makeSequence(numInBatch, batchSize);
//   auto dataPrefix = exclusivePrefixSum(data, numInBatch, batchSize);

//   auto data_dev = toDeviceNonTemporary(&res, data, stream);

//   auto tempSize = getBatchExclusivePrefixSumTempSize(numInBatch, batchSize);
//   EXPECT_GT(tempSize, 0);

//   auto prefix_dev = DeviceTensor<uint32_t, 1, true>(
//     &res, makeDevAlloc(AllocType::Other, stream),
//     {(int) (numInBatch * batchSize)});

//   auto temp_dev = DeviceTensor<uint32_t, 1, true>(
//     &res, makeDevAlloc(AllocType::Other, stream),
//     {(int) tempSize});

//   batchExclusivePrefixSum<uint32_t>(data_dev.data(),
//                                     prefix_dev.data(),
//                                     temp_dev.data(),
//                                     numInBatch,
//                                     batchSize,
//                                     stream);

//   auto gpuDataPrefix = prefix_dev.copyToVector(stream);

//   for (int i = 0; i < dataPrefix.size(); ++i) {
//     if (dataPrefix[i] != gpuDataPrefix[i]) {
//       printf("mismatch on %d: %u %u\n", i, dataPrefix[i], gpuDataPrefix[i]);
//       break;
//     }
//   }

//   // EXPECT_EQ(dataPrefix, gpuDataPrefix);
// }
