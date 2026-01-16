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

#include "dietgpu/ans/GpuANSStatistics.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
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

std::vector<uint32_t> histogram(const std::vector<uint8_t>& data) {
  auto counts = std::vector<uint32_t>(256);

  for (auto v : data) {
    counts[v]++;
  }

  return counts;
}

TEST(ANSStatisticsTest, Histogram) {
  auto res = makeStackMemory();
  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

  for (auto size :
       {1,
        2,
        11,
        32,
        55,
        1000,
        1001,
        1000000,
        1024 * 1024,
        1000001,
        12345677}) {
    int numInBatch = 3;

    auto data = std::vector<uint8_t>();

    auto histograms = std::vector<std::vector<uint32_t>>();

    int stride = 11;
    auto strideData = std::vector<uint8_t>(stride);

    for (int b = 0; b < numInBatch; ++b) {
      auto gen = generateSymbols(size, 20.0 + b * 2);
      histograms.push_back(histogram(gen));

      data.insert(data.end(), gen.begin(), gen.end());

      // Add some stride padding
      data.insert(data.end(), strideData.begin(), strideData.end());
    }

    auto data_dev = res.copyAlloc(stream, data);
    auto hist_dev = res.alloc<uint32_t>(stream, numInBatch * kNumSymbols);

    auto inProvider = BatchProviderStride(data_dev.data(), size + stride, size);

    ansHistogramBatch(numInBatch, inProvider, hist_dev.data(), stream);

    auto hist_host = hist_dev.copyToHost(stream);

    for (int b = 0; b < numInBatch; ++b) {
      for (int i = 0; i < kNumSymbols; ++i) {
        EXPECT_EQ(histograms[b][i], hist_host[b * kNumSymbols + i]);
      }
    }
  }
}

std::vector<uint4> dataToANSTable(
    const std::vector<uint8_t>& data,
    int probBits = 10) {
  auto res = makeStackMemory();
  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

  auto data_dev = res.copyAlloc(stream, data);

  // Get histogram
  auto hist_dev = res.alloc<uint32_t>(stream, kNumSymbols);
  auto inProvider =
      BatchProviderStride(data_dev.data(), data.size(), data.size());

  ansHistogramBatch(1, inProvider, hist_dev.data(), stream);

  // Get ANS table from histogram (post-normalization)
  auto table_dev = res.alloc<uint4>(stream, kNumSymbols);

  ansCalcWeights(
      1,
      probBits,
      BatchProviderStride(hist_dev.data(), data.size(), data.size()),
      hist_dev.data(),
      table_dev.data(),
      stream);

  return table_dev.copyToHost(stream);
}

TEST(ANSStatisticsTest, Normalization_NonZero) {
  // Ensure that non-zero count symbols get non-zero weight
  auto data = std::vector<uint8_t>(10000);

  for (int i = 0; i < 256; ++i) {
    data[i] = uint8_t(i);
  }

  for (int i = 256; i < data.size(); ++i) {
    data[i] = 1;
  }

  int probBits = 10;
  auto table = dataToANSTable(data, probBits);

  for (int i = 0; i < kNumSymbols; ++i) {
    if (i != 1) {
      EXPECT_EQ(table[i].x, 1);
    } else {
      EXPECT_EQ(table[i].x, (1 << probBits) - 255);
    }
  }
}

TEST(ANSStatisticsTest, Normalization_EqualWeight) {
  // Ensure that non-zero count symbols get non-zero weight
  auto data = std::vector<uint8_t>(kNumSymbols * 64);

  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < kNumSymbols; ++j) {
      data[i * kNumSymbols + j] = uint8_t(j);
    }
  }

  int probBits = 10;
  auto table = dataToANSTable(data, probBits);

  for (int i = 0; i < kNumSymbols; ++i) {
    EXPECT_EQ(table[i].x, (1 << probBits) / kNumSymbols);
  }
}

TEST(ANSStatisticsTest, Normalization) {
  auto data = generateSymbols(12345, 40.0f);

  // Count true distribution
  auto hist = histogram(data);

  int probBits = 11;
  auto table = dataToANSTable(data, probBits);

  uint32_t totalSum = 0;
  uint32_t totalWeight = 1 << probBits;

  for (int i = 0; i < kNumSymbols; ++i) {
    auto count = hist[i];
    auto pdf = table[i].x;

    totalSum += pdf;

    if (count == 0) {
      EXPECT_EQ(pdf, 0) << "failed on " << i;
    } else if (count > 0) {
      EXPECT_GT(pdf, 0);
      // The normalized prob should be within some small factor of the real
      // count
      float prob = float(count) / float(data.size());
      float normalizedProb = float(pdf) / float(totalWeight);

      EXPECT_GE(normalizedProb, prob * 0.5f);

      // Only relevant if the prob is > 1/totalWeight (i.e., we
      // weren't rounded up to be a non-zero weight)
      if (prob > 1.0f / float(totalWeight)) {
        EXPECT_LE(normalizedProb, prob * 2.0f);
      }
    }
  }

  EXPECT_EQ(totalSum, totalWeight);
}
