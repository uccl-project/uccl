/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <glog/logging.h>
#include "dietgpu/ans/GpuANSUtils.cuh"

namespace dietgpu {

struct BatchWriter {
  inline __device__ BatchWriter(void* out)
      : out_((uint8_t*)out), outBlock_(nullptr) {}

  inline __device__ void setBlock(uint32_t block) {
    outBlock_ = out_ + block * kDefaultBlockSize;
  }

  inline __device__ void write(uint32_t offset, uint8_t sym) {
    outBlock_[offset] = sym;
  }

  // template <typename Vec>
  // inline __device__ void writeVec(uint32_t offset, Vec symV) {
  //   ((Vec*)outBlock_)[offset] = symV;
  // }

  // __device__ void preload(uint32_t offset) {}

  uint8_t* out_;
  uint8_t* outBlock_;
};

struct BatchProviderStride {
  using Writer = BatchWriter;

  __host__ BatchProviderStride(
      void* ptr_dev,
      uint32_t batchStride,
      uint32_t batchCapacity = 0)
      : ptr_dev_(ptr_dev),
        batchStride_(batchStride),
        batchCapacity_(batchCapacity) {}

  __device__ void* getBatchStart(uint32_t batch) {
    return ((uint8_t*)ptr_dev_) + batchStride_ * batch;
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    return ((uint8_t*)ptr_dev_) + batchStride_ * batch;
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return batchCapacity_;
  }

  void* ptr_dev_;
  uint32_t batchStride_;
  uint32_t batchCapacity_;
};

struct BatchProviderSplitSize {
  using Writer = BatchWriter;

  __host__ BatchProviderSplitSize(
      void* ptr_dev,
      const uint32_t* splitSize_dev,
      // Exclusive prefix sum of splitSize_dev
      const uint32_t* splitSizePrefix_dev,
      uint32_t wordSize)
      : ptr_dev_(ptr_dev),
        splitSize_dev_(splitSize_dev),
        splitSizePrefix_dev_(splitSizePrefix_dev),
        wordSize_(wordSize) {}

  __device__ void* getBatchStart(uint32_t batch) {
    return ((uint8_t*)ptr_dev_) + splitSizePrefix_dev_[batch] * wordSize_;
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    return ((uint8_t*)ptr_dev_) + splitSizePrefix_dev_[batch] * wordSize_;
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return splitSize_dev_[batch];
  }

  void* ptr_dev_;
  const uint32_t* splitSize_dev_;
  const uint32_t* splitSizePrefix_dev_;
  uint32_t wordSize_;
};

struct BatchProviderPointer {
  using Writer = BatchWriter;

  __host__ BatchProviderPointer(
      void** ptr_dev,
      const uint32_t* capacity_dev = nullptr)
      : ptr_dev_(ptr_dev), capacity_dev_(capacity_dev) {}

  __device__ void* getBatchStart(uint32_t batch) {
    return ptr_dev_[batch];
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    return ptr_dev_[batch];
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return capacity_dev_[batch];
  }

  void** ptr_dev_;
  const uint32_t* capacity_dev_;
};

template <int N>
struct BatchProviderInlinePointer {
  using Writer = BatchWriter;

  __host__ BatchProviderInlinePointer(int num, void** ptr_host) {
    CHECK_LE(num, N);
    for (int i = 0; i < num; ++i) {
      ptr_dev_[i] = ptr_host[i];
    }
  }

  __device__ void* getBatchStart(uint32_t batch) {
    return ptr_dev_[batch];
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    return ptr_dev_[batch];
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  void* ptr_dev_[N];
};

template <int N>
struct BatchProviderInlinePointerCapacity {
  using Writer = BatchWriter;

  __host__ BatchProviderInlinePointerCapacity(
      int num,
      void** ptr_host,
      const uint32_t* capacity_host) {
    CHECK_LE(num, N);
    for (int i = 0; i < num; ++i) {
      ptr_dev_[i] = ptr_host[i];
      capacity_dev_[i] = capacity_host[i];
    }
  }

  __device__ void* getBatchStart(uint32_t batch) {
    return ptr_dev_[batch];
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    return ptr_dev_[batch];
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return capacity_dev_[batch];
  }

  void* ptr_dev_[N];
  uint32_t capacity_dev_[N];
};

} // namespace dietgpu
