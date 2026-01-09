/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/utils/StackDeviceMemory.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include "dietgpu/utils/DeviceUtils.h"

namespace dietgpu {

namespace {

size_t adjustStackSize(size_t sz) {
  if (sz == 0) {
    return 0;
  } else {
    // ensure that we have at least kSDMAlignment bytes, as all allocations are
    // bumped up to it
    return std::max(sz, kSDMAlignment);
  }
}

} // namespace

//
// StackDeviceMemory
//

StackDeviceMemory::Stack::Stack(int d, size_t sz)
    : device_(d),
      alloc_(nullptr),
      allocSize_(adjustStackSize(sz)),
      start_(nullptr),
      end_(nullptr),
      head_(nullptr),
      overflowSize_(0),
      maxSeenSize_(0) {
  if (allocSize_ == 0) {
    return;
  }

  DeviceScope s(device_);
  CUDA_VERIFY(cudaMalloc(&alloc_, allocSize_));
  CHECK(alloc_);

  // In order to disambiguate between our entire region of temporary memory
  // versus the first allocation in the temporary memory region, ensure that the
  // first address returned is +kSDMAlignment bytes from the beginning
  start_ = alloc_;
  head_ = start_;
  end_ = alloc_ + allocSize_;
}

StackDeviceMemory::Stack::Stack(int device, void* p, size_t size)
    : device_(device),
      alloc_(nullptr),
      allocSize_(adjustStackSize(size)),
      start_(nullptr),
      end_(nullptr),
      head_(nullptr),
      overflowSize_(0),
      maxSeenSize_(0) {
  CHECK(p || size == 0);

  // the minimum size that can be provided (see adjustStackSize), if we are
  // allocating memory internally
  CHECK(size == 0 || size >= kSDMAlignment);

  // alloc_ is not used, as we don't own this allocation
  start_ = (char*)p;
  head_ = start_;
  end_ = p ? (char*)p + allocSize_ : nullptr;
}

StackDeviceMemory::Stack::~Stack() {
  // Make sure there are no outstanding memory allocations
  CHECK_EQ(head_, start_);
  CHECK(overflowAllocs_.empty());
  CHECK_EQ(overflowSize_, 0);

  // Did we own the stack buffer?
  if (alloc_) {
    DeviceScope s(device_);
    CUDA_VERIFY(cudaFree(alloc_));
  }
}

size_t StackDeviceMemory::Stack::getSizeAvailable() const {
  return (end_ - head_);
}

size_t StackDeviceMemory::Stack::getSizeTotal() const {
  return (end_ - start_);
}

size_t StackDeviceMemory::Stack::getStackSizeUsed() const {
  return (head_ - start_);
}

void* StackDeviceMemory::Stack::getAlloc(
    size_t size,
    cudaStream_t stream,
    AllocType type) {
  // All allocations should have been adjusted to a multiple of kSDMAlignment
  // bytes
  CHECK_GE(size, kSDMAlignment);
  CHECK_EQ(size % kSDMAlignment, 0);

  void* out = nullptr;

  size_t stackMemUsed = head_ - start_;
  auto sizeRemaining = getSizeAvailable();

  if (size > sizeRemaining || type == AllocType::Permanent) {
    // No space in the stack, fallback to cudaMalloc
    if (type == AllocType::Temporary) {
      // Current memory used after this allocation
      size_t curUsed = overflowSize_ + size + stackMemUsed;

      std::cerr << "WARNING: StackDeviceMemory: attempting to allocate " << size
                << " bytes with " << sizeRemaining
                << " bytes available; calling cudaMalloc. "
                << "Resize temp memory to >= "
                << std::max(maxSeenSize_, curUsed)
                << " bytes to avoid performance problems. "
                << "(Current usage: " << getStackSizeUsed() << " bytes stack "
                << overflowSize_ << " bytes overflow)\n";
    }

    CUDA_VERIFY(cudaMalloc(&out, size));
    CHECK(out);

    overflowAllocs_[out] = size;
    overflowSize_ += size;
  } else {
    // Space is available in the stack
    CHECK(head_);
    out = head_;

    head_ = head_ + size;
    CHECK_LE(head_, end_);
  }

  maxSeenSize_ = std::max(maxSeenSize_, stackMemUsed + overflowSize_);

  return out;
}

void StackDeviceMemory::Stack::returnAlloc(
    void* p,
    size_t size,
    cudaStream_t stream) {
  auto it = overflowAllocs_.find(p);
  if (it != overflowAllocs_.end()) {
    // This allocation was not made on the stack
    CHECK_EQ(it->second, size);

    CUDA_VERIFY(cudaFree(p));
    overflowAllocs_.erase(it);
    CHECK_GE(overflowSize_, size);
    overflowSize_ -= size;

    return;
  }

  // Otherwise, this is on our stack
  char* pc = static_cast<char*>(p);

  // Otherwise, this allocation should be within ourselves
  CHECK(pc >= start_ && pc < end_);

  // All allocations should have been adjusted
  CHECK_EQ(size % kSDMAlignment, 0);

  // Allocations should be freed in the reverse order they are made
  CHECK_EQ(pc + size, head_);

  head_ = pc;
}

std::string StackDeviceMemory::Stack::toString() const {
  std::stringstream s;

  s << "SDM device " << device_ << ": Total memory " << allocSize_ << " ["
    << (void*)start_ << ", " << (void*)end_ << ")\n";
  s << "     Available memory " << (size_t)(end_ - head_) << " ["
    << (void*)head_ << ", " << (void*)end_ << ")\n";
  s << "     Maximum seen mem usage " << maxSeenSize_ << "\n";

  return s.str();
}

StackDeviceMemory::StackDeviceMemory(int device, size_t allocPerDevice)
    : device_(device), stack_(device, allocPerDevice) {}

StackDeviceMemory::StackDeviceMemory(int device, void* p, size_t size)
    : device_(device), stack_(device, p, size) {}

StackDeviceMemory::~StackDeviceMemory() = default;

int StackDeviceMemory::getDevice() const {
  return device_;
}

size_t StackDeviceMemory::getSizeAvailable() const {
  return stack_.getSizeAvailable();
}

size_t StackDeviceMemory::getSizeTotal() const {
  return stack_.getSizeTotal();
}

size_t StackDeviceMemory::getMaxMemoryUsage() const {
  return stack_.maxSeenSize_;
}

void StackDeviceMemory::resetMaxMemoryUsage() {
  stack_.maxSeenSize_ = 0;
}

std::string StackDeviceMemory::toString() const {
  return stack_.toString();
}

void* StackDeviceMemory::allocPointer(
    cudaStream_t stream,
    size_t size,
    AllocType type) {
  return stack_.getAlloc(size, stream, type);
}

void StackDeviceMemory::deallocPointer(
    int device,
    cudaStream_t stream,
    size_t size,
    void* p) {
  CHECK(p);
  CHECK_EQ(device, device_);

  stack_.returnAlloc(p, size, stream);
}

StackDeviceMemory makeStackMemory(size_t bytes) {
  return StackDeviceMemory(getCurrentDevice(), bytes);
}

} // namespace dietgpu
