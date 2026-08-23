// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <stdexcept>

namespace mscclpp::lite {

// Describes the allocation backing a Span/Rows view.  The host variants are
// kept distinct because only pinned/mapped allocations provide the CUDA
// properties needed by genuinely asynchronous host/device data movement.
enum class MemoryType { Host, HostPinned, HostMapped, Device };

template <typename T>
struct Span {
  T* data = nullptr;
  size_t count = 0;
  int numaNode = -1;
  int device = -1;

  size_t bytes() const { return count * sizeof(T); }

  Span subspan(size_t offset, size_t length) const {
    if (offset > count || length > count - offset ||
        (data == nullptr && length != 0)) {
      throw std::out_of_range("CpuSwitch span is out of bounds");
    }
    return {data == nullptr ? nullptr : data + offset, length, numaNode,
            device};
  }
};

template <typename T>
struct Rows {
  T* data = nullptr;
  size_t rowCount = 0;
  size_t columnCount = 0;
  size_t stride = 0;  // In terms of elements, not bytes.
  int numaNode = -1;
  int device = -1;

  Span<T> row(size_t index) const {
    if (index >= rowCount || stride < columnCount || data == nullptr) {
      throw std::out_of_range("CpuSwitch row is out of bounds");
    }
    return {data + index * stride, columnCount, numaNode, device};
  }
};

}  // namespace mscclpp::lite
