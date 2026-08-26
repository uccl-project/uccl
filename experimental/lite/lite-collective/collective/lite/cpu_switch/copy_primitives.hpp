// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "completion.hpp"
#include "types.hpp"

#include <cstring>
#include <type_traits>

namespace mscclpp::lite {
namespace copy_detail {

template <MemoryType Kind>
inline constexpr bool IsHostMemory =
    Kind == MemoryType::Host || Kind == MemoryType::HostPinned ||
    Kind == MemoryType::HostMapped;

template <MemoryType Kind>
inline constexpr bool IsDeviceMemory = Kind == MemoryType::Device;

template <MemoryType Src, MemoryType Dst>
constexpr cudaMemcpyKind cudaCopyKind() {
  static_assert(!(IsHostMemory<Src> && IsHostMemory<Dst>),
                "host-to-host copies do not use CUDA");
  static_assert((IsHostMemory<Src> || IsDeviceMemory<Src>) &&
                    (IsHostMemory<Dst> || IsDeviceMemory<Dst>),
                "unsupported CpuSwitch memory type");
  if constexpr (IsDeviceMemory<Src> && IsDeviceMemory<Dst>) {
    return cudaMemcpyDeviceToDevice;
  } else if constexpr (IsDeviceMemory<Src>) {
    return cudaMemcpyDeviceToHost;
  } else {
    return cudaMemcpyHostToDevice;
  }
}

}  // namespace copy_detail

template <typename T>
class CopyPrimitives {
 public:
  template <MemoryType Src, MemoryType Dst, typename SrcT>
  Completion copy(Span<SrcT> source, Span<T> destination,
                  cudaStream_t stream = nullptr) const {
    enqueueCopy<Src, Dst>(source, destination, stream);
    if constexpr (copy_detail::IsHostMemory<Src> &&
                  copy_detail::IsHostMemory<Dst>) {
      return {};
    } else {
      return Completion::record(stream);
    }
  }

  /**
  Submit a copy without allocating a completion event. Host-to-host copies
  finish before returning; CUDA copies are only enqueued on the given stream.
  The caller must keep buffers alive and use its existing stream/event/flag
  protocol before reading the result or reusing either buffer.
  */
  template <MemoryType Src, MemoryType Dst, typename SrcT>
  void enqueueCopy(Span<SrcT> source, Span<T> destination,
                   cudaStream_t stream = nullptr) const {
    static_assert(std::is_same_v<std::remove_const_t<SrcT>, T>,
                  "CpuSwitch copy element types must match");
    checkCopy(source, destination);
    size_t bytes = destination.bytes();
    if constexpr (copy_detail::IsHostMemory<Src> &&
                  copy_detail::IsHostMemory<Dst>) {
      std::memmove(destination.data, source.data, bytes);
    } else {
      throwCudaError(cudaMemcpyAsync(destination.data, source.data, bytes,
                                     copy_detail::cudaCopyKind<Src, Dst>(),
                                     stream),
                     "CpuSwitch copy");
    }
  }

  template <MemoryType Src, MemoryType Dst, typename SrcT>
  Completion copyRows(Rows<SrcT> source, Rows<T> destination,
                      cudaStream_t stream = nullptr) const {
    enqueueCopyRows<Src, Dst>(source, destination, stream);
    if constexpr (copy_detail::IsHostMemory<Src> &&
                  copy_detail::IsHostMemory<Dst>) {
      return {};
    } else {
      return Completion::record(stream);
    }
  }

  /* Same completion/lifetime contract as enqueueCopy; strides are in elements. */
  template <MemoryType Src, MemoryType Dst, typename SrcT>
  void enqueueCopyRows(Rows<SrcT> source, Rows<T> destination,
                       cudaStream_t stream = nullptr) const {
    static_assert(std::is_same_v<std::remove_const_t<SrcT>, T>,
                  "CpuSwitch row-copy element types must match");
    checkRows(source);
    checkRows(destination);
    if (source.rowCount != destination.rowCount ||
        source.columnCount != destination.columnCount) {
      throw std::invalid_argument("CpuSwitch row-copy shapes do not match");
    }
    size_t width = source.columnCount * sizeof(T);
    if constexpr (copy_detail::IsHostMemory<Src> &&
                  copy_detail::IsHostMemory<Dst>) {
      for (size_t row = 0; row < source.rowCount; ++row) {
        std::memmove(destination.data + row * destination.stride,
                     source.data + row * source.stride, width);
      }
    } else {
      throwCudaError(
          cudaMemcpy2DAsync(destination.data, destination.stride * sizeof(T),
                            source.data, source.stride * sizeof(T), width,
                            source.rowCount,
                            copy_detail::cudaCopyKind<Src, Dst>(), stream),
          "CpuSwitch row copy");
    }
  }

 private:
  template <typename SrcT>
  static void checkCopy(Span<SrcT> source, Span<T> destination) {
    if (source.count != destination.count ||
        (source.data == nullptr && source.count != 0) ||
        (destination.data == nullptr && destination.count != 0)) {
      throw std::invalid_argument("CpuSwitch copy spans do not match");
    }
  }

  template <typename RowT>
  static void checkRows(Rows<RowT> rows) {
    // Each row (allows padding) must fit within its stride, and non-empty rows need valid data.
    if (rows.stride < rows.columnCount || (rows.data == nullptr && rows.rowCount * rows.columnCount != 0)) {
      throw std::invalid_argument("CpuSwitch rows are invalid");
    }
  }
};

}  // namespace mscclpp::lite
