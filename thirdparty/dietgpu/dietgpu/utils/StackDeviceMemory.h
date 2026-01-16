/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <dietgpu/utils/DeviceUtils.h>
#include <dietgpu/utils/StaticUtils.h>
#include <glog/logging.h>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace dietgpu {

// All memory allocations are aligned to this boundary and are a multiple of
// this size in bytes
constexpr size_t kSDMAlignment = 256;

class StackDeviceMemory;

enum class AllocType {
  Temporary,
  Permanent,
};

/// A RAII object that manages a temporary memory request
template <typename T>
struct GpuMemoryReservation {
  GpuMemoryReservation()
      : res(nullptr),
        device(0),
        stream(nullptr),
        ptr(nullptr),
        num(0),
        sizeAllocated(0) {}

  GpuMemoryReservation(
      StackDeviceMemory* r,
      int dev,
      cudaStream_t str,
      void* p,
      size_t n,
      size_t szAlloc)
      : res(r),
        device(dev),
        stream(str),
        ptr(p),
        num(n),
        sizeAllocated(szAlloc) {}

  GpuMemoryReservation(GpuMemoryReservation&& m) noexcept {
    res = m.res;
    m.res = nullptr;
    device = m.device;
    m.device = 0;
    stream = m.stream;
    m.stream = nullptr;
    ptr = m.ptr;
    m.ptr = nullptr;
    num = m.num;
    m.num = 0;
    sizeAllocated = m.sizeAllocated;
    m.sizeAllocated = 0;
  }

  ~GpuMemoryReservation();

  GpuMemoryReservation& operator=(GpuMemoryReservation&& m) {
    // Can't be both a valid allocation and the same allocation
    CHECK(!(res && res == m.res && device == m.device && ptr == m.ptr));

    release();
    res = m.res;
    m.res = nullptr;
    device = m.device;
    m.device = 0;
    stream = m.stream;
    m.stream = nullptr;
    ptr = m.ptr;
    m.ptr = nullptr;
    num = m.num;
    m.num = 0;
    sizeAllocated = m.sizeAllocated;
    m.sizeAllocated = 0;

    return *this;
  }

  T* data() {
    return reinterpret_cast<T*>(ptr);
  }

  const T* data() const {
    return reinterpret_cast<const T*>(ptr);
  }

  // Copy from the device to a host std::vector<T>, ordered wrt stream
  std::vector<T> copyToHost(cudaStream_t stream_2) const {
    auto out = std::vector<T>(num);

    CUDA_VERIFY(cudaMemcpyAsync(
        out.data(), data(), num * sizeof(T), cudaMemcpyDeviceToHost, stream_2));

    return out;
  }

  void release();

  StackDeviceMemory* res;
  int device;
  cudaStream_t stream;
  void* ptr;
  // number of valid sizeof(T) words available
  size_t num;
  // size allocated in bytes
  size_t sizeAllocated;
};

/// Device memory manager that provides temporary memory allocations
/// out of a region of memory, for a single device
class StackDeviceMemory {
 public:
  /// Allocate a new region of memory that we manage
  StackDeviceMemory(int device, size_t allocPerDevice);

  /// Manage a region of memory for a particular device, without ownership
  StackDeviceMemory(int device, void* p, size_t size);
  ~StackDeviceMemory();

  int getDevice() const;

  // Allocate a chunk of memory on our device ordered wrt the given stream
  // of size sizeof(T) * num bytes
  template <typename T>
  GpuMemoryReservation<T> alloc(
      cudaStream_t stream,
      size_t num,
      AllocType type = AllocType::Temporary) {
    // All allocations are aligned to this size/boundary
    size_t sizeToAlloc = roundUp(num * sizeof(T), kSDMAlignment);
    sizeToAlloc = std::max(sizeToAlloc, kSDMAlignment);

    return GpuMemoryReservation<T>(
        this,
        device_,
        stream,
        allocPointer(stream, sizeToAlloc, type),
        num,
        sizeToAlloc);
  }

  // Copy a T* array from the host to our device, with the memory allocated
  // from ourselves, ordered wrt the given stream
  template <typename T>
  GpuMemoryReservation<T> copyAlloc(
      cudaStream_t stream,
      const T* ptr,
      size_t num,
      AllocType type = AllocType::Temporary) {
    auto size = num * sizeof(T);
    auto mem = alloc<T>(stream, size, type);

    CUDA_VERIFY(
        cudaMemcpyAsync(mem.data(), ptr, size, cudaMemcpyHostToDevice, stream));

    return mem;
  }

  // Copy a std::vector from the host to our device, with the memory allocated
  // from ourselves, ordered wrt the given stream
  template <typename T>
  GpuMemoryReservation<T> copyAlloc(
      cudaStream_t stream,
      const std::vector<T>& v,
      AllocType type = AllocType::Temporary) {
    return copyAlloc<T>(stream, v.data(), v.size(), type);
  }

  /// All allocations requested should be a multiple of kSDMAlignment bytes
  void* allocPointer(cudaStream_t stream, size_t size, AllocType type);
  void deallocPointer(int device, cudaStream_t, size_t size, void* p);

  size_t getSizeAvailable() const;
  size_t getSizeTotal() const;
  std::string toString() const;

  size_t getMaxMemoryUsage() const;
  void resetMaxMemoryUsage();

 protected:
  /// Previous allocation ranges and the streams for which
  /// synchronization is required
  struct Range {
    inline Range(char* s, char* e, cudaStream_t str)
        : start_(s), end_(e), stream_(str) {}

    // References a memory range [start, end)
    char* start_;
    char* end_;
    cudaStream_t stream_;
  };

  struct Stack {
    /// Constructor that allocates memory via cudaMalloc
    Stack(int device, size_t size);

    /// Constructor that uses an externally-provided region of memory
    Stack(int device, void* p, size_t size);

    ~Stack();

    /// Returns how much size is available for an allocation without
    /// calling cudaMalloc
    size_t getSizeAvailable() const;

    /// Returns how large our temporary buffer is in total
    size_t getSizeTotal() const;

    /// Returns how much stack memory is in use
    size_t getStackSizeUsed() const;

    /// Obtains an allocation; all allocations are guaranteed to be 16
    /// byte aligned
    void* getAlloc(size_t size, cudaStream_t stream, AllocType type);

    /// Returns an allocation
    void returnAlloc(void* p, size_t size, cudaStream_t stream);

    /// Returns the stack state
    std::string toString() const;

    /// Device this allocation is on
    int device_;

    /// Where our temporary memory buffer is allocated; we allocate starting 16
    /// bytes into this
    char* alloc_;

    /// Total size of our allocation
    size_t allocSize_;

    /// Our temporary memory region; [start_, end_) is valid
    char* start_;
    char* end_;

    /// Stack head within [start, end)
    char* head_;

    /// Free allocations via cudaMalloc that we made that couldn't fit inside
    /// our stack
    std::unordered_map<void*, size_t> overflowAllocs_;

    /// How much memory we currently have in overflowAllocs_
    size_t overflowSize_;

    /// The current maximum seen memory usage, including both stack usage and
    /// overflow allocations
    size_t maxSeenSize_;
  };

  /// Our device
  int device_;

  /// Memory stack
  Stack stack_;
};

template <typename T>
GpuMemoryReservation<T>::~GpuMemoryReservation() {
  if (ptr) {
    CHECK(res);
    res->deallocPointer(device, stream, sizeAllocated, ptr);
  }
}

template <typename T>
void GpuMemoryReservation<T>::release() {
  if (ptr) {
    CHECK(res);
    res->deallocPointer(device, stream, sizeAllocated, ptr);
    res = nullptr;
    device = 0;
    stream = nullptr;
    ptr = nullptr;
    num = 0;
    sizeAllocated = 0;
  }
}

// Construct a StackDeviceMemory for the current device pre-allocating the given
// amount of memory
StackDeviceMemory makeStackMemory(size_t bytes = 256 * 1024 * 1024);

} // namespace dietgpu
