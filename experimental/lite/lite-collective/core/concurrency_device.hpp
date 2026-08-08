// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONCURRENCY_DEVICE_HPP_
#define MSCCLPP_CONCURRENCY_DEVICE_HPP_

#include "atomic_device.hpp"
#include "poll_device.hpp"

namespace mscclpp {

/// A device-wide barrier.
/// This barrier can be used to synchronize multiple thread blocks within a
/// kernel. It uses atomic operations to ensure that all threads in the same
/// kernel reach the barrier before proceeding and they can safely read data
/// written by other threads in different blocks.
///
/// Example usage of DeviceSyncer:
/// ```cpp
/// __global__ void myKernel(mscclpp::DeviceSyncer* syncer, int numBlocks) {
///   // Do some work here
///   // ...
///   // Synchronize all blocks
///   syncer->sync(numBlocks);
///   // All blocks have reached this point
///   // ...
/// }
/// ```
struct DeviceSyncer {
 public:
  /// Construct a new DeviceSyncer object.
  MSCCLPP_INLINE DeviceSyncer() = default;

  /// Destroy the DeviceSyncer object.
  MSCCLPP_INLINE ~DeviceSyncer() = default;

  /// The number of sync counters.
  static unsigned int const NumCounters = 3U;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Synchronize all threads inside a kernel. Guarantee that all previous work
  /// of all threads in cooperating blocks is finished.
  /// @param blockNum The number of blocks that will synchronize.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  MSCCLPP_DEVICE_INLINE void sync(
      int blockNum, [[maybe_unused]] int64_t maxSpinCount = 100000000) {
    unsigned int targetCnt = blockNum;
    __syncthreads();
    if (blockNum == 1) return;
    if (threadIdx.x == 0) {
      unsigned int countIdx = (currentCountIdx_ + 1) % NumCounters;
      unsigned int nextCountIdx = (countIdx + 1) % NumCounters;
      unsigned int* count = &count_[countIdx];
      count_[nextCountIdx] = 0;
      atomicFetchAdd<unsigned int, scopeDevice>(count, 1U, memoryOrderRelease);
      POLL_MAYBE_JAILBREAK((atomicLoad<unsigned int, scopeDevice>(
                                count, memoryOrderAcquire) != targetCnt),
                           maxSpinCount);
      currentCountIdx_ = countIdx;
    }
    // We need sync here because only a single thread is checking whether
    // the flag is flipped.
    __syncthreads();
  }
#endif  // !defined(MSCCLPP_DEVICE_COMPILE)

 private:
  /// The counter of synchronized blocks.
  unsigned int count_[NumCounters];
  /// Index of the current counter being used.
  unsigned int currentCountIdx_;
};

/// A device-wide semaphore.
/// This semaphore can be used to control access to a resource across multiple
/// threads or blocks. It uses atomic operations to ensure that the semaphore
/// value is updated correctly across threads. The semaphore value is an integer
/// that can be set, acquired, and released.
///
/// Example usage of DeviceSemaphore:
/// ```cpp
/// __global__ void myKernel(mscclpp::DeviceSemaphore* semaphore) {
///   // Initialize the semaphore (allow up to 3 threads access the resource
///   simultaneously) if (blockIdx.x == 0 && threadIdx.x == 0) {
///     semaphore->set(3);
///   }
///   // Each block acquires the semaphore before accessing the shared resource
///   if (threadIdx.x == 0) {
///     semaphore->acquire();
///   }
///   __syncthreads();
///   // Access the shared resource
///   // ...
///   __syncthreads();
///   // Release the semaphore after accessing the shared resource
///   if (threadIdx.x == 0) {
///     semaphore->release();
///   }
/// }
/// ```
struct DeviceSemaphore {
 public:
  /// Construct a new DeviceSemaphore object.
  MSCCLPP_INLINE DeviceSemaphore() = default;

  // / Construct a new DeviceSemaphore object with an initial value.
  /// @param initialValue The initial value of the semaphore.
  MSCCLPP_INLINE DeviceSemaphore(int initialValue) : semaphore_(initialValue) {}

  /// Destroy the DeviceSemaphore object.
  MSCCLPP_INLINE ~DeviceSemaphore() = default;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Set the semaphore value. This function is used to initialize or reset the
  /// semaphore value. The initial value is typically set to a positive integer
  /// to allow acquiring the semaphore.
  /// @param value The value to set.
  MSCCLPP_DEVICE_INLINE void set(int value) {
    atomicStore<int, scopeDevice>(&semaphore_, value, memoryOrderRelease);
  }

  /// Acquire the semaphore (decrease the semaphore value by 1).
  /// This function will spin until the semaphore is acquired or the maximum
  /// spin count is reached.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  MSCCLPP_DEVICE_INLINE void acquire([[maybe_unused]] int maxSpinCount = -1) {
    int oldVal =
        atomicFetchAdd<int, scopeDevice>(&semaphore_, -1, memoryOrderAcquire);
    if (oldVal <= 0) {
      POLL_MAYBE_JAILBREAK((atomicLoad<int, scopeDevice>(
                                &semaphore_, memoryOrderAcquire) < oldVal),
                           maxSpinCount);
    }
  }

  /// Release the semaphore (increase the semaphore value by 1).
  MSCCLPP_DEVICE_INLINE void release() {
    atomicFetchAdd<int, scopeDevice>(&semaphore_, 1, memoryOrderRelease);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

 private:
  /// The semaphore value.
  int semaphore_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONCURRENCY_DEVICE_HPP_
