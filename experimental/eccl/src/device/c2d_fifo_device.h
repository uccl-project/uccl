#pragma once

#include "fifo_util.hpp"
#include <cstddef>
#include <iterator>
#include <memory>

namespace mscclpp {

/// Device-side handle for CpuToGpuFifo.
template <typename T>
struct C2DDeviceHandle {
  T* buffer;       // T FIFO on device
  uint64_t* head;  // host-pinned, updated by CPU
  uint64_t* tail;  // device, atomically consumed by GPU
  int size;        // Fifo Size

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Try to get a pointer to the next unconsumed task.
  /// @return Pointer to task if available, nullptr otherwise.
  MSCCLPP_DEVICE_INLINE T* poll() {
    uint64_t currentTail =
        atomicLoad(tail, memoryOrderRelaxed);  // relaxed，acquire？
    // Use acquire to ensure we see the data written by CPU before head update
    uint64_t currentHead = atomicLoad(head, memoryOrderAcquire);
    if (currentTail >= currentHead) return nullptr;
    return &buffer[currentTail % size];
  }

  /// Consume the task at tail (advance tail by 1).
  /// Only call after poll() returns non-null.
  MSCCLPP_DEVICE_INLINE void pop() {
    atomicFetchAdd<uint64_t, scopeDevice>(tail, 1, memoryOrderRelease);
  }
#endif  // MSCCLPP_DEVICE_COMPILE
};

}  // namespace mscclpp
