
#pragma once

#include "fifo_util.hpp"
#include "task.h"

namespace mscclpp {

template <typename T>
struct SmDeviceHandle {
  T* buffer;               // T FIFO on device
  uint64_t* head_reserve;  // atomicAdd reservation, device, updated by other SM
  uint64_t* head_publish;  // publish boundary, device, updated by other SM
  uint64_t* tail;          // device, atomically consumed by cur SM
  int size;                // Fifo Size

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Try to get a pointer to the next unconsumed task.
  /// @return Pointer to task if available, nullptr otherwise.
  MSCCLPP_DEVICE_INLINE T* poll() {
    uint64_t currentTail = atomicLoad(tail, memoryOrderRelaxed);
    // Use acquire on head ensures we see buffer writes before head update
    uint64_t currentHead = atomicLoad(head_publish, memoryOrderAcquire);
    if (currentTail >= currentHead) return nullptr;
    return &buffer[currentTail % size];
  }

  /// Consume the task at tail (advance tail by 1).
  /// Only call after poll() returns non-null.
  MSCCLPP_DEVICE_INLINE void pop() {
    atomicFetchAdd<uint64_t, scopeDevice>(tail, 1, memoryOrderRelease);
  }

  /// Push a Task to the FIFO.
  /// @param task Task to push.
  /// @param maxSpinCount Max spin count before assert. Never assert if
  /// negative.
  /// @return Previous head of the FIFO where the task was pushed.
  MSCCLPP_DEVICE_INLINE uint64_t push(T task, int64_t maxSpinCount = 1000000) {
    // reserve slot
    uint64_t pos = atomicFetchAdd<uint64_t, scopeDevice>(head_reserve, 1,
                                                         memoryOrderRelaxed);

    uint64_t t = atomicLoad(tail, memoryOrderAcquire);
    if (pos >= t + size) {
      // full, spin wait
      sync(pos - size, maxSpinCount);
    }

    T* slotPtr = &buffer[pos % size];

    // write buffer payload (release store)
#if defined(MSCCLPP_DEVICE_CUDA)
    asm volatile("st.global.release.sys.v2.u64 [%0], {%1,%2};"
                 :
                 : "l"(slotPtr), "l"(task.fst), "l"(task.snd));
#else
    atomicStore(&(slotPtr->snd), task.snd, memoryOrderRelaxed);
    atomicStore(&(slotPtr->fst), task.fst, memoryOrderRelease);
#endif

    // wait until previous publishes complete
    while (atomicLoad(head_publish, memoryOrderAcquire) != pos)
      ;

    // publish head AFTER data is visible
    // Must ensure consumer sees buffer before head_publish update
    atomicStore<uint64_t, scopeDevice>(head_publish, pos + 1,
                                       memoryOrderRelease);

    return pos;
  }

  /// Wait until a specific task is popped from the FIFO.
  /// @param fifoHead FIFO head where the task was pushed.
  /// @param maxSpinCount Max spin count before assert. Never assert if
  /// negative.
  MSCCLPP_DEVICE_INLINE void sync(
      uint64_t fifoHead, [[maybe_unused]] int64_t maxSpinCount = 1000000) {
    uint64_t val;
    POLL_MAYBE_JAILBREAK(
        (fifoHead >= (val = atomicLoad(tail, memoryOrderAcquire))),
        maxSpinCount);
  }

#endif  // MSCCLPP_DEVICE_COMPILE
};

}  // namespace mscclpp