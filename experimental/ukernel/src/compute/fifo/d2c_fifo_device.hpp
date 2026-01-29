// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_DEVICE_HPP_
#define MSCCLPP_FIFO_DEVICE_HPP_

#include "../task.h"
#include "fifo_util.hpp"
#include <cstdint>

namespace mscclpp {

/// Pair of 64-bit unsigned integers used as a trigger for the proxy.
/// Used as a work element in the concurrent FIFO.
/// Most significant bit of snd is reserved.
union alignas(16) ProxyTrigger {
  struct {
    uint64_t fst;
    uint64_t snd;
  };

  struct {
    uint64_t type : UKernel::Compute::TaskTypeSize;
    uint64_t dataType : UKernel::Compute::DataTypeSize;
    uint64_t blockId : UKernel::Compute::BlockIdSize;
    uint64_t : (64 - UKernel::Compute::TaskTypeSize -
                UKernel::Compute::DataTypeSize - UKernel::Compute::BlockIdSize);
    uint64_t argsId : UKernel::Compute::TaskArgsIndexSize;
    uint64_t : (64 - UKernel::Compute::TaskArgsIndexSize);
  } fields;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Default constructor.
  MSCCLPP_INLINE ProxyTrigger() = default;

  /// Constructor.
  /// @param type The type of the Task.
  /// @param dType The type of Data.
  /// @param blockIndex Which block the task will be dispatched to.
  /// @param argsIndex The Args Id of Task (in TaskManager).
  MSCCLPP_DEVICE_INLINE ProxyTrigger(UKernel::Compute::TaskType type,
                                     UKernel::Compute::DataType dType,
                                     uint32_t blockIndex, uint32_t argsIndex) {
    const uint64_t t = static_cast<uint64_t>(type);
    const uint64_t dt = static_cast<uint64_t>(dType);
    const uint64_t bi = static_cast<uint64_t>(blockIndex);
    const uint64_t ai = static_cast<uint64_t>(argsIndex);

    MSCCLPP_ASSERT_DEVICE(t < (1ULL << UKernel::Compute::TaskTypeSize),
                          "type is too large");
    MSCCLPP_ASSERT_DEVICE(dt < (1ULL << UKernel::Compute::DataTypeSize),
                          "dType is too large");
    MSCCLPP_ASSERT_DEVICE(bi < (1ULL << UKernel::Compute::BlockIdSize),
                          "blockIndex is too large");
    MSCCLPP_ASSERT_DEVICE(ai < (1ULL << UKernel::Compute::TaskArgsIndexSize),
                          "argsIndex is too large");

    constexpr uint64_t maskType = (1ULL << UKernel::Compute::TaskTypeSize) - 1;
    constexpr uint64_t maskDType = (1ULL << UKernel::Compute::DataTypeSize) - 1;
    constexpr uint64_t maskBlockId =
        (1ULL << UKernel::Compute::BlockIdSize) - 1;
    constexpr uint64_t maskArgs =
        (1ULL << UKernel::Compute::TaskArgsIndexSize) - 1;
    fst =
        (t & maskType) | ((dt & maskDType) << UKernel::Compute::TaskTypeSize) |
        ((bi & maskBlockId)
         << (UKernel::Compute::TaskTypeSize + UKernel::Compute::DataTypeSize));
    snd = (ai & maskArgs);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

/// Concurrent FIFO where multiple device threads (the number of threads should
/// not exceed the FIFO size) to push Head pointer is on device, tail pointer is
/// on host (readable by device). The FIFO’s capacity is limited only by
/// MAX_UINT64—effectively infinite for practical use. Exceeding this limit will
/// overflow the counter and lead to undefined behavior.
struct FifoDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a trigger to the FIFO.
  /// @param trigger Trigger to push.
  /// @param maxSpinCount Max spin count before assert. Never assert if
  /// negative.
  /// @return Previous head of the FIFO where the trigger was pushed.
  MSCCLPP_DEVICE_INLINE uint64_t push(ProxyTrigger trigger,
                                      int64_t maxSpinCount = 1000000) {
    uint64_t prevHead =
        atomicFetchAdd<uint64_t, scopeDevice>(head, 1, memoryOrderRelaxed);

    // Flip the last bit for safe polling; host will revert.
    constexpr uint64_t flipMask = uint64_t{1} << uint64_t{63};
    trigger.snd ^= flipMask;

    // Wait until the trigger is freed by the host.
    if (prevHead >= size + *tailCache) {
      sync(prevHead - size, maxSpinCount);
    }

    ProxyTrigger* triggerPtr = &(triggers[prevHead % size]);

#if defined(MSCCLPP_DEVICE_CUDA)
#if __CUDA_ARCH__ == 800
    // This is faster than release for A100.
    __threadfence_system();
    asm volatile(
        "st.global.relaxed.sys.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr),
        "l"(trigger.fst), "l"(trigger.snd));
#else
    asm volatile(
        "st.global.release.sys.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr),
        "l"(trigger.fst), "l"(trigger.snd));
#endif
#else   // !defined(MSCCLPP_DEVICE_CUDA)
    // Store snd no later than fst.
    atomicStore(&(triggerPtr->snd), trigger.snd, memoryOrderRelaxed);
    atomicStore(&(triggerPtr->fst), trigger.fst, memoryOrderRelease);
#endif  // !defined(MSCCLPP_DEVICE_CUDA)

    return prevHead;
  }

  /// Poll whether a specific trigger is popped from the FIFO.
  /// @param fifoHead FIFO head where the trigger was pushed.
  /// @return True if the trigger is popped; false otherwise.
  MSCCLPP_DEVICE_INLINE bool poll(uint64_t fifoHead) {
    uint64_t val;
    if (fifoHead <
        (val = atomicLoad(tail,
                          memoryOrderAcquire))) {  // GPU 可以从host ptr load
      // Same as in sync(), this may write a stale value to tailCache.
      *tailCache = val;
      return true;
    }
    return false;
  }

  /// Wait until a specific trigger is popped from the FIFO.
  /// @param fifoHead FIFO head where the trigger was pushed.
  /// @param maxSpinCount Max spin count before assert. Never assert if
  /// negative.
  MSCCLPP_DEVICE_INLINE void sync(
      uint64_t fifoHead, [[maybe_unused]] int64_t maxSpinCount = 1000000) {
    uint64_t val;
    POLL_MAYBE_JAILBREAK(
        (fifoHead >= (val = atomicLoad(tail, memoryOrderAcquire))),
        maxSpinCount);
    // If multiple threads sync in parallel, this may write a stale value to
    // tailCache. This is fine, as the tailCache is for avoiding unnecessary
    // syncs from the push(), which can work as long as the tailCache is not
    // stale by the length of the FIFO.
    *tailCache = val;
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  /// FIFO buffer on host.
  ProxyTrigger* triggers;
  /// FIFO head on device.
  uint64_t* head;
  /// FIFO tail on host.
  uint64_t* tail;
  /// Cached tail value.
  uint64_t* tailCache;
  /// FIFO size.
  int size;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_DEVICE_HPP_