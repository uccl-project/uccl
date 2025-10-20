// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_DEVICE_HPP_
#define MSCCLPP_FIFO_DEVICE_HPP_

#include "fifo_util.hpp"
#include "ring_buffer.cuh"
#include <cstdint>

namespace mscclpp {

using TriggerType = uint64_t;
constexpr TriggerType TriggerData = 0x1;  // Trigger a data transfer.
constexpr TriggerType TriggerFlag = 0x2;  // Trigger a signaling.
constexpr TriggerType TriggerSync = 0x4;  // Trigger a flush.

constexpr unsigned int TriggerBitsSize = 32;
constexpr unsigned int TriggerBitsOffset = 32;
constexpr unsigned int TriggerBitsMemoryId = 9;
constexpr unsigned int TriggerBitsType = 3;
constexpr unsigned int TriggerBitsSemaphoreId = 10;
constexpr unsigned int TriggerBitsFifoReserved = 1;

/// Pair of 64-bit unsigned integers used as a trigger for the proxy.
/// Used as a work element in the concurrent FIFO.
/// Most significant bit of snd is reserved.
union alignas(16) ProxyTrigger {
  struct {
    uint64_t fst;
    uint64_t snd;
  };
  // The summation of number of bits must be 128 or less.
  struct {
    // First 64 bits: value[0]
    uint64_t size : TriggerBitsSize;
    uint64_t srcOffset : TriggerBitsOffset;
    uint64_t : (64 - TriggerBitsSize -
                TriggerBitsOffset);  // ensure 64-bit alignment
    // Second 64 bits: value[1]
    uint64_t dstOffset : TriggerBitsOffset;
    uint64_t srcMemoryId : TriggerBitsMemoryId;
    uint64_t dstMemoryId : TriggerBitsMemoryId;
    uint64_t type : TriggerBitsType;
    uint64_t semaphoreId : TriggerBitsSemaphoreId;
    uint64_t : (64 - TriggerBitsOffset - TriggerBitsMemoryId -
                TriggerBitsMemoryId - TriggerBitsType - TriggerBitsSemaphoreId -
                TriggerBitsFifoReserved);  // ensure 64-bit alignment
    uint64_t reserved : TriggerBitsFifoReserved;
  } fields;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Default constructor.
  MSCCLPP_INLINE ProxyTrigger() = default;

  MSCCLPP_DEVICE_INLINE ProxyTrigger
  ProxyTriggerFromTransferCmd(TransferCmd const& c, uint32_t semaphoreId = 0) {
    // Map CmdType -> TriggerType (3 bits)
    TriggerType type;
    switch (c.cmd_type) {
      case CmdType::WRITE:
        type = TriggerData;
        break;
      case CmdType::ATOMIC:
        type = TriggerData;
        break;  // still a data op for the proxy
      case CmdType::QUIET:
        type = TriggerFlag;
        break;
      case CmdType::BARRIER:
        type = TriggerSync;
        break;
      default:
        type = TriggerFlag;
        break;
    }
    /*
     * NOTE: Fields in TransferCmd that are **NOT** captured by
     * ProxyTriggerFromTransferCmd()
     * ----------------------------------------------------------------------------
     * The current mapping only encodes:
     *   - bytes        -> fields.size              (32 bits, max 4 GiB - 1)
     *   - req_lptr     -> fields.srcOffset         (32 bits; must fit < 2^32)
     *   - req_rptr     -> fields.dstOffset         (32 bits; must fit < 2^32)
     *   - dst_rank     -> fields.dstMemoryId       (9 bits; must be < 512)
     *   - dst_gpu      -> fields.srcMemoryId       (9 bits; must be < 512)
     *   - cmd_type     -> fields.type              (3 bits; values 0..7)  [only
     * if we map it]
     *   - semaphoreId  -> fields.semaphoreId       (10 bits)              [only
     * if we map it]
     *
     * Everything else in TransferCmd is currently **omitted** from the 128-bit
     * trigger:
     *   - cmd                         (64 bits)
     *   - src_ptr                     (64-bit pointer)
     *   - warp_id                     (32 bits)
     *   - expert_idx                  (32 bits)
     *   - lane_id                     (32 bits)
     *   - message_idx                 (32 bits)
     *   - is_atomic                   (1 bit)
     *   - value                       (32 bits)
     *   - is_combine                  (1 bit)
     *   - low_latency_buffer_idx      (32 bits)
     *   - atomic_offset               (64 bits)
     *   - atomic_val                  (64 bits)
     */

    // Derive IDs and offsets from TransferCmd
    const uint32_t dstId = c.dst_rank;      // -> src/dstMemoryId (9 bits each)
    const uint32_t srcId = c.dst_gpu;       // -> src/dstMemoryId (9 bits each)
    const uint64_t dstOffset = c.req_rptr;  // -> dstOffset (32 bits)
    const uint64_t srcOffset = c.req_lptr;  // -> srcOffset (32 bits)
    const uint64_t bytes = c.bytes;         // -> size (32 bits)

    // Bit-width checks (mirror MSCCL constructor)
    MSCCLPP_ASSERT_DEVICE(type < (1ULL << TriggerBitsType),
                          "type is too large");
    MSCCLPP_ASSERT_DEVICE(dstId < (1ULL << TriggerBitsMemoryId),
                          "dstId is too large");
    MSCCLPP_ASSERT_DEVICE(dstOffset < (1ULL << TriggerBitsOffset),
                          "dstOffset is too large");
    MSCCLPP_ASSERT_DEVICE(srcId < (1ULL << TriggerBitsMemoryId),
                          "srcId is too large");
    MSCCLPP_ASSERT_DEVICE(srcOffset < (1ULL << TriggerBitsOffset),
                          "srcOffset is too large");
    MSCCLPP_ASSERT_DEVICE(bytes != 0, "bytes must not be zero");
    MSCCLPP_ASSERT_DEVICE(bytes < (1ULL << TriggerBitsSize),
                          "bytes is too large");
    MSCCLPP_ASSERT_DEVICE(semaphoreId < (1ULL << TriggerBitsSemaphoreId),
                          "semaphoreId is too large");

    // Masks
    constexpr uint64_t maskSize = (1ULL << TriggerBitsSize) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskType = (1ULL << TriggerBitsType) - 1;
    constexpr uint64_t maskSemaphoreId = (1ULL << TriggerBitsSemaphoreId) - 1;

    ProxyTrigger trig;
    // fst: [ srcOffset (32b) | bytes (32b) ]
    trig.fst =
        (((srcOffset & maskSrcOffset) << TriggerBitsSize) | (bytes & maskSize));

    // snd: [ dstOffset | srcId | dstId | type | semId ] packed high→low per
    // MSCCL
    trig.snd = (((((((((static_cast<uint64_t>(semaphoreId) & maskSemaphoreId)
                       << TriggerBitsType) |
                      (static_cast<uint64_t>(type) & maskType))
                     << TriggerBitsMemoryId) |
                    (static_cast<uint64_t>(dstId) & maskDstMemoryId))
                   << TriggerBitsMemoryId) |
                  (static_cast<uint64_t>(srcId) & maskSrcMemoryId))
                 << TriggerBitsOffset) |
                (dstOffset & maskDstOffset));

    return trig;
  }

  /// Constructor.
  /// @param type The type of the trigger.
  /// @param dstId The destination ID of memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The source ID of memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param bytes The bytes of the transfer.
  /// @param semaphoreId The ID of the semaphore.
  MSCCLPP_DEVICE_INLINE ProxyTrigger(TriggerType type, uint32_t dstId,
                                     uint64_t dstOffset, uint32_t srcId,
                                     uint64_t srcOffset, uint64_t bytes,
                                     uint32_t semaphoreId) {
    MSCCLPP_ASSERT_DEVICE(type < (1ULL << TriggerBitsType),
                          "type is too large");
    MSCCLPP_ASSERT_DEVICE(dstId < (1ULL << TriggerBitsMemoryId),
                          "dstId is too large");
    MSCCLPP_ASSERT_DEVICE(dstOffset < (1ULL << TriggerBitsOffset),
                          "dstOffset is too large");
    MSCCLPP_ASSERT_DEVICE(srcId < (1ULL << TriggerBitsMemoryId),
                          "srcId is too large");
    MSCCLPP_ASSERT_DEVICE(srcOffset < (1ULL << TriggerBitsOffset),
                          "srcOffset is too large");
    MSCCLPP_ASSERT_DEVICE(bytes != 0, "bytes must not be zero");
    MSCCLPP_ASSERT_DEVICE(bytes < (1ULL << TriggerBitsSize),
                          "bytes is too large");
    MSCCLPP_ASSERT_DEVICE(semaphoreId < (1ULL << TriggerBitsSemaphoreId),
                          "semaphoreId is too large");
    constexpr uint64_t maskSize = (1ULL << TriggerBitsSize) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskType = (1ULL << TriggerBitsType) - 1;
    constexpr uint64_t maskSemaphoreId = (1ULL << TriggerBitsSemaphoreId) - 1;
    fst =
        (((srcOffset & maskSrcOffset) << TriggerBitsSize) + (bytes & maskSize));
    snd = (((((((((semaphoreId & maskSemaphoreId) << TriggerBitsType) +
                 ((uint64_t)type & maskType))
                << TriggerBitsMemoryId) +
               (dstId & maskDstMemoryId))
              << TriggerBitsMemoryId) +
             (srcId & maskSrcMemoryId))
            << TriggerBitsOffset) +
           (dstOffset & maskDstOffset));
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
    if (fifoHead < (val = atomicLoad(tail, memoryOrderAcquire))) {
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