// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_DEVICE_HPP_

#include "fifo_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// Numeric ID of Semaphore. ProxyService has an internal array indexed by these
/// handles mapping to the actual semaphores.
using SemaphoreId = uint32_t;

/// Numeric ID of RegisteredMemory. ProxyService has an internal array indexed
/// by these handles mapping to the actual.
using MemoryId = uint32_t;

struct BasePortChannelDeviceHandle {
  SemaphoreId semaphoreId_;

  Host2DeviceSemaphoreDeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  FifoDeviceHandle fifo_;

  MSCCLPP_INLINE BasePortChannelDeviceHandle() = default;

  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandle(
      SemaphoreId semaphoreId, Host2DeviceSemaphoreDeviceHandle semaphore,
      FifoDeviceHandle fifo)
      : semaphoreId_(semaphoreId), semaphore_(semaphore), fifo_(fifo) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a TriggerData to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dstId, uint64_t dstOffset,
                                 MemoryId srcId, uint64_t srcOffset,
                                 uint64_t size) {
    fifo_.push(
        {TriggerData, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
  }

  /// Push a TriggerData to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory
  /// regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dstId, MemoryId srcId,
                                 uint64_t offset, uint64_t size) {
    put(dstId, offset, srcId, offset, size);
  }

  /// Push a TriggerFlag to the FIFO.
  MSCCLPP_DEVICE_INLINE void signal() {
    fifo_.push({TriggerFlag, 0, 0, 0, 0, 1, semaphoreId_});
  }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, uint64_t dstOffset,
                                           MemoryId srcId, uint64_t srcOffset,
                                           uint64_t size) {
    fifo_.push({TriggerData | TriggerFlag, dstId, dstOffset, srcId, srcOffset,
                size, semaphoreId_});
  }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory
  /// regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, MemoryId srcId,
                                           uint64_t offset, uint64_t size) {
    putWithSignal(dstId, offset, srcId, offset, size);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to
  /// the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(
      MemoryId dstId, uint64_t dstOffset, MemoryId srcId, uint64_t srcOffset,
      uint64_t size, int64_t maxSpinCount = 1000000) {
    uint64_t curFifoHead =
        fifo_.push({TriggerData | TriggerFlag | TriggerSync, dstId, dstOffset,
                    srcId, srcOffset, size, semaphoreId_});
    fifo_.sync(curFifoHead, maxSpinCount);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to
  /// the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory
  /// regions.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(
      MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size,
      int64_t maxSpinCount = 1000000) {
    putWithSignalAndFlush(dstId, offset, srcId, offset, size, maxSpinCount);
  }

  /// Push a TriggerSync to the FIFO.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    uint64_t curFifoHead =
        fifo_.push({TriggerSync, 0, 0, 0, 0, 1, semaphoreId_});
    fifo_.sync(curFifoHead, maxSpinCount);
  }

  /// Check if the port channel has been signaled.
  /// @return true if the port channel has been signaled.
  MSCCLPP_DEVICE_INLINE bool poll() { return semaphore_.poll(); }

  /// Wait for the port channel to be signaled.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) {
    semaphore_.wait(maxSpinCount);
  }

#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

struct PortChannelDeviceHandle : public BasePortChannelDeviceHandle {
  MemoryId dst_;
  MemoryId src_;

  MSCCLPP_INLINE PortChannelDeviceHandle() = default;

  MSCCLPP_HOST_DEVICE_INLINE PortChannelDeviceHandle(
      SemaphoreId semaphoreId, Host2DeviceSemaphoreDeviceHandle semaphore,
      FifoDeviceHandle fifo, MemoryId dst, MemoryId src)
      : BasePortChannelDeviceHandle(semaphoreId, semaphore, fifo),
        dst_(dst),
        src_(src) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a TriggerData to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset,
                                 uint64_t size) {
    BasePortChannelDeviceHandle::put(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a TriggerData to the FIFO.
  /// @param offset The common offset into the destination and source memory
  /// regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) {
    put(offset, offset, size);
  }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset,
                                           uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandle::putWithSignal(dst_, dstOffset, src_, srcOffset,
                                               size);
  }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory
  /// regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) {
    putWithSignal(offset, offset, size);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to
  /// the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting.
  /// Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(
      uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
      int64_t maxSpinCount = 1000000) {
    BasePortChannelDeviceHandle::putWithSignalAndFlush(
        dst_, dstOffset, src_, srcOffset, size, maxSpinCount);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to
  /// the FIFO.
  /// @param offset The common offset into the destination and source memory
  /// regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset,
                                                   uint64_t size) {
    putWithSignalAndFlush(offset, offset, size);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
