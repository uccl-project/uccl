// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_MEMORY_CHANNEL_DEVICE_HPP_
#define MSCCLPP_MEMORY_CHANNEL_DEVICE_HPP_

#include "semaphore_device.hpp"
#if defined(MSCCLPP_DEVICE_COMPILE)
#include "copy_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

/// Device-side handle of a MemoryChannel without specific source and
/// destination.
struct BaseMemoryChannelDeviceHandle {
  MemoryDevice2DeviceSemaphoreDeviceHandle semaphore_;

  MSCCLPP_INLINE BaseMemoryChannelDeviceHandle() = default;

  MSCCLPP_HOST_DEVICE_INLINE BaseMemoryChannelDeviceHandle(
      MemoryDevice2DeviceSemaphoreDeviceHandle semaphore)
      : semaphore_(semaphore) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Signal the remote semaphore.
  ///
  /// This function guarantees that all the memory operation before this
  /// function is completed before the remote semaphore is signaled.
  ///
  MSCCLPP_DEVICE_INLINE void signal() { semaphore_.signal(); }

  /// Signal the remote semaphore.
  ///
  /// This function is a relaxed version of signal() and provides no guarantee
  /// on the completion of memory operations. User requires to call proper
  /// fencing before using this function.
  ///
  MSCCLPP_DEVICE_INLINE void relaxedSignal() { semaphore_.relaxedSignal(); }

  /// Check if the remote semaphore has signaled.
  /// @return true if the remote semaphore has signaled.
  MSCCLPP_DEVICE_INLINE bool poll() { return semaphore_.poll(); }

  /// Wait for the remote semaphore to send a signal.
  /// @param maxSpinCount The maximum number of spins before asserting. Never
  /// assert if negative.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) {
    semaphore_.wait(maxSpinCount);
  }

  /// Wait for the remote semaphore to send a signal.
  ///
  /// This function is a relaxed version of signal() and provides no guarantee
  /// on the completion of memory operations. User requires to call proper
  /// fencing before using this function.
  ///
  /// @param maxSpinCount The maximum number of spins before asserting. Never
  /// assert if negative.
  MSCCLPP_DEVICE_INLINE void relaxedWait(int64_t maxSpinCount = 10000000) {
    semaphore_.relaxedWait(maxSpinCount);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

/// Device-side handle of a MemoryChannel.
struct MemoryChannelDeviceHandle : public BaseMemoryChannelDeviceHandle {
  void* dst_;
  void* src_;
  void* packetBuffer_;

  MSCCLPP_INLINE MemoryChannelDeviceHandle() = default;

  MSCCLPP_HOST_DEVICE_INLINE MemoryChannelDeviceHandle(
      MemoryDevice2DeviceSemaphoreDeviceHandle semaphore, void* dst, void* src,
      void* packetBuffer)
      : BaseMemoryChannelDeviceHandle(semaphore),
        dst_(dst),
        src_(src),
        packetBuffer_(packetBuffer) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Load a value from the remote memory.
  /// @tparam T The type of the value to be loaded.
  /// @param index The index of the value to be loaded. The offset in bytes is
  /// calculated as index * sizeof(T).
  /// @return The value loaded.
  template <typename T>
  MSCCLPP_DEVICE_INLINE T read(uint64_t index) {
    return *(reinterpret_cast<T*>(dst_) + index);
  }

  /// Write a value to the remote memory.
  /// @tparam T The type of the value to be written.
  /// @param index The index of the value to be written. The offset in bytes is
  /// calculated as index * sizeof(T).
  /// @param v The value to be written.
  template <typename T>
  MSCCLPP_DEVICE_INLINE void write(uint64_t index, const T& v) {
    *(reinterpret_cast<T*>(dst_) + index) = v;
  }

  /// Copy data from the local memory (origin) to the remote memory (target).
  ///
  /// This function is intended to be collectively called by multiple threads.
  /// Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses.
  /// Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of
  /// bytes is not a multiple of @p Alignment.
  /// @param targetOffset The offset in bytes of the remote address. Should be a
  /// multiple of @p Alignment.
  /// @param originOffset The offset in bytes of the local address. Should be a
  /// multiple of @p Alignment.
  /// @param originBytes Bytes of the origin to be copied. Should be a multiple
  /// of @p Alignment.
  /// @param threadId The index of the current thread among all threads running
  /// this function. This is different from the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void put(uint64_t targetOffset, uint64_t originOffset,
                                 uint64_t originBytes, uint32_t threadId,
                                 uint32_t numThreads) {
    copy<Alignment, CopyRemainder>(reinterpret_cast<char*>(dst_) + targetOffset,
                                   reinterpret_cast<char*>(src_) + originOffset,
                                   originBytes, threadId, numThreads);
  }

  /// Wrapper of put() with the same offset for target and origin.
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t originBytes,
                                 uint32_t threadId, uint32_t numThreads) {
    put<Alignment, CopyRemainder>(offset, offset, originBytes, threadId,
                                  numThreads);
  }

  /// Copy data from the remote memory (origin) to the local memory (target).
  ///
  /// This function is intended to be collectively called by multiple threads.
  /// Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses.
  /// Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of
  /// bytes is not a multiple of @p Alignment.
  /// @param targetOffset The offset in bytes of the local address. Should be a
  /// multiple of @p Alignment.
  /// @param originOffset The offset in bytes of the remote address. Should be a
  /// multiple of @p Alignment.
  /// @param originBytes Bytes of the origin to be copied. Should be a multiple
  /// of @p Alignment.
  /// @param threadId The index of the current thread among all threads running
  /// this function. This is different from the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void get(uint64_t targetOffset, uint64_t originOffset,
                                 uint64_t originBytes, uint32_t threadId,
                                 uint32_t numThreads) {
    copy<Alignment, CopyRemainder>(reinterpret_cast<char*>(src_) + targetOffset,
                                   reinterpret_cast<char*>(dst_) + originOffset,
                                   originBytes, threadId, numThreads);
  }

  /// Wrapper of get() with the same offset for target and origin.
  template <int Alignment = 16, bool CopyRemainder = true>
  MSCCLPP_DEVICE_INLINE void get(uint64_t offset, uint64_t originBytes,
                                 uint32_t threadId, uint32_t numThreads) {
    get<Alignment, CopyRemainder>(offset, offset, originBytes, threadId,
                                  numThreads);
  }

  /// Copy data from the local memory (origin) to the remote memory (target)
  /// using packets.
  ///
  /// This function is intended to be collectively called by multiple threads.
  /// Each thread copies a part of data.
  ///
  /// @tparam PacketType The packet type. It should be either LL16Packet or
  /// LL8Packet.
  /// @param targetOffset The offset in bytes of the remote address.
  /// @param originOffset The offset in bytes of the local address.
  /// @param originBytes Bytes of the origin to be copied.
  /// @param threadId The index of the current thread among all threads running
  /// this function. This is different from the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  /// @param flag The flag to write.
  ///
  template <typename PacketType = LL16Packet>
  MSCCLPP_DEVICE_INLINE void putPackets(uint64_t targetOffset,
                                        uint64_t originOffset,
                                        uint64_t originBytes, uint32_t threadId,
                                        uint32_t numThreads, uint32_t flag) {
    static_assert(std::is_same<PacketType, LL16Packet>::value ||
                      std::is_same<PacketType, LL8Packet>::value,
                  "Unsupported packet type");
    copyToPackets<PacketType>(reinterpret_cast<char*>(dst_) + targetOffset,
                              reinterpret_cast<char*>(src_) + originOffset,
                              originBytes, threadId, numThreads, flag);
  }

  /// Wrapper of putPackets() with the same offset for target and origin.
  template <typename PacketType = LL16Packet>
  MSCCLPP_DEVICE_INLINE void putPackets(uint64_t offset, uint64_t originBytes,
                                        uint32_t threadId, uint32_t numThreads,
                                        uint32_t flag) {
    putPackets<PacketType>(offset, offset, originBytes, threadId, numThreads,
                           flag);
  }

  /// Retrieve data from a packet in the local packet buffer.
  ///
  /// @tparam PacketType The packet type. It should be either LL16Packet or
  /// LL8Packet.
  /// @param index The index of the packet to be read. The offset in bytes is
  /// calculated as index * sizeof(PacketType).
  /// @param flag The flag to read.
  /// @param maxSpinCount The maximum number of spins before asserting. Never
  /// assert if negative.
  /// @return The value read from the packet. The type of the value depends on
  /// the packet type.
  ///
  template <typename PacketType = LL16Packet>
  MSCCLPP_DEVICE_INLINE auto unpackPacket(uint64_t index, uint32_t flag,
                                          int64_t maxSpinCount = -1) {
    MSCCLPP_ASSERT_DEVICE(packetBuffer_ != nullptr, "Packet buffer is null");
    return reinterpret_cast<PacketType*>(packetBuffer_)[index].read(
        flag, maxSpinCount);
  }

  /// Retrieve data from packets in the local packet buffer (target) and write
  /// to the local memory (origin).
  ///
  /// This function is intended to be collectively called by multiple threads.
  /// Each thread copies a part of data.
  ///
  /// @tparam PacketType The packet type. It should be either LL16Packet or
  /// LL8Packet.
  /// @param targetOffset The offset in bytes of the local packet buffer.
  /// @param originOffset The offset in bytes of the local address.
  /// @param originBytes Bytes of the origin to be copied.
  /// @param threadId The index of the current thread among all threads running
  /// this function. This is different from the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  /// @param flag The flag to write.
  /// @param maxSpinCount The maximum number of spins before asserting. Never
  /// assert if negative.
  ///
  template <typename PacketType = LL16Packet>
  MSCCLPP_DEVICE_INLINE void unpackPackets(uint64_t targetOffset,
                                           uint64_t originOffset,
                                           uint64_t originBytes,
                                           uint32_t threadId,
                                           uint32_t numThreads, uint32_t flag,
                                           int64_t maxSpinCount = -1) {
    static_assert(std::is_same<PacketType, LL16Packet>::value ||
                      std::is_same<PacketType, LL8Packet>::value,
                  "Unsupported packet type");
    MSCCLPP_ASSERT_DEVICE(packetBuffer_ != nullptr, "Packet buffer is null");
    copyFromPackets<PacketType>(
        reinterpret_cast<char*>(src_) + originOffset,
        reinterpret_cast<char*>(packetBuffer_) + targetOffset, originBytes,
        threadId, numThreads, flag, maxSpinCount);
  }

  /// Wrapper of unpackPackets() with the same offset for target and origin.
  template <typename PacketType = LL16Packet>
  MSCCLPP_DEVICE_INLINE void unpackPackets(uint64_t offset,
                                           uint64_t originBytes,
                                           uint32_t threadId,
                                           uint32_t numThreads, uint32_t flag,
                                           int64_t maxSpinCount = -1) {
    unpackPackets<PacketType>(offset, offset, originBytes, threadId, numThreads,
                              flag, maxSpinCount);
  }

  template <typename PacketType = LL16Packet>
  [[deprecated("Use unpackPackets() instead.")]] MSCCLPP_DEVICE_INLINE void
  getPackets(uint64_t targetOffset, uint64_t originOffset, uint64_t originBytes,
             uint32_t threadId, uint32_t numThreads, uint32_t flag) {
    unpackPackets<PacketType>(targetOffset, originOffset, originBytes, threadId,
                              numThreads, flag, 100000000);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

/// @deprecated Use MemoryChannelDeviceHandle instead.
[[deprecated(
    "Use MemoryChannelDeviceHandle "
    "instead.")]] typedef MemoryChannelDeviceHandle SmChannelDeviceHandle;

}  // namespace mscclpp

#endif  // MSCCLPP_MEMORY_CHANNEL_DEVICE_HPP_
