// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_MEMORY_CHANNEL_HPP_
#define MSCCLPP_MEMORY_CHANNEL_HPP_

#include "core.hpp"
#include "memory_channel_device.hpp"
#include "semaphore.hpp"
#include <type_traits>

namespace mscclpp {

/// Memory channel without specifying source/destination memory regions.
struct BaseMemoryChannel {
 protected:
  std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore_;

 public:
  /// Constructor.
  BaseMemoryChannel() = default;

  /// Constructor.
  /// @param semaphore Semaphore used to synchronize the communication.
  BaseMemoryChannel(std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore);

  /// Constructor.
  /// @param semaphore Semaphore used to synchronize the communication.
  BaseMemoryChannel(Semaphore const& semaphore);

  /// Constructor.
  /// @param other Other BaseMemoryChannel to copy from.
  BaseMemoryChannel(BaseMemoryChannel const& other) = default;

  BaseMemoryChannel& operator=(BaseMemoryChannel& other) = default;

  /// Device-side handle for BaseMemoryChannel.
  using DeviceHandle = BaseMemoryChannelDeviceHandle;

  /// Returns the device-side handle.
  /// User should make sure the BaseMemoryChannel is not released when using the
  /// returned handle.
  /// @return The device-side handle.
  DeviceHandle deviceHandle() const;
};

/// Channel for accessing peer memory directly from GPU threads.
struct MemoryChannel : public BaseMemoryChannel {
 private:
  RegisteredMemory dst_;
  RegisteredMemory src_;
  void* packetBuffer_;

 public:
  /// Constructor.
  MemoryChannel() = default;

  /// Constructor.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param dst Registered memory of the destination.
  /// @param src Registered memory of the source.
  /// @param packetBuffer A buffer used to store packets. @p packetBuffer is
  /// optional and if it is nullptr, unpackPacket() and unpackPackets() methods
  /// are not available.
  MemoryChannel(std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore,
                RegisteredMemory dst, RegisteredMemory src,
                void* packetBuffer = nullptr);

  /// Constructor.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param dst Registered memory of the destination.
  /// @param src Registered memory of the source.
  /// @param packetBuffer A buffer used to store packets. @p packetBuffer is
  /// optional and if it is nullptr, unpackPacket() and unpackPackets() methods
  /// are not available.
  MemoryChannel(Semaphore const& semaphore, RegisteredMemory dst,
                RegisteredMemory src, void* packetBuffer = nullptr);

  /// Device-side handle for MemoryChannel.
  using DeviceHandle = MemoryChannelDeviceHandle;

  /// Returns the device-side handle.
  /// User should make sure the MemoryChannel is not released when using the
  /// returned handle.
  /// @return The device-side handle.
  DeviceHandle deviceHandle() const;
};

/// @deprecated Use MemoryChannel instead.
[[deprecated("Use MemoryChannel instead.")]] typedef MemoryChannel SmChannel;

}  // namespace mscclpp

#endif  // MSCCLPP_MEMORY_CHANNEL_HPP_
