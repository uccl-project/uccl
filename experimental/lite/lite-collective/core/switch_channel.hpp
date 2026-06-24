// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SWITCH_CHANNEL_HPP_
#define MSCCLPP_SWITCH_CHANNEL_HPP_

#include "gpu_utils.hpp"
#include "switch_channel_device.hpp"

namespace mscclpp {

class NvlsConnection;

struct SwitchChannel {
 private:
  void* devicePtr_;
  std::shared_ptr<void> mcPtr_;
  size_t bufferSize_;

 public:
  using DeviceHandle = SwitchChannelDeviceHandle;
  SwitchChannel(void* devicePtr, std::shared_ptr<void> mcPtr, size_t bufferSize)
      : devicePtr_(devicePtr), mcPtr_(mcPtr), bufferSize_(bufferSize) {}
  DeviceHandle deviceHandle() const;
  void* getDevicePtr();

  friend class NvlsConnection;
};

class NvlsConnection {
 public:
  NvlsConnection(size_t bufferSize, int numDevices);
  NvlsConnection(std::vector<char> const& data);
  NvlsConnection() = delete;
  std::vector<char> serialize();

  /// Bind the memory allocated via mscclpp::GpuBuffer to the multicast handle.
  /// The behavior is undefined if the devicePtr is not allocated by
  /// mscclpp::GpuBuffer.
  /// @param devicePtr The device pointer returned by
  /// `mscclpp::GpuBuffer::data()`.
  /// @param size The bytes of the memory to bind to the multicast handle.
  /// @return SwitchChannel with devicePtr, mcPtr and bufferSize
  SwitchChannel bindAllocatedMemory(CUdeviceptr devicePtr, size_t size);

 private:
  class Impl;
  std::shared_ptr<Impl> pimpl_;
};

class Communicator;

/// Connect to NVLS on setup.
///
/// This function used to connect to NVLS on setup. NVLS collective using
/// multicast operations to send/recv data. Here we need to put all involved
/// ranks into the collective group.
///
/// @param comm The communicator.
/// @param allRanks The ranks of all processes involved in the collective.
/// @param config The configuration for the local endpoint.
/// @return std::shared_ptr<NvlsConnection> A shared pointer to the NVLS
/// connection.
std::shared_ptr<NvlsConnection> connectNvlsCollective(
    std::shared_ptr<Communicator> comm, std::vector<int> allRanks,
    size_t bufferSize);

}  // namespace mscclpp

#endif  // MSCCLPP_SWITCH_CHANNEL_HPP_
