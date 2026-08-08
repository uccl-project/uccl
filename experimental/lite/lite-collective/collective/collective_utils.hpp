// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_COLLECTIVE_UTILS_HPP_
#define MSCCLPP_EXT_COLLECTIVE_UTILS_HPP_

#include "concurrency_device.hpp"
#include "core.hpp"
#include "env.hpp"
#include "memory_channel.hpp"
#include "port_channel.hpp"
#include "semaphore.hpp"
#include "switch_channel.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#define __syncwarp() __builtin_amdgcn_wave_barrier()
#else
#define WARP_SIZE 32
#endif

namespace mscclpp {

namespace collective {
constexpr int NUM_NVLS_CONNECTION = 8;
constexpr int NUM_SEMAPHORES = 64;

constexpr int MAX_NRANKS_PER_NODE = 8;

constexpr int SCRATCH_SIZE =
    2 * 1024 * 1024 *
    70;  // double buffer * 35 thread-blocks * 8 ranks * 256KB = 70MB

std::vector<RegisteredMemory> setupRemoteMemories(
    std::shared_ptr<Communicator> comm, int rank, RegisteredMemory localMemory);

std::vector<MemoryChannel> setupMemoryChannels(
    std::vector<Connection> const& connections,
    std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> const&
        memorySemaphores,
    std::vector<RegisteredMemory> const& remoteMemories,
    RegisteredMemory localMemory, int nChannelsPerConnection);

std::vector<Connection> setupConnections(std::shared_ptr<Communicator> comm);
std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>>
setupMemorySemaphores(std::shared_ptr<Communicator> comm,
                      std::vector<Connection> const& connections,
                      int nChannelsPerConnection);

std::shared_ptr<DeviceHandle<MemoryChannel>> setupMemoryChannelDeviceHandles(
    std::vector<MemoryChannel> const& memoryChannels);

std::vector<std::shared_ptr<NvlsConnection>> setupNvlsConnections(
    std::shared_ptr<Communicator> comm, size_t size, int numConnections);

std::vector<SwitchChannel> setupNvlsChannels(
    std::vector<std::shared_ptr<NvlsConnection>> conns, void* buffer,
    size_t bufferSize, int nSwitchChannels);

std::shared_ptr<DeviceHandle<SwitchChannel>> setupNvlsChannelDeviceHandles(
    std::vector<SwitchChannel> const& nvlsChannels);

std::vector<BaseMemoryChannel> setupBaseMemoryChannels(
    std::vector<Connection> const& connections,
    std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> const&
        memorySemaphores,
    int nChannelsPerConnection);

std::shared_ptr<DeviceHandle<BaseMemoryChannel>>
setupBaseMemoryChannelDeviceHandles(
    std::vector<BaseMemoryChannel> const& baseMemoryChannels);

/// Context holding resources for algorithm execution.
///
/// This struct contains all the channels, semaphores, and memory handles
/// needed for executing a native algorithm. It is created once per unique
/// buffer configuration and cached for reuse.
class AlgorithmCtx {
 public:
  int rank;
  int workSize;
  int nRanksPerNode;

  std::vector<RegisteredMemory> registeredMemories;
  std::vector<MemoryChannel> memoryChannels;
  std::vector<SwitchChannel> switchChannels;
  std::vector<PortChannel> portChannels;
  std::shared_ptr<DeviceHandle<MemoryChannel>> memoryChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<SwitchChannel>> switchChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<PortChannel>> portChannelDeviceHandles;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> hostSemaphores;
  std::shared_ptr<void*> remoteMemoryHandles;
  std::unordered_map<std::string, std::shared_ptr<void>> extras;
};

}  // namespace collective
}  // namespace mscclpp
#endif  // MSCCLPP_EXT_COLLECTIVE_UTILS_HPP_