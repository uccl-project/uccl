// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "collective_utils.hpp"
#include "algorithm.hpp"
#include "core.hpp"
#include "memory_channel.hpp"
#include "switch_channel.hpp"
#include <algorithm>

namespace mscclpp {
namespace collective {
std::vector<mscclpp::RegisteredMemory> setupRemoteMemories(
    std::shared_ptr<mscclpp::Communicator> comm, int rank,
    mscclpp::RegisteredMemory localMemory) {
  std::vector<mscclpp::RegisteredMemory> remoteMemories;
  std::vector<std::shared_future<mscclpp::RegisteredMemory>>
      remoteRegMemoryFutures;
  for (int i = 0; i < comm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    remoteRegMemoryFutures.push_back(comm->recvMemory(i));
    comm->sendMemory(localMemory, i);
  }
  std::transform(remoteRegMemoryFutures.begin(), remoteRegMemoryFutures.end(),
                 std::back_inserter(remoteMemories),
                 [](auto const& future) { return future.get(); });
  return remoteMemories;
}

std::vector<mscclpp::MemoryChannel> setupMemoryChannels(
    std::vector<mscclpp::Connection> const& connections,
    std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> const&
        memorySemaphores,
    std::vector<mscclpp::RegisteredMemory> const& remoteMemories,
    mscclpp::RegisteredMemory localMemory, int nChannelsPerConnection) {
  std::vector<mscclpp::MemoryChannel> channels;
  size_t nConnections = connections.size();
  for (int idx = 0; idx < nChannelsPerConnection; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (connections[cid].transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(memorySemaphores[idx * nConnections + cid],
                              remoteMemories[cid], localMemory, nullptr);
      }
    }
  }
  return channels;
}

std::vector<mscclpp::Connection> setupConnections(
    std::shared_ptr<mscclpp::Communicator> comm) {
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures;
  for (int i = 0; i < comm->bootstrap()->getNranks(); i++) {
    if (i == comm->bootstrap()->getRank()) continue;
    connectionFutures.push_back(comm->connect(mscclpp::Transport::CudaIpc, i));
  }
  std::vector<mscclpp::Connection> connections;
  std::transform(connectionFutures.begin(), connectionFutures.end(),
                 std::back_inserter(connections),
                 [](auto const& future) { return future.get(); });
  return connections;
}

std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>
setupMemorySemaphores(std::shared_ptr<mscclpp::Communicator> comm,
                      std::vector<mscclpp::Connection> const& connections,
                      int nChannelsPerConnection) {
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>
      memorySemaphores;
  for (int idx = 0; idx < nChannelsPerConnection; ++idx) {
    for (size_t cid = 0; cid < connections.size(); ++cid) {
      if (connections[cid].transport() == mscclpp::Transport::CudaIpc) {
        memorySemaphores.emplace_back(
            std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(
                *(comm), connections[cid]));
      }
    }
  }
  return memorySemaphores;
}

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>>
setupMemoryChannelDeviceHandles(
    std::vector<mscclpp::MemoryChannel> const& memoryChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::MemoryChannel>>
      memoryChannelDeviceHandles;
  std::transform(memoryChannels.begin(), memoryChannels.end(),
                 std::back_inserter(memoryChannelDeviceHandles),
                 [](mscclpp::MemoryChannel const& memoryChannel) {
                   return mscclpp::deviceHandle(memoryChannel);
                 });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>> ptr =
      mscclpp::detail::gpuCallocShared<
          mscclpp::DeviceHandle<mscclpp::MemoryChannel>>(
          memoryChannelDeviceHandles.size());
  mscclpp::gpuMemcpy<mscclpp::DeviceHandle<mscclpp::MemoryChannel>>(
      ptr.get(), memoryChannelDeviceHandles.data(),
      memoryChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

std::vector<std::shared_ptr<mscclpp::NvlsConnection>> setupNvlsConnections(
    std::shared_ptr<mscclpp::Communicator> comm, size_t size,
    int numConnections) {
  // for nvls connection
  std::vector<std::shared_ptr<mscclpp::NvlsConnection>> nvlsConnections;
  int nRanks = comm->bootstrap()->getNranks();
  std::vector<int> ranks;
  for (int i = 0; i < nRanks; i++) {
    ranks.push_back(i);
  }
  for (int i = 0; i < numConnections; i++) {
    std::shared_ptr<mscclpp::NvlsConnection> nvlsConnection =
        mscclpp::connectNvlsCollective(comm, ranks, size);
    nvlsConnections.push_back(nvlsConnection);
  }
  return nvlsConnections;
}

std::vector<mscclpp::SwitchChannel> setupNvlsChannels(
    std::vector<std::shared_ptr<mscclpp::NvlsConnection>> conns, void* buffer,
    size_t bufferSize, int nSwitchChannels) {
  std::vector<mscclpp::SwitchChannel> channels;

  for (int idx = 0; idx < nSwitchChannels; ++idx) {
    std::shared_ptr<mscclpp::NvlsConnection> nvlsConnection = conns[idx];
    mscclpp::SwitchChannel switchChannel =
        nvlsConnection->bindAllocatedMemory((CUdeviceptr)buffer, bufferSize);
    channels.push_back(switchChannel);
  }
  return channels;
}

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>>
setupNvlsChannelDeviceHandles(
    std::vector<mscclpp::SwitchChannel> const& nvlsChannels) {
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> ptr =
      mscclpp::detail::gpuCallocShared<
          mscclpp::DeviceHandle<mscclpp::SwitchChannel>>(nvlsChannels.size());
  std::vector<mscclpp::DeviceHandle<mscclpp::SwitchChannel>>
      nvlsChannelDeviceHandles;
  std::transform(nvlsChannels.begin(), nvlsChannels.end(),
                 std::back_inserter(nvlsChannelDeviceHandles),
                 [](mscclpp::SwitchChannel const& nvlsChannel) {
                   return mscclpp::deviceHandle(nvlsChannel);
                 });
  mscclpp::gpuMemcpy<mscclpp::DeviceHandle<mscclpp::SwitchChannel>>(
      ptr.get(), nvlsChannelDeviceHandles.data(),
      nvlsChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

std::vector<mscclpp::BaseMemoryChannel> setupBaseMemoryChannels(
    std::vector<mscclpp::Connection> const& connections,
    std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> const&
        memorySemaphores,
    int nChannelsPerConnection) {
  std::vector<mscclpp::BaseMemoryChannel> channels;
  size_t nConnections = connections.size();
  for (int idx = 0; idx < nChannelsPerConnection; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (connections[cid].transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(memorySemaphores[idx * nConnections + cid]);
      }
    }
  }
  return channels;
}

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>>
setupBaseMemoryChannelDeviceHandles(
    std::vector<mscclpp::BaseMemoryChannel> const& baseMemoryChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>>
      memoryChannelDeviceHandles;
  std::transform(baseMemoryChannels.begin(), baseMemoryChannels.end(),
                 std::back_inserter(memoryChannelDeviceHandles),
                 [](mscclpp::BaseMemoryChannel const& memoryChannel) {
                   return mscclpp::deviceHandle(memoryChannel);
                 });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>> ptr =
      mscclpp::detail::gpuCallocShared<
          mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>>(
          memoryChannelDeviceHandles.size());
  mscclpp::gpuMemcpy<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>>(
      ptr.get(), memoryChannelDeviceHandles.data(),
      memoryChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

}  // namespace collective

}  // namespace mscclpp