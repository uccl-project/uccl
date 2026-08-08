// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NCCL_COMMON_HPP_
#define NCCL_COMMON_HPP_

#include "concurrency_device.hpp"
#include "core.hpp"
#include "env.hpp"
#include "memory_channel.hpp"
#include "port_channel.hpp"
#include "switch_channel.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace mscclpp {

struct AlgorithmCtx {
  int rank = 0;
  int workSize = 0;
  int nRanksPerNode = 0;

  std::vector<RegisteredMemory> registeredMemories;
  std::vector<MemoryChannel> memoryChannels;
  std::vector<SwitchChannel> switchChannels;
  std::vector<PortChannel> portChannels;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections;
  std::shared_ptr<DeviceHandle<MemoryChannel>> memoryChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<SwitchChannel>> switchChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<PortChannel>> portChannelDeviceHandles;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> hostSemaphores;

  std::unordered_map<std::string, std::shared_ptr<void>> extras;
};

}  // namespace mscclpp

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#define __syncwarp() __builtin_amdgcn_wave_barrier()
#else
#define WARP_SIZE 32
#endif

constexpr int NUM_NVLS_CONNECTION = 8;
constexpr int NUM_SEMAPHORES = 64;

constexpr int MAX_NRANKS_PER_NODE = 8;

constexpr int SCRATCH_SIZE =
    2 * 1024 * 1024 *
    70;  // double buffer * 35 thread-blocks * 8 ranks * 256KB = 70MB
static bool mscclppDisableChannelCache = mscclpp::env()->disableChannelCache;

__device__ mscclpp::DeviceSyncer deviceSyncer;
__device__ mscclpp::DeviceSemaphore deviceSemaphore[NUM_SEMAPHORES];

std::vector<mscclpp::RegisteredMemory> setupRemoteMemories(
    std::shared_ptr<mscclpp::Communicator> comm, int rank,
    mscclpp::RegisteredMemory localMemory);

std::vector<mscclpp::MemoryChannel> setupMemoryChannels(
    std::vector<mscclpp::Connection> const& connections,
    std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> const&
        memorySemaphores,
    std::vector<mscclpp::RegisteredMemory> const& remoteMemories,
    mscclpp::RegisteredMemory localMemory, int nChannelsPerConnection);

std::vector<mscclpp::Connection> setupConnections(
    std::shared_ptr<mscclpp::Communicator> comm);

std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>
setupMemorySemaphores(std::shared_ptr<mscclpp::Communicator> comm,
                      std::vector<mscclpp::Connection> const& connections,
                      int nChannelsPerConnection);

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>>
setupMemoryChannelDeviceHandles(
    std::vector<mscclpp::MemoryChannel> const& memoryChannels);

std::vector<std::shared_ptr<mscclpp::NvlsConnection>> setupNvlsConnections(
    std::shared_ptr<mscclpp::Communicator> comm, size_t size,
    int numConnections);

std::vector<mscclpp::SwitchChannel> setupNvlsChannels(
    std::vector<std::shared_ptr<mscclpp::NvlsConnection>> conns, void* buffer,
    size_t bufferSize, int nSwitchChannels);

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>>
setupNvlsChannelDeviceHandles(
    std::vector<mscclpp::SwitchChannel> const& nvlsChannels);

std::vector<mscclpp::BaseMemoryChannel> setupBaseMemoryChannels(
    std::vector<mscclpp::Connection> const& connections,
    std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> const&
        memorySemaphores,
    int nChannelsPerConnection);

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>>
setupBaseMemoryChannelDeviceHandles(
    std::vector<mscclpp::BaseMemoryChannel> const& baseMemoryChannels);

#include "algorithm.hpp"
#include "gpu_data_types.hpp"
#include "nccl.h"

namespace mscclpp {
namespace nccl_port {

inline CommResult ncclResultToCommResult(ncclResult_t r) {
  if (r == ncclSuccess) {
    return CommResult::CommSuccess;
  }
  if (r == ncclInvalidArgument) {
    return CommResult::CommInvalidArgument;
  }
  return CommResult::CommInternalError;
}

inline ncclRedOp_t reduceOpToNccl(ReduceOp op) {
  switch (op) {
    case ReduceOp::SUM:
      return ncclSum;
    case ReduceOp::MIN:
      return ncclMin;
    default:
      return ncclNumOps;
  }
}

inline std::unordered_map<std::string, std::shared_ptr<void>>
legacyExtrasWithOp(ReduceOp op) {
  ncclRedOp_t nrop = reduceOpToNccl(op);
  auto box = std::make_shared<ncclRedOp_t>(nrop);
  std::unordered_map<std::string, std::shared_ptr<void>> ex;
  ex["op"] = std::shared_ptr<void>(box.get(), [box](void*) {});
  return ex;
}

inline std::unordered_map<std::string, std::shared_ptr<void>>
legacyExtrasWithRoot(int const& root) {
  auto box = std::make_shared<int>(root);
  std::unordered_map<std::string, std::shared_ptr<void>> ex;
  ex["root"] = std::shared_ptr<void>(box.get(), [box](void*) {});
  return ex;
}

}  // namespace nccl_port
}  // namespace mscclpp

#endif  // NCCL_COMMON_HPP_
