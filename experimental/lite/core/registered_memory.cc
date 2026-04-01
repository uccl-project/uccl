// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "registered_memory.hpp"
#include "api.h"
#include "context.hpp"
#include "gpu_utils.hpp"
#include "logger.hpp"
#include "serialization.hpp"
#include "unix_socket.hpp"
#include "utils_internal.hpp"
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <sys/syscall.h>
#include <unistd.h>

#define MSCCLPP_CULOG_WARN(cmd)                                         \
  do {                                                                  \
    CUresult err = cmd;                                                 \
    if (err != CUDA_SUCCESS) {                                          \
      char const* errStr;                                               \
      if (cuGetErrorString(err, &errStr) != CUDA_SUCCESS) {             \
        errStr = "failed to get error string";                          \
      }                                                                 \
      WARN(mscclpp::GPU, "Call to " #cmd " failed, error is ", errStr); \
    }                                                                   \
  } while (false)

namespace mscclpp {

RegisteredMemory::Impl::Impl(void* data, size_t size, TransportFlags transports,
                             Context::Impl& contextImpl)
    : data(data),
      originalDataPtr(data),
      size(size),
      hostHash(getHostHash()),
      pidHash(getPidHash()),
      transports(transports) {
  if (transports.has(Transport::CudaIpc)) {
    CudaDeviceGuard deviceGuard(detail::gpuIdFromAddress(data));

    localGpuIpcMemHandle =
        GpuIpcMemHandle::create(reinterpret_cast<CUdeviceptr>(data));
    TransportInfo transportInfo;
    transportInfo.transport = Transport::CudaIpc;
    transportInfo.gpuIpcMemHandle = *localGpuIpcMemHandle;
    this->transportInfos.emplace_back(transportInfo);
  }
  if ((transports & AllIBTransports).any()) {
    auto addIb = [&](Transport ibTransport) {
      TransportInfo transportInfo;
      transportInfo.transport = ibTransport;
      this->ibMrMap[ibTransport] =
          contextImpl.getIbContext(ibTransport)->registerMr(data, size);
      transportInfo.ibMr = this->ibMrMap[ibTransport].get();
      transportInfo.ibLocal = true;
      transportInfo.ibMrInfo = this->ibMrMap[ibTransport]->getInfo();
      this->transportInfos.push_back(transportInfo);
      INFO(NET, "IB mr for address ", data, " with size ", size,
           " is registered");
    };
    if (transports.has(Transport::IB0)) addIb(Transport::IB0);
    if (transports.has(Transport::IB1)) addIb(Transport::IB1);
    if (transports.has(Transport::IB2)) addIb(Transport::IB2);
    if (transports.has(Transport::IB3)) addIb(Transport::IB3);
    if (transports.has(Transport::IB4)) addIb(Transport::IB4);
    if (transports.has(Transport::IB5)) addIb(Transport::IB5);
    if (transports.has(Transport::IB6)) addIb(Transport::IB6);
    if (transports.has(Transport::IB7)) addIb(Transport::IB7);
  }
}

MSCCLPP_API_CPP RegisteredMemory::RegisteredMemory(std::shared_ptr<Impl> pimpl)
    : pimpl_(pimpl) {}

MSCCLPP_API_CPP RegisteredMemory::~RegisteredMemory() = default;

MSCCLPP_API_CPP void* RegisteredMemory::data() const { return pimpl_->data; }

MSCCLPP_API_CPP void* RegisteredMemory::originalDataPtr() const {
  return pimpl_->originalDataPtr;
}

MSCCLPP_API_CPP size_t RegisteredMemory::size() const { return pimpl_->size; }

MSCCLPP_API_CPP TransportFlags RegisteredMemory::transports() const {
  return pimpl_->transports;
}

MSCCLPP_API_CPP std::vector<char> RegisteredMemory::serialize() const {
  std::vector<char> result;
  detail::serialize(result, pimpl_->originalDataPtr);
  detail::serialize(result, pimpl_->size);
  detail::serialize(result, pimpl_->hostHash);
  detail::serialize(result, pimpl_->pidHash);
  detail::serialize(result, pimpl_->transports);
  if (pimpl_->transportInfos.size() >
      static_cast<size_t>(std::numeric_limits<int8_t>::max())) {
    throw Error("Too many transport info entries", ErrorCode::InternalError);
  }
  int8_t transportCount = pimpl_->transportInfos.size();
  detail::serialize(result, transportCount);
  for (auto& entry : pimpl_->transportInfos) {
    detail::serialize(result, entry.transport);
    if (entry.transport == Transport::CudaIpc) {
      detail::serialize(result, entry.gpuIpcMemHandle);
    } else if (AllIBTransports.has(entry.transport)) {
      detail::serialize(result, entry.ibMrInfo);
    } else {
      throw Error("Unknown transport", ErrorCode::InternalError);
    }
  }
  return result;
}

MSCCLPP_API_CPP RegisteredMemory
RegisteredMemory::deserialize(std::vector<char> const& data) {
  return RegisteredMemory(std::make_shared<Impl>(data));
}

RegisteredMemory::Impl::Impl(const std::vector<char>::const_iterator& begin,
                             const std::vector<char>::const_iterator& end) {
  auto it = begin;
  it = detail::deserialize(it, this->originalDataPtr);
  it = detail::deserialize(it, this->size);
  it = detail::deserialize(it, this->hostHash);
  it = detail::deserialize(it, this->pidHash);
  it = detail::deserialize(it, this->transports);
  int8_t transportCount;
  it = detail::deserialize(it, transportCount);
  for (int i = 0; i < transportCount; ++i) {
    TransportInfo transportInfo;
    it = detail::deserialize(it, transportInfo.transport);
    if (transportInfo.transport == Transport::CudaIpc) {
      it = detail::deserialize(it, transportInfo.gpuIpcMemHandle);
    } else if (AllIBTransports.has(transportInfo.transport)) {
      it = detail::deserialize(it, transportInfo.ibMrInfo);
      transportInfo.ibLocal = false;
    } else {
      throw Error("Unknown transport", ErrorCode::InternalError);
    }
    this->transportInfos.emplace_back(transportInfo);
  }
  if (it != end) {
    throw Error("Serialization failed", ErrorCode::InternalError);
  }

  // Next decide how to set this->data
  this->data = nullptr;
  if (getHostHash() == this->hostHash && getPidHash() == this->pidHash) {
    // The memory is local to the process, so originalDataPtr is valid as is
    this->data = this->originalDataPtr;
    if (transports.has(Transport::CudaIpc)) {
      auto entry = getTransportInfo(Transport::CudaIpc);
      if ((entry.gpuIpcMemHandle.typeFlags &
           GpuIpcMemHandle::Type::RuntimeIpc) == 0) {
        // Query which device owns this memory
        int gpuId = detail::gpuIdFromAddress(this->data);
        int currentDevice = -1;
        MSCCLPP_CUDATHROW(cudaGetDevice(&currentDevice));

        // Only set access if we're on a different device than where memory was
        // allocated
        if (gpuId != currentDevice) {
          detail::setReadWriteMemoryAccess(this->data,
                                           entry.gpuIpcMemHandle.baseSize);
        }
      }
    }
  } else if (transports.has(Transport::CudaIpc)) {
    auto entry = getTransportInfo(Transport::CudaIpc);
    auto gpuIpcMem = GpuIpcMem::create(entry.gpuIpcMemHandle);
    // Create a memory map for the remote GPU memory. The memory map will keep
    // the GpuIpcMem instance alive.
    this->remoteMemMap = gpuIpcMem->map();
    this->data = this->remoteMemMap.get();
  }
  if (this->data != nullptr) {
    INFO(GPU, "Opened CUDA IPC handle at pointer ", this->data);
  }
}

RegisteredMemory::Impl::Impl(std::vector<char> const& serialization)
    : Impl(serialization.begin(), serialization.end()) {}

TransportInfo const& RegisteredMemory::Impl::getTransportInfo(
    Transport transport) const {
  for (auto& entry : transportInfos) {
    if (entry.transport == transport) {
      return entry;
    }
  }
  throw Error("Transport data not found", ErrorCode::InternalError);
}

MSCCLPP_API_CPP void RegisteredMemory::getIbMrInfo(
    Transport transport, IbMr const** outMr, IbMrInfo* outMrInfo) const {
  auto const& info = pimpl_->getTransportInfo(transport);
  if (outMr) {
    *outMr = info.ibLocal ? info.ibMr : nullptr;
  }
  if (outMrInfo) {
    *outMrInfo = info.ibMrInfo;
  }
}

}  // namespace mscclpp
