// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "semaphore.hpp"
#include "api.h"
#include "atomic.hpp"
#include "connection.hpp"
#include "context.hpp"
#include "gpu_utils.hpp"
#include "registered_memory.hpp"
#include "serialization.hpp"

namespace mscclpp {

struct SemaphoreStub::Impl {
  Impl(Connection const& connection);

  Impl(RegisteredMemory const& idMemory, Device const& device);

  Impl(std::vector<char> const& data);

  std::shared_ptr<uint64_t> gpuCallocToken(std::shared_ptr<Context> context);

  Connection connection_;
  std::shared_ptr<uint64_t> token_;
  RegisteredMemory idMemory_;
  Device device_;
};

std::shared_ptr<uint64_t> SemaphoreStub::Impl::gpuCallocToken(
    [[maybe_unused]] std::shared_ptr<Context> context) {
#if (CUDA_NVLS_API_AVAILABLE)
  if (isNvlsSupported()) {
    return context->pimpl_->getToken();
  }
#endif  // CUDA_NVLS_API_AVAILABLE
#if defined(MSCCLPP_USE_ROCM)
  return detail::gpuCallocUncachedShared<uint64_t>();
#else   // !defined(MSCCLPP_USE_ROCM)
  return detail::gpuCallocShared<uint64_t>();
#endif  // !defined(MSCCLPP_USE_ROCM)
}

SemaphoreStub::Impl::Impl(Connection const& connection)
    : connection_(connection) {
  // Allocate a semaphore ID on the local device
  Device const& localDevice = connection_.localDevice();
  if (localDevice.type == DeviceType::CPU) {
    token_ = std::make_shared<uint64_t>(0);
  } else if (localDevice.type == DeviceType::GPU) {
    if (localDevice.id < 0) {
      throw Error("Local GPU ID is not provided", ErrorCode::InvalidUsage);
    }
    CudaDeviceGuard deviceGuard(localDevice.id);
    token_ = gpuCallocToken(connection_.context());
  } else {
    throw Error("Unsupported local device type", ErrorCode::InvalidUsage);
  }
  idMemory_ = std::move(connection_.context()->registerMemory(
      token_.get(), sizeof(uint64_t), connection_.transport()));
}

SemaphoreStub::Impl::Impl(RegisteredMemory const& idMemory,
                          Device const& device)
    : idMemory_(idMemory), device_(device) {}

SemaphoreStub::SemaphoreStub(std::shared_ptr<Impl> pimpl)
    : pimpl_(std::move(pimpl)) {}

MSCCLPP_API_CPP SemaphoreStub::SemaphoreStub(Connection const& connection)
    : pimpl_(std::make_shared<Impl>(connection)) {}

MSCCLPP_API_CPP std::vector<char> SemaphoreStub::serialize() const {
  auto data = pimpl_->idMemory_.serialize();
  detail::serialize(data, pimpl_->device_);
  return data;
}

MSCCLPP_API_CPP SemaphoreStub
SemaphoreStub::deserialize(std::vector<char> const& data) {
  Device device;
  auto memEnd = data.end() - sizeof(device);
  RegisteredMemory idMemory(
      std::make_shared<RegisteredMemory::Impl>(data.begin(), memEnd));
  auto it = detail::deserialize(memEnd, device);
  if (it != data.end()) {
    throw Error("SemaphoreStub deserialize failed", ErrorCode::InvalidUsage);
  }
  return SemaphoreStub(std::make_shared<Impl>(std::move(idMemory), device));
}

MSCCLPP_API_CPP RegisteredMemory const& SemaphoreStub::memory() const {
  return pimpl_->idMemory_;
}

struct Semaphore::Impl {
  Impl(SemaphoreStub const& localStub, RegisteredMemory const& remoteStubMemory)
      : localStub_(localStub), remoteStubMemory_(remoteStubMemory) {}

  SemaphoreStub localStub_;
  RegisteredMemory remoteStubMemory_;
};

Semaphore::Semaphore(SemaphoreStub const& localStub,
                     SemaphoreStub const& remoteStub) {
  auto remoteMemImpl = remoteStub.memory().pimpl_;
  if (remoteMemImpl->hostHash == getHostHash() &&
      remoteMemImpl->pidHash == getPidHash()) {
    pimpl_ = std::make_shared<Impl>(
        localStub,
        RegisteredMemory::deserialize(remoteStub.memory().serialize()));
  } else {
    pimpl_ = std::make_shared<Impl>(localStub, remoteStub.memory());
  }
}

MSCCLPP_API_CPP Connection& Semaphore::connection() {
  return pimpl_->localStub_.pimpl_->connection_;
}

MSCCLPP_API_CPP RegisteredMemory const& Semaphore::localMemory() const {
  return pimpl_->localStub_.memory();
}

MSCCLPP_API_CPP RegisteredMemory const& Semaphore::remoteMemory() const {
  return pimpl_->remoteStubMemory_;
}

static Semaphore buildSemaphoreFromConnection(Communicator& communicator,
                                              Connection const& connection) {
  auto semaphoreFuture = communicator.buildSemaphore(
      connection, communicator.remoteRankOf(connection),
      communicator.tagOf(connection));
  return semaphoreFuture.get();
}

MSCCLPP_API_CPP Host2DeviceSemaphore::Host2DeviceSemaphore(
    Semaphore const& semaphore)
    : semaphore_(semaphore),
      expectedInboundToken_(detail::gpuCallocUnique<uint64_t>()),
      outboundToken_(std::make_unique<uint64_t>()) {
  if (connection().localDevice().type != DeviceType::GPU) {
    throw Error(
        "Local endpoint device type of Host2DeviceSemaphore should be GPU",
        ErrorCode::InvalidUsage);
  }
  BaseConnection::getImpl(connection())
      ->setRemoteUpdateDstAddr(
          reinterpret_cast<uint64_t>(semaphore_.localMemory().data()));
}

MSCCLPP_API_CPP Host2DeviceSemaphore::Host2DeviceSemaphore(
    Communicator& communicator, Connection const& connection)
    : Host2DeviceSemaphore(
          buildSemaphoreFromConnection(communicator, connection)) {}

MSCCLPP_API_CPP Connection& Host2DeviceSemaphore::connection() {
  return semaphore_.connection();
}

MSCCLPP_API_CPP void Host2DeviceSemaphore::signal() {
  connection().updateAndSync(semaphore_.remoteMemory(), 0, outboundToken_.get(),
                             *outboundToken_ + 1);
}

MSCCLPP_API_CPP Host2DeviceSemaphore::DeviceHandle
Host2DeviceSemaphore::deviceHandle() const {
  Host2DeviceSemaphore::DeviceHandle device;
  device.inboundToken =
      reinterpret_cast<uint64_t*>(semaphore_.localMemory().data());
  device.expectedInboundToken = expectedInboundToken_.get();
  return device;
}

MSCCLPP_API_CPP Host2HostSemaphore::Host2HostSemaphore(
    Semaphore const& semaphore)
    : semaphore_(semaphore),
      expectedInboundToken_(std::make_unique<uint64_t>()),
      outboundToken_(std::make_unique<uint64_t>()) {
  if (connection().transport() == Transport::CudaIpc) {
    throw Error("Host2HostSemaphore cannot be used with CudaIpc transport",
                ErrorCode::InvalidUsage);
  }
  if (connection().localDevice().type != DeviceType::CPU) {
    throw Error(
        "Local endpoint device type of Host2HostSemaphore should be CPU",
        ErrorCode::InvalidUsage);
  }
  BaseConnection::getImpl(connection())
      ->setRemoteUpdateDstAddr(
          reinterpret_cast<uint64_t>(semaphore_.localMemory().data()));
}

MSCCLPP_API_CPP Host2HostSemaphore::Host2HostSemaphore(
    Communicator& communicator, Connection const& connection)
    : Host2HostSemaphore(
          buildSemaphoreFromConnection(communicator, connection)) {}

MSCCLPP_API_CPP Connection& Host2HostSemaphore::connection() {
  return semaphore_.connection();
}

MSCCLPP_API_CPP void Host2HostSemaphore::signal() {
  connection().updateAndSync(semaphore_.remoteMemory(), 0, outboundToken_.get(),
                             *outboundToken_ + 1);
}

MSCCLPP_API_CPP bool Host2HostSemaphore::poll() {
  bool signaled =
      (atomicLoad(reinterpret_cast<uint64_t*>(semaphore_.localMemory().data()),
                  memoryOrderAcquire) > (*expectedInboundToken_));
  if (signaled) (*expectedInboundToken_) += 1;
  return signaled;
}

MSCCLPP_API_CPP void Host2HostSemaphore::wait(int64_t maxSpinCount) {
  (*expectedInboundToken_) += 1;
  int64_t spinCount = 0;
  while (
      atomicLoad(reinterpret_cast<uint64_t*>(semaphore_.localMemory().data()),
                 memoryOrderAcquire) < (*expectedInboundToken_)) {
    if (maxSpinCount >= 0 && spinCount++ == maxSpinCount) {
      throw Error("Host2HostSemaphore::wait timed out", ErrorCode::Timeout);
    }
  }
}

MSCCLPP_API_CPP MemoryDevice2DeviceSemaphore::MemoryDevice2DeviceSemaphore(
    Semaphore const& semaphore)
    : semaphore_(semaphore),
      expectedInboundToken_(detail::gpuCallocUnique<uint64_t>()),
      outboundToken_(detail::gpuCallocUnique<uint64_t>()) {
  if (connection().localDevice().type != DeviceType::GPU) {
    throw Error(
        "Local endpoint device type of MemoryDevice2DeviceSemaphore should be "
        "GPU",
        ErrorCode::InvalidUsage);
  }
}

MSCCLPP_API_CPP MemoryDevice2DeviceSemaphore::MemoryDevice2DeviceSemaphore(
    Communicator& communicator, Connection const& connection)
    : MemoryDevice2DeviceSemaphore(
          buildSemaphoreFromConnection(communicator, connection)) {}

MSCCLPP_API_CPP Connection& MemoryDevice2DeviceSemaphore::connection() {
  return semaphore_.connection();
}

MSCCLPP_API_CPP MemoryDevice2DeviceSemaphore::DeviceHandle
MemoryDevice2DeviceSemaphore::deviceHandle() const {
  MemoryDevice2DeviceSemaphore::DeviceHandle device;
  device.remoteInboundToken =
      reinterpret_cast<uint64_t*>(semaphore_.remoteMemory().data());
  device.inboundToken =
      reinterpret_cast<uint64_t*>(semaphore_.localMemory().data());
  device.expectedInboundToken = expectedInboundToken_.get();
  device.outboundToken = outboundToken_.get();
  return device;
};

}  // namespace mscclpp
