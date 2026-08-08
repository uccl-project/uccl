// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPU_IPC_MEM_HPP_
#define MSCCLPP_GPU_IPC_MEM_HPP_

#include "gpu.hpp"
#include <bitset>
#include <memory>
#include <ostream>

namespace mscclpp {

/// GpuIpcMemHandle is a generic GPU memory handle that covers all existing
/// methods for GPU memory export/import, including the original CUDA IPC
/// methods (`RuntimeIpc` type) and the later ones using a POSIX file descriptor
/// (`PosixFd` type) or a fabric handle by the NVIDIA IMEX service (`Fabric`
/// type). When a GPU memory pointer is given, RegisteredMemory creates and owns
/// a GpuIpcMemHandle that consists of all types of handles that are available
/// on the local environment, so that the remote side can choose the most
/// suitable one for import. Note that multiple types of handles can be present
/// in a single GpuIpcMemHandle, and the `typeFlags` field indicates which types
/// are available. Note that InfiniBand memory registration is not covered by
/// GpuIpcMemHandle.
struct GpuIpcMemHandle {
  struct Type {
    static constexpr uint8_t None = 0;
    static constexpr uint8_t RuntimeIpc = 1;
    static constexpr uint8_t PosixFd = 2;
    static constexpr uint8_t Fabric = 4;
  };

  using TypeFlags = uint8_t;

  TypeFlags typeFlags;
  size_t baseSize;
  size_t offsetFromBase;

  struct {
    cudaIpcMemHandle_t handle;
  } runtimeIpc;

  struct {
    int pid;
    int fd;
  } posixFd;

  struct {
    char handle[64];
  } fabric;

  static void deleter(GpuIpcMemHandle* handle);

  // We make GpuIpcMemHandle trivially copyable for easy serialization,
  // and thus it cannot have explicit destructors.
  // We use a custom deleter for unique_ptr to handle cleanup without a
  // destructor.
  struct UniquePtr
      : public std::unique_ptr<GpuIpcMemHandle,
                               decltype(&GpuIpcMemHandle::deleter)> {
    using Base =
        std::unique_ptr<GpuIpcMemHandle, decltype(&GpuIpcMemHandle::deleter)>;

    // Default constructor
    UniquePtr() : Base(nullptr, &GpuIpcMemHandle::deleter) {}

    // Inherit other constructors
    using Base::Base;

    // Allow implicit conversion from Base
    UniquePtr(Base&& other) : Base(std::move(other)) {}
  };

  static UniquePtr create(const CUdeviceptr ptr);
  static UniquePtr createMulticast(size_t bufferSize, int numDevices);
};

using UniqueGpuIpcMemHandle = GpuIpcMemHandle::UniquePtr;

std::ostream& operator<<(std::ostream& os,
                         GpuIpcMemHandle::TypeFlags const& typeFlags);

static_assert(std::is_trivially_copyable_v<GpuIpcMemHandle>);

/// GpuIpcMem represents a GPU memory region that has been imported using a
/// GpuIpcMemHandle. If a RegisteredMemory instance represents an imported GPU
/// memory, it will manage a unique GpuIpcMem instance for that memory region.
class GpuIpcMem : public std::enable_shared_from_this<GpuIpcMem> {
 public:
  /// Create a GpuIpcMem instance from a GpuIpcMemHandle.
  /// @param handle The handle to import.
  /// @return A shared_ptr to the created GpuIpcMem instance.
  static std::shared_ptr<GpuIpcMem> create(GpuIpcMemHandle const& handle);

  ~GpuIpcMem();

  /// Map the imported GPU memory for access. Subsequent calls to map() will
  /// simply create a new mapping to the same memory, which is not a desired
  /// usage pattern.
  /// @return A shared_ptr to the mapped memory. When all references are
  /// released,
  ///         the memory is automatically unmapped.
  std::shared_ptr<void> map();

  /// Map multicast memory at the given offset.
  /// @param numDevices Number of devices participating in multicast.
  /// @param mcOffset Offset in the multicast buffer.
  /// @param bufferAddr Device pointer to bind.
  /// @param bufferSize Size of the buffer to bind.
  /// @return A shared_ptr to the mapped multicast memory. When all references
  /// are released,
  ///         the memory is automatically unmapped and unbound.
  std::shared_ptr<void> mapMulticast(int numDevices, size_t mcOffset,
                                     CUdeviceptr bufferAddr, size_t bufferSize);

 private:
  GpuIpcMem(GpuIpcMemHandle const& handle);

  GpuIpcMemHandle handle_;
  CUmemGenericAllocationHandle allocHandle_;
  int multicastAddedDeviceId_;
  uint8_t type_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_GPU_IPC_MEM_HPP_
