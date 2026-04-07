// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpu_ipc_mem.hpp"
#include "gpu_utils.hpp"
#include "logger.hpp"
#include "unix_socket.hpp"
#include <cstring>
#include <unistd.h>

namespace mscclpp {

std::ostream& operator<<(std::ostream& os,
                         GpuIpcMemHandle::TypeFlags const& typeFlags) {
  bool first = true;
  if (typeFlags & GpuIpcMemHandle::Type::RuntimeIpc) {
    os << "RuntimeIpc";
    first = false;
  }
  if (typeFlags & GpuIpcMemHandle::Type::PosixFd) {
    if (!first) os << "|";
    os << "PosixFd";
    first = false;
  }
  if (typeFlags & GpuIpcMemHandle::Type::Fabric) {
    if (!first) os << "|";
    os << "Fabric";
    first = false;
  }
  if (first) {
    os << "None";
  }
  return os;
}

[[maybe_unused]] static bool isFabricMemHandleAvailable() {
#if (CUDA_NVLS_API_AVAILABLE)
  static int resultCache =
      -1;  // -1: uninitialized, 0: not available, 1: available
  if (resultCache != -1) {
    return resultCache == 1;
  }
  CUdevice currentDevice;
  int isFabricSupported;
  MSCCLPP_CUTHROW(cuCtxGetDevice(&currentDevice));
  MSCCLPP_CUTHROW(cuDeviceGetAttribute(
      &isFabricSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
      currentDevice));
  if (isFabricSupported == 0) {
    resultCache = 0;
    return false;
  }

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = currentDevice;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t minGran;
  MSCCLPP_CUTHROW(cuMemGetAllocationGranularity(
      &minGran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  // try allocating minimal amount of memory
  CUmemGenericAllocationHandle memHandle;
  CUresult result = cuMemCreate(&memHandle, minGran, &prop, 0);
  if (result == CUDA_ERROR_NOT_PERMITTED ||
      result == CUDA_ERROR_NOT_SUPPORTED) {
    // unprivileged user or old kernel version
    resultCache = 0;
    return false;
  } else {
    MSCCLPP_CUTHROW(result);
  }

  // it worked; cleanup now
  MSCCLPP_CUTHROW(cuMemRelease(memHandle));
  resultCache = 1;
  return true;
#else   // !(CUDA_NVLS_API_AVAILABLE)
  return false;
#endif  // !(CUDA_NVLS_API_AVAILABLE)
}

#if defined(MSCCLPP_DEVICE_HIP)

// Custom hash and equality for cudaIpcMemHandle_t
struct CudaIpcMemHandleHash {
  size_t operator()(cudaIpcMemHandle_t const& handle) const {
    std::string_view view(handle.reserved, sizeof(handle.reserved));
    return std::hash<std::string_view>{}(view);
  }
};

struct CudaIpcMemHandleEqual {
  bool operator()(cudaIpcMemHandle_t const& lhs,
                  cudaIpcMemHandle_t const& rhs) const noexcept {
    return std::memcmp(lhs.reserved, rhs.reserved, sizeof(lhs.reserved)) == 0;
  }
};

static std::unordered_map<cudaIpcMemHandle_t, void*, CudaIpcMemHandleHash,
                          CudaIpcMemHandleEqual>
    openCudaIpcMemHandleMap;

static std::mutex openCudaIpcMemHandleMapMutex;

// Cache open ipc handles to avoid opening multiple times (ROCm may exceed
// system limit on vm.max_map_count).
static inline cudaError_t cudaIpcOpenMemHandleWrapper(
    void** addr, cudaIpcMemHandle_t ipcHandle) {
  std::lock_guard<std::mutex> lock(openCudaIpcMemHandleMapMutex);
  auto it = openCudaIpcMemHandleMap.find(ipcHandle);
  if (it != openCudaIpcMemHandleMap.end()) {
    *addr = it->second;
    return cudaSuccess;
  }
  cudaError_t err =
      cudaIpcOpenMemHandle(addr, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
  if (err == cudaSuccess) {
    openCudaIpcMemHandleMap[ipcHandle] = *addr;
  }
  return err;
}

static inline cudaError_t cudaIpcCloseMemHandleWrapper(
    void* addr, cudaIpcMemHandle_t ipcHandle) {
  std::lock_guard<std::mutex> lock(openCudaIpcMemHandleMapMutex);
  openCudaIpcMemHandleMap.erase(ipcHandle);
  return cudaIpcCloseMemHandle(addr);
}

#else  // !defined(MSCCLPP_DEVICE_HIP)

static inline cudaError_t cudaIpcOpenMemHandleWrapper(
    void** addr, cudaIpcMemHandle_t ipcHandle) {
  return cudaIpcOpenMemHandle(addr, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
}

static inline cudaError_t cudaIpcCloseMemHandleWrapper(
    void* addr, [[maybe_unused]] cudaIpcMemHandle_t ipcHandle) {
  return cudaIpcCloseMemHandle(addr);
}

#endif  // !defined(MSCCLPP_DEVICE_HIP)

void GpuIpcMemHandle::deleter(GpuIpcMemHandle* handle) {
  if (handle) {
    if (handle->typeFlags & GpuIpcMemHandle::Type::PosixFd) {
      UnixSocketServer::instance().unregisterFd(handle->posixFd.fd);
      ::close(handle->posixFd.fd);
    }
    delete handle;
  }
}

UniqueGpuIpcMemHandle GpuIpcMemHandle::create(const CUdeviceptr ptr) {
  auto handle =
      UniqueGpuIpcMemHandle(new GpuIpcMemHandle(), &GpuIpcMemHandle::deleter);
  handle->typeFlags = GpuIpcMemHandle::Type::None;
  handle->posixFd.fd = -1;

  CUdeviceptr basePtr;
  size_t sz;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&basePtr, &sz, ptr));
  if (sz == 0) {
    // No valid memory range found
    WARN(GPU,
         "Failed to create GpuIpcMemHandle: cuMemGetAddressRange returned size "
         "0 for pointer ",
         (void*)ptr);
    return handle;
  }
  handle->baseSize = sz;
  handle->offsetFromBase = size_t(ptr) - size_t(basePtr);

  // Runtime IPC handle
  cudaError_t err =
      cudaIpcGetMemHandle(&handle->runtimeIpc.handle, (void*)basePtr);
  if (err == cudaSuccess) {
    handle->typeFlags |= GpuIpcMemHandle::Type::RuntimeIpc;
  } else {
    (void)cudaGetLastError();
  }

#if !defined(MSCCLPP_DEVICE_HIP)  // Remove when HIP fully supports virtual
                                  // memory management APIs
  CUmemGenericAllocationHandle allocHandle;
  CUresult res = cuMemRetainAllocationHandle(&allocHandle, (void*)basePtr);
  if (res == CUDA_ERROR_NOT_SUPPORTED || res == CUDA_ERROR_INVALID_VALUE) {
    // Not supported on this platform or not mapped by cuMem API
    return handle;
  }
  MSCCLPP_CUTHROW(res);

  // POSIX FD handle
  int fileDesc;
  if (cuMemExportToShareableHandle(&fileDesc, allocHandle,
                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                   0) == CUDA_SUCCESS) {
    handle->posixFd.fd = UnixSocketServer::instance().registerFd(fileDesc);
    handle->posixFd.pid = ::getpid();
    handle->typeFlags |= GpuIpcMemHandle::Type::PosixFd;
  }

  // FABRIC handle
  if (cuMemExportToShareableHandle(&(handle->fabric.handle), allocHandle,
                                   CU_MEM_HANDLE_TYPE_FABRIC,
                                   0) == CUDA_SUCCESS) {
    handle->typeFlags |= GpuIpcMemHandle::Type::Fabric;
  }

  MSCCLPP_CUTHROW(cuMemRelease(allocHandle));
#endif  // !defined(MSCCLPP_DEVICE_HIP)

  return handle;
}

UniqueGpuIpcMemHandle GpuIpcMemHandle::createMulticast(
    [[maybe_unused]] size_t bufferSize, [[maybe_unused]] int numDevices) {
#if (CUDA_NVLS_API_AVAILABLE)
  if (bufferSize == 0) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "Multicast buffer size should be positive");
  }
  if (numDevices < 1) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "Number of multicasting devices should be positive");
  }
  auto handle =
      UniqueGpuIpcMemHandle(new GpuIpcMemHandle(), &GpuIpcMemHandle::deleter);

  bool isFabricAvailable = isFabricMemHandleAvailable();

  // get granularity
  size_t recMcGran;
  CUmulticastObjectProp prop = {};
  prop.size = bufferSize;
  prop.numDevices = numDevices;
  if (isFabricAvailable) {
    prop.handleTypes =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC;
  } else {
    prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(
      &recMcGran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

  CUmemGenericAllocationHandle allocHandle;
  size_t baseSize = ((bufferSize + recMcGran - 1) / recMcGran) * recMcGran;
  prop.size = baseSize;
  MSCCLPP_CUTHROW(cuMulticastCreate(&allocHandle, &prop));

  handle->baseSize = baseSize;
  handle->offsetFromBase = 0;
  handle->typeFlags = GpuIpcMemHandle::Type::None;
  handle->posixFd.fd = -1;

  // POSIX FD handle
  int fileDesc;
  if (cuMemExportToShareableHandle(&fileDesc, allocHandle,
                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                   0) == CUDA_SUCCESS) {
    handle->posixFd.fd = UnixSocketServer::instance().registerFd(fileDesc);
    handle->posixFd.pid = ::getpid();
    handle->typeFlags |= GpuIpcMemHandle::Type::PosixFd;
  }

  // FABRIC handle
  if (isFabricAvailable &&
      (cuMemExportToShareableHandle(&(handle->fabric.handle), allocHandle,
                                    CU_MEM_HANDLE_TYPE_FABRIC,
                                    0) == CUDA_SUCCESS)) {
    handle->typeFlags |= GpuIpcMemHandle::Type::Fabric;
  }

  if (handle->typeFlags == GpuIpcMemHandle::Type::None) {
    cuMemRelease(allocHandle);
    THROW(GPU, Error, ErrorCode::SystemError,
          "createMulticast failed: neither POSIX FD nor FABRIC handle was "
          "created");
  }

  // Release the local allocation handle. The exported POSIX FD / Fabric handle
  // keeps the multicast object alive. Each importer will get its own handle via
  // cuMemImportFromShareableHandle.
  MSCCLPP_CUTHROW(cuMemRelease(allocHandle));
  return handle;
#else   // !(CUDA_NVLS_API_AVAILABLE)
  THROW(GPU, Error, ErrorCode::InvalidUsage,
        "NVLS is not supported on this device (requires CUDA version >= 12.3 "
        "and Linux kernel version >= 5.6.0)");
#endif  // !(CUDA_NVLS_API_AVAILABLE)
}

std::shared_ptr<GpuIpcMem> GpuIpcMem::create(GpuIpcMemHandle const& handle) {
  return std::shared_ptr<GpuIpcMem>(new GpuIpcMem(handle));
}

GpuIpcMem::GpuIpcMem(GpuIpcMemHandle const& handle)
    : handle_(handle),
      allocHandle_(0),
      multicastAddedDeviceId_(-1),
      type_(GpuIpcMemHandle::Type::None) {
  if (handle_.typeFlags == GpuIpcMemHandle::Type::None) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "GpuIpcMemHandle type is None, cannot create GpuIpcMem");
  }
  if ((type_ == GpuIpcMemHandle::Type::None) &&
      (handle_.typeFlags & GpuIpcMemHandle::Type::Fabric)) {
    if (cuMemImportFromShareableHandle(
            &allocHandle_, (void*)handle_.fabric.handle,
            CU_MEM_HANDLE_TYPE_FABRIC) == CUDA_SUCCESS) {
      type_ = GpuIpcMemHandle::Type::Fabric;
    }
  }
  if ((type_ == GpuIpcMemHandle::Type::None) &&
      (handle_.typeFlags & GpuIpcMemHandle::Type::PosixFd)) {
    int fileDesc = UnixSocketClient::instance().requestFd(
        UnixSocketServer::generateSocketPath(handle_.posixFd.pid),
        static_cast<uint32_t>(handle_.posixFd.fd));
    if (cuMemImportFromShareableHandle(
            &allocHandle_, reinterpret_cast<void*>(fileDesc),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS) {
      type_ = GpuIpcMemHandle::Type::PosixFd;
    }
    ::close(fileDesc);
  }
  if ((type_ == GpuIpcMemHandle::Type::None) &&
      (handle_.typeFlags & GpuIpcMemHandle::Type::RuntimeIpc)) {
    type_ = GpuIpcMemHandle::Type::RuntimeIpc;
  }
  if (type_ == GpuIpcMemHandle::Type::None) {
    THROW(GPU, Error, ErrorCode::Aborted,
          "Failed to open GpuIpcMemHandle (type: ", handle_.typeFlags, ")");
  }
}

GpuIpcMem::~GpuIpcMem() {
  if (type_ == GpuIpcMemHandle::Type::PosixFd ||
      type_ == GpuIpcMemHandle::Type::Fabric) {
    CUresult res;
    char const* errStr;
    res = cuMemRelease(allocHandle_);
    if (res != CUDA_SUCCESS) {
      (void)cuGetErrorString(res, &errStr);
      WARN(GPU, "Failed to release CUDA memory allocation handle: ", errStr);
    }
  }
}

std::shared_ptr<void> GpuIpcMem::map() {
  if (type_ == GpuIpcMemHandle::Type::None) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "GpuIpcMemHandle type is None, cannot map memory");
  }

  if (type_ == GpuIpcMemHandle::Type::RuntimeIpc) {
    // RuntimeIpc: Open handle and return shared_ptr with cleanup in deleter
    void* basePtr = nullptr;
    MSCCLPP_CUDATHROW(
        cudaIpcOpenMemHandleWrapper(&basePtr, handle_.runtimeIpc.handle));
    void* dataPtr = static_cast<void*>(static_cast<char*>(basePtr) +
                                       handle_.offsetFromBase);
    cudaIpcMemHandle_t ipcHandle = handle_.runtimeIpc.handle;
    return std::shared_ptr<void>(
        dataPtr, [self = shared_from_this(), basePtr, ipcHandle](void*) {
          cudaError_t err = cudaIpcCloseMemHandleWrapper(basePtr, ipcHandle);
          if (err != cudaSuccess) {
            WARN(GPU, "Failed to close CUDA IPC handle at pointer ", basePtr,
                 ": ", cudaGetErrorString(err));
            (void)cudaGetLastError();
          }
        });
  }

  size_t pageSize = getpagesize();
  if (handle_.baseSize % pageSize) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "Tried to map remote GPU memory with size ", handle_.baseSize,
          " that is not a multiple of the local host page size ", pageSize);
  }

  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));

  CUdeviceptr base;
  size_t minGran;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = deviceId;
  MSCCLPP_CUTHROW(cuMemGetAllocationGranularity(
      &minGran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  MSCCLPP_CUTHROW(cuMemAddressReserve(&base, handle_.baseSize, minGran, 0, 0));
  MSCCLPP_CUTHROW(cuMemMap(base, handle_.baseSize, 0, allocHandle_, 0));

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess(base, handle_.baseSize, &accessDesc, 1));

  void* basePtr = (void*)base;
  size_t baseSize = handle_.baseSize;
  void* dataPtr =
      static_cast<void*>(static_cast<char*>(basePtr) + handle_.offsetFromBase);

  // Return shared_ptr with deleter that unmaps and frees memory
  return std::shared_ptr<void>(
      dataPtr, [self = shared_from_this(), basePtr, baseSize](void*) {
        CUresult res;
        const char* errStr;

        res = cuMemUnmap((CUdeviceptr)basePtr, baseSize);
        if (res != CUDA_SUCCESS) {
          (void)cuGetErrorString(res, &errStr);
          WARN(GPU, "Failed to unmap CUDA memory at pointer ", basePtr, ": ",
               errStr);
        }

        res = cuMemAddressFree((CUdeviceptr)basePtr, baseSize);
        if (res != CUDA_SUCCESS) {
          (void)cuGetErrorString(res, &errStr);
          WARN(GPU, "Failed to free CUDA memory at pointer ", basePtr, ": ",
               errStr);
        }
        // self release will trigger ~GpuIpcMem() which releases allocHandle_
      });
}

std::shared_ptr<void> GpuIpcMem::mapMulticast(
    [[maybe_unused]] int numDevices, [[maybe_unused]] size_t mcOffset,
    [[maybe_unused]] CUdeviceptr bufferAddr,
    [[maybe_unused]] size_t bufferSize) {
#if (CUDA_NVLS_API_AVAILABLE)
  if (type_ != GpuIpcMemHandle::Type::PosixFd &&
      type_ != GpuIpcMemHandle::Type::Fabric) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "GpuIpcMemHandle type is not PosixFd or Fabric, cannot map multicast "
          "memory");
  }
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  if (multicastAddedDeviceId_ == -1) {
    MSCCLPP_CUTHROW(cuMulticastAddDevice(allocHandle_, deviceId));
    multicastAddedDeviceId_ = deviceId;
  } else if (multicastAddedDeviceId_ != deviceId) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "Multicast device ID mismatch: expected ", multicastAddedDeviceId_,
          ", but got ", deviceId);
  }

  size_t minMcGran;
  CUmulticastObjectProp prop = {};
  prop.size = handle_.baseSize;
  prop.numDevices = numDevices;
  prop.handleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC;

  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGran, &prop,
                                            CU_MULTICAST_GRANULARITY_MINIMUM));

  if (!isCuMemMapAllocated((void*)bufferAddr)) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "This NVLS connection tried to bind a buffer that was not allocated "
          "with cuMemMap");
  }
  if ((uintptr_t)bufferAddr % minMcGran != 0) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "This NVLS connection tried to bind a buffer that is not aligned to "
          "the minimum granularity ",
          minMcGran);
  }
  if (bufferSize == 0) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "NVLS buffer size should be larger than zero.");
  }
  if (bufferSize % minMcGran != 0) {
    THROW(GPU, Error, ErrorCode::InvalidUsage,
          "Tried to bind a multicast buffer that is not aligned to the minimum "
          "granularity ",
          minMcGran, ", buffer size: ", bufferSize);
  }

  // Bind the buffer at the specified offset in the multicast handle
  // This will block until all devices call cuMulticastAddDevice()
  MSCCLPP_CUTHROW(
      cuMulticastBindAddr(allocHandle_, mcOffset, bufferAddr, bufferSize, 0));

  // cuMemMap requires offset to be 0 for multicast handles, so we map the
  // entire range [0, mcOffset + bufferSize) and return a pointer at mcPtr +
  // mcOffset. This only consumes extra virtual address space for the mcOffset
  // region; no additional physical memory is used.
  size_t mapSize = mcOffset + bufferSize;
  CUdeviceptr mcPtr;
  MSCCLPP_CUTHROW(cuMemAddressReserve(&mcPtr, mapSize, minMcGran, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap(mcPtr, mapSize, 0, allocHandle_, 0));

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess(mcPtr, mapSize, &accessDesc, 1));

  // Return shared_ptr with custom deleter that unmaps and unbinds
  CUmemGenericAllocationHandle allocHandle = allocHandle_;
  return std::shared_ptr<void>(
      reinterpret_cast<void*>(mcPtr + mcOffset),
      [self = shared_from_this(), mcPtr, mapSize, mcOffset, bufferSize,
       allocHandle](void*) {
        CUresult res;
        const char* errStr;

        res = cuMemUnmap(mcPtr, mapSize);
        if (res != CUDA_SUCCESS) {
          (void)cuGetErrorString(res, &errStr);
          WARN(GPU, "Failed to unmap CUDA memory at pointer ", (void*)mcPtr,
               ": ", errStr);
        }

        res = cuMemAddressFree(mcPtr, mapSize);
        if (res != CUDA_SUCCESS) {
          (void)cuGetErrorString(res, &errStr);
          WARN(GPU, "Failed to free CUDA memory at pointer ", (void*)mcPtr,
               ": ", errStr);
        }

        int deviceId;
        CUdevice device;
        if (cudaGetDevice(&deviceId) == cudaSuccess &&
            cuDeviceGet(&device, deviceId) == CUDA_SUCCESS) {
          (void)cuMulticastUnbind(allocHandle, device, mcOffset, bufferSize);
        }
      });
#else   // !(CUDA_NVLS_API_AVAILABLE)
  THROW(GPU, Error, ErrorCode::InvalidUsage,
        "NVLS is not supported on this device (requires CUDA version >= 12.3 "
        "and Linux kernel version >= 5.6.0)");
#endif  // !(CUDA_NVLS_API_AVAILABLE)
}

}  // namespace mscclpp
