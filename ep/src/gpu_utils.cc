// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstring>
#include "gpu.hpp"
#include "gpu_utils.hpp"

static inline bool isCudaTeardownError(cudaError_t err) {
#if defined(__HIP_PLATFORM_AMD__)
  return err == cudaErrorContextIsDestroyed || err == cudaErrorInvalidDevice;
#else   // !defined(__HIP_PLATFORM_AMD__)
  return err == cudaErrorCudartUnloading || err == cudaErrorContextIsDestroyed || err == cudaErrorInitializationError ||
         err == cudaErrorInvalidDevice || err == cudaErrorLaunchFailure;
#endif  // !defined(__HIP_PLATFORM_AMD__)
}

static inline bool isCuTeardownError(CUresult r) {
  return r == CUDA_ERROR_DEINITIALIZED || r == CUDA_ERROR_CONTEXT_IS_DESTROYED || r == CUDA_ERROR_LAUNCH_FAILED;
}

#define MSCCLPP_CUDATHROW_IGNORE_TEARDOWN(cmd) \
  do {                                         \
    cudaError_t __e = cmd;                     \
    if (isCudaTeardownError(__e)) {            \
      (void)cudaGetLastError();                \
    } else {                                   \
      MSCCLPP_CUDATHROW(__e);                  \
    }                                          \
  } while (false)

#define MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cmd) \
  do {                                       \
    CUresult __e = cmd;                      \
    if (!isCuTeardownError(__e)) {           \
      MSCCLPP_CUTHROW(__e);                  \
    }                                        \
  } while (false)

#define MSCCLPP_CUTHROW_IGNORE(cmd)                                        \
  do {                                                                     \
    CUresult __e = cmd;                                                    \
    if (__e != CUDA_SUCCESS) {                                             \
      const char* errStr;                                                  \
      cuGetErrorString(__e, &errStr);                                      \
      printf("%s:%d Cuda failure %d '%s'", __FILE__, __LINE__, __e, errStr); \
    }                                                                      \
  } while (false)

namespace mscclpp {

AvoidCudaGraphCaptureGuard::AvoidCudaGraphCaptureGuard() : mode_(cudaStreamCaptureModeRelaxed), active_(true) {
  cudaError_t res = cudaThreadExchangeStreamCaptureMode(&mode_);
  if (isCudaTeardownError(res)) {
    // Runtime is going away; just mark inactive so destructor skips restoring.
    active_ = false;
    (void)cudaGetLastError();
  } else {
    MSCCLPP_CUDATHROW(res);
  }
}

AvoidCudaGraphCaptureGuard::~AvoidCudaGraphCaptureGuard() {
  if (!active_) return;
  (void)cudaThreadExchangeStreamCaptureMode(&mode_);
}

CudaStreamWithFlags::CudaStreamWithFlags() : stream_(nullptr) { MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId_)); }

CudaStreamWithFlags::CudaStreamWithFlags(unsigned int flags) {
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId_));
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream_, flags));
}

CudaStreamWithFlags::~CudaStreamWithFlags() {
  if (!empty()) (void)cudaStreamDestroy(stream_);
}

void CudaStreamWithFlags::set(unsigned int flags) {
  if (!empty()) throw Error("CudaStreamWithFlags already set", ErrorCode::InvalidUsage);
  int originalDeviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&originalDeviceId));  // Save the current device
  MSCCLPP_CUDATHROW(cudaSetDevice(deviceId_));
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream_, flags));
  MSCCLPP_CUDATHROW(cudaSetDevice(originalDeviceId));  // Restore the original device
}

bool CudaStreamWithFlags::empty() const { return stream_ == nullptr; }

GpuStream::GpuStream(std::shared_ptr<GpuStreamPool> pool, std::shared_ptr<CudaStreamWithFlags> stream)
    : pool_(pool), stream_(stream) {}

GpuStream::~GpuStream() { pool_->streams_[deviceId()].push_back(stream_); }

GpuStreamPool::GpuStreamPool() {}

GpuStream GpuStreamPool::getStream() {
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  auto& streamVec = streams_[deviceId];
  if (!streamVec.empty()) {
    auto stream = streamVec.back();
    streamVec.pop_back();
    return GpuStream(gpuStreamPool(), stream);
  }
  return GpuStream(gpuStreamPool(), std::make_shared<CudaStreamWithFlags>(cudaStreamNonBlocking));
}

void GpuStreamPool::clear() { streams_.clear(); }

// A global pool instance
static std::shared_ptr<GpuStreamPool> gGpuStreamPool_;

std::shared_ptr<GpuStreamPool> gpuStreamPool() {
  if (!gGpuStreamPool_) {
    gGpuStreamPool_ = std::make_shared<GpuStreamPool>();
  }
  return gGpuStreamPool_;
}

namespace detail {

CUmemAllocationHandleType nvlsCompatibleMemHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

/// set memory access permission to read-write
/// @param base Base memory pointer.
/// @param size Size of the memory.
void setReadWriteMemoryAccess(void* base, size_t size) {
  CUmemAccessDesc accessDesc = {};
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess((CUdeviceptr)base, size, &accessDesc, 1));
}

void* gpuCalloc(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  auto stream = gpuStreamPool()->getStream();
  MSCCLPP_CUDATHROW(cudaMalloc(&ptr, bytes));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, bytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}

void* gpuCallocHost(size_t bytes, unsigned int flags) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  MSCCLPP_CUDATHROW(cudaHostAlloc(&ptr, bytes, flags));
  ::memset(ptr, 0, bytes);
  return ptr;
}

#if defined(__HIP_PLATFORM_AMD__)
void* gpuCallocUncached(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  auto stream = gpuStreamPool()->getStream();
  MSCCLPP_CUDATHROW(hipExtMallocWithFlags((void**)&ptr, bytes, hipDeviceMallocUncached));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, bytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}
#endif  // defined(__HIP_PLATFORM_AMD__)

#if (CUDA_NVLS_API_AVAILABLE)
size_t getMulticastGranularity(size_t size, CUmulticastGranularity_flags granFlag) {
  size_t gran = 0;
  int numDevices = 0;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&numDevices));

  CUmulticastObjectProp prop = {};
  prop.size = size;
  // This is a dummy value, it might affect the granularity in the future
  prop.numDevices = numDevices;
  prop.handleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC);
  prop.flags = 0;
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&gran, &prop, granFlag));
  return gran;
}

void* gpuCallocPhysical(size_t bytes, size_t gran, size_t align) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  int deviceId = -1;
  CUdevice currentDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  MSCCLPP_CUTHROW(cuDeviceGet(&currentDevice, deviceId));

  int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  int isFabricSupported;
  MSCCLPP_CUTHROW(
      cuDeviceGetAttribute(&isFabricSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDevice));
  if (isFabricSupported) {
    requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = (CUmemAllocationHandleType)(requestedHandleTypes);
  prop.location.id = currentDevice;

  if (gran == 0) {
    gran = getMulticastGranularity(bytes, CU_MULTICAST_GRANULARITY_RECOMMENDED);
  }

  // allocate physical memory
  CUmemGenericAllocationHandle memHandle;
  size_t nbytes = (bytes + gran - 1) / gran * gran;
  CUresult result = cuMemCreate(&memHandle, nbytes, &prop, 0);
  if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC &&
      (result == CUDA_ERROR_NOT_PERMITTED || result == CUDA_ERROR_NOT_SUPPORTED)) {
    requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.requestedHandleTypes = (CUmemAllocationHandleType)requestedHandleTypes;
    MSCCLPP_CUTHROW(cuMemCreate(&memHandle, nbytes, &prop, 0));
  } else {
    MSCCLPP_CUTHROW(result);
  }
  nvlsCompatibleMemHandleType = (CUmemAllocationHandleType)requestedHandleTypes;

  if (align == 0) {
    align = getMulticastGranularity(nbytes, CU_MULTICAST_GRANULARITY_MINIMUM);
  }

  void* devicePtr = nullptr;
  MSCCLPP_CUTHROW(cuMemAddressReserve((CUdeviceptr*)&devicePtr, nbytes, align, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap((CUdeviceptr)devicePtr, nbytes, 0, memHandle, 0));
  setReadWriteMemoryAccess(devicePtr, nbytes);
  auto stream = gpuStreamPool()->getStream();
  MSCCLPP_CUDATHROW(cudaMemsetAsync(devicePtr, 0, nbytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  return devicePtr;
}
#endif  // CUDA_NVLS_API_AVAILABLE

void gpuFree(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW_IGNORE_TEARDOWN(cudaFree(ptr));
}

void gpuFreeHost(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW_IGNORE_TEARDOWN(cudaFreeHost(ptr));
}

#if (CUDA_NVLS_API_AVAILABLE)
void gpuFreePhysical(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cuMemRetainAllocationHandle(&handle, ptr));
  MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cuMemRelease(handle));
  MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  MSCCLPP_CUTHROW_IGNORE(cuMemUnmap((CUdeviceptr)ptr, size));
  MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cuMemRelease(handle));
  MSCCLPP_CUTHROW_IGNORE(cuMemAddressFree((CUdeviceptr)ptr, size));
}
#endif  // CUDA_NVLS_API_AVAILABLE

void gpuMemcpyAsync(void* dst, const void* src, size_t bytes, cudaStream_t stream, cudaMemcpyKind kind) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, bytes, kind, stream));
}

void gpuMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}

}  // namespace detail

bool isNvlsSupported() {
  // Environment check removed - not needed for this benchmark
  [[maybe_unused]] static bool result = false;
  [[maybe_unused]] static bool isChecked = false;
#if (CUDA_NVLS_API_AVAILABLE)
  if (!isChecked) {
    int deviceId;
    int isMulticastSupported;
    CUdevice dev;
    MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
    MSCCLPP_CUTHROW(cuDeviceGet(&dev, deviceId));
    MSCCLPP_CUTHROW(cuDeviceGetAttribute(&isMulticastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
    return isMulticastSupported == 1;
  }
  return result;
#endif
  return false;
}

bool isCuMemMapAllocated([[maybe_unused]] void* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  return false;
#else
  CUmemGenericAllocationHandle handle;
  CUresult result = cuMemRetainAllocationHandle(&handle, ptr);
  if (result != CUDA_SUCCESS) {
    return false;
  }
  MSCCLPP_CUTHROW(cuMemRelease(handle));
  if (!isNvlsSupported()) {
    throw Error("cuMemMap is used in env without NVLS support", ErrorCode::InvalidUsage);
  }
  return true;
#endif
}

}  // namespace mscclpp