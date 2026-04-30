#pragma once

#ifndef __HIP_PLATFORM_AMD__
#include <cuda.h>
#include <cuda_runtime.h>
#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#define gpuGetErrorString cudaGetErrorString
#define gpuStream_t cudaStream_t
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamDestroy cudaStreamDestroy
#define gpuDeviceProp cudaDeviceProp
#define gpuSetDevice cudaSetDevice
#define gpuDeviceMapHost cudaDeviceMapHost
#define gpuSetDeviceFlags cudaSetDeviceFlags
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuDeviceGetPCIBusId cudaDeviceGetPCIBusId
#define gpuDeviceReset cudaDeviceReset
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuIpcMemHandle_t cudaIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#define gpuIpcOpenMemHandle cudaIpcOpenMemHandle
#define gpuIpcGetMemHandle cudaIpcGetMemHandle
#define gpuIpcCloseMemHandle cudaIpcCloseMemHandle
#define gpuHostMalloc cudaMallocHost  // no cudaHostMalloc API in CUDA
#define gpuMallocHost cudaMallocHost
#define gpuHostAlloc cudaHostAlloc
#define gpuHostAllocMapped cudaHostAllocMapped
#define gpuHostFree cudaFreeHost
#define gpuFreeHost cudaFreeHost
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMallocAsync cudaMallocAsync
#define gpuFreeAsync cudaFreeAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyPeerAsync cudaMemcpyPeerAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemset cudaMemset
#define gpuMemsetAsync cudaMemsetAsync
inline gpuError_t gpuMemGetAddressRange(void** base_ptr, size_t* size,
                                        void* ptr) {
  if (ptr == nullptr) return cudaErrorInvalidValue;
  CUdeviceptr base = 0;
  CUresult result = cuMemGetAddressRange(&base, size, (CUdeviceptr)ptr);
  if (result == CUDA_SUCCESS) {
    if (base_ptr != nullptr) *base_ptr = reinterpret_cast<void*>(base);
    return gpuSuccess;
  }
  return static_cast<gpuError_t>(result);
}
#define gpuGetLastError cudaGetLastError
#define gpuErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
#define gpuErrorNotReady cudaErrorNotReady
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventQuery cudaEventQuery
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuStreamWaitEvent cudaStreamWaitEvent
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDefault cudaEventDefault
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuEventInterprocess cudaEventInterprocess
#define gpuIpcEventHandle_t cudaIpcEventHandle_t
#define gpuIpcGetEventHandle cudaIpcGetEventHandle
#define gpuIpcOpenEventHandle cudaIpcOpenEventHandle
#define gpuIpcCloseEventHandle cudaIpcCloseEventHandle
#define gpuLaunchKernel cudaLaunchKernel
#define gpuDeviceSynchronize cudaDeviceSynchronize
// DMA-BUF / GPU driver types for GPUDirect RDMA
#define gpuDriverResult_t CUresult
#define gpuDevicePtr_t CUdeviceptr
#define gpuDriverSuccess CUDA_SUCCESS
#define gpuMemRangeHandleType CUmemRangeHandleType
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
#define gpuPointerAttributes cudaPointerAttributes
#define gpuPointerAttribute_t cudaPointerAttributes
#define gpuPointerGetAttributes cudaPointerGetAttributes
#define gpuMemoryTypeDevice cudaMemoryTypeDevice
#define gpuMemoryTypeManaged cudaMemoryTypeManaged
#define GPU_DRIVER_LIB_NAME "libcuda.so.1"
#define GPU_DRIVER_LIB_NAME_FALLBACK "libcuda.so"
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME \
  "cuMemGetHandleForAddressRange"
// gpu dirver api : for fifo_gdrcopy later
#define gpuDrvResult_t CUresult
#define gpuDrvSuccess CUDA_SUCCESS
#define gpuDrvDevicePtr CUdeviceptr
#define gpuDrvInit(flags) cuInit(flags)
#define gpuDrvDevice_t CUdevice
#define gpuDrvCtx_t CUcontext
#define gpuDrvDeviceGet(pdev, ordinal) cuDeviceGet(pdev, ordinal)
#define gpuDrvDeviceGetAttribute(pi, attrib, dev) \
  cuDeviceGetAttribute(pi, attrib, dev)
#define gpuDrvDevicePrimaryCtxRetain(pctx, dev) \
  cuDevicePrimaryCtxRetain(pctx, dev)
#define gpuDrvCtxSetCurrent(ctx) cuCtxSetCurrent(ctx)
#define gpuDrvMemAlloc(pdevptr, bytes) cuMemAlloc(pdevptr, bytes)
#define gpuDrvMemFree(devptr) cuMemFree(devptr)
#define gpuDrvMemsetD8(devptr, value, bytes) cuMemsetD8(devptr, value, bytes)
inline char const* gpuDrvGetErrorString(gpuDrvResult_t r) {
  char const* s = nullptr;
  (void)cuGetErrorString(r, &s);
  return s ? s : "Unknown CUDA driver error";
}
#else
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#define gpuGetErrorString hipGetErrorString
#define gpuStream_t hipStream_t
#define gpuStreamNonBlocking hipStreamNonBlocking
#define gpuStreamCreate hipStreamCreate
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamDestroy hipStreamDestroy
#define gpuSetDevice hipSetDevice
#define gpuDeviceMapHost hipDeviceMapHost
#define gpuSetDeviceFlags hipSetDeviceFlags
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuDevAttrMultiProcessorCount hipDevAttrMultiProcessorCount
#define gpuDeviceProp hipDeviceProp_t
#define gpuDeviceReset hipDeviceReset
#define gpuDeviceGetPCIBusId hipDeviceGetPCIBusId
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define gpuIpcMemHandle_t hipIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define gpuIpcOpenMemHandle hipIpcOpenMemHandle
#define gpuIpcGetMemHandle hipIpcGetMemHandle
#define gpuIpcCloseMemHandle hipIpcCloseMemHandle
#define gpuHostMalloc hipHostMalloc
#define gpuMallocHost hipHostMalloc  // cudaMallocHost deprecated in ROCm
#define gpuHostAlloc hipHostAlloc
#define gpuHostFree hipHostFree
#define gpuHostAllocMapped hipHostAllocMapped
#define gpuFreeHost hipFreeHost
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMallocAsync hipMallocAsync
#define gpuFreeAsync hipFreeAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyPeerAsync hipMemcpyPeerAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
#define gpuMemset hipMemset
#define gpuMemsetAsync hipMemsetAsync
#define gpuGetLastError hipGetLastError
#define gpuErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define gpuErrorNotReady hipErrorNotReady
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventQuery hipEventQuery
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuStreamWaitEvent hipStreamWaitEvent
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDefault hipEventDefault
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuEventInterprocess hipEventInterprocess
#define gpuIpcEventHandle_t hipIpcEventHandle_t
#define gpuIpcGetEventHandle hipIpcGetEventHandle
#define gpuIpcOpenEventHandle hipIpcOpenEventHandle
#define gpuIpcCloseEventHandle(handle) (gpuSuccess)
#define gpuMemGetAddressRange hipMemGetAddressRange
#define gpuLaunchKernel hipLaunchKernel
#define gpuDeviceSynchronize hipDeviceSynchronize
// DMA-BUF / GPU driver types for GPUDirect RDMA
#define gpuDriverResult_t hipError_t
#define gpuDevicePtr_t hipDeviceptr_t
#define gpuDriverSuccess hipSuccess
#define gpuMemRangeHandleType hipMemRangeHandleType
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD hipMemRangeHandleTypeDmaBufFd
#define gpuPointerAttributes hipPointerAttribute_t
#define gpuPointerAttribute_t hipPointerAttribute_t
#define gpuPointerGetAttributes hipPointerGetAttributes
#define gpuMemoryTypeDevice hipMemoryTypeDevice
#define gpuMemoryTypeManaged hipMemoryTypeManaged
#define GPU_DRIVER_LIB_NAME "libamdhip64.so"
#define GPU_DRIVER_LIB_NAME_FALLBACK "libamdhip64.so"
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME \
  "hipMemGetHandleForAddressRange"
// gpu dirver api : for fifo_gdrcopy later
#define gpuDrvResult_t hipError_t
#define gpuDrvSuccess hipSuccess
#define gpuDrvDevicePtr hipDeviceptr_t
#define gpuDrvInit(flags) hipInit(flags)
#define gpuDrvDevice_t hipDevice_t
#define gpuDrvCtx_t hipCtx_t
#define gpuDrvDeviceGet(pdev, ordinal) hipDeviceGet(pdev, ordinal)
inline gpuDrvResult_t gpuDrvDeviceGetAttribute(int* value, int attrib,
                                               gpuDrvDevice_t dev) {
  return hipDeviceGetAttribute(value, static_cast<hipDeviceAttribute_t>(attrib),
                               dev);
}
#define gpuDrvDevicePrimaryCtxRetain(pctx, dev) \
  hipDevicePrimaryCtxRetain(pctx, dev)
#define gpuDrvCtxSetCurrent(ctx) hipCtxSetCurrent(ctx)
inline gpuDrvResult_t gpuDrvMemAlloc(void** p, size_t bytes) {
  return hipMalloc(p, bytes);
}
inline gpuDrvResult_t gpuDrvMemFree(void* p) { return hipFree(p); }
inline gpuDrvResult_t gpuDrvMemsetD8(void* p, unsigned char v, size_t bytes) {
  return hipMemset(p, (int)v, bytes);
}
inline char const* gpuDrvGetErrorString(gpuDrvResult_t r) {
  return hipGetErrorString(r);
}
#endif

// Function pointer type for DMA-BUF handle export (loaded via dlsym).
typedef gpuDriverResult_t (*gpuMemGetHandleForAddressRange_fn)(
    void*, gpuDevicePtr_t, size_t, gpuMemRangeHandleType, unsigned long long);

#define GPU_RT_CHECK(call)                                         \
  do {                                                             \
    gpuError_t err__ = (call);                                     \
    if (err__ != gpuSuccess) {                                     \
      fprintf(stderr, "GPU error %s:%d: %s\n", __FILE__, __LINE__, \
              gpuGetErrorString(err__));                           \
      std::abort();                                                \
    }                                                              \
  } while (0)

#define GPU_RT_CHECK_ERRORS(msg)                              \
  do {                                                        \
    gpuError_t __err = gpuGetLastError();                     \
    if (__err != gpuSuccess) {                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
              gpuGetErrorString(__err), __FILE__, __LINE__);  \
      fprintf(stderr, "*** FAILED - ABORTING\n");             \
      exit(1);                                                \
    }                                                         \
  } while (0)

#define GPU_DRV_CHECK(call)                                                 \
  do {                                                                      \
    gpuDrvResult_t _r = (call);                                             \
    if (_r != gpuDrvSuccess) {                                              \
      fprintf(stderr, "GPU DRV error %s:%d: %s (%d)\n", __FILE__, __LINE__, \
              gpuDrvGetErrorString(_r), (int)_r);                           \
      std::abort();                                                         \
    }                                                                       \
  } while (0)
