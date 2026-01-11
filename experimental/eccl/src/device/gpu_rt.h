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
#define gpuDeviceGetPCIBusId cudaDeviceGetPCIBusId
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuIpcMemHandle_t cudaIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#define gpuIpcOpenMemHandle cudaIpcOpenMemHandle
#define gpuIpcGetMemHandle cudaIpcGetMemHandle
#define gpuIpcCloseMemHandle cudaIpcCloseMemHandle
#define gpuHostMalloc cudaHostMalloc
#define gpuHostAlloc cudaHostAlloc
#define gpuHostAllocMapped cudaHostAllocMapped
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
#define gpuGetLastError cudaGetLastError
#define gpuErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventQuery cudaEventQuery
#define gpuEventSynchronize cudaEventSynchronize
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
// gpu dirver api : for fifo_gdrcopy later
#define gpuDrvResult_t CUresult
#define gpuDrvSuccess CUDA_SUCCESS
#define gpuDrvDevicePtr CUdeviceptr
#define gpuDrvInit(flags) cuInit(flags)
#define gpuDrvDevice_t CUdevice
#define gpuDrvCtx_t CUcontext
#define gpuDrvDeviceGet(pdev, ordinal) cuDeviceGet(pdev, ordinal)
#define gpuDrvDevicePrimaryCtxRetain(pctx, dev) cuDevicePrimaryCtxRetain(pctx, dev)
#define gpuDrvCtxSetCurrent(ctx) cuCtxSetCurrent(ctx)
#define gpuDrvMemAlloc(pdevptr, bytes) cuMemAlloc(pdevptr, bytes)
#define gpuDrvMemFree(devptr) cuMemFree(devptr)
#define gpuDrvMemsetD8(devptr, value, bytes) cuMemsetD8(devptr, value, bytes)
inline const char* gpuDrvGetErrorString(gpuDrvResult_t r) {
  const char* s = nullptr;
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
#define gpuDeviceProp hipDeviceProp_t
#define gpuDeviceGetPCIBusId hipDeviceGetPCIBusId
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define gpuIpcMemHandle_t hipIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define gpuIpcOpenMemHandle hipIpcOpenMemHandle
#define gpuIpcGetMemHandle hipIpcGetMemHandle
#define gpuIpcCloseMemHandle hipIpcCloseMemHandle
#define gpuHostMalloc hipHostMalloc
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
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventQuery hipEventQuery
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
// gpu dirver api : for fifo_gdrcopy later
#define gpuDrvResult_t hipError_t
#define gpuDrvSuccess hipSuccess
#define gpuDrvDevicePtr hipDeviceptr_t
#define gpuDrvInit(flags) hipInit(flags)
#define gpuDrvDevice_t hipDevice_t
#define gpuDrvCtx_t hipCtx_t
#define gpuDrvDeviceGet(pdev, ordinal) hipDeviceGet(pdev, ordinal)
#define gpuDrvDevicePrimaryCtxRetain(pctx, dev) hipDevicePrimaryCtxRetain(pctx, dev)
#define gpuDrvCtxSetCurrent(ctx) hipCtxSetCurrent(ctx)
inline gpuDrvResult_t gpuDrvMemAlloc(void** p, size_t bytes) { return hipMalloc(p, bytes); }
inline gpuDrvResult_t gpuDrvMemFree(void* p) { return hipFree(p); }
inline gpuDrvResult_t gpuDrvMemsetD8(void* p, unsigned char v, size_t bytes) {
  return hipMemset(p, (int)v, bytes);
}
inline const char* gpuDrvGetErrorString(gpuDrvResult_t r) {
  return hipGetErrorString(r);
}
#endif

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

#define GPU_DRV_CHECK(call)                                                     \
  do {                                                                          \
    gpuDrvResult_t _r = (call);                                                 \
    if (_r != gpuDrvSuccess) {                                                  \
      fprintf(stderr, "GPU DRV error %s:%d: %s (%d)\n", __FILE__, __LINE__,     \
              gpuDrvGetErrorString(_r), (int)_r);                               \
      std::abort();                                                             \
    }                                                                           \
  } while (0)