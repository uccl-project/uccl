#pragma once

#if defined(__CAMBRICON_PLATFORM_MLU__)
// Cambricon MLU: map the gpu* abstraction onto CNRT (cnrt*) / CNDRV (cn*);
// direct macros where signatures match, inline wrappers where they differ.
#include <climits>  // CNRT headers do not pull in PATH_MAX like cuda_runtime.h
#include <cn_api.h>
#include <cnrt.h>
#define gpuSuccess cnrtSuccess
#define gpuError_t cnrtRet_t
#define gpuGetErrorString cnrtGetErrorStr
#define gpuGetLastError cnrtGetLastError
#define gpuErrorNotReady cnrtErrorNotReady
// Queue stands in for CUDA stream.
#define gpuStream_t cnrtQueue_t
#define gpuStreamNonBlocking 0
#define gpuStreamLegacy ((cnrtQueue_t)0)
#define gpuStreamPerThread ((cnrtQueue_t)0)
#define gpuStreamCreate cnrtQueueCreate
#define gpuStreamSynchronize cnrtQueueSync
#define gpuStreamDestroy cnrtQueueDestroy
#define gpuLaunchHostFunc cnrtInvokeHostFunc  // (queue, fn, data) order matches
#define gpuHostFn_t cnrtHostFn_t
#define gpuSetDevice cnrtSetDevice
#define gpuGetDevice cnrtGetDevice
#define gpuDeviceGetPCIBusId cnrtDeviceGetPCIBusId
// IPC memory: Acquire/Map/UnMap correspond to CUDA Get/Open/Close.
#define gpuIpcMemHandle_t cnrtIpcMemHandle
#define gpuIpcMemLazyEnablePeerAccess 0
#define gpuIpcGetMemHandle cnrtAcquireMemHandle
#define gpuIpcOpenMemHandle cnrtMapMemHandle
#define gpuIpcCloseMemHandle cnrtUnMapMemHandle
#define gpuMalloc cnrtMalloc
#define gpuFree cnrtFree
#define gpuMemcpy cnrtMemcpy  // (dst, src, bytes, dir) order matches
#define gpuMemcpyPeerAsync cnrtMemcpyPeerAsync
#define gpuMemcpyHostToDevice cnrtMemcpyHostToDev
#define gpuMemcpyDeviceToHost cnrtMemcpyDevToHost
#define gpuMemcpyDeviceToDevice cnrtMemcpyDevToDev
// Notifier stands in for CUDA event.
#define gpuEvent_t cnrtNotifier_t
#define gpuEventDestroy cnrtNotifierDestroy
#define gpuEventRecord cnrtPlaceNotifier
#define gpuEventQuery cnrtQueryNotifier
#define gpuEventDisableTiming CNRT_NOTIFIER_DISABLE_TIMING
#define gpuPointerAttribute_t cnrtPointerAttributes_t
#define gpuPointerGetAttributes cnrtPointerGetAttributes
#define gpuMemoryTypeDevice cnrtMemTypeDevice
// DMA-BUF / driver-level types for GPUDirect RDMA (inter-node path only).
#define gpuDriverResult_t CNresult
#define gpuDevicePtr_t CNaddr
#define gpuDriverSuccess CN_SUCCESS
#define gpuMemRangeHandleType int
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD 0
#define GPU_DRIVER_LIB_NAME "libcndrv.so"
#define GPU_DRIVER_LIB_NAME_FALLBACK "libcndrv.so.2"
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME \
  "cnMemGetHandleForAddressRange"
// Wrappers for APIs whose argument order/flags differ from CUDA.
inline gpuError_t gpuStreamCreateWithFlags(cnrtQueue_t* q, unsigned int) {
  return cnrtQueueCreate(q);
}
inline gpuError_t gpuEventCreateWithFlags(cnrtNotifier_t* n, unsigned int) {
  return cnrtNotifierCreate(n);
}
inline gpuError_t gpuStreamWaitEvent(cnrtQueue_t q, cnrtNotifier_t n,
                                     unsigned int flags) {
  return cnrtQueueWaitNotifier(n, q, flags);
}
inline gpuError_t gpuMemcpyAsync(void* dst, void const* src, size_t count,
                                 cnrtMemTransDir_t kind, cnrtQueue_t q) {
  return cnrtMemcpyAsync(dst, const_cast<void*>(src), count, q, kind);
}
inline gpuError_t gpuDeviceCanAccessPeer(int* can_access, int dev, int peer) {
  unsigned int c = 0;
  cnrtRet_t r = cnrtGetPeerAccessibility(&c, dev, peer);
  *can_access = (int)c;
  return r;
}
inline gpuError_t gpuGetDeviceCount(int* count) {
  unsigned int c = 0;
  cnrtRet_t r = cnrtGetDeviceCount(&c);
  *count = (int)c;
  return r;
}
inline gpuError_t gpuMemGetAddressRange(void** base_ptr, size_t* size,
                                        void* ptr) {
  cnrtPointerAttributes_t attr;
  cnrtRet_t r = cnrtPointerGetAttributes(&attr, ptr);
  if (r == cnrtSuccess) {
    *base_ptr = attr.deviceBasePointer;
    *size = attr.size;
  }
  return r;
}
#elif !defined(__HIP_PLATFORM_AMD__)
#include <cuda.h>
#include <cuda_runtime.h>
#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#define gpuGetErrorString cudaGetErrorString
#define gpuStream_t cudaStream_t
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpuStreamLegacy cudaStreamLegacy
#define gpuStreamPerThread cudaStreamPerThread
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamDestroy cudaStreamDestroy
#define gpuLaunchHostFunc cudaLaunchHostFunc
#define gpuHostFn_t cudaHostFn_t
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
#define gpuHostMalloc cudaMallocHost  // no cudaHostMalloc API in CUDA
#define gpuHostAlloc cudaHostAlloc
#define gpuHostAllocMapped cudaHostAllocMapped
#define gpuMalloc cudaMalloc
#define gpuMallocAsync cudaMallocAsync
#define gpuMallocHost cudaMallocHost
#define gpuFree cudaFree
#define gpuFreeAsync cudaFreeAsync
#define gpuFreeHost cudaFreeHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyPeerAsync cudaMemcpyPeerAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemsetAsync cudaMemsetAsync
#define gpuGetLastError cudaGetLastError
#define gpuErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
#define gpuErrorNotReady cudaErrorNotReady
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
// DMA-BUF / GPU driver types for GPUDirect RDMA
#define gpuDriverResult_t CUresult
#define gpuDevicePtr_t CUdeviceptr
#define gpuDriverSuccess CUDA_SUCCESS
#define gpuMemRangeHandleType CUmemRangeHandleType
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
#define gpuPointerAttribute_t cudaPointerAttributes
#define gpuPointerGetAttributes cudaPointerGetAttributes
#define gpuMemoryTypeDevice cudaMemoryTypeDevice
#define GPU_DRIVER_LIB_NAME "libcuda.so.1"
#define GPU_DRIVER_LIB_NAME_FALLBACK "libcuda.so"
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME \
  "cuMemGetHandleForAddressRange"
inline gpuError_t gpuMemGetAddressRange(void** base_ptr, size_t* size,
                                        void* ptr) {
  CUdeviceptr base;
  CUresult result = cuMemGetAddressRange(&base, size, (CUdeviceptr)ptr);
  if (result == CUDA_SUCCESS) {
    *base_ptr = (void*)base;
    return gpuSuccess;
  }
  return gpuError_t(result);
}
#else
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#define gpuGetErrorString hipGetErrorString
#define gpuStream_t hipStream_t
#define gpuStreamNonBlocking hipStreamNonBlocking
// Hygon DTK lacks hipStreamLegacy; hipStreamDefault has equivalent semantics.
#ifdef __UCCL_DTK__
#define gpuStreamLegacy hipStreamDefault
#else
#define gpuStreamLegacy hipStreamLegacy
#endif
#define gpuStreamPerThread hipStreamPerThread
#define gpuStreamCreate hipStreamCreate
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamDestroy hipStreamDestroy
#define gpuLaunchHostFunc hipLaunchHostFunc
#define gpuHostFn_t hipHostFn_t
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
#define gpuMalloc hipMalloc
#define gpuMallocAsync hipMallocAsync
#define gpuMallocHost hipHostMalloc  // cudaMallocHost Deprecated in ROCm
#define gpuFree hipFree
#define gpuFreeAsync hipFreeAsync
#define gpuFreeHost hipFreeHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyPeerAsync hipMemcpyPeerAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
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
#define gpuStreamWaitEvent hipStreamWaitEvent
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDefault hipEventDefault
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuEventInterprocess hipEventInterprocess
#define gpuIpcEventHandle_t hipIpcEventHandle_t
#define gpuIpcGetEventHandle hipIpcGetEventHandle
#define gpuIpcOpenEventHandle hipIpcOpenEventHandle
#define gpuIpcCloseEventHandle(handle) (gpuSuccess)
// DMA-BUF / GPU driver types for GPUDirect RDMA
#define gpuDriverResult_t hipError_t
#define gpuDevicePtr_t hipDeviceptr_t
#define gpuDriverSuccess hipSuccess
// Hygon DTK lacks hipMemGetHandleForAddressRange (DMA-BUF export).
#ifndef __UCCL_DTK__
#define gpuMemRangeHandleType hipMemRangeHandleType
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD hipMemRangeHandleTypeDmaBufFd
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME \
  "hipMemGetHandleForAddressRange"
#else
typedef int gpuMemRangeHandleType;
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD 0
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME ""
#endif
#define gpuPointerAttribute_t hipPointerAttribute_t
#define gpuPointerGetAttributes hipPointerGetAttributes
#define gpuMemoryTypeDevice hipMemoryTypeDevice
#define GPU_DRIVER_LIB_NAME "libamdhip64.so"
#define GPU_DRIVER_LIB_NAME_FALLBACK "libamdhip64.so"
#define gpuMemGetAddressRange hipMemGetAddressRange
#endif

// Function pointer type for DMA-BUF handle export (loaded via dlsym)
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
