#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#define gpuErrorInvalidValue hipErrorInvalidValue
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
#define gpuMemoryTypeManaged hipMemoryTypeManaged
#ifdef __UCCL_DTK__
#define gpuMemTypeOf(a) (a).memoryType
#else
#define gpuMemTypeOf(a) (a).type
#endif
#define GPU_DRIVER_LIB_NAME "libamdhip64.so"
#define GPU_DRIVER_LIB_NAME_FALLBACK "libamdhip64.so"
#define gpuMemGetAddressRange hipMemGetAddressRange
