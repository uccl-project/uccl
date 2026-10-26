#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#define gpuErrorInvalidValue cudaErrorInvalidValue
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
#define gpuMemoryTypeManaged cudaMemoryTypeManaged
#define gpuMemTypeOf(a) (a).type
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
