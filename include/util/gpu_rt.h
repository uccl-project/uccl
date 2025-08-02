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
#define gpuGetLastError cudaGetLastError
#define gpuErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
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
#define gpuGetLastError hipGetLastError
#define gpuErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
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
