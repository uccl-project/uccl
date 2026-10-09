#pragma once

#include <cstdint>
#include <musa.h>
#include <musa_runtime.h>

#define gpuSuccess musaSuccess
#define gpuError_t musaError_t
#define gpuGetErrorString musaGetErrorString
#define gpuGetLastError musaGetLastError
#define gpuErrorNotReady musaErrorNotReady
#define gpuErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
#define gpuStream_t musaStream_t
#define gpuStreamNonBlocking musaStreamNonBlocking
#define gpuStreamLegacy musaStreamLegacy
#define gpuStreamPerThread musaStreamPerThread
#define gpuStreamCreate musaStreamCreate
#define gpuStreamCreateWithFlags musaStreamCreateWithFlags
#define gpuStreamSynchronize musaStreamSynchronize
#define gpuStreamDestroy musaStreamDestroy
#define gpuStreamWaitEvent musaStreamWaitEvent
#define gpuLaunchHostFunc musaLaunchHostFunc
#define gpuHostFn_t musaHostFn_t
#define gpuDeviceProp musaDeviceProp
#define gpuSetDevice musaSetDevice
#define gpuDeviceMapHost musaDeviceMapHost
#define gpuSetDeviceFlags musaSetDeviceFlags
#define gpuGetDevice musaGetDevice
#define gpuGetDeviceCount musaGetDeviceCount
#define gpuGetDeviceProperties musaGetDeviceProperties
#define gpuDeviceGetPCIBusId musaDeviceGetPCIBusId
#define gpuDeviceCanAccessPeer musaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#define gpuIpcMemHandle_t musaIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess musaIpcMemLazyEnablePeerAccess
#define gpuIpcGetMemHandle musaIpcGetMemHandle
#define gpuIpcOpenMemHandle musaIpcOpenMemHandle
#define gpuIpcCloseMemHandle musaIpcCloseMemHandle
#define gpuHostMalloc musaMallocHost
#define gpuHostAlloc musaHostAlloc
#define gpuHostAllocMapped musaHostAllocMapped
#define gpuMalloc musaMalloc
#define gpuMallocAsync musaMallocAsync
#define gpuMallocHost musaMallocHost
#define gpuFree musaFree
#define gpuFreeAsync musaFreeAsync
#define gpuFreeHost musaFreeHost
#define gpuMemcpyHostToDevice musaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost musaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define gpuMemcpy musaMemcpy
#define gpuMemcpyAsync musaMemcpyAsync
#define gpuMemcpyPeerAsync musaMemcpyPeerAsync
#define gpuMemcpyFromSymbol musaMemcpyFromSymbol
#define gpuMemsetAsync musaMemsetAsync
#define gpuEvent_t musaEvent_t
#define gpuEventCreate musaEventCreate
#define gpuEventCreateWithFlags musaEventCreateWithFlags
#define gpuEventDestroy musaEventDestroy
#define gpuEventRecord musaEventRecord
#define gpuEventQuery musaEventQuery
#define gpuEventSynchronize musaEventSynchronize
#define gpuEventDefault musaEventDefault
#define gpuEventDisableTiming musaEventDisableTiming
#define gpuEventInterprocess musaEventInterprocess
#define gpuIpcEventHandle_t musaIpcEventHandle_t
#define gpuIpcGetEventHandle musaIpcGetEventHandle
#define gpuIpcOpenEventHandle musaIpcOpenEventHandle
#define gpuIpcCloseEventHandle musaEventDestroy
#define gpuPointerAttribute_t musaPointerAttributes
#define gpuPointerGetAttributes musaPointerGetAttributes
#define gpuMemoryTypeDevice musaMemoryTypeDevice
#define gpuMemoryTypeManaged musaMemoryTypeManaged
#define gpuMemTypeOf(a) (a).type

inline gpuError_t gpuMemGetAddressRange(void** base_ptr, size_t* size,
                                        void* ptr) {
  if (!base_ptr || !size) return musaErrorInvalidValue;
  MUdeviceptr base = 0;
  size_t bytes = 0;
  MUresult result =
      muMemGetAddressRange(&base, &bytes, reinterpret_cast<MUdeviceptr>(ptr));
  if (result != MUSA_SUCCESS) return musaErrorInvalidValue;
  *base_ptr = reinterpret_cast<void*>(base);
  *size = bytes;
  return gpuSuccess;
}

// IPC exports allocations, not arbitrary aligned addresses or tensor views.
inline gpuError_t gpuExportIpcRange(gpuIpcMemHandle_t* handle,
                                    uintptr_t* offset, void* ptr, size_t size) {
  if (!handle || !offset || !ptr) return musaErrorInvalidValue;
  void* base = nullptr;
  size_t allocation_size = 0;
  auto err = gpuMemGetAddressRange(&base, &allocation_size, ptr);
  if (err != gpuSuccess) return err;
  auto address = reinterpret_cast<uintptr_t>(ptr);
  auto allocation = reinterpret_cast<uintptr_t>(base);
  if (address < allocation || size > allocation_size ||
      address - allocation > allocation_size - size)
    return musaErrorInvalidValue;
  gpuIpcMemHandle_t exported{};
  err = gpuIpcGetMemHandle(&exported, base);
  if (err != gpuSuccess) return err;
  *handle = exported;
  *offset = address - allocation;
  return gpuSuccess;
}
