#pragma once

// Cambricon MLU: map the gpu* abstraction onto CNRT (cnrt*) / CNDRV (cn*);
// direct macros where signatures match, inline wrappers where they differ.
#include <climits>  // CNRT headers do not pull in PATH_MAX like cuda_runtime.h
#include <map>
#include <mutex>
#include <utility>
#include <cn_api.h>
#include <cnrt.h>
#define gpuSuccess cnrtSuccess
#define gpuError_t cnrtRet_t
#define gpuErrorInvalidValue cnrtErrorArgsInvalid
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
// CNRT has no managed memory type; alias to device so the check degenerates.
#define gpuMemoryTypeManaged cnrtMemTypeDevice
#define gpuMemTypeOf(a) (a).type
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
