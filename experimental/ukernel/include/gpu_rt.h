#pragma once

#include "util/gpu_rt.h"

// UKernel supplement layer:
// Keep a single include entry (`gpu_rt.h`) inside ukernel, while reusing
// shared definitions from `include/util/gpu_rt.h` and only adding missing ones.

#ifdef __HIP_PLATFORM_AMD__

#ifndef gpuDeviceGetAttribute
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#endif

#ifndef gpuDevAttrMultiProcessorCount
#define gpuDevAttrMultiProcessorCount hipDevAttrMultiProcessorCount
#endif

#ifndef gpuDeviceReset
#define gpuDeviceReset hipDeviceReset
#endif

#ifndef gpuMallocHost
#define gpuMallocHost hipHostMalloc
#endif

#ifndef gpuHostFree
#define gpuHostFree hipHostFree
#endif

#ifndef gpuMemset
#define gpuMemset hipMemset
#endif

#ifndef gpuEventElapsedTime
#define gpuEventElapsedTime hipEventElapsedTime
#endif

#ifndef gpuLaunchKernel
#define gpuLaunchKernel hipLaunchKernel
#endif

#ifndef gpuDeviceSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#endif

#ifndef gpuDrvResult_t
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
  return hipMemset(p, static_cast<int>(v), bytes);
}
inline char const* gpuDrvGetErrorString(gpuDrvResult_t r) {
  return hipGetErrorString(r);
}
#endif  // gpuDrvResult_t

#else

#ifndef gpuDevAttrMultiProcessorCount
#define gpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#endif

#ifndef gpuDeviceGetAttribute
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#endif

#ifndef gpuDeviceReset
#define gpuDeviceReset cudaDeviceReset
#endif

#ifndef gpuMallocHost
#define gpuMallocHost cudaMallocHost
#endif

#ifndef gpuHostFree
#define gpuHostFree cudaFreeHost
#endif

#ifndef gpuMemset
#define gpuMemset cudaMemset
#endif

#ifndef gpuEventElapsedTime
#define gpuEventElapsedTime cudaEventElapsedTime
#endif

#ifndef gpuLaunchKernel
#define gpuLaunchKernel cudaLaunchKernel
#endif

#ifndef gpuDeviceSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#endif

#ifndef gpuDrvResult_t
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
#endif  // gpuDrvResult_t

#endif  // __HIP_PLATFORM_AMD__

#ifndef GPU_DRV_CHECK
#define GPU_DRV_CHECK(call)                                                 \
  do {                                                                      \
    gpuDrvResult_t _r = (call);                                             \
    if (_r != gpuDrvSuccess) {                                              \
      fprintf(stderr, "GPU DRV error %s:%d: %s (%d)\n", __FILE__, __LINE__, \
              gpuDrvGetErrorString(_r), (int)_r);                           \
      std::abort();                                                         \
    }                                                                       \
  } while (0)
#endif
