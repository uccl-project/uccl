#pragma once

// Compatibility facade shared by P2P, EP and GIN. SDK mappings live in the
// selected runtime adapter; consumers keep the existing gpu* API.
#include "util/device/backend.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>
#include UCCL_DEVICE_RUNTIME_HEADER

struct GpuIpcExport {
  gpuIpcMemHandle_t handle{};
  uintptr_t offset = 0;
  int owner_device = -1;
};

struct GpuIpcNoCache {};
struct GpuIpcHandleCache {
  std::mutex cache_mu;
  std::map<std::pair<int, uintptr_t>, gpuIpcMemHandle_t> handle_cache;
};
using GpuIpcExportContext =
    std::conditional_t<uccl::device::Backend::cache_ipc_exports,
                       GpuIpcHandleCache, GpuIpcNoCache>;

template <typename Context>
inline gpuError_t gpuIpcGetMemHandleForExport(Context* context,
                                              gpuIpcMemHandle_t* handle,
                                              void* base, int device) {
  if constexpr (std::is_same_v<Context, GpuIpcHandleCache>) {
    if (context == nullptr) return gpuErrorInvalidValue;
    std::lock_guard<std::mutex> lock(context->cache_mu);
    auto key = std::make_pair(device, reinterpret_cast<uintptr_t>(base));
    auto it = context->handle_cache.find(key);
    if (it != context->handle_cache.end()) {
      *handle = it->second;
      return gpuSuccess;
    }
    gpuError_t err = gpuIpcGetMemHandle(handle, base);
    if (err == gpuSuccess) context->handle_cache.emplace(key, *handle);
    return err;
  } else {
    return gpuIpcGetMemHandle(handle, base);
  }
}

inline gpuError_t gpuExportIpcRange(GpuIpcExportContext* context, void* ptr,
                                    size_t size, GpuIpcExport* result) {
  if (context == nullptr || ptr == nullptr || result == nullptr)
    return gpuErrorInvalidValue;

  void* base = nullptr;
  size_t allocation_size = 0;
  gpuError_t err = gpuMemGetAddressRange(&base, &allocation_size, ptr);
  if (err != gpuSuccess) return err;

  uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t allocation = reinterpret_cast<uintptr_t>(base);
  if (address < allocation || size > allocation_size ||
      address - allocation > allocation_size - size) {
    return gpuErrorInvalidValue;
  }

  GpuIpcExport exported;
  if constexpr (uccl::device::Backend::ipc_export_owner) {
    gpuPointerAttribute_t attributes{};
    err = gpuPointerGetAttributes(&attributes, ptr);
    if (err != gpuSuccess) return err;
    exported.owner_device = attributes.device;
  }
  err = gpuIpcGetMemHandleForExport(context, &exported.handle, base,
                                    exported.owner_device);
  if (err != gpuSuccess) return err;
  exported.offset = address - allocation;
  *result = exported;
  return gpuSuccess;
}

// Keep the driver ABI available to EP/GIN even when P2P cannot export DMA-BUF.
// DTK provides these types but uses an empty symbol name for runtime fallback.
#ifdef GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME
typedef gpuDriverResult_t (*gpuMemGetHandleForAddressRange_fn)(
    void*, gpuDevicePtr_t, size_t, gpuMemRangeHandleType, unsigned long long);
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
