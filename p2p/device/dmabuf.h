#pragma once
#include "util/debug.h"
#include "util/gpu_rt.h"
#include <dlfcn.h>
#include <unistd.h>

namespace uccl::device {
// Owns the exported handle until the consumer has registered it. Export
// support is a build capability; driver/device support is checked by open().
class DmaBuf {
 public:
  DmaBuf() = default;
  DmaBuf(DmaBuf const&) = delete;
  DmaBuf& operator=(DmaBuf const&) = delete;
  ~DmaBuf() {
    if (fd >= 0) close(fd);
  }
  bool open(void* addr, size_t size) {
#if UCCL_DEVICE_HAS_DMABUF
    // Allocation export granularity belongs to the device adapter.
    static constexpr size_t kDmabufGranularity = Backend::dmabuf_granularity;

    static gpuMemGetHandleForAddressRange_fn gpuGetHandleForRange_func =
        nullptr;
    static std::once_flag init_flag;

    std::call_once(init_flag, []() {
      void* handle = dlopen(GPU_DRIVER_LIB_NAME, RTLD_LAZY);
      if (!handle) handle = dlopen(GPU_DRIVER_LIB_NAME_FALLBACK, RTLD_LAZY);
      if (handle) {
        gpuGetHandleForRange_func = (gpuMemGetHandleForAddressRange_fn)dlsym(
            handle, GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME);
      }
    });

    if (!gpuGetHandleForRange_func) {
      UCCL_LOG(ERROR) << "GPU Driver API not available for DMA-BUF";
      return false;
    }

    // The DMA-BUF export API requires the address to be at the start of a
    // GPU allocation and the size aligned to GPU page granularity. The caller
    // may pass a sub-region, so we find the allocation base and compute offset.
    void* alloc_base_ptr = nullptr;
    size_t alloc_size = 0;
    gpuError_t gpu_err =
        gpuMemGetAddressRange(&alloc_base_ptr, &alloc_size, addr);
    if (gpu_err != gpuSuccess) {
      UCCL_LOG(ERROR) << "gpuMemGetAddressRange failed ("
                      << gpuGetErrorString(gpu_err) << ")";
      return false;
    }

    gpuDevicePtr_t alloc_base = (gpuDevicePtr_t)(uintptr_t)alloc_base_ptr;
    offset = (uintptr_t)addr - (uintptr_t)alloc_base;

    // Export the entire allocation as a DMA-BUF fd (aligned to granularity).
    size_t export_size =
        ((alloc_size + kDmabufGranularity - 1) / kDmabufGranularity) *
        kDmabufGranularity;

    UCCL_LOG(INFO) << "DMA-BUF: gpu_buf=" << addr << " bytes=" << size
                   << " alloc_base=0x" << std::hex << (uintptr_t)alloc_base
                   << std::dec << " alloc_size=" << alloc_size
                   << " offset=" << offset << " export_size=" << export_size;

    // Export DMA-BUF fd from the allocation base.

    gpuDriverResult_t drv_err = gpuGetHandleForRange_func(
        &fd, alloc_base, export_size, GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);

    if (drv_err != gpuDriverSuccess) {
      UCCL_LOG(ERROR) << "gpuMemGetHandleForAddressRange failed (error="
                      << (int)drv_err << "), DMA-BUF unavailable";
      return false;
    }

    return true;
#else
    UCCL_LOG(ERROR) << Backend::name
                    << " DMA-BUF export is not enabled; "
                       "use a validated peer-memory NIC";
    return false;
#endif
  }
  int fd = -1;
  size_t offset = 0;
};
}  // namespace uccl::device
