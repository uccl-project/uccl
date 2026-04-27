#pragma once

#include <cuda.h>
#include <deep_ep/common/exception.cuh>

#include "lazy_driver.hpp"

namespace deep_ep::shared_memory {

union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;
    CUmemFabricHandle cu_mem_fabric_handle;
};

struct MemHandle {
    MemHandleInner inner;
    size_t size;
};

static void cu_mem_set_access_all(void* ptr, size_t size) {
    int device_count;
    CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&device_count));

    constexpr int kMaxDeviceCount = 8;
    EP_HOST_ASSERT(0 < device_count and device_count <= kMaxDeviceCount);

    CUmemAccessDesc access_desc[kMaxDeviceCount];
    for (int i = 0; i < device_count; ++ i) {
        access_desc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[i].location.id = i;
        access_desc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(reinterpret_cast<CUdeviceptr>(ptr), size, access_desc, device_count));
}

static void cu_mem_free(void* ptr) {
    CUmemGenericAllocationHandle handle;
    CUDA_DRIVER_CHECK(lazy_cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CUDA_DRIVER_CHECK(lazy_cuMemGetAddressRange_v2(nullptr, &size, reinterpret_cast<CUdeviceptr>(ptr)));

    CUDA_DRIVER_CHECK(lazy_cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size));
    CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size));
    CUDA_DRIVER_CHECK(lazy_cuMemRelease(handle));
}

class SharedMemoryAllocator {
public:
    explicit SharedMemoryAllocator(const bool& use_fabric) : use_fabric(use_fabric) {}

    void malloc(void** ptr, size_t size) const {
        if (use_fabric) {
            CUdevice device;
            CUDA_DRIVER_CHECK(lazy_cuCtxGetDevice(&device));

            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
            prop.location.id = device;

            size_t alignment = 0;
            EP_HOST_ASSERT(size > 0);
            CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&alignment, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
            size = ((size + alignment - 1) / alignment) * alignment;

            CUmemGenericAllocationHandle handle;
            CUDA_DRIVER_CHECK(lazy_cuMemCreate(&handle, size, &prop, 0));
            CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(ptr), size, alignment, 0, 0));
            CUDA_DRIVER_CHECK(lazy_cuMemMap(reinterpret_cast<CUdeviceptr>(*ptr), size, 0, handle, 0));
            cu_mem_set_access_all(*ptr, size);
        } else {
            CUDA_RUNTIME_CHECK(cudaMalloc(ptr, size));
        }
    }

    void free(void* ptr) const {
        if (use_fabric) {
            cu_mem_free(ptr);
        } else {
            CUDA_RUNTIME_CHECK(cudaFree(ptr));
        }
    }

    void get_mem_handle(MemHandle* mem_handle, void* ptr) const {
        size_t size = 0;
        CUDA_DRIVER_CHECK(lazy_cuMemGetAddressRange_v2(nullptr, &size, reinterpret_cast<CUdeviceptr>(ptr)));
        mem_handle->size = size;

        if (use_fabric) {
            CUmemGenericAllocationHandle handle;
            CUDA_DRIVER_CHECK(lazy_cuMemRetainAllocationHandle(&handle, ptr));
            CUDA_DRIVER_CHECK(lazy_cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        } else {
            CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
        }
    }

    void open_mem_handle(void** ptr, MemHandle* mem_handle) const {
        if (use_fabric) {
            size_t size = mem_handle->size;

            CUmemGenericAllocationHandle handle;
            CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

            CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(ptr), size, 0, 0, 0));
            CUDA_DRIVER_CHECK(lazy_cuMemMap(reinterpret_cast<CUdeviceptr>(*ptr), size, 0, handle, 0));
            cu_mem_set_access_all(*ptr, size);
        } else {
            CUDA_RUNTIME_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
        }
    }

    void close_mem_handle(void* ptr) const {
        if (use_fabric) {
            cu_mem_free(ptr);
        } else {
            CUDA_RUNTIME_CHECK(cudaIpcCloseMemHandle(ptr));
        }
    }

private:
    bool use_fabric;
};

}  // namespace deep_ep::shared_memory
