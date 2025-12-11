#pragma once

#include "define.h"
#include "rdma_context.h"

// GPU runtime support
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#elif defined(__CUDACC__) || defined(__NVCC__)
#include <cuda_runtime.h>
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#endif

class MemoryAllocator {
 public:
  explicit MemoryAllocator() : page_size_(sysconf(_SC_PAGESIZE)) {
    if (page_size_ <= 0) {
      page_size_ = 4096;  // fallback to 4KB
    }
  }

  ~MemoryAllocator() = default;

  std::shared_ptr<RegMemBlock> allocate(
      size_t size, MemoryType type,
      const std::shared_ptr<RdmaContext> ctx = nullptr) {
    void* addr = nullptr;
    struct ibv_mr* mr = nullptr;

    if (type == MemoryType::GPU) {
      addr = allocateGPU(size);
    } else {  // MemoryType::HOST
      addr = allocateHost(size);
    }

    if (!addr) {
      throw std::runtime_error("Failed to allocate memory");
    }

    if (ctx) {
      mr = ctx->regMem(addr, size);

      if (!mr) {
        deallocateRaw(addr, type);
        throw std::runtime_error("Failed to register memory with RDMA");
      }
    }

    // Create RegMemBlock with custom deleter
    auto deleter = [this, type](RegMemBlock* block) {
      if (block) {
        std::cout << "memory freed" << std::endl;
        if (block->mr) RdmaContext::deregMem(block->mr);
        if (block->addr) deallocateRaw(block->addr, type);
        delete block;
      }
    };

    auto block = new RegMemBlock(addr, size, type, mr);
    return std::shared_ptr<RegMemBlock>(block, deleter);
  }

 private:
  long page_size_;

  void* allocateHost(size_t size) {
    void* addr = nullptr;
    int ret = posix_memalign(&addr, page_size_, size);
    if (ret != 0) {
      std::cerr << "posix_memalign failed with error: " << ret << std::endl;
      return nullptr;
    }
    memset(addr, 0, size);  // Initialize memory to zero
    return addr;
  }
  void* allocateGPU(size_t size) {
    void* addr = nullptr;
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || \
    defined(__CUDACC__) || defined(__NVCC__)
    gpuError_t err = gpuMalloc(&addr, size);
    if (err != gpuSuccess) {
      std::cerr << "gpuMalloc failed: " << gpuGetErrorString(err) << std::endl;
      return nullptr;
    }
#else
    std::cerr << "GPU support not available (neither CUDA nor HIP)"
              << std::endl;
    return nullptr;
#endif
    return addr;
  }

  // Deallocate memory without RDMA deregistration
  void deallocateRaw(void* addr, MemoryType type) {
    if (!addr) return;

    if (type == MemoryType::GPU) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || \
    defined(__CUDACC__) || defined(__NVCC__)
      gpuError_t err = gpuFree(addr);
      if (err != gpuSuccess) {
        std::cerr << "gpuFree failed: " << gpuGetErrorString(err) << std::endl;
      }
#else
      std::cerr << "GPU support not available (neither CUDA nor HIP)"
                << std::endl;
#endif
    } else {  // MemoryType::HOST
      free(addr);
    }
  }
};
