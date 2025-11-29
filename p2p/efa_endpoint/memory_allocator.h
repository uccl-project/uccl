#pragma once

#include "define.h"
#include "rdma_context.h"

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
    cudaError_t err = cudaMalloc(&addr, size);
    if (err != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err)
                << std::endl;
      return nullptr;
    }
    return addr;
  }

  // Deallocate memory without RDMA deregistration
  void deallocateRaw(void* addr, MemoryType type) {
    if (!addr) return;

    if (type == MemoryType::GPU) {
      cudaError_t err = cudaFree(addr);
      if (err != cudaSuccess) {
        std::cerr << "cudaFree failed: " << cudaGetErrorString(err)
                  << std::endl;
      }
    } else {  // MemoryType::HOST
      free(addr);
    }
  }
};
