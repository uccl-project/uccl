#include "memory_allocator.h"
#include "util/gpu_rt.h"

MemoryAllocator::MemoryAllocator() : page_size_(sysconf(_SC_PAGESIZE)) {
  if (page_size_ <= 0) {
    page_size_ = 4096;  // fallback to 4KB
  }
}

MemoryAllocator::~MemoryAllocator() = default;

std::shared_ptr<RegMemBlock> MemoryAllocator::allocate(
    size_t size, MemoryType type, std::shared_ptr<RdmaContext> const ctx) {
  void* addr = nullptr;
  struct ibv_mr* mr = nullptr;

  if (type == MemoryType::GPU) {
    addr = allocate_gpu(size);
  } else {  // MemoryType::HOST
    addr = allocate_host(size);
  }

  if (!addr) {
    throw std::runtime_error("Failed to allocate memory");
  }

  // Create RegMemBlock with custom deleter
  auto deleter = [this, type](RegMemBlock* block) {
    if (block) {
      if (block->addr) deallocate_raw(block->addr, type);
      delete block;
    }
  };

  auto block = new RegMemBlock(addr, size, type);
  if (ctx) {
    mr = ctx->reg_mem(addr, size);

    if (!mr) {
      deallocate_raw(addr, type);
      throw std::runtime_error("Failed to register memory with RDMA");
    }
    block->set_mr_by_context_id(ctx->get_context_id(), mr);
  }

  return std::shared_ptr<RegMemBlock>(block, deleter);
}

void* MemoryAllocator::allocate_host(size_t size) {
  void* addr = nullptr;
  int ret = posix_memalign(&addr, page_size_, size);
  if (ret != 0) {
    std::cerr << "posix_memalign failed with error: " << ret << std::endl;
    return nullptr;
  }
  memset(addr, 0, size);  // Initialize memory to zero
  return addr;
}

void* MemoryAllocator::allocate_gpu(size_t size) {
  void* addr = nullptr;
  gpuError_t err = gpuMalloc(&addr, size);
  if (err != gpuSuccess) {
    std::cerr << "gpuMalloc failed: " << gpuGetErrorString(err) << std::endl;
    return nullptr;
  }
  return addr;
}

void MemoryAllocator::deallocate_raw(void* addr, MemoryType type) {
  if (!addr) return;

  if (type == MemoryType::GPU) {
    gpuError_t err = gpuFree(addr);
    if (err != gpuSuccess) {
      std::cerr << "gpuFree failed: " << gpuGetErrorString(err) << std::endl;
    }
  } else {  // MemoryType::HOST
    free(addr);
  }
}
