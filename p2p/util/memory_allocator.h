#pragma once

#include "common.h"
#include "rdma_context.h"
#include "util/gpu_rt.h"

class MemoryAllocator {
 public:
  explicit MemoryAllocator();

  ~MemoryAllocator();

  std::shared_ptr<RegMemBlock> allocate(
      size_t size, MemoryType type,
      std::shared_ptr<RdmaContext> const ctx = nullptr);

 private:
  long page_size_;

  void* allocateHost(size_t size);

  void* allocateGPU(size_t size);

  // Deallocate memory without RDMA deregistration
  void deallocateRaw(void* addr, MemoryType type);
};
