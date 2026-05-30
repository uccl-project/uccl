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

  void* allocate_host(size_t size);

  void* allocate_gpu(size_t size);

  // Deallocate memory without RDMA deregistration
  void deallocate_raw(void* addr, MemoryType type);
};
