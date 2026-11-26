#pragma once
#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

namespace uccl_lite::cpu_router {
enum class MemoryKind { Gpu, NumaPinned, NumaHost };
struct MemoryRegistration;
struct MemorySpan {
  void* address = nullptr;
  size_t bytes = 0;
  MemoryKind kind = MemoryKind::Gpu;
  int device = -1;
  int numa = -1;
  std::shared_ptr<MemoryRegistration> registration;
  MemorySpan subspan(size_t offset, size_t length) const;
  void* gpuAddress(int accessingDevice) const;
  static MemorySpan borrowedGpu(void* address, size_t bytes, int device);
  static MemorySpan borrowedPinned(void* address, size_t bytes, int numa);
};
class RegisteredMemory {
 public:
  static RegisteredMemory gpu(size_t bytes, int device);
  static RegisteredMemory pinned(size_t bytes, int numa);
  static RegisteredMemory registerPinned(void* address, size_t bytes, int numa);
  static RegisteredMemory registerGpu(void* address, size_t bytes, int device);
  // NumaHost is CPU-only scratch; pinned is mapped and CUDA-portable.
  void allocate(size_t bytes, int numa, bool pinned = true);
  void* data() const;
  size_t bytes() const;
  MemorySpan span() const;

 private:
  std::shared_ptr<MemoryRegistration> registration_;
};
}  // namespace uccl_lite::cpu_router
