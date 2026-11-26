#include "memory.hpp"
#include <linux/mempolicy.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite::cpu_router {
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct MemoryRegistration {
  void* address = nullptr;
  size_t bytes = 0;
  MemoryKind kind = MemoryKind::Gpu;
  int device = -1, numa = -1;
  bool owned = false, registered = false;
  cudaStream_t reclaim = nullptr;
  ~MemoryRegistration() {
    if (registered) cudaHostUnregister(address);
    if (owned && address) {
      if (kind == MemoryKind::Gpu) {
        cudaSetDevice(device);
        cudaFreeAsync(address, reclaim);
        cudaStreamDestroy(reclaim);
      } else
        munmap(address, bytes);
    }
  }
};
MemorySpan MemorySpan::subspan(size_t offset, size_t length) const {
  if (offset > bytes || length > bytes - offset)
    throw std::out_of_range("memory span bounds");
  auto s = *this;
  s.address = static_cast<char*>(address) + offset;
  s.bytes = length;
  return s;
}
void* MemorySpan::gpuAddress(int accessingDevice) const {
  if (kind == MemoryKind::NumaHost)
    throw std::invalid_argument("CPU scratch is not GPU registered");
  if (kind == MemoryKind::Gpu) {
    if (device != accessingDevice) {
      int peer;
      check(cudaDeviceCanAccessPeer(&peer, accessingDevice, device));
      if (!peer)
        throw std::invalid_argument(
            "direct GPU channel needs peer access; route through host spans");
      check(cudaSetDevice(accessingDevice));
      auto e = cudaDeviceEnablePeerAccess(device, 0);
      if (e == cudaErrorPeerAccessAlreadyEnabled)
        cudaGetLastError();
      else
        check(e);
    }
    return address;
  }
  check(cudaSetDevice(accessingDevice));
  void* mapped;
  check(cudaHostGetDevicePointer(&mapped, address, 0));
  return mapped;
}
MemorySpan MemorySpan::borrowedGpu(void* p, size_t n, int d) {
  return {p, n, MemoryKind::Gpu, d, -1, {}};
}
MemorySpan MemorySpan::borrowedPinned(void* p, size_t n, int node) {
  return {p, n, MemoryKind::NumaPinned, -1, node, {}};
}
void RegisteredMemory::allocate(size_t n, int node, bool pinned) {
  if (!n || node < 0 || node >= 64)
    throw std::invalid_argument("invalid NUMA allocation");
  auto r = std::make_shared<MemoryRegistration>();
  r->bytes = (n + 4095) / 4096 * 4096;
  r->kind = pinned ? MemoryKind::NumaPinned : MemoryKind::NumaHost;
  r->numa = node;
  r->owned = true;
  r->address = mmap(nullptr, r->bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (r->address == MAP_FAILED) {
    r->address = nullptr;
    throw std::runtime_error("NUMA mmap failed");
  }
  unsigned long mask = 1UL << node;
  if (syscall(SYS_mbind, r->address, r->bytes, MPOL_BIND, &mask, 64, 0))
    throw std::runtime_error("NUMA mbind failed");
  std::memset(r->address, 0, r->bytes);
  if (pinned) {
    check(cudaHostRegister(r->address, r->bytes,
                           cudaHostRegisterPortable | cudaHostRegisterMapped));
    r->registered = true;
  }
  registration_ = std::move(r);
}
RegisteredMemory RegisteredMemory::gpu(size_t n, int d) {
  if (!n) throw std::invalid_argument("empty GPU allocation");
  RegisteredMemory m;
  auto r = std::make_shared<MemoryRegistration>();
  r->device = d;
  r->bytes = n;
  r->owned = true;
  check(cudaSetDevice(d));
  check(cudaStreamCreateWithFlags(&r->reclaim, cudaStreamNonBlocking));
  check(cudaMalloc(&r->address, n));
  m.registration_ = std::move(r);
  return m;
}
RegisteredMemory RegisteredMemory::pinned(size_t n, int node) {
  RegisteredMemory m;
  m.allocate(n, node);
  return m;
}
RegisteredMemory RegisteredMemory::registerPinned(void* p, size_t n, int node) {
  if (!p || !n) throw std::invalid_argument("empty host registration");
  RegisteredMemory m;
  auto r = std::make_shared<MemoryRegistration>();
  r->address = p;
  r->bytes = n;
  r->numa = node;
  r->kind = MemoryKind::NumaPinned;
  check(cudaHostRegister(p, n,
                         cudaHostRegisterPortable | cudaHostRegisterMapped));
  r->registered = true;
  m.registration_ = std::move(r);
  return m;
}
RegisteredMemory RegisteredMemory::registerGpu(void* p, size_t n, int d) {
  cudaPointerAttributes a{};
  check(cudaPointerGetAttributes(&a, p));
  if (a.type != cudaMemoryTypeDevice || a.device != d || !n)
    throw std::invalid_argument("invalid GPU registration");
  RegisteredMemory m;
  auto r = std::make_shared<MemoryRegistration>();
  r->address = p;
  r->bytes = n;
  r->device = d;
  m.registration_ = std::move(r);
  return m;
}
void* RegisteredMemory::data() const {
  return registration_ ? registration_->address : nullptr;
}
size_t RegisteredMemory::bytes() const {
  return registration_ ? registration_->bytes : 0;
}
MemorySpan RegisteredMemory::span() const {
  if (!registration_) return {};
  auto& r = *registration_;
  return {r.address, r.bytes, r.kind, r.device, r.numa, registration_};
}
}  // namespace uccl_lite::cpu_router
