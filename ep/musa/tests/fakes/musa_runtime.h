#pragma once

#include "mpi.h"
#include <atomic>
#include <cstdlib>
#include <cstring>

using musaError_t = int;
using musaStream_t = void*;
using musaEvent_t = void*;
constexpr int musaSuccess = 0, musaErrorInvalidValue = 1;
constexpr int musaMemcpyHostToDevice = 1, musaMemcpyDeviceToDevice = 2;
namespace fake {
inline std::atomic<int> fail_alloc{-1}, allocations{0}, live_allocations{0};
inline std::atomic<int> waits{0}, syncs{0};
}  // namespace fake
inline int musaMalloc(void** p, size_t bytes) {
  int expected = fake::rank;
  if (fake::fail_alloc.compare_exchange_strong(expected, -1)) {
    *p = nullptr;
    return 2;
  }
  *p = std::malloc(bytes);
  if (!*p) return 2;
  ++fake::allocations;
  ++fake::live_allocations;
  return musaSuccess;
}
inline int musaFree(void* p) {
  if (p) {
    --fake::live_allocations;
    std::free(p);
  }
  return musaSuccess;
}
inline int musaStreamCreate(musaStream_t* p) {
  *p = reinterpret_cast<void*>(1);
  return musaSuccess;
}
inline int musaStreamDestroy(musaStream_t) { return musaSuccess; }
inline int musaStreamSynchronize(musaStream_t) {
  ++fake::syncs;
  return musaSuccess;
}
inline int musaStreamWaitEvent(musaStream_t, musaEvent_t e, unsigned) {
  assert(e);
  ++fake::waits;
  return musaSuccess;
}
inline int musaMemcpyAsync(void* dst, void const* src, size_t bytes, int,
                           musaStream_t) {
  if (bytes) std::memmove(dst, src, bytes);
  return musaSuccess;
}
