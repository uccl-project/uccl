#pragma once
#include "signal.hpp"
#include <cuda_runtime.h>
namespace uccl_lite::cpu_router {
struct DeviceSignalRef {
  GenerationSignal* word;
  __device__ Generation acquire() const {
    Generation g;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(g)
                 : "l"(&word->value)
                 : "memory");
    return g;
  }
  __device__ void publish(Generation g) const {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(&word->value),
                 "r"(g)
                 : "memory");
  }
  __device__ void wait(Generation g) const {
    while (acquire() < g) __nanosleep(64);
  }
};
struct DevicePacketOwnership {
  DeviceSignalRef ready, released;
  GenerationSignal* readers;
  __device__ void acquire(Generation previous) const {
    if (previous) released.wait(previous);
  }
  __device__ void publish(Generation g) const { ready.publish(g); }
  __device__ void consume(unsigned reader, Generation g) const {
    DeviceSignalRef{readers + reader}.publish(g);
  }
};
// Device-side channel operations compose with signaling and caller CTA
// barriers.
struct DeviceChannel {
  struct alignas(32) Vector32 {
    uint4 lo, hi;
  };
  // Packet callers check 32-byte alignment and retain ownership until readers
  // finish; control signals remain independent 32-bit system-scope accesses.
  __device__ static Vector32 get32(void const* p) {
    Vector32 v;
#if __CUDA_ARCH__ >= 1000 &&      \
    (__CUDACC_VER_MAJOR__ > 12 || \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
    asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(v.lo.x), "=r"(v.lo.y), "=r"(v.lo.z), "=r"(v.lo.w),
                   "=r"(v.hi.x), "=r"(v.hi.y), "=r"(v.hi.z), "=r"(v.hi.w)
                 : "l"(p)
                 : "memory");
#else
    v.lo = get16(p);
    v.hi = get16(static_cast<char const*>(p) + 16);
#endif
    return v;
  }
  __device__ static void put32(void* p, Vector32 v) {
#if __CUDA_ARCH__ >= 1000 &&      \
    (__CUDACC_VER_MAJOR__ > 12 || \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
    asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(p),
                 "r"(v.lo.x), "r"(v.lo.y), "r"(v.lo.z), "r"(v.lo.w),
                 "r"(v.hi.x), "r"(v.hi.y), "r"(v.hi.z), "r"(v.hi.w)
                 : "memory");
#else
    put16(p, v.lo);
    put16(static_cast<char*>(p) + 16, v.hi);
#endif
  }
  __device__ static uint4 get16(void const* p) {
    return *static_cast<uint4 const*>(p);
  }
  __device__ static uint16_t get2(void const* p) {
    uint16_t v;
    asm volatile("ld.global.u16 %0,[%1];" : "=h"(v) : "l"(p) : "memory");
    return v;
  }
  __device__ static void put16(void* p, uint4 value) {
    *static_cast<uint4*>(p) = value;
  }
  __device__ static void put2(void* p, uint16_t value) {
    asm volatile("st.global.u16 [%0],%1;" ::"l"(p), "h"(value) : "memory");
  }
  __device__ static void put(void* dst, void const* src, size_t bytes,
                             size_t lane, size_t stride) {
    if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) &
         15) == 0) {
      for (size_t i = lane; i < bytes / 16; i += stride)
        put16(static_cast<char*>(dst) + i * 16,
              get16(static_cast<char const*>(src) + i * 16));
      for (size_t i = bytes / 16 * 16 + lane; i < bytes; i += stride)
        static_cast<char*>(dst)[i] = static_cast<char const*>(src)[i];
    } else
      for (size_t i = lane; i < bytes; i += stride)
        static_cast<char*>(dst)[i] = static_cast<char const*>(src)[i];
  }
  __device__ static void get(void* dst, void const* src, size_t bytes,
                             size_t lane, size_t stride) {
    put(dst, src, bytes, lane, stride);
  }
};
}  // namespace uccl_lite::cpu_router
