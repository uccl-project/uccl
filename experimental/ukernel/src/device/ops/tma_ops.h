#pragma once

#include "reduce_ops.h"
#include "task.h"

namespace UKernel {
namespace Device {

__device__ __forceinline__ bool is_tma_supported() {
#if __CUDA_ARCH__ >= 900
  return true;
#else
  return false;
#endif
}

struct TmaSemaphore {
  uint64_t expect_bytes;
  uint32_t phase;
  uint32_t padding[5];
};

__device__ __forceinline__ void tma_init_semaphore(TmaSemaphore& sem,
                                                   uint32_t initial_phase) {
  sem.expect_bytes = 0;
  sem.phase = initial_phase;
}

#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void tma_expect_bytes(TmaSemaphore& sem,
                                                 uint32_t bytes) {
  uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(sem_ptr),
      "r"(bytes));
}

__device__ __forceinline__ void tma_arrive(TmaSemaphore& sem,
                                           uint32_t count = 1) {
  uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"r"(sem_ptr),
               "r"(count)
               : "memory");
}

__device__ __forceinline__ void tma_wait(TmaSemaphore& sem, int phase) {
  uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
  asm volatile(
      "{ .reg .pred P1; LAB_WAIT: "
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1; "
      "@P1 bra.uni DONE; bra.uni LAB_WAIT; DONE: }" ::"r"(sem_ptr),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void tma_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}
template <int N = 0>
__device__ __forceinline__ void tma_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void tma_fence_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void tma_fence() { __threadfence(); }

template <typename T>
__device__ void tma_load(void* smem_dst, void const* gmem_src, uint64_t bytes,
                         TmaSemaphore& sem) {
  uint32_t size_bytes = static_cast<uint32_t>(bytes);
  tma_expect_bytes(sem, size_bytes);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];\n" ::"r"(dst_ptr),
      "l"(gmem_src), "r"(size_bytes), "r"(sem_ptr)
      : "memory");
}

template <typename T>
__device__ void tma_store(void* gmem_dst, void const* smem_src,
                          uint64_t bytes) {
  uint32_t size_bytes = static_cast<uint32_t>(bytes);
  tma_fence_async();
  uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_src));
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(
          gmem_dst),
      "r"(src_ptr), "r"(size_bytes)
      : "memory");
  tma_commit_group();
}
#else
__device__ __forceinline__ void tma_expect_bytes(TmaSemaphore& sem,
                                                 uint32_t bytes) {}
__device__ __forceinline__ void tma_arrive(TmaSemaphore& sem,
                                           uint32_t count = 1) {}
__device__ __forceinline__ void tma_wait(TmaSemaphore& sem, int phase) {}
__device__ __forceinline__ void tma_commit_group() {}
template <int N = 0>
__device__ __forceinline__ void tma_wait_group() {}
__device__ __forceinline__ void tma_fence_async() {}
__device__ __forceinline__ void tma_fence() {}

template <typename T>
__device__ void tma_load(void* smem_dst, void const* gmem_src, uint64_t bytes,
                         TmaSemaphore& sem) {}

template <typename T>
__device__ void tma_store(void* gmem_dst, void const* smem_src,
                          uint64_t bytes) {}
#endif

}  // namespace Device
}  // namespace UKernel