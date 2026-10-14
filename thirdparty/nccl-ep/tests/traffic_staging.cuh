// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "device/ht_scheduler.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace nccl_ep::hybridep::traffic {

#if defined(HT_TRAFFIC_TMEM)
constexpr bool tensor_compiled = true;
#else
constexpr bool tensor_compiled = false;
#endif

enum class Path { Copy, Direct, Tma, TensorInput, TensorOutput };

inline char const* name(Path path) {
  switch (path) {
    case Path::Copy: return "copy";
    case Path::Direct: return "direct";
    case Path::Tma: return "tma";
    case Path::TensorInput: return "tmem-input";
    case Path::TensorOutput: return "tmem-output";
  }
  return "invalid";
}

template <Path P>
constexpr bool tensor = P == Path::TensorInput;

__device__ inline void wait_barrier(uint32_t barrier, uint32_t parity) {
  asm volatile("{ .reg .pred done; wait_loop: "
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;"
               "@!done bra wait_loop; }" :: "r"(barrier), "r"(parity) : "memory");
}

__device__ inline uint4 load_words(uint32_t const* source, std::size_t at,
                                  std::size_t words, bool aligned) {
  uint4 value{};
  if (aligned && at + 4 <= words) {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(source + at) : "memory");
  } else {
    auto src = reinterpret_cast<uint32_t const volatile*>(source);
    value.x = at < words ? src[at] : 0;
    value.y = at + 1 < words ? src[at + 1] : 0;
    value.z = at + 2 < words ? src[at + 2] : 0;
    value.w = at + 3 < words ? src[at + 3] : 0;
  }
  return value;
}

__device__ inline void store_words(uint32_t* destination, std::size_t at,
                                   std::size_t words, uint4 value, bool aligned) {
  if (aligned && at + 4 <= words) {
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(destination + at), "r"(value.x), "r"(value.y),
                    "r"(value.z), "r"(value.w) : "memory");
  } else {
    auto dst = reinterpret_cast<uint32_t volatile*>(destination);
    if (at < words) dst[at] = value.x;
    if (at + 1 < words) dst[at + 1] = value.y;
    if (at + 2 < words) dst[at + 2] = value.z;
    if (at + 3 < words) dst[at + 3] = value.w;
  }
}

// 128 threads own their warp's 32 TMEM rows. Allocation spans the whole
// grid-stride loop, rather than allocating and freeing for every 2 KiB tile.
template <Path P>
__global__ void copy(uint32_t const* source, uint32_t* destination,
                     std::size_t words, int passes) {
#if !defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || __CUDA_ARCH_FAMILY_SPECIFIC__ < 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1100
  if constexpr (tensor<P>) { __trap(); return; }
#endif
  __shared__ __align__(16) uint32_t tile[512];
  __shared__ uint32_t address;
  __shared__ __align__(8) uint64_t barrier;
  const uint32_t tid = threadIdx.x;
  if (std::size_t(blockIdx.x) * 512 >= words) return;
  const uint32_t shared = uint32_t(__cvta_generic_to_shared(tile));
  const uint32_t bar = uint32_t(__cvta_generic_to_shared(&barrier));
  if constexpr (P == Path::Tma || P == Path::TensorInput) {
    if (tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if constexpr (tensor<P>) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100
    if (tid < 32) {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                   :: "r"(uint32_t(__cvta_generic_to_shared(&address))) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
#else
    __trap();
#endif
  }
  if constexpr (P != Path::Direct) __syncthreads();
  uint32_t parity = 0;
  const bool aligned = (uintptr_t(source) | uintptr_t(destination)) % 16 == 0;
  for (int pass = 0; pass < passes; ++pass) {
    for (std::size_t first = std::size_t(blockIdx.x) * 512; first < words;
         first += std::size_t(gridDim.x) * 512) {
      const std::size_t at = first + tid * 4;
      const bool full = aligned && words - first >= 512;
      uint4 value;
      if constexpr (P == Path::Tma || P == Path::TensorInput) {
        if (full) {
          if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 2048;"
                         :: "r"(bar) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1], 2048, [%2];"
                         :: "r"(shared), "l"(source + first), "r"(bar) : "memory");
          }
          wait_barrier(bar, parity); parity ^= 1;
        } else {
          value = load_words(source, at, words, aligned);
          tile[tid * 4] = value.x; tile[tid * 4 + 1] = value.y;
          tile[tid * 4 + 2] = value.z; tile[tid * 4 + 3] = value.w;
          if constexpr (P == Path::TensorInput)
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          __syncthreads();
        }
        if constexpr (P == Path::Tma) {
          value = {tile[tid * 4], tile[tid * 4 + 1], tile[tid * 4 + 2], tile[tid * 4 + 3]};
        }
      } else {
        value = load_words(source, at, words, aligned);
      }
      if constexpr (tensor<P>) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100
        uint32_t warp_address = address + ((tid / 32 * 32) << 16);
        if (tid == 0) {
          uint64_t desc = (uint64_t(shared) >> 4) | (uint64_t{8} << 16) |
                          (uint64_t{8} << 32) | (uint64_t{1} << 46);
          asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;"
                       :: "r"(address), "l"(desc) : "memory");
          asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                       :: "r"(bar) : "memory");
        }
        wait_barrier(bar, parity); parity ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                     : "r"(warp_address) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
      }
      store_words(destination, at, words, value, aligned);
      if constexpr (tensor<P>) asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      if constexpr (P != Path::Direct) __syncthreads();
      if constexpr (tensor<P>) asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
  }
  if constexpr (tensor<P>) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100
    if (tid < 32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                             :: "r"(address) : "memory");
#endif
  }
  if constexpr (P == Path::Tma || P == Path::TensorInput) {
    if (tid == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
  }
}

// Two TMEM slots hand registers from the producer warpgroup to the consumer.
// A slot becomes reusable after the consumer's TMEM load, independently of
// the subsequent SMEM->HBM write. Only consumers use named CTA barrier 1.
__global__ void tensor_output(uint32_t const* source, uint32_t* destination,
                              std::size_t words, int passes) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100
  __shared__ uint32_t address;
  __shared__ __align__(16) uint32_t tile[512];
  __shared__ __align__(8) uint64_t ready[2], consumed[2];
  const uint32_t tid = threadIdx.x;
  if (std::size_t(blockIdx.x) * 512 >= words) return;
  if (tid == 0) {
    for (int slot = 0; slot < 2; ++slot) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" :: "r"(uint32_t(__cvta_generic_to_shared(ready + slot))) : "memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" :: "r"(uint32_t(__cvta_generic_to_shared(consumed + slot))) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                 :: "r"(uint32_t(__cvta_generic_to_shared(&address))) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  __syncthreads();
  const uint32_t lane = tid % 128;
  const uint32_t rows = (lane / 32 * 32) << 16;
  const bool aligned = (uintptr_t(source) | uintptr_t(destination)) % 16 == 0;
  uint32_t iteration = 0;
  for (int pass = 0; pass < passes; ++pass) {
    for (std::size_t first = std::size_t(blockIdx.x) * 512; first < words;
         first += std::size_t(gridDim.x) * 512, ++iteration) {
      uint32_t slot = iteration % 2;
      uint32_t publication = uint32_t(__cvta_generic_to_shared(ready + slot));
      uint32_t reuse = uint32_t(__cvta_generic_to_shared(consumed + slot));
      uint32_t taddr = address + rows + slot * 4;
      std::size_t at = first + lane * 4;
      if (tid < 128) {
        if (iteration >= 2) {
          wait_barrier(reuse, (iteration / 2 - 1) % 2);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
        uint4 value = load_words(source, at, words, aligned);
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
                     :: "r"(taddr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
        asm volatile("tcgen05.wait::st.sync.aligned; tcgen05.fence::before_thread_sync;" ::: "memory");
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(publication) : "memory");
      } else {
        wait_barrier(publication, iteration / 2 % 2);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        uint4 value;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w) : "r"(taddr) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned; tcgen05.fence::before_thread_sync;" ::: "memory");
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(reuse) : "memory");
        if (aligned && words - first >= 512) {
          tile[lane * 4] = value.x; tile[lane * 4 + 1] = value.y;
          tile[lane * 4 + 2] = value.z; tile[lane * 4 + 3] = value.w;
          asm volatile("fence.proxy.async.shared::cta; bar.sync 1, 128;" ::: "memory");
          if (lane == 0) {
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 2048;"
                         :: "l"(destination + first), "r"(uint32_t(__cvta_generic_to_shared(tile))) : "memory");
            asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::: "memory");
          }
          asm volatile("bar.sync 1, 128;" ::: "memory");
        } else store_words(destination, at, words, value, aligned);
      }
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (tid < 32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(address) : "memory");
  if (tid == 0) for (int slot = 0; slot < 2; ++slot) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(uint32_t(__cvta_generic_to_shared(ready + slot))) : "memory");
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(uint32_t(__cvta_generic_to_shared(consumed + slot))) : "memory");
  }
#else
  __trap();
#endif
}

template <class Function>
void select(Path path, Function function) {
  switch (path) {
    case Path::Direct: function(copy<Path::Direct>); break;
    case Path::Tma: function(copy<Path::Tma>); break;
    case Path::TensorInput: function(copy<Path::TensorInput>); break;
    case Path::TensorOutput: function(tensor_output); break;
    case Path::Copy: break;
  }
}

inline cudaError_t launch(Path path, void const* input, void* output,
                          std::size_t bytes, int ctas, int passes, cudaStream_t stream) {
  cudaError_t status = cudaSuccess;
  if (path == Path::Copy) {
    for (int pass = 0; pass < passes; ++pass) {
      auto status = cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToDevice, stream);
      if (status != cudaSuccess) return status;
    }
  } else {
    int grid = int(std::min(std::size_t(ctas), (bytes / 4 + 511) / 512));
    auto source = static_cast<uint32_t const*>(input);
    auto destination = static_cast<uint32_t*>(output);
    std::size_t words = bytes / 4;
    void* args[] = {&source, &destination, &words, &passes};
    select(path, [&](auto kernel) {
      status = cudaLaunchKernel(reinterpret_cast<void const*>(kernel), dim3(grid),
                                dim3(path == Path::TensorOutput ? 256 : 128), args, 0, stream);
    });
  }
  return status;
}

inline cudaError_t query(Path path, int ctas, cudaDeviceProp const& device,
                         KernelResources& result) {
  result.sms = device.multiProcessorCount;
  result.threads = path == Path::TensorOutput ? 256 : 128; result.blocks = ctas;
  result.smem_per_sm = device.sharedMemPerMultiprocessor;
  result.smem_per_block = device.sharedMemPerBlockOptin;
  cudaError_t status = cudaGetDevice(&result.device);
  if (status != cudaSuccess) return status;
  select(path, [&](auto kernel) {
    cudaFuncAttributes attributes;
    status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) return;
    result.registers = attributes.numRegs; result.static_smem = attributes.sharedSizeBytes;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&result.resident_blocks, kernel, result.threads, 0);
  });
  return status;
}

}  // namespace nccl_ep::hybridep::traffic
