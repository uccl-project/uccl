/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 * Adapted from NCCL unpack kernel logic
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "../include/unpack_descriptor.h"
#include "unpack_launch.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

// Forward declaration for device visibility barrier used by probe kernel
namespace tcpx {
namespace device {
__device__ __forceinline__ void devmem_visibility_barrier(void const* flag_ptr);
}
}  // namespace tcpx

namespace tcpx {
// Define a minimal staging kernel for debugging: read 1 byte and write to
// dst[0]
extern "C" __global__ void tcpxUnpackKernelProbeByte(
    tcpx::rx::UnpackDescriptorBlock const* desc_block) {
  // Only thread 0 does the visibility load, but all threads must sync
  if (threadIdx.x == 0) {
    device::devmem_visibility_barrier(desc_block->ready_flag);
  }
  __syncthreads();  // All threads sync here

  if (blockIdx.x == 0 && threadIdx.x == 0 && desc_block->count > 0) {
    tcpx::rx::UnpackDescriptor const& d = desc_block->descriptors[0];
    char const* src =
        static_cast<char const*>(desc_block->bounce_buffer) + d.src_off;
    char* dst = static_cast<char*>(desc_block->dst_buffer) + d.dst_off;
    char volatile v = *src;
    *dst = v;
  }
}

namespace device {

// CUDA memory access primitives (adapted from NCCL)
template <int BYTES>
struct BytePack;

template <>
struct BytePack<1> {
  uint8_t u8;
};
template <>
struct BytePack<2> {
  uint16_t u16;
};
template <>
struct BytePack<4> {
  uint32_t u32;
};
template <>
struct BytePack<8> {
  uint64_t u64;
};
template <>
struct BytePack<16> {
  uint64_t u64[2];
};

// Vectorized load/store functions
template <int BYTES>
__device__ __forceinline__ BytePack<BYTES> ld_volatile_global(uintptr_t addr);

template <>
__device__ __forceinline__ BytePack<1> ld_volatile_global<1>(uintptr_t addr) {
  BytePack<1> val;
  val.u8 = *reinterpret_cast<uint8_t volatile*>(addr);
  return val;
}

template <>
__device__ __forceinline__ BytePack<2> ld_volatile_global<2>(uintptr_t addr) {
  BytePack<2> val;
  val.u16 = *reinterpret_cast<uint16_t volatile*>(addr);
  return val;
}

template <>
__device__ __forceinline__ BytePack<4> ld_volatile_global<4>(uintptr_t addr) {
  BytePack<4> val;
  val.u32 = *reinterpret_cast<uint32_t volatile*>(addr);
  return val;
}

template <>
__device__ __forceinline__ BytePack<8> ld_volatile_global<8>(uintptr_t addr) {
  BytePack<8> val;
  val.u64 = *reinterpret_cast<uint64_t volatile*>(addr);
  return val;
}

template <>
__device__ __forceinline__ BytePack<16> ld_volatile_global<16>(uintptr_t addr) {
  BytePack<16> val;
  const volatile uint64_t* ptr =
      reinterpret_cast<const volatile uint64_t*>(addr);
  val.u64[0] = ptr[0];
  val.u64[1] = ptr[1];
  return val;
}

template <int BYTES>
__device__ __forceinline__ void st_global(uintptr_t addr, BytePack<BYTES> val);

template <>
__device__ __forceinline__ void st_global<1>(uintptr_t addr, BytePack<1> val) {
  *reinterpret_cast<uint8_t*>(addr) = val.u8;
}

template <>
__device__ __forceinline__ void st_global<2>(uintptr_t addr, BytePack<2> val) {
  *reinterpret_cast<uint16_t*>(addr) = val.u16;
}

template <>
__device__ __forceinline__ void st_global<4>(uintptr_t addr, BytePack<4> val) {
  *reinterpret_cast<uint32_t*>(addr) = val.u32;
}

template <>
__device__ __forceinline__ void st_global<8>(uintptr_t addr, BytePack<8> val) {
  *reinterpret_cast<uint64_t*>(addr) = val.u64;
}

template <>
__device__ __forceinline__ void st_global<16>(uintptr_t addr,
                                              BytePack<16> val) {
  uint64_t* ptr = reinterpret_cast<uint64_t*>(addr);
  ptr[0] = val.u64[0];
  ptr[1] = val.u64[1];
}

// Constants
#define DATA_LOAD_SIZE 16
#define WARP_SIZE 32
constexpr int kWarpShmPageCnt = 4;

// Device-side visibility barrier similar to NCCL load64gpu on cnt
// NOTE: This function does NOT include __syncthreads() - caller must sync if
// needed
__device__ __forceinline__ void devmem_visibility_barrier(
    void const* flag_ptr) {
  if (!flag_ptr) return;
#if __CUDA_ARCH__ >= 700
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.u64 {%0}, [%1];"
               : "=l"(v)
               : "l"(flag_ptr)
               : "memory");
#else
  unsigned long long volatile* p = (unsigned long long volatile*)flag_ptr;
  (void)*p;
#endif
}

// Bulk copy template (adapted from NCCL)
// IMPORTANT: Loop condition must match NCCL exactly: data_s + DATA_LOAD_SIZE -
// 1 < len
template <int BYTES>
__device__ void bulkCopy(int lane, uint32_t len, char* src, char* dst) {
  int const elements_per_thread = DATA_LOAD_SIZE / BYTES;
  BytePack<BYTES> reg[elements_per_thread];

  // Match NCCL's loop condition exactly
  for (uint32_t offset = lane * DATA_LOAD_SIZE;
       offset + DATA_LOAD_SIZE - 1 <
       len;  // Fixed: was "offset + DATA_LOAD_SIZE <= len"
       offset += WARP_SIZE * DATA_LOAD_SIZE) {
// Load data
#pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
      reg[i] = ld_volatile_global<BYTES>(
          reinterpret_cast<uintptr_t>(src + offset) + i * BYTES);
    }

// Store data
#pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
      st_global<BYTES>(reinterpret_cast<uintptr_t>(dst + offset) + i * BYTES,
                       reg[i]);
    }
  }
}

// Single descriptor unpack kernel
// Matches NCCL logic from unpack.h:251-275
__device__ void unpackSingleDescriptor(int lane,
                                       tcpx::rx::UnpackDescriptor const& desc,
                                       char* bounce_buffer, char* dst_buffer) {
  char* src = bounce_buffer + desc.src_off;
  char* dst = dst_buffer + desc.dst_off;
  uint32_t len = desc.len;

  // Fast path for data >= 16 bytes (matches NCCL line 251)
  if (len >= DATA_LOAD_SIZE) {
    // Determine optimal alignment for vectorized access (matches NCCL line
    // 255-256)
    uint8_t align_off = (desc.src_off | desc.dst_off) % DATA_LOAD_SIZE;
    align_off = align_off & (-align_off);  // Keep lowest bit

    // Select bulk copy size based on alignment (matches NCCL line 257-267)
    if (align_off == 0) {
      bulkCopy<16>(lane, len, src, dst);
    } else if (align_off & 0x8) {
      bulkCopy<8>(lane, len, src, dst);
    } else if (align_off & 0x4) {
      bulkCopy<4>(lane, len, src, dst);
    } else if (align_off & 0x2) {
      bulkCopy<2>(lane, len, src, dst);
    } else {
      bulkCopy<1>(lane, len, src, dst);
    }
  }

  // Handle remaining bytes (< DATA_LOAD_SIZE) (matches NCCL line 271-275)
  // This handles both:
  // 1. Tail bytes after bulk copy (e.g., 23 bytes = 16 bulk + 7 tail)
  // 2. Entire payload if len < 16 (e.g., 7 bytes = 0 bulk + 7 tail)
  uint32_t remaining_start = (len / DATA_LOAD_SIZE) * DATA_LOAD_SIZE;
  if (lane < len % DATA_LOAD_SIZE) {
    char volatile* src_ptr = src + remaining_start + lane;
    char volatile* dst_ptr = dst + remaining_start + lane;
    *dst_ptr = *src_ptr;
  }
}

// Main unpack kernel
extern "C" __global__ void tcpxUnpackKernel(
    tcpx::rx::UnpackDescriptorBlock const* desc_block) {
  int tid = threadIdx.x;

  if (tid == 0) {
    devmem_visibility_barrier(desc_block->ready_flag);
  }
  __syncthreads();

  char* bounce_buffer = static_cast<char*>(desc_block->bounce_buffer);
  char* dst_buffer = static_cast<char*>(desc_block->dst_buffer);

  int const lane = tid & (WARP_SIZE - 1);
  int const warp_id = tid / WARP_SIZE;
  int const warps_per_block = blockDim.x / WARP_SIZE;
  int const total_warps = gridDim.x * warps_per_block;

  extern __shared__ unsigned char smem[];
  auto* warp_cache = reinterpret_cast<tcpx::rx::UnpackDescriptor*>(smem) +
                     warp_id * kWarpShmPageCnt;

  int const warp_global = blockIdx.x * warps_per_block + warp_id;
  for (int base = warp_global * kWarpShmPageCnt; base < desc_block->count;
       base += total_warps * kWarpShmPageCnt) {
    int batch = desc_block->count - base;
    if (batch > kWarpShmPageCnt) batch = kWarpShmPageCnt;
    if (batch <= 0) continue;

    if (lane < batch) {
      warp_cache[lane] = desc_block->descriptors[base + lane];
    }
    __syncwarp();

    for (int i = 0; i < batch; ++i) {
      tcpx::rx::UnpackDescriptor const& desc = warp_cache[i];
      unpackSingleDescriptor(lane, desc, bounce_buffer, dst_buffer);
    }
    __syncwarp();
  }
}

// Optimized kernel for small descriptors (single warp per descriptor)
// Note: Unlike NCCL which has warps process multiple descriptors in a loop,
// we assign one warp per descriptor. No inter-warp sync needed since each
// warp works on independent memory regions (different dst_off ranges).
extern "C" __global__ void tcpxUnpackKernelSmall(
    tcpx::rx::UnpackDescriptorBlock const* desc_block) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  // Issue a device-side visibility barrier before reading bounce buffer
  // Only thread 0 in each block does the visibility load, but all threads must
  // sync
  if (threadIdx.x == 0) {
    devmem_visibility_barrier(desc_block->ready_flag);
  }
  __syncthreads();  // All threads in block sync here

  char* bounce_buffer = static_cast<char*>(desc_block->bounce_buffer);
  char* dst_buffer = static_cast<char*>(desc_block->dst_buffer);

  // Each warp processes one descriptor
  // No sync needed after processing since descriptors have non-overlapping
  // dst_off ranges
  if (warp_id < desc_block->count) {
    tcpx::rx::UnpackDescriptor const& desc = desc_block->descriptors[warp_id];
    unpackSingleDescriptor(lane_id, desc, bounce_buffer, dst_buffer);
  }
}

// Kernel launch parameters calculation (use header definition)

// Calculate optimal launch parameters
extern "C" __host__ tcpx::device::KernelLaunchParams calculateLaunchParams(
    tcpx::rx::UnpackDescriptorBlock const& desc_block) {
  tcpx::device::KernelLaunchParams params;

  if (desc_block.count == 0) {
    params.grid_size = dim3(0);
    params.block_size = dim3(0);
    return params;
  }

  // Determine if we should use small kernel (warp-per-descriptor)
  bool use_small_kernel = true;
  for (uint32_t i = 0; i < desc_block.count; ++i) {
    if (desc_block.descriptors[i].len > 1024) {
      use_small_kernel = false;
      break;
    }
  }

  if (use_small_kernel) {
    // Warp per descriptor
    int warps_needed = desc_block.count;
    int threads_per_block = 256;  // 8 warps per block
    int blocks_needed =
        (warps_needed * WARP_SIZE + threads_per_block - 1) / threads_per_block;

    params.grid_size = dim3(blocks_needed);
    params.block_size = dim3(threads_per_block);
    params.shared_mem_size = 0;
  } else {
    // Block per descriptor
    params.grid_size = dim3(std::max<uint32_t>(1, desc_block.count));
    params.block_size = dim3(256);  // 8 warps per block
    int warps_per_block = params.block_size.x / WARP_SIZE;
    params.shared_mem_size =
        warps_per_block * kWarpShmPageCnt * sizeof(tcpx::rx::UnpackDescriptor);
  }
  params.use_small_kernel = use_small_kernel;

  return params;
}

}  // namespace device
}  // namespace tcpx
