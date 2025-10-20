#pragma once

#define NUM_MAX_NVL_PEERS 8
#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

#define ENABLE_FAST_DEBUG
#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull  // 20G cycles ~= 10s
#endif

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900  // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__     // NOLINT(*-reserved-identifier)
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cstdint>
#include <cuda_runtime.h>


#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// #include <hip/hip_bfloat16.h>
// #include <hip/hip_fp8.h>
#define nv_bfloat16 hip_bfloat16
#define __nv_fp8x2_storage_t __hip_fp8x2_storage_t
#define __nv_fp8_storage_t __hip_fp8_storage_t
#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2
#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_E4M3 __HIP_E4M3_FNUZ
#define WARP_SIZE 64
#define WARP_MASK 0xffffffffffffffff
#define MAX_NTHREADS 1024
#define MAX_GROUPS (MAX_NTHREADS/WARP_SIZE)

#else
#include <cuda_bf16.h>
#define WARP_SIZE 32
#define WARP_MASK 0xffffffff
#ifndef DISABLE_SM90_FEATURES
#include <cuda_fp8.h>
#else
// Ampere does not support FP8 features
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int __nv_fp8_interpretation_t;
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif
#endif
