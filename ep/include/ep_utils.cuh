#pragma once
#include "ep_configs.cuh"
#include "ep_util.hpp"
#include <atomic>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include "amd_nanosleep.cuh"
#ifndef clock64
#define clock64 wall_clock64
#endif

using i64_gptr = __attribute__((address_space(1))) int64_t*;
using i32_gptr = __attribute__((address_space(1))) int*;
using u8_gptr = __attribute__((address_space(1))) uint8_t*;
using f32_gptr = __attribute__((address_space(1))) float*;

#ifndef DISABLE_AGGRESSIVE_ATOMIC
#define HIP_ATOMIC_LOAD(ptr, order, scope) \
  __hip_atomic_load((ptr), __ATOMIC_RELAXED, (scope))
#define HIP_ATOMIC_STORE(val, ptr, order, scope) \
  __hip_atomic_store((ptr), (val), __ATOMIC_RELAXED, (scope))
#define HIP_ATOMIC_ADD(ptr, val, order, scope) \
  __hip_atomic_fetch_add((ptr), (val), __ATOMIC_RELAXED, (scope))
#else
#define HIP_ATOMIC_LOAD(ptr, order, scope) \
  __hip_atomic_load((ptr), (order), (scope))
#define HIP_ATOMIC_STORE(val, ptr, order, scope) \
  __hip_atomic_store((ptr), (val), (order), (scope))
#define HIP_ATOMIC_ADD(ptr, val, order, scope) \
  __hip_atomic_fetch_add((ptr), (val), (order), (scope))
#endif

#ifdef DISABLE_BUILTIN_SHLF_SYNC
// use __shlf instead of __shlf_sync will speed up combine performance
#define __shfl_xor_sync(mask, var, srcLane) __shfl_xor((var), (srcLane))
#define __shfl_sync(mask, var, srcLane) __shfl((var), (srcLane))
#endif

// workgroup-level barrier sync used shared memory
namespace amd {

struct SharedData {
  uint32_t barrier[MAX_NUM_BARRIERS];
};

__shared__ SharedData shared_data;

__device__ __forceinline__ void barrier_init(int barrier_id) {
  shared_data.barrier[barrier_id] = 0;
}

template <typename T>
__device__ __forceinline__ T barrier_arrive(T* bar_ptr, int num_participants) {
  __threadfence_block();
  T v = __hip_atomic_fetch_add(bar_ptr, 1U, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_WORKGROUP);

  if ((v & MAX_GROUPS_MASK) == num_participants - 1)
    __hip_atomic_fetch_add(bar_ptr, MAX_GROUPS - num_participants,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);

  return v & ~MAX_GROUPS_MASK;
}

template <typename T>
__device__ __forceinline__ void barrier_wait(T* bar_ptr, T target) {
  while ((__hip_atomic_load(bar_ptr, __ATOMIC_ACQUIRE,
                            __HIP_MEMORY_SCOPE_WORKGROUP) &
          ~MAX_GROUPS_MASK) == target)
    __builtin_amdgcn_s_sleep(1);
}

template <typename T>
__device__ __forceinline__ void barrier_sync(T* bar_ptr,
                                             uint32_t num_participants) {
  // bound check
  if (num_participants >= MAX_GROUPS) {
    __syncthreads();
    return;
  }
  if (num_participants == 1) {
    __syncwarp();
    return;
  }

  auto const lane_id = __lane_id();
  if (lane_id == 0) {
    barrier_wait(bar_ptr, barrier_arrive(bar_ptr, num_participants));
  }
  __syncwarp();
}

template <typename T, int MemoryOrder = __ATOMIC_RELAXED,
          int MemoryScope = __HIP_MEMORY_SCOPE_AGENT>
__device__ __forceinline__ void grid_sync(T* bar_ptr, int num_participants) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    __hip_atomic_fetch_add(bar_ptr, 1, MemoryOrder, MemoryScope);
    while (__hip_atomic_load(bar_ptr, MemoryOrder, MemoryScope) <
           num_participants)
      __builtin_amdgcn_s_sleep(1);

    asm volatile("s_wakeup");
  }

  __syncthreads();
}
}  // namespace amd
#endif

__forceinline__ __device__ int get_lane_id() {
  int lane_id;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  lane_id = __lane_id();
#else
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
#endif
  return lane_id;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) && defined(__gfx942__)
constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 240.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 240.0f;
#else
constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;
#endif

template <int kBytes>
struct VecInt {};
template <>
struct VecInt<1> {
  using vec_t = int8_t;
};
template <>
struct VecInt<2> {
  using vec_t = int16_t;
};
template <>
struct VecInt<4> {
  using vec_t = int;
};
template <>
struct VecInt<8> {
  using vec_t = int64_t;
};
template <>
struct VecInt<16> {
  using vec_t = int4;
};

// Unified reduction function
template <uint32_t kNumLanes, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
  EP_STATIC_ASSERT(kNumLanes == 64 or kNumLanes == 32 or kNumLanes == 16 or
                       kNumLanes == 8 or kNumLanes == 4 or kNumLanes == 2 or
                       kNumLanes == 1,
                   "Invalid number of lanes");
  if constexpr (kNumLanes >= 64)
    value = op(value, __shfl_xor_sync(WARP_MASK, value, 32));
  if constexpr (kNumLanes >= 32)
    value = op(value, __shfl_xor_sync(WARP_MASK, value, 16));
  if constexpr (kNumLanes >= 16)
    value = op(value, __shfl_xor_sync(WARP_MASK, value, 8));
  if constexpr (kNumLanes >= 8)
    value = op(value, __shfl_xor_sync(WARP_MASK, value, 4));
  if constexpr (kNumLanes >= 4)
    value = op(value, __shfl_xor_sync(WARP_MASK, value, 2));
  if constexpr (kNumLanes >= 2)
    value = op(value, __shfl_xor_sync(WARP_MASK, value, 1));
  return value;
}

template <typename T>
struct ReduceSum {
  __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
  __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
  __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};

template <uint32_t kNumLanes = WARP_SIZE, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
  return warp_reduce<kNumLanes, T>(value, ReduceMin<T>{});
}

template <uint32_t kNumLanes = WARP_SIZE, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
  return warp_reduce<kNumLanes, T>(value, ReduceMax<T>{});
}

__device__ __forceinline__ float log2f_approx(float const& x) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return __builtin_amdgcn_logf(x);
#else
  float ret;
  asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
#endif
}

__device__ __forceinline__ void tma_store_fence() {
#if __CUDA_ARCH__
  asm volatile("fence.proxy.async.shared::cta;");
#else
  EP_DEVICE_ASSERT(false);
#endif
}

template <int N = 0>
__device__ __forceinline__ void tma_store_wait() {
#if __CUDA_ARCH__
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
#else
  EP_DEVICE_ASSERT(false);
#endif
}

__device__ __forceinline__ void fence_view_async_shared() {
#if __CUDA_ARCH__
  asm volatile("fence.proxy.async.shared::cta; \n" ::);
#else
  EP_DEVICE_ASSERT(false);
#endif
}

__device__ __forceinline__ void fence_barrier_init() {
#if __CUDA_ARCH__
  asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
#else
  EP_DEVICE_ASSERT(false);
#endif
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(dtype_b_t const& packed, dtype_a_t& x,
                                        dtype_a_t& y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t),
                   "Invalid dtypes");
  auto unpacked_ptr = reinterpret_cast<dtype_a_t const*>(&packed);
  x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename FuncT>
struct PatternVisitor {
  FuncT func;

  __device__ __host__ explicit PatternVisitor(FuncT&& func)
      : func(std::forward<FuncT>(func)) {}

  __device__ __host__ auto operator[](uint32_t const& i) { return func(i); }
};

constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(
    uint64_t* mbar_ptr, int num_bytes) {
#if __CUDA_ARCH__
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(
          num_bytes),
      "r"(mbar_int_ptr));
#else
  EP_DEVICE_ASSERT(false);
#endif
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr,
                                              uint32_t& phase) {
#if __CUDA_ARCH__
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile(
      "{\n\t"
      ".reg .pred       P1; \n\t"
      "LAB_WAIT: \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
      "@P1 bra DONE; \n\t"
      "bra     LAB_WAIT; \n\t"
      "DONE: \n\t"
      "}" ::"r"(mbar_int_ptr),
      "r"(phase), "r"(0x989680));
  phase ^= 1;

#else
  EP_DEVICE_ASSERT(false);
#endif
}

__device__ __forceinline__ void tma_store_1d(void const* smem_ptr,
                                             void const* gmem_ptr,
                                             int num_bytes,
                                             bool evict_first = true) {
#if __CUDA_ARCH__
  auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  auto const cache_hint = evict_first ? kEvictFirst : kEvictNormal;
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], "
      "%2, %3;\n" ::"l"(gmem_ptr),
      "r"(smem_int_ptr), "r"(num_bytes), "l"(cache_hint)
      : "memory");
  asm volatile("cp.async.bulk.commit_group;");
#else
  EP_DEVICE_ASSERT(false);
#endif
}

__device__ __forceinline__ void tma_load_1d(void const* smem_ptr,
                                            void const* gmem_ptr,
                                            uint64_t* mbar_ptr, int num_bytes,
                                            bool evict_first = true) {
#if __CUDA_ARCH__
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  auto const cache_hint = evict_first ? kEvictFirst : kEvictNormal;
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::"
      "cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
      "l"(gmem_ptr), "r"(num_bytes), "r"(mbar_int_ptr), "l"(cache_hint)
      : "memory");
#else
  EP_DEVICE_ASSERT(false);
#endif
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr,
                                              uint32_t arrive_count) {
#if __CUDA_ARCH__
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count),
               "r"(mbar_int_ptr));
#else
  EP_DEVICE_ASSERT(false);
#endif
}

// Convenience aliases
template <uint32_t kNumLanes = WARP_SIZE, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
  return warp_reduce<kNumLanes, T>(value, ReduceSum<T>{});
}

__forceinline__ __device__ int fast_log2_ceil(float x) {
  auto bits_x = *reinterpret_cast<uint32_t*>(&x);
  auto exp_x = (bits_x >> 23) & 0xff;
  auto man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

__device__ __forceinline__ float exp2f_approx(float const& x) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return __builtin_amdgcn_exp2f(x);
#else
  float ret;
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
#endif
}

__forceinline__ __device__ float fast_pow2(int x) {
  // We can ensure `-126 <= x and x <= 127`
  uint32_t bits_x = (x + 127) << 23;
  return *reinterpret_cast<float*>(&bits_x);
}

__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale,
                                                     float& scale_inv,
                                                     bool round_scale) {
  if (round_scale) {
    auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
    scale = fast_pow2(-exp_scale_inv);
    scale_inv = fast_pow2(exp_scale_inv);
  } else {
    scale_inv = amax * kFinfoAmaxInvE4M3;
    scale = kFinfoAmaxE4M3 / amax;
  }
}

// `ld.global.nc.L1::no_allocate` will be translated into
// `LDG.E.NA.[width].CONSTANT` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into
// `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(dtype_t const* ptr) {
  auto ret = ld_nc_global(
      reinterpret_cast<typename VecInt<sizeof(dtype_t)>::vec_t const*>(ptr));
  return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(uint8_t const* ptr) {
  uint16_t ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __builtin_nontemporal_load((u8_gptr)ptr);
#else
  // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit
  // constraint letter (`h` below means unsigned 16-bit)
  asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
#endif
  return static_cast<uint8_t>(ret);
}

template <>
__device__ __forceinline__ int ld_nc_global(int const* ptr) {
  int ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __builtin_nontemporal_load((i32_gptr)ptr);
#else
  asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
  return ret;
}

template <>
__device__ __forceinline__ int64_t ld_nc_global(int64_t const* ptr) {
  int64_t ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __builtin_nontemporal_load((i64_gptr)ptr);
#else
  asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
  return ret;
}

template <>
__device__ __forceinline__ float ld_nc_global(float const* ptr) {
  float ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __builtin_nontemporal_load((f32_gptr)ptr);
#else
  asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
#endif
  return ret;
}

template <>
__device__ __forceinline__ int2 ld_nc_global(int2 const* ptr) {
  int2 ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret.x = __builtin_nontemporal_load((i32_gptr)&ptr->x);
  ret.y = __builtin_nontemporal_load((i32_gptr)&ptr->y);
#else
  asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];"
               : "=r"(ret.x), "=r"(ret.y)
               : "l"(ptr));
#endif
  return ret;
}

template <>
__device__ __forceinline__ int4 ld_nc_global(int4 const* ptr) {
  int4 ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret.x = __builtin_nontemporal_load((i32_gptr)&ptr->x);
  ret.y = __builtin_nontemporal_load((i32_gptr)&ptr->y);
  ret.z = __builtin_nontemporal_load((i32_gptr)&ptr->z);
  ret.w = __builtin_nontemporal_load((i32_gptr)&ptr->w);
#else
  asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
#endif
  return ret;
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif
template <typename dtype_t>
__device__ __forceinline__ void st_na_global(dtype_t const* ptr,
                                             dtype_t const& value) {
  st_na_global(
      reinterpret_cast<typename VecInt<sizeof(dtype_t)>::vec_t const*>(ptr),
      *reinterpret_cast<typename VecInt<sizeof(dtype_t)>::vec_t const*>(
          &value));
}

template <>
__device__ __forceinline__ void st_na_global(int const* ptr, int const& value) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  *(i32_gptr)ptr = value;
#else
  asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(int64_t const* ptr,
                                             int64_t const& value) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  *(i64_gptr)ptr = value;
#else
  asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(float const* ptr,
                                             float const& value) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  *(f32_gptr)ptr = value;
#else
  asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(int4 const* ptr,
                                             int4 const& value) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  *(i32_gptr)&ptr->x = value.x;
  *(i32_gptr)&ptr->y = value.y;
  *(i32_gptr)&ptr->z = value.z;
  *(i32_gptr)&ptr->w = value.w;
#else
  asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),
               "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
#endif
}

__device__ __forceinline__ int ld_acquire_sys_global(int const* ptr) {
  int ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
  return ret;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(dtype_a_t const& x,
                                           dtype_a_t const& y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t),
                   "Invalid dtypes");
  dtype_b_t packed;
  auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
  unpacked_ptr[0] = x, unpacked_ptr[1] = y;
  return packed;
}

template <bool kIsUE8M0,
          typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t
extract_required_scale_format(float value) {
  if constexpr (kIsUE8M0) {
    return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
  } else {
    return value;
  }
}

// 32-bit system-consistent load with acquire semantics (GH200-safe)
__device__ __forceinline__ uint32_t
ld_acquire_sys_global(uint32_t const volatile* p) {
  uint32_t v;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  v = HIP_ATOMIC_LOAD(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
#endif
  return v;
}

__device__ __forceinline__ uint64_t
ld_acquire_sys_global(uint64_t const volatile* p) {
  uint64_t v;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  v = HIP_ATOMIC_LOAD(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
#endif
  return v;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t const* ptr) {
  uint64_t ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
  return ret;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align(dtype_t a, dtype_t b) {
  return ceil_div<dtype_t>(a, b) * b;
}

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC,     \
                           ST_FUNC)                                          \
  {                                                                          \
    constexpr int kLoopStride = WARP_SIZE * (UNROLL_FACTOR);                 \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type       \
        unrolled_values[(UNROLL_FACTOR)];                                    \
    auto __src = (SRC);                                                      \
    auto __dst = (DST);                                                      \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride;       \
         __i += kLoopStride) {                                               \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)      \
          unrolled_values[__j] = LD_FUNC(__src + __i + __j * WARP_SIZE);     \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)      \
          ST_FUNC(__dst + __i + __j * WARP_SIZE, unrolled_values[__j]);      \
    }                                                                        \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); \
         __i += WARP_SIZE)                                                   \
      ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                            \
  }

__device__ __forceinline__ int atomic_add_release_global(int const* ptr,
                                                         int value) {
  int ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = HIP_ATOMIC_ADD(const_cast<int*>(ptr), value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_AGENT);
#else
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
#endif
  return ret;
}

__device__ __forceinline__ uint32_t elect_one_sync(int lane_id) {
  uint32_t pred = 0;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return lane_id == 0;
#else
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "      elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "      mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(lane_id), "+r"(pred)
      : "r"(WARP_MASK));
#endif
  return pred;
}

__device__ __forceinline__ int ld_acquire_global(int const* ptr) {
  int ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = HIP_ATOMIC_LOAD(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#else
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
  return ret;
}

__device__ __forceinline__ void st_release_sys_global(int const* ptr, int val) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  HIP_ATOMIC_STORE(val, const_cast<int*>(ptr), __ATOMIC_RELEASE,
                   __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ void st_release_cta(int const* ptr, int val) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_WORKGROUP);
#else
  asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
#endif
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

template <bool kUseUnsafeSync = false>
__device__ inline void workgroup_sync_barrier(int barrier_id, int num_threads) {
  // If __syncthreads is feasible in kernel,
  // using __syncthreads directly will be better than shared memory based
  // barrier.
  if constexpr (kUseUnsafeSync) {
    // maybe stuck in __syncthreads
    num_threads >= WARP_SIZE ? __syncthreads() : __syncwarp();
  } else {
    EP_DEVICE_ASSERT(num_threads % WARP_SIZE == 0 and
                     "invalid number of threads");

    auto* bar_ptr = amd::shared_data.barrier + barrier_id;
    auto const num_participants =
        static_cast<uint32_t>(num_threads / WARP_SIZE);
    amd::barrier_sync(bar_ptr, num_participants);
  }
}
#endif

template <bool kUseUnsafeSync = false>
__device__ inline void sync_barrier(int barrier_id, int num_threads) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  workgroup_sync_barrier<kUseUnsafeSync>(barrier_id, num_threads);
#else
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#endif
}

template <bool kUseUnsafedSync = false>
__device__ inline void sync_barrier_1(int num_threads) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  workgroup_sync_barrier<kUseUnsafedSync>(1, num_threads);
#else
  asm volatile("bar.sync 1, %0;" ::"r"(num_threads));
#endif
}

__device__ inline void sys_membar() {
#if __CUDA_ARCH__
  asm volatile("membar.sys;" ::: "memory");
#endif
}

__device__ __forceinline__ void trap() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  abort();
#else
  asm("trap;");
#endif
}

__device__ __forceinline__ int ld_volatile_global(int const* ptr) {
  int ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __atomic_load_n(const_cast<int*>(ptr), __ATOMIC_RELAXED);
#else
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
  return ret;
}

__device__ __forceinline__ float ld_volatile_global(float const* ptr) {
  float ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  __atomic_load(const_cast<float*>(ptr), &ret, __ATOMIC_RELAXED);
#else
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
#endif
  return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(int64_t const* ptr) {
  int64_t ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __atomic_load_n(const_cast<int64_t*>(ptr), __ATOMIC_RELAXED);
#else
  asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
  return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(uint64_t const* ptr) {
  int64_t ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __atomic_load_n(const_cast<uint64_t*>(ptr), __ATOMIC_RELAXED);
#else
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
  return ret;
}

__device__ __forceinline__ void memory_fence() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  __threadfence_system();
#else
  asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
}

__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // TODO: __hip_atomic_compare_exchange_strong or
  // __hip_atomic_compare_exchange_weak
  __hip_atomic_compare_exchange_strong(addr, &x, y, __ATOMIC_ACQUIRE,
                                       __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_WORKGROUP);
  return x;
#else
  int ret;
  asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;"
               : "=r"(ret)
               : "l"(addr), "r"(x), "r"(y)
               : "memory");
  return ret;
#endif
}

__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
  int ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __hip_atomic_exchange(addr, x, __ATOMIC_RELEASE,
                              __HIP_MEMORY_SCOPE_WORKGROUP);
#else
  asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;"
               : "=r"(ret)
               : "l"(addr), "r"(x)
               : "memory");
#endif
  return ret;
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs,
                                              int rank) {
  auto thread_id = static_cast<int>(threadIdx.x);

  // For non-sync-only cases, the memory operations by other threads in the
  // block must be visible to the `sys` scope
  if constexpr (not kSyncOnly) {
    memory_fence();
    __syncthreads();
  }

  // Add self-ranks, sub other ranks
  if (thread_id < kNumRanks) {
    atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
    atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
  }
  EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);

  // Check timeout
  auto start_time = clock64();
  while (true) {
    auto value = thread_id < kNumRanks
                     ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id)
                     : 0;
    if (__all_sync(WARP_MASK, value <= 0)) break;

    if (clock64() - start_time > NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
      printf(
          "DeepEP timeout check failed: rank = %d, thread = %d, value = "
          "%d)\n",
          rank, thread_id, value);
      trap();
    }
  }
  __syncthreads();
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens,
                                                       int num_sms, int sm_id,
                                                       int& token_start_idx,
                                                       int& token_end_idx) {
  int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
  token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
  token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
  EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
  auto send_int_values = reinterpret_cast<int*>(&ptr);
  int recv_int_values[sizeof(dtype_t) / sizeof(int)];
#pragma unroll
  for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
    recv_int_values[i] =
        __shfl_sync(WARP_MASK, send_int_values[i], src_lane_idx);
  return *reinterpret_cast<dtype_t*>(recv_int_values);
}

__device__ __forceinline__ void memory_fence_gpu() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  __threadfence();
#else
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}

__device__ __forceinline__ void memory_fence_cta() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  __threadfence_block();
#else
  asm volatile("fence.acq_rel.cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void st_relaxed_sys_global(int const* ptr, int val) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  HIP_ATOMIC_STORE(val, const_cast<int*>(ptr), __ATOMIC_RELAXED,
                   __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ int ld_acquire_cta(int const* ptr) {
  int ret;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP);
#else
  asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
  return ret;
}

__forceinline__ __device__ void acquire_lock(int* mutex) {
  // To make later memory operations valid, we must use `acquire` for memory
  // semantics
  while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)
    ;
}

__forceinline__ __device__ void release_lock(int* mutex) {
  // To make previous memory operations visible to other threads, we must
  // use `release` for memory semantics
  atomic_exch_cta_release(mutex, 0);
}
