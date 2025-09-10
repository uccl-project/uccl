#pragma once
#include "ep_util.hpp"

// TODO(MaoZiming): The whole thing is very nvidia-specific.
__forceinline__ __device__ int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;

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
  EP_STATIC_ASSERT(kNumLanes == 32 or kNumLanes == 16 or kNumLanes == 8 or
                       kNumLanes == 4 or kNumLanes == 2 or kNumLanes == 1,
                   "Invalid number of lanes");

  if constexpr (kNumLanes >= 32)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 16));
  if constexpr (kNumLanes >= 16)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 8));
  if constexpr (kNumLanes >= 8)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 4));
  if constexpr (kNumLanes >= 4)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 2));
  if constexpr (kNumLanes >= 2)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 1));
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

template <uint32_t kNumLanes = 32, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
  return warp_reduce<kNumLanes, T>(value, ReduceMin<T>{});
}

template <uint32_t kNumLanes = 32, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
  return warp_reduce<kNumLanes, T>(value, ReduceMax<T>{});
}

__device__ __forceinline__ float log2f_approx(float const& x) {
  float ret;
  asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
}

__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;");
}

template <int N = 0>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void fence_view_async_shared() {
  asm volatile("fence.proxy.async.shared::cta; \n" ::);
}

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
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
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(
          num_bytes),
      "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr,
                                              uint32_t& phase) {
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
}

__device__ __forceinline__ void tma_store_1d(void const* smem_ptr,
                                             void const* gmem_ptr,
                                             int num_bytes,
                                             bool evict_first = true) {
  auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  auto const cache_hint = evict_first ? kEvictFirst : kEvictNormal;
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], "
      "%2, %3;\n" ::"l"(gmem_ptr),
      "r"(smem_int_ptr), "r"(num_bytes), "l"(cache_hint)
      : "memory");
  asm volatile("cp.async.bulk.commit_group;");
}

__device__ __forceinline__ void tma_load_1d(void const* smem_ptr,
                                            void const* gmem_ptr,
                                            uint64_t* mbar_ptr, int num_bytes,
                                            bool evict_first = true) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  auto const cache_hint = evict_first ? kEvictFirst : kEvictNormal;
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::"
      "cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
      "l"(gmem_ptr), "r"(num_bytes), "r"(mbar_int_ptr), "l"(cache_hint)
      : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr,
                                              uint32_t arrive_count) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count),
               "r"(mbar_int_ptr));
}

// Convenience aliases
template <uint32_t kNumLanes = 32, typename T>
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
  float ret;
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
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
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(dtype_t const* ptr) {
  auto ret = ld_nc_global(
      reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
  return *reinterpret_cast<dtype_t*>(&ret);
}

__device__ __forceinline__ float ld_cg_global(float const* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ int ld_cg_global(int const* p) {
  int v;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ uint2 ld_cg_global(uint2 const* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
               : "=r"(v.x), "=r"(v.y)
               : "l"(p));
  return v;
}

__device__ __forceinline__ int4 ld_cg_global(int4 const* p) {
  int4 v;
  asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void st_cg_global(int* p, int v) {
  asm volatile("st.global.cg.s32 [%0], %1;" : : "l"(p), "r"(v));
}

__device__ __forceinline__ void st_cg_global(int4* p, int4 v) {
  asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(dtype_t const* ptr,
                                             dtype_t const& value) {
  st_na_global(
      reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
      *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(
          &value));
}

__device__ __forceinline__ int ld_acquire_sys_global(int const* ptr) {
  int ret;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
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

__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t const* ptr) {
  uint64_t ret;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align(dtype_t a, dtype_t b) {
  return ceil_div<dtype_t>(a, b) * b;
}

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC,     \
                           ST_FUNC)                                          \
  {                                                                          \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                        \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type       \
        unrolled_values[(UNROLL_FACTOR)];                                    \
    auto __src = (SRC);                                                      \
    auto __dst = (DST);                                                      \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride;       \
         __i += kLoopStride) {                                               \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)      \
          unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);            \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)      \
          ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);             \
    }                                                                        \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); \
         __i += 32)                                                          \
      ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                            \
  }

__device__ __forceinline__ int atomic_add_release_global(int const* ptr,
                                                         int value) {
  int ret;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

#ifndef DISABLE_SM90_FEATURES

__device__ __forceinline__ uint32_t elect_one_sync(int lane_id) {
  uint32_t pred = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "      elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "      mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(lane_id), "+r"(pred)
      : "r"(0xffffffff));
  return pred;
}

#endif