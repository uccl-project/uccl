#pragma once
// Inline FP4 (E2M1) quantization helpers for the low-latency dispatch and
// combine kernels.
//
// Two block-scaled formats are supported:
//   - NVFP4: 16 elements per block, one E4M3 block scale, plus one FP32
//     per-token decode scale `gs = amax(token) / (6 * 448)` so that every
//     block scale fits in E4M3.  x ~= e2m1 * e4m3(block) * gs.
//   - MXFP4: 32 elements per block, one UE8M0 block scale (power of two,
//     rounded up so no element saturates).  x ~= e2m1 * 2^(ue8m0 - 127).
//
// Packing: two E2M1 values per byte, the even element in the low nibble
// (the same order as `cvt.rn.satfinite.e2m1x2.f32 d, odd, even`).
//
// For finite inputs, the software encoder matches hardware round-to-nearest
// conversion with saturation to [-6, 6]. E2M1 has no NaN encoding.
// Architecture- and family-specific Blackwell targets use the hardware path.

#include "mx_cvt.cuh"
#include <cstdint>
#include <cuda_fp8.h>

namespace uccl {
namespace fp4 {

enum Mode : int { kNone = 0, kNVFP4 = 1, kMXFP4 = 2 };

template <int kMode>
struct Traits;
template <>
struct Traits<kNVFP4> {
  static constexpr int kBlock = 16;
  static constexpr bool kHasTokenScale = true;
};
template <>
struct Traits<kMXFP4> {
  static constexpr int kBlock = 32;
  static constexpr bool kHasTokenScale = false;
};

constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;

// Bytes of one quantized row of `hidden` elements: data, then block scales,
// then (NVFP4 only, when `with_token_scale`) one FP32 token scale; padded to
// 16 bytes.
__host__ __device__ constexpr int data_bytes(int hidden) { return hidden / 2; }
__host__ __device__ constexpr int num_block_scales(int mode, int hidden) {
  return mode == kNVFP4 ? hidden / 16 : (mode == kMXFP4 ? hidden / 32 : 0);
}
__host__ __device__ constexpr int row_bytes(int mode, int hidden,
                                            bool with_token_scale) {
  int b = data_bytes(hidden) + num_block_scales(mode, hidden) +
          ((mode == kNVFP4 and with_token_scale) ? 4 : 0);
  return (b + 15) / 16 * 16;
}

// Element conversions (and the UCCL_EP_FP4_HW_CVT switch) live in
// mx_cvt.cuh; the names below keep the call sites in internode_ll.cu.
using mx::e2m1_mag_code;
using mx::e2m1_soft;

// Pack 8 floats (already multiplied by the inverse scale) into 8 nibbles.
__device__ __forceinline__ uint32_t pack8_e2m1(float const (&v)[8]) {
  return mx::e2m1x8_rn(v);
}

__device__ __forceinline__ uint32_t pack8_e2m1_soft(float const (&v)[8]) {
  uint32_t r = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) r |= e2m1_soft(v[j]) << (4 * j);
  return r;
}

__device__ __forceinline__ float e2m1_decode(uint32_t nib) {
  uint32_t const m = nib & 7u;
  // 0, 0.5 for m < 2; otherwise (1 + (m & 1) / 2) * 2^((m >> 1) - 1).
  float const mag = m < 2 ? 0.5f * static_cast<float>(m)
                          : (1.0f + 0.5f * static_cast<float>(m & 1u)) *
                                static_cast<float>(1u << ((m >> 1) - 1));
  return (nib & 8u) ? -mag : mag;
}

__device__ __forceinline__ void unpack8_e2m1(uint32_t w, float scale,
                                             float (&out)[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) out[j] = e2m1_decode(w >> (4 * j)) * scale;
}

// MXFP4: smallest e with 6 * 2^e >= amax, clamped to the UE8M0 range.
__device__ __forceinline__ int mx_block_exp(float amax) {
  if (!(amax > 0.0f)) return -127;
  int e;
  float const m = frexpf(amax / kE2M1Max, &e);  // v = m * 2^e, m in [0.5, 1)
  if (m == 0.5f) e -= 1;                        // exact power of two
  if (ldexpf(kE2M1Max, e) < amax) e += 1;       // guard the division rounding
  return max(-127, min(127, e));
}

__device__ __forceinline__ float e4m3_to_float(__nv_fp8_storage_t s) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
  return __half2float(*reinterpret_cast<__half*>(&h));
}

// NVFP4 per-token decode scale.
__device__ __forceinline__ float nv_token_scale(float token_amax) {
  return token_amax > 0.0f ? token_amax / (kE2M1Max * kE4M3Max) : 1.0f;
}

// NVFP4 block scale: returns the E4M3 byte and the inverse effective scale.
__device__ __forceinline__ uint8_t nv_block_scale(float block_amax, float gs,
                                                  float& inv_scale) {
  float const s = block_amax / (kE2M1Max * gs);
  __nv_fp8_storage_t const q =
      __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
  float const eff = e4m3_to_float(q) * gs;
  inv_scale = eff > 0.0f ? 1.0f / eff : 0.0f;
  return static_cast<uint8_t>(q);
}

__device__ __forceinline__ uint8_t mx_block_scale(float block_amax,
                                                  float& inv_scale) {
  int const e = mx_block_exp(block_amax);
  inv_scale = ldexpf(1.0f, -e);
  return static_cast<uint8_t>(e + 127);
}

__device__ __forceinline__ float mx_decode_scale(uint8_t b) {
  return ldexpf(1.0f, static_cast<int>(b) - 127);
}

}  // namespace fp4
}  // namespace uccl
