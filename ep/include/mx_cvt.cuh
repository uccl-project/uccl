#pragma once
// Element and scale conversions for the block-scaled formats used by the
// low-latency kernels: E4M3 (FP8), E2M1 (FP4) and UE8M0 (power-of-two
// scales).
//
// Packing order: for every x2 form the first argument (`lo`) lands in the
// low byte / low nibble, i.e. `cvt d, hi, lo` in PTX operand order.
//
// The hardware forms are used only where ptxas accepts them:
//   - e4m3x2: sm_89 and newer.
//   - ue8m0x2 and e2m1x2: arch- or family-specific Blackwell targets
//     (sm_100a, sm_120a, ...).  A plain `compute_120` / `sm_120` target has
//     neither, so those builds use the software path below.
// Define UCCL_EP_FP4_SOFT_CVT to force the software path everywhere (tests).
//
// The software UE8M0 encoder works on the exponent field: NaN -> 0xFF,
// +-inf -> 0xFE (satfinite), rz keeps the exponent field (clamped at code 0
// = 2^-127), rp rounds up to the next power of two and saturates at 0xFE.
// The sign is ignored; callers pass amax-style non-negative values.
// ep/tests/mx_cvt_test.cu compares both paths with CUDA's own conversions.

#include <cstdint>
#include <cuda_fp8.h>

#if defined(__CUDA_ARCH__) && !defined(UCCL_EP_FP4_SOFT_CVT) && \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) ||                     \
     defined(__CUDA_ARCH_FEAT_SM101_ALL) ||                     \
     defined(__CUDA_ARCH_FEAT_SM103_ALL) ||                     \
     defined(__CUDA_ARCH_FEAT_SM110_ALL) ||                     \
     defined(__CUDA_ARCH_FEAT_SM120_ALL) ||                     \
     defined(__CUDA_ARCH_FEAT_SM121_ALL) ||                     \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) &&                 \
      __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000))
#define UCCL_EP_FP4_HW_CVT 1
#endif

#if defined(__CUDA_ARCH__) && !defined(UCCL_EP_FP4_SOFT_CVT) && \
    __CUDA_ARCH__ >= 890
#define UCCL_EP_E4M3_HW_CVT 1
#endif

namespace uccl {
namespace mx {

enum class ScaleRound : int {
  kUp = 0,    // rp: smallest power of two >= x (no element saturates)
  kZero = 1,  // rz: largest power of two <= x (OCP default)
};

// ---------------------------------------------------------------- UE8M0

template <ScaleRound kRound>
__device__ __forceinline__ uint32_t ue8m0_soft(float x) {
  uint32_t const b = __float_as_uint(x) & 0x7fffffffu;
  if (b > 0x7f800000u) return 0xffu;   // NaN
  if (b == 0x7f800000u) return 0xfeu;  // inf, satfinite
  uint32_t e = b >> 23;
  uint32_t const m = b & 0x7fffffu;
  if (kRound == ScaleRound::kUp) {
    // Normal: round up when the mantissa is non-zero.  Subnormal: code 0 is
    // 2^-127 = 0x00400000, so only values above it round up to code 1.
    if (e != 0u ? m != 0u : m > 0x400000u) e += 1u;
  }
  return e > 0xfeu ? 0xfeu : e;
}

// Two UE8M0 codes: `lo` in bits [7:0], `hi` in bits [15:8].
template <ScaleRound kRound>
__device__ __forceinline__ uint16_t ue8m0x2(float lo, float hi) {
#ifdef UCCL_EP_FP4_HW_CVT
  uint16_t r;
  if (kRound == ScaleRound::kUp)
    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;"
        : "=h"(r)
        : "f"(hi), "f"(lo));
  else
    asm("cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2;"
        : "=h"(r)
        : "f"(hi), "f"(lo));
  return r;
#else
  return static_cast<uint16_t>(ue8m0_soft<kRound>(lo) |
                               (ue8m0_soft<kRound>(hi) << 8));
#endif
}

// 2^k for k in [-127, 127]; 2^-127 is subnormal (do not build with ftz).
__device__ __forceinline__ float pow2_f32(int k) {
  return k >= -126 ? __uint_as_float(static_cast<uint32_t>(k + 127) << 23)
                   : __uint_as_float(0x00400000u);
}

// Decode scale 2^(code - 127) and its inverse 2^(127 - code); 0xFF is NaN.
__device__ __forceinline__ float ue8m0_to_f32(uint32_t code) {
  return code == 0xffu ? __uint_as_float(0x7fc00000u)
                       : pow2_f32(static_cast<int>(code) - 127);
}
__device__ __forceinline__ float ue8m0_to_inv_f32(uint32_t code) {
  return code == 0xffu ? __uint_as_float(0x7fc00000u)
                       : pow2_f32(127 - static_cast<int>(code));
}

// OCP MXFP8/E4M3 scale: floor_pow2(amax) / 256, clamped to UE8M0.
// Multiplication by a power of two is exact in the supported normal range.
__device__ __forceinline__ uint8_t mxfp8_scale(float amax) {
  return static_cast<uint8_t>(ue8m0_soft<ScaleRound::kZero>(amax * 0x1p-8f));
}

// ----------------------------------------------------------------- E4M3

__device__ __forceinline__ uint16_t e4m3x2_rn(float lo, float hi) {
#ifdef UCCL_EP_E4M3_HW_CVT
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
#else
  return static_cast<uint16_t>(static_cast<uint32_t>(__nv_cvt_float_to_fp8(
                                   lo, __NV_SATFINITE, __NV_E4M3)) |
                               (static_cast<uint32_t>(__nv_cvt_float_to_fp8(
                                    hi, __NV_SATFINITE, __NV_E4M3))
                                << 8));
#endif
}

// v0 in bits [7:0] ... v3 in bits [31:24].
__device__ __forceinline__ uint32_t e4m3x4_rn(float v0, float v1, float v2,
                                              float v3) {
  return static_cast<uint32_t>(e4m3x2_rn(v0, v1)) |
         (static_cast<uint32_t>(e4m3x2_rn(v2, v3)) << 16);
}

// ----------------------------------------------------------------- E2M1

// |a| -> 3-bit E2M1 magnitude code, round to nearest even, saturating.
// Grid: 0, 0.5, 1, 1.5, 2, 3, 4, 6.  Ties go to the even code.
__device__ __forceinline__ uint32_t e2m1_mag_code(float a) {
  return static_cast<uint32_t>(a > 0.25f) + static_cast<uint32_t>(a >= 0.75f) +
         static_cast<uint32_t>(a > 1.25f) + static_cast<uint32_t>(a >= 1.75f) +
         static_cast<uint32_t>(a > 2.5f) + static_cast<uint32_t>(a >= 3.5f) +
         static_cast<uint32_t>(a > 5.0f);
}

__device__ __forceinline__ uint32_t e2m1_soft(float x) {
  uint32_t const sign = (__float_as_uint(x) >> 31) << 3;
  return sign | e2m1_mag_code(fabsf(x));
}

// Two E2M1 codes: `lo` in bits [3:0], `hi` in bits [7:4].  The PTX
// destination must be a .b8 register.
__device__ __forceinline__ uint32_t e2m1x2_rn(float lo, float hi) {
#ifdef UCCL_EP_FP4_HW_CVT
  uint32_t r;
  asm("{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n"
      "cvt.u32.u8 %0, t;\n}"
      : "=r"(r)
      : "f"(hi), "f"(lo));
  return r;
#else
  return e2m1_soft(lo) | (e2m1_soft(hi) << 4);
#endif
}

// Eight E2M1 codes, v[0] in bits [3:0] ... v[7] in bits [31:28].
__device__ __forceinline__ uint32_t e2m1x8_rn(float const (&v)[8]) {
#ifdef UCCL_EP_FP4_HW_CVT
  uint32_t r;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(r)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]),
        "f"(v[6]), "f"(v[7]));
  return r;
#else
  uint32_t r = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) r |= e2m1_soft(v[j]) << (4 * j);
  return r;
#endif
}

}  // namespace mx
}  // namespace uccl
