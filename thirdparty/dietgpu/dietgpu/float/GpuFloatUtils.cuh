/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StaticUtils.h"

#include <cuda.h>
#include "util/debug.h"

namespace dietgpu {

// magic number to verify archive integrity
constexpr uint32_t kFloatMagic = 0xf00f;

// current DietGPU version number
constexpr uint32_t kFloatVersion = 0x0001;

// Header on our compressed floating point data
struct __align__(16) GpuFloatHeader {
  __host__ __device__ void setMagicAndVersion() {
    magicAndVersion = (kFloatMagic << 16) | kFloatVersion;
  }

  __host__ __device__ void checkMagicAndVersion() const {
    assert((magicAndVersion >> 16) == kFloatMagic);
    assert((magicAndVersion & 0xffffU) == kFloatVersion);
  }

  __host__ __device__ FloatType getFloatType() const {
    return FloatType(options & 0xf);
  }

  __host__ __device__ void setFloatType(FloatType ft) {
    assert(uint32_t(ft) <= 0xf);
    options = (options & 0xfffffff0U) | uint32_t(ft);
  }

  __host__ __device__ bool getUseChecksum() const {
    return options & 0x10;
  }

  __host__ __device__ void setUseChecksum(bool uc) {
    options = (options & 0xffffffef) | (uint32_t(uc) << 4);
  }

  __host__ __device__ uint32_t getChecksum() const {
    return checksum;
  }

  __host__ __device__ void setChecksum(uint32_t c) {
    checksum = c;
  }

  // (16: magic)(16: version)
  uint32_t magicAndVersion;

  // Number of floating point words of the given float type in the archive
  uint32_t size;

  // (27: unused)(1: use checksum)(4: float type)
  uint32_t options;

  // Optional checksum computed on the input data
  uint32_t checksum;
};

static_assert(sizeof(GpuFloatHeader) == 16, "");

struct __align__(16) uint32x4 {
  uint32_t x[4];
};

struct __align__(16) uint16x8 {
  uint16_t x[8];
};

struct __align__(8) uint16x4 {
  uint16_t x[4];
};

struct __align__(8) uint8x8 {
  uint8_t x[8];
};

struct __align__(4) uint8x4 {
  uint8_t x[4];
};

// Convert FloatType to word size/type
template <FloatType FT>
struct FloatTypeInfo;

template <>
struct FloatTypeInfo<FloatType::kFloat16> {
  using WordT = uint16_t;
  using CompT = uint8_t;
  using NonCompT = uint8_t;

  // 16 byte vector type
  using VecT = uint16x8;
  using CompVecT = uint8x8;
  using NonCompVecT = uint8x8;

  static __device__ void split(WordT in, CompT& comp, NonCompT& nonComp) {
    // don't bother extracting the specific exponent
    comp = in >> 8;
    nonComp = in & 0xff;
  }

  static __device__ WordT join(CompT comp, NonCompT nonComp) {
    return WordT(comp) * WordT(256) + WordT(nonComp);
  }

  // How many bytes of data are in the non-compressed portion past the float
  // header?
  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    // The size of the uncompressed data is always a multiple of 16 bytes, to
    // guarantee alignment for proceeding data segments
    return roundUp(size, 16 / sizeof(NonCompT));
  }
};

template <>
struct FloatTypeInfo<FloatType::kBFloat16> {
  using WordT = uint16_t;
  using CompT = uint8_t;
  using NonCompT = uint8_t;

  // 16 byte vector type
  using VecT = uint16x8;
  using CompVecT = uint8x8;
  using NonCompVecT = uint8x8;

  static __device__ void split(WordT in, CompT& comp, NonCompT& nonComp) {
    uint32_t v = uint32_t(in) * 65536U + uint32_t(in);

    v = rotateLeft(v, 1);
    comp = v >> 24;
    nonComp = v & 0xff;
  }

  static __device__ WordT join(CompT comp, NonCompT nonComp) {
    uint32_t lo = uint32_t(comp) * 256U + uint32_t(nonComp);
    lo <<= 16;
    uint32_t hi = nonComp;

    uint32_t out;

#if defined(__HIP_PLATFORM_AMD__)
    out = (lo >> 1) | (hi << 31);
    // Emulate funnel shift right: concatenate lo:hi and shift right by 1
    // out = static_cast<uint32_t>(((uint64_t(lo) << 32) | uint64_t(hi)) >> 1);
#else
    asm("shf.r.clamp.b32 %0, %1, %2, %3;"
        : "=r"(out)
        : "r"(lo), "r"(hi), "r"(1));
#endif

    return out >>= 16;
  }

  // How many bytes of data are in the non-compressed portion past the float
  // header?
  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    // The size of the uncompressed data is always a multiple of 16 bytes, to
    // guarantee alignment for proceeding data segments
    return roundUp(size, 16 / sizeof(NonCompT));
  }
};

template <>
struct FloatTypeInfo<FloatType::kFloat32> {
  using WordT = uint32_t;
  using CompT = uint8_t;
  using NonCompT = uint32_t;

  // 16 byte vector type
  using VecT = uint32x4;
  using CompVecT = uint8x4;
  using NonCompVecT = uint32x4;

  static __device__ void split(WordT in, CompT& comp, NonCompT& nonComp) {
    auto v = rotateLeft(in, 1);
    comp = v >> 24;
    nonComp = v & 0xffffffU;
  }

  static __device__ WordT join(CompT comp, NonCompT nonComp) {
    uint32_t v = (uint32_t(comp) * 16777216U) + uint32_t(nonComp);
    return rotateRight(v, 1);
  }

  // How many bytes of data are in the non-compressed portion past the float
  // header?
  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    // The size of the uncompressed data is always a multiple of 16 bytes, to
    // guarantee alignment for proceeding data segments
    // We store the low order 2 bytes first, then the high order uncompressed
    // byte afterwards.
    // Both sections should be 16 byte aligned
    return 2 * roundUp(size, 8) + // low order 2 bytes
        roundUp(size, 16); // high order 1 byte, starting at an aligned address
                           // after the low 2 byte segment
  }
};

// Pack two fp8 e4m3fn values (8 bits each) into a single bf16-sized 16-bit word.
// Layout: [15]=a.sign, [14:11]=a.exp, [10:7]=b.exp, [6:4]=a.mant, [3]=b.sign, [2:0]=b.mant
__device__ __forceinline__ uint16_t packTwoFp8(uint8_t a, uint8_t b) {
  const uint16_t a_sign = static_cast<uint16_t>((a >> 7) & 0x1);
  const uint16_t a_exp = static_cast<uint16_t>((a >> 3) & 0xF);
  const uint16_t a_man = static_cast<uint16_t>(a & 0x7);

  const uint16_t b_sign = static_cast<uint16_t>((b >> 7) & 0x1);
  const uint16_t b_exp = static_cast<uint16_t>((b >> 3) & 0xF);
  const uint16_t b_man = static_cast<uint16_t>(b & 0x7);

  uint16_t out = 0;
  out |= (a_sign << 15);
  out |= (a_exp << 11);
  out |= (b_exp << 7);
  out |= (a_man << 4);
  out |= (b_sign << 3);
  out |= b_man;
  return out;
}

// Unpack a bf16-sized 16-bit word back to two fp8 e4m3fn values.
__device__ __forceinline__ void
unpackTwoFp8(uint16_t in, uint8_t& a, uint8_t& b) {
  const uint8_t a_sign = static_cast<uint8_t>((in >> 15) & 0x1);
  const uint8_t a_exp = static_cast<uint8_t>((in >> 11) & 0xF);
  const uint8_t b_exp = static_cast<uint8_t>((in >> 7) & 0xF);
  const uint8_t a_man = static_cast<uint8_t>((in >> 4) & 0x7);
  const uint8_t b_sign = static_cast<uint8_t>((in >> 3) & 0x1);
  const uint8_t b_man = static_cast<uint8_t>(in & 0x7);

  a = static_cast<uint8_t>((a_sign << 7) | (a_exp << 3) | a_man);
  b = static_cast<uint8_t>((b_sign << 7) | (b_exp << 3) | b_man);
}

// FP8 E4M3FN: reads pairs of fp8 as uint16_t, fuses pack/unpack with bf16
// split/join so there is no intermediate buffer.
template <>
struct FloatTypeInfo<FloatType::kFloat8E4M3FN> {
  // Each "word" is a pair of fp8 bytes read as uint16_t (little-endian)
  using WordT = uint16_t;
  using CompT = uint8_t;
  using NonCompT = uint8_t;

  using VecT = uint16x8;
  using CompVecT = uint8x8;
  using NonCompVecT = uint8x8;

  static __device__ void split(WordT in, CompT& comp, NonCompT& nonComp) {
    // 'in' is two fp8 bytes in little-endian: low = fp8[0], high = fp8[1]
    uint8_t a = static_cast<uint8_t>(in & 0xFF);
    uint8_t b = static_cast<uint8_t>(in >> 8);
    uint16_t packed = packTwoFp8(a, b);
    // Reuse bf16 split on the packed value
    FloatTypeInfo<FloatType::kBFloat16>::split(packed, comp, nonComp);
  }

  static __device__ WordT join(CompT comp, NonCompT nonComp) {
    // Reconstruct packed bf16 via bf16 join
    uint16_t packed = FloatTypeInfo<FloatType::kBFloat16>::join(comp, nonComp);
    // Unpack back to two fp8 bytes
    uint8_t a, b;
    unpackTwoFp8(packed, a, b);
    // Return little-endian uint16: low = fp8[0], high = fp8[1]
    return static_cast<uint16_t>(a) | (static_cast<uint16_t>(b) << 8);
  }

  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    return roundUp(size, 16 / sizeof(NonCompT));
  }
};

// FP8 E5M2: same packing strategy as E4M3FN.
// E5M2 layout: [7]=sign, [6:2]=exp(5 bits), [1:0]=mantissa(2 bits)
// We take the top 4 exponent bits [6:3] and pack them the same way,
// treating [2:0] (lowest exp bit + 2 mantissa bits) as the non-compressed part.
template <>
struct FloatTypeInfo<FloatType::kFloat8E5M2> {
  using WordT = uint16_t;
  using CompT = uint8_t;
  using NonCompT = uint8_t;

  using VecT = uint16x8;
  using CompVecT = uint8x8;
  using NonCompVecT = uint8x8;

  static __device__ void split(WordT in, CompT& comp, NonCompT& nonComp) {
    uint8_t a = static_cast<uint8_t>(in & 0xFF);
    uint8_t b = static_cast<uint8_t>(in >> 8);
    uint16_t packed = packTwoFp8(a, b);
    FloatTypeInfo<FloatType::kBFloat16>::split(packed, comp, nonComp);
  }

  static __device__ WordT join(CompT comp, NonCompT nonComp) {
    uint16_t packed = FloatTypeInfo<FloatType::kBFloat16>::join(comp, nonComp);
    uint8_t a, b;
    unpackTwoFp8(packed, a, b);
    return static_cast<uint16_t>(a) | (static_cast<uint16_t>(b) << 8);
  }

  static __host__ __device__ uint32_t getUncompDataSize(uint32_t size) {
    return roundUp(size, 16 / sizeof(NonCompT));
  }
};

inline size_t getWordSizeFromFloatType(FloatType ft) {
  switch (ft) {
    case FloatType::kFloat8E4M3FN:
    case FloatType::kFloat8E5M2:
      return sizeof(uint8_t);
    case FloatType::kFloat16:
    case FloatType::kBFloat16:
      return sizeof(uint16_t);
    case FloatType::kFloat32:
      return sizeof(uint32_t);
    default:
      UCCL_CHECK(false);
      return 0;
  }
}

} // namespace dietgpu
