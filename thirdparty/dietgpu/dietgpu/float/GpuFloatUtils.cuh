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
#include <glog/logging.h>

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
    asm("shf.r.clamp.b32 %0, %1, %2, %3;"
        : "=r"(out)
        : "r"(lo), "r"(hi), "r"(1));
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

inline size_t getWordSizeFromFloatType(FloatType ft) {
  switch (ft) {
    case FloatType::kFloat16:
    case FloatType::kBFloat16:
      return sizeof(uint16_t);
    case FloatType::kFloat32:
      return sizeof(uint32_t);
    default:
      CHECK(false);
      return 0;
  }
}

} // namespace dietgpu
