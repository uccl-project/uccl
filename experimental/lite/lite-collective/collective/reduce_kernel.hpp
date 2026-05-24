// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_REDUCE_KERNEL_HPP_
#define MSCCLPP_REDUCE_KERNEL_HPP_

#include "algorithm.hpp"
#include "gpu_data_types.hpp"
#include <type_traits>

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE)

// Generic element-wise calculation helper
template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE T cal_elements(const T& a, const T& b) {
  if constexpr (OpType == SUM) {
    return a + b;
  } else if constexpr (OpType == MIN) {
    return mscclpp::min(a, b);
  }
  static_assert(OpType == SUM || OpType == MIN, "Unsupported ReduceOp");
}

// Generic vector reduction helpers
template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE int4 cal_vector_helper(int4 const& a, int4 const& b) {
  int4 ret;
  ret.w = bit_cast<int, T>(
      cal_elements<T, OpType>(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(
      cal_elements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(
      cal_elements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(
      cal_elements<T, OpType>(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE uint2 cal_vector_helper(uint2 const& a, uint2 const& b) {
  uint2 ret;
  ret.x = bit_cast<uint32_t, T>(cal_elements<T, OpType>(
      bit_cast<T, uint32_t>(a.x), bit_cast<T, uint32_t>(b.x)));
  ret.y = bit_cast<uint32_t, T>(cal_elements<T, OpType>(
      bit_cast<T, uint32_t>(a.y), bit_cast<T, uint32_t>(b.y)));
  return ret;
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE int cal_vector_helper(int const& a, int const& b) {
  return bit_cast<int, T>(
      cal_elements<T, OpType>(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE uint32_t cal_vector_helper(uint32_t const& a,
                                                 uint32_t const& b) {
  return bit_cast<uint32_t, T>(cal_elements<T, OpType>(
      bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

// cal_vector wrapper - converts scalar types to vector types and calls
// cal_vector_helper
template <typename T, ReduceOp OpType, typename DataType>
MSCCLPP_DEVICE_INLINE DataType cal_vector(DataType const& a,
                                          DataType const& b) {
  // Define the vectorized computation type based on the element type
  static_assert(sizeof(DataType) % sizeof(T) == 0,
                "DataType size must be multiple of T size");
  static_assert(sizeof(DataType) >= 4,
                "DataType size must be at least 4 bytes");
  using CompType = typename std::conditional_t<
      std::is_same_v<T, __half>, f16x2,
      std::conditional_t<
          std::is_same_v<T, __bfloat16>, bf16x2,
          std::conditional_t<
              std::is_same_v<T, uint8_t>, u8x4,
#if defined(__FP8_TYPES_EXIST__)
              std::conditional_t<
                  std::is_same_v<T, __fp8_e4m3>, f8_e4m3x4,
                  std::conditional_t<std::is_same_v<T, __fp8_e5m2>, f8_e5m2x4,
#endif
                                     T
#if defined(__FP8_TYPES_EXIST__)
                                     >>>>>;
#else
              >>>;
#endif
  return cal_vector_helper<CompType, OpType>(a, b);
}

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

}  // namespace mscclpp

#endif  // MSCCLPP_REDUCE_KERNEL_HPP_
