// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ALLREDUCE_COMMOM_HPP_
#define MSCCLPP_ALLREDUCE_COMMOM_HPP_

#include "algorithm.hpp"
#include "gpu_data_types.hpp"
#include "packet_device.hpp"
#include "reduce_kernel.hpp"
#include <cmath>
#include <type_traits>

#if defined(ENABLE_NPKIT)
#include "npkit.hpp"
#endif

namespace mscclpp {

namespace collective {
constexpr ReduceOp SUM = ReduceOp::SUM;
constexpr ReduceOp MIN = ReduceOp::MIN;

#if defined(MSCCLPP_DEVICE_COMPILE)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <class T>
MSCCLPP_DEVICE_INLINE constexpr std::size_t calcVectorSize() {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<U, std::int32_t> ||
                std::is_same_v<U, std::uint32_t>) {
    return 1;
  } else {
    static_assert(16 % sizeof(U) == 0,
                  "16 bytes must be divisible by sizeof(T).");
    return 16 / sizeof(U);
  }
}

template <typename T>
MSCCLPP_DEVICE_INLINE void handleMultiLoadReduceStore(T* src, T* dst,
                                                      size_t srcOffset,
                                                      size_t dstOffset,
                                                      size_t size, int tid,
                                                      int nThreads) {
  // nvls can only handle 4 bytes alignment
  MSCCLPP_ASSERT_DEVICE(size % 4 == 0, "size must be 4 bytes aligned");
  constexpr size_t nElem = calcVectorSize<T>();
  // For integer types, use 1-element vectors since multimem doesn't support
  // vectorized integer operations
  constexpr size_t vecSize =
      (std::is_same_v<T, int> || std::is_same_v<T, int32_t> ||
       std::is_same_v<T, unsigned int> || std::is_same_v<T, uint32_t>)
          ? 1
          : nElem;
  using vectorType = mscclpp::VectorType<T, vecSize>;
  const size_t nVec = size / sizeof(vectorType);
  const size_t srcOffset4 = srcOffset / sizeof(vectorType);
  const size_t dstOffset4 = dstOffset / sizeof(vectorType);
  vectorType* src4 = (vectorType*)src;
  vectorType* dst4 = (vectorType*)dst;
  for (size_t idx = tid; idx < nVec; idx += nThreads) {
    auto val = mscclpp::SwitchChannelDeviceHandle::multimemLoadReduce(
        src4 + srcOffset4 + idx);
    mscclpp::SwitchChannelDeviceHandle::multimemStore(val,
                                                      dst4 + dstOffset4 + idx);
  }
  // handle rest of data
  size_t processed = nVec * sizeof(vectorType);
  constexpr size_t nRestElem = 4 / sizeof(T);
  using restVectorType = mscclpp::VectorType<T, nRestElem>;
  const size_t startIdx = (srcOffset + processed) / sizeof(restVectorType);
  const size_t endIdx = (srcOffset + size) / sizeof(restVectorType);
  for (size_t idx = tid + startIdx; idx < endIdx; idx += nThreads) {
    auto val = mscclpp::SwitchChannelDeviceHandle::multimemLoadReduce(
        (restVectorType*)src + idx);
    mscclpp::SwitchChannelDeviceHandle::multimemStore(
        val, (restVectorType*)dst + idx);
  }
}
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

using AllreduceFunc = std::function<cudaError_t(
    void const*, void*, void*, void*, void*,
    mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
    mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t, int,
    int, int, size_t, cudaStream_t, void*, uint32_t, uint32_t, int, int)>;

template <template <ReduceOp, typename> class Adapter>
AllreduceFunc dispatch(ReduceOp op, mscclpp::DataType dtype) {
  if (op == SUM) {
    if (dtype == mscclpp::DataType::FLOAT16) {
      return Adapter<SUM, half>::call;
    } else if (dtype == mscclpp::DataType::FLOAT32) {
      return Adapter<SUM, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::BFLOAT16) {
      return Adapter<SUM, __bfloat16>::call;
#endif
#if defined(__FP8_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::FLOAT8_E4M3) {
      return Adapter<SUM, __fp8_e4m3>::call;
    } else if (dtype == mscclpp::DataType::FLOAT8_E5M2) {
      return Adapter<SUM, __fp8_e5m2>::call;
#endif
    } else if (dtype == mscclpp::DataType::INT32 ||
               dtype == mscclpp::DataType::UINT32) {
      return Adapter<SUM, int>::call;
    } else if (dtype == mscclpp::DataType::UINT8) {
      return Adapter<SUM, uint8_t>::call;
    } else {
      return nullptr;
    }
  } else if (op == MIN) {
    if (dtype == mscclpp::DataType::FLOAT16) {
      return Adapter<MIN, half>::call;
    } else if (dtype == mscclpp::DataType::FLOAT32) {
      return Adapter<MIN, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::BFLOAT16) {
      return Adapter<MIN, __bfloat16>::call;
#endif
#if defined(__FP8_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::FLOAT8_E4M3) {
      return Adapter<MIN, __fp8_e4m3>::call;
    } else if (dtype == mscclpp::DataType::FLOAT8_E5M2) {
      return Adapter<MIN, __fp8_e5m2>::call;
#endif
    } else if (dtype == mscclpp::DataType::INT32 ||
               dtype == mscclpp::DataType::UINT32) {
      return Adapter<MIN, int>::call;
    } else if (dtype == mscclpp::DataType::UINT8) {
      return Adapter<MIN, uint8_t>::call;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}
}  // namespace collective
}  // namespace mscclpp

#endif  // MSCCLPP_ALLREDUCE_COMMON_HPP_