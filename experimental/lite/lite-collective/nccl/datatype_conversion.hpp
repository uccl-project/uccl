// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DATATYPE_CONVERSION_HPP_
#define MSCCLPP_DATATYPE_CONVERSION_HPP_

#include "algorithm.hpp"
#include "gpu_data_types.hpp"
#include "nccl.h"
#include <cstddef>

// Convert ncclDataType_t to mscclpp::DataType without throwing.
inline bool tryNcclDataTypeToMscclpp(ncclDataType_t dtype,
                                     mscclpp::DataType* out) {
  switch (dtype) {
    case ncclInt32:
      *out = mscclpp::DataType::INT32;
      return true;
    case ncclUint32:
      *out = mscclpp::DataType::UINT32;
      return true;
    case ncclUint8:
      *out = mscclpp::DataType::UINT8;
      return true;
    case ncclFloat16:
      *out = mscclpp::DataType::FLOAT16;
      return true;
    case ncclFloat32:
      *out = mscclpp::DataType::FLOAT32;
      return true;
    case ncclBfloat16:
      *out = mscclpp::DataType::BFLOAT16;
      return true;
#ifdef __FP8_TYPES_EXIST__
    case ncclFloat8e4m3:
      *out = mscclpp::DataType::FLOAT8_E4M3;
      return true;
    case ncclFloat8e5m2:
      *out = mscclpp::DataType::FLOAT8_E5M2;
      return true;
#endif
    default:
      return false;
  }
}

// Convert ncclDataType_t to mscclpp::DataType
inline mscclpp::DataType ncclDataTypeToMscclpp(ncclDataType_t dtype) {
  mscclpp::DataType out;
  if (tryNcclDataTypeToMscclpp(dtype, &out)) return out;
  throw mscclpp::Error("Unsupported ncclDataType_t: " + std::to_string(dtype),
                       mscclpp::ErrorCode::InvalidUsage);
}

// Get the size in bytes of a data type
inline size_t getDataTypeSize(mscclpp::DataType dtype) {
  switch (dtype) {
    case mscclpp::DataType::UINT8:
    case mscclpp::DataType::FLOAT8_E4M3:
    case mscclpp::DataType::FLOAT8_E5M2:
      return 1;
    case mscclpp::DataType::FLOAT16:
    case mscclpp::DataType::BFLOAT16:
      return 2;
    case mscclpp::DataType::INT32:
    case mscclpp::DataType::UINT32:
    case mscclpp::DataType::FLOAT32:
      return 4;
    default:
      return 0;
  }
}

static inline ncclDataType_t mscclppToNcclDataType(mscclpp::DataType dtype) {
  switch (dtype) {
    case mscclpp::DataType::INT32:
      return ncclInt32;
    case mscclpp::DataType::UINT32:
      return ncclUint32;
    case mscclpp::DataType::UINT8:
      return ncclUint8;
    case mscclpp::DataType::FLOAT16:
      return ncclFloat16;
    case mscclpp::DataType::FLOAT32:
      return ncclFloat32;
    case mscclpp::DataType::BFLOAT16:
      return ncclBfloat16;
#ifdef __FP8_TYPES_EXIST__
    case mscclpp::DataType::FLOAT8_E4M3:
      return ncclFloat8e4m3;
    case mscclpp::DataType::FLOAT8_E5M2:
      return ncclFloat8e5m2;
#endif
    default:
      throw mscclpp::Error("Unsupported mscclpp::DataType: " +
                               std::to_string(static_cast<int>(dtype)),
                           mscclpp::ErrorCode::InvalidUsage);
  }
}

inline mscclpp::ReduceOp ncclRedOpToMscclpp(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:
      return mscclpp::ReduceOp::SUM;
    case ncclMin:
      return mscclpp::ReduceOp::MIN;
    default:
      return mscclpp::ReduceOp::NOP;
  }
}

#endif  // MSCCLPP_DATATYPE_CONVERSION_HPP_
