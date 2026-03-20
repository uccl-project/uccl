// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DATATYPE_CONVERSION_HPP_
#define MSCCLPP_DATATYPE_CONVERSION_HPP_

#include "algorithm.hpp"
#include "errors.hpp"
#include "gpu_data_types.hpp"
#include "nccl.h"
#include <cstddef>

// Convert ncclDataType_t to mscclpp::DataType
inline mscclpp::DataType ncclDataTypeToMscclpp(ncclDataType_t dtype) {
  switch (dtype) {
    case ncclInt32:
      return mscclpp::DataType::INT32;
    case ncclUint32:
      return mscclpp::DataType::UINT32;
    case ncclUint8:
      return mscclpp::DataType::UINT8;
    case ncclFloat16:
      return mscclpp::DataType::FLOAT16;
    case ncclFloat32:
      return mscclpp::DataType::FLOAT32;
    case ncclBfloat16:
      return mscclpp::DataType::BFLOAT16;
#ifdef __FP8_TYPES_EXIST__
    case ncclFloat8e4m3:
      return mscclpp::DataType::FLOAT8_E4M3;
    case ncclFloat8e5m2:
      return mscclpp::DataType::FLOAT8_E5M2;
#endif
    default:
      throw mscclpp::Error(
          "Unsupported ncclDataType_t: " + std::to_string(dtype),
          mscclpp::ErrorCode::InvalidUsage);
  }
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
