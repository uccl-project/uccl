// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "errors.hpp"
#include "gpu.hpp"
#include <cstring>

namespace mscclpp {

std::string errorToString(enum ErrorCode error) {
  switch (error) {
    case ErrorCode::SystemError:
      return "SystemError";
    case ErrorCode::InternalError:
      return "InternalError";
    case ErrorCode::InvalidUsage:
      return "InvalidUsage";
    case ErrorCode::Timeout:
      return "Timeout";
    case ErrorCode::Aborted:
      return "Aborted";
    case ErrorCode::ExecutorError:
      return "ExecutorError";
    default:
      return "UnknownError";
  }
}

BaseError::BaseError(std::string const& message, int errorCode)
    : std::runtime_error(""), message_(message), errorCode_(errorCode) {}

BaseError::BaseError(int errorCode)
    : std::runtime_error(""), errorCode_(errorCode) {}

int BaseError::getErrorCode() const { return errorCode_; }

char const* BaseError::what() const noexcept { return message_.c_str(); }

Error::Error(std::string const& message, ErrorCode errorCode)
    : BaseError(static_cast<int>(errorCode)) {
  message_ = message + " (Mscclpp failure: " + errorToString(errorCode) + ")";
}

ErrorCode Error::getErrorCode() const {
  return static_cast<ErrorCode>(errorCode_);
}

SysError::SysError(std::string const& message, int errorCode)
    : BaseError(errorCode) {
  message_ = message + " (System failure: " + std::strerror(errorCode) + ")";
}

CudaError::CudaError(std::string const& message, int errorCode)
    : BaseError(errorCode) {
  message_ = message + " (Cuda failure: " +
             cudaGetErrorString(static_cast<cudaError_t>(errorCode)) + ")";
}

CuError::CuError(std::string const& message, int errorCode)
    : BaseError(errorCode) {
  char const* errStr;
  if (cuGetErrorString(static_cast<CUresult>(errorCode), &errStr) !=
      CUDA_SUCCESS) {
    errStr = "failed to get error string";
  }
  message_ = message + " (Cu failure: " + errStr + ")";
}

IbError::IbError(std::string const& message, int errorCode)
    : BaseError(errorCode) {
  message_ = message + " (Ib failure: " + std::strerror(errorCode) + ")";
}

};  // namespace mscclpp