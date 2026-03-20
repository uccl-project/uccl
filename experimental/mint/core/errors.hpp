// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ERRORS_HPP_
#define MSCCLPP_ERRORS_HPP_

#include <stdexcept>

namespace mscclpp {

/// Enumeration of error codes used by MSCCL++.
enum class ErrorCode {
  SystemError,    // A system error occurred.
  InternalError,  // An MSCCL++ internal error occurred.
  RemoteError,    // An error occurred on a remote system.
  InvalidUsage,   // The function was used incorrectly.
  Timeout,        // The operation timed out.
  Aborted,        // The operation was aborted.
  ExecutorError,  // An error occurred in the MSCCL++ executor.
};

/// Convert an error code to a string.
///
/// @param error The error code to convert.
/// @return The string representation of the error code.
std::string errorToString(enum ErrorCode error);

/// Base class for all errors thrown by MSCCL++.
class BaseError : public std::runtime_error {
 public:
  /// Constructor of BaseError.
  ///
  /// @param message The error message.
  /// @param errorCode The error code.
  BaseError(std::string const& message, int errorCode);

  /// Constructor of BaseError.
  ///
  /// @param errorCode The error code.
  explicit BaseError(int errorCode);

  /// Virtual destructor for BaseError.
  virtual ~BaseError() = default;

  /// Get the error code.
  ///
  /// @return The error code.
  int getErrorCode() const;

  /// Get the error message.
  ///
  /// @return The error message.
  char const* what() const noexcept override;

 protected:
  std::string message_;
  int errorCode_;
};

/// A generic error.
class Error : public BaseError {
 public:
  Error(std::string const& message, ErrorCode errorCode);
  virtual ~Error() = default;
  ErrorCode getErrorCode() const;
};

/// An error from a system call that sets `errno`.
class SysError : public BaseError {
 public:
  SysError(std::string const& message, int errorCode);
  virtual ~SysError() = default;
};

/// An error from a CUDA runtime library call.
class CudaError : public BaseError {
 public:
  CudaError(std::string const& message, int errorCode);
  virtual ~CudaError() = default;
};

/// An error from a CUDA driver library call.
class CuError : public BaseError {
 public:
  CuError(std::string const& message, int errorCode);
  virtual ~CuError() = default;
};

/// An error from an ibverbs library call.
class IbError : public BaseError {
 public:
  IbError(std::string const& message, int errorCode);
  virtual ~IbError() = default;
};

};  // namespace mscclpp

#endif  // MSCCLPP_ERRORS_HPP_
