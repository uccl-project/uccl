// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 UCCL Contributors
//
// Logging abstraction layer for UCCL.
//
// Provides glog-compatible macros with a lightweight fallback when glog is
// not available. This allows gradual migration away from glog dependency.
//
// Usage:
//   Replace: #include <glog/logging.h>
//   With:    #include "util/logging.h"
//
// The behavior is controlled by the USE_GLOG macro:
//   - If USE_GLOG is defined: Uses glog for all logging
//   - Otherwise: Uses lightweight stderr-based logging

#pragma once

#ifdef USE_GLOG
#include <glog/logging.h>
#else

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace uccl {
namespace logging {

enum LogLevel { INFO = 0, WARNING = 1, ERROR = 2, FATAL = 3 };

inline const char* LogLevelName(LogLevel level) {
  switch (level) {
    case INFO:
      return "INFO";
    case WARNING:
      return "WARNING";
    case ERROR:
      return "ERROR";
    case FATAL:
      return "FATAL";
    default:
      return "UNKNOWN";
  }
}

// Minimal log message class that outputs to stderr
class LogMessage {
 public:
  LogMessage(const char* file, int line, LogLevel level)
      : level_(level), file_(file), line_(line) {}

  ~LogMessage() {
    std::cerr << "[" << LogLevelName(level_) << "] " << file_ << ":" << line_
              << "] " << stream_.str() << std::endl;
    if (level_ == FATAL) {
      std::abort();
    }
  }

  std::ostream& stream() { return stream_; }

 private:
  LogLevel level_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

// Null stream for disabled logging (e.g., VLOG when verbosity is too low)
class NullStream {
 public:
  template <typename T>
  NullStream& operator<<(const T&) {
    return *this;
  }
  // Handle stream manipulators like std::endl
  NullStream& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
};

// Check message that aborts on destruction if condition was false
class CheckMessage {
 public:
  CheckMessage(const char* file, int line, const char* condition)
      : file_(file), line_(line), condition_(condition) {}

  ~CheckMessage() {
    std::cerr << "[FATAL] " << file_ << ":" << line_ << "] Check failed: "
              << condition_ << " " << stream_.str() << std::endl;
    std::abort();
  }

  std::ostream& stream() { return stream_; }

 private:
  const char* file_;
  int line_;
  const char* condition_;
  std::ostringstream stream_;
};

// Verbose logging level (default: 0, can be set via UCCL_VLOG_LEVEL env var)
inline int VerboseLevel() {
  static int level = []() {
    const char* env = std::getenv("UCCL_VLOG_LEVEL");
    return env ? std::atoi(env) : 0;
  }();
  return level;
}

}  // namespace logging
}  // namespace uccl

// LOG macros
#define LOG(level) \
  uccl::logging::LogMessage(__FILE__, __LINE__, uccl::logging::level).stream()

// VLOG macro - verbose logging controlled by UCCL_VLOG_LEVEL env var
#define VLOG(level)                                                 \
  (uccl::logging::VerboseLevel() >= (level))                        \
      ? uccl::logging::LogMessage(__FILE__, __LINE__,               \
                                  uccl::logging::INFO)              \
            .stream()                                               \
      : uccl::logging::NullStream()

// CHECK macros
#define CHECK(condition)                                              \
  (condition) ? (void)0                                               \
              : (void)(uccl::logging::CheckMessage(__FILE__, __LINE__, \
                                                   #condition)         \
                           .stream())

#define CHECK_NOTNULL(ptr)                                                 \
  ([&]() -> decltype(ptr) {                                                \
    if ((ptr) == nullptr) {                                                \
      uccl::logging::CheckMessage(__FILE__, __LINE__, #ptr " != nullptr")  \
              .stream()                                                    \
          << "Pointer is null";                                            \
    }                                                                      \
    return (ptr);                                                          \
  }())

// DCHECK macros - only active in debug builds
#ifdef NDEBUG
#define DCHECK(condition) \
  while (false) CHECK(condition)
#define DCHECK_EQ(a, b) \
  while (false) CHECK((a) == (b))
#define DCHECK_NE(a, b) \
  while (false) CHECK((a) != (b))
#define DCHECK_LT(a, b) \
  while (false) CHECK((a) < (b))
#define DCHECK_LE(a, b) \
  while (false) CHECK((a) <= (b))
#define DCHECK_GT(a, b) \
  while (false) CHECK((a) > (b))
#define DCHECK_GE(a, b) \
  while (false) CHECK((a) >= (b))
#else
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(a, b) CHECK((a) == (b))
#define DCHECK_NE(a, b) CHECK((a) != (b))
#define DCHECK_LT(a, b) CHECK((a) < (b))
#define DCHECK_LE(a, b) CHECK((a) <= (b))
#define DCHECK_GT(a, b) CHECK((a) > (b))
#define DCHECK_GE(a, b) CHECK((a) >= (b))
#endif

// CHECK comparison macros
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))

#endif  // USE_GLOG
