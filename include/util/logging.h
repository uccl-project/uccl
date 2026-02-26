// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 UCCL Contributors
//
// NCCL-compatible logging abstraction layer for UCCL.
//
// Provides NCCL-style logging macros (INFO, WARN, TRACE) that integrate
// with NCCL's debug logging infrastructure. The logger function is passed
// to plugins via pluginInit(ncclDebugLogger_t logFunction).
//
// Usage:
//   1. In plugin init: uccl_log_func = logFunction;
//   2. Throughout code: INFO(UCCL_NET, "message %d", value);
//                       WARN("error: %s", msg);
//                       TRACE(UCCL_INIT, "trace msg");

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

// Include NCCL types if building as plugin, otherwise define our own
#ifdef NCCL_NET_H_
// Already included from nccl_net.h
#else

// NCCL debug log levels
typedef enum {
  NCCL_LOG_NONE = 0,
  NCCL_LOG_VERSION = 1,
  NCCL_LOG_WARN = 2,
  NCCL_LOG_INFO = 3,
  NCCL_LOG_ABORT = 4,
  NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

// NCCL debug subsystems (can be OR'd together)
typedef enum {
  NCCL_INIT = 0x1,
  NCCL_COLL = 0x2,
  NCCL_P2P = 0x4,
  NCCL_SHM = 0x8,
  NCCL_NET = 0x10,
  NCCL_GRAPH = 0x20,
  NCCL_TUNING = 0x40,
  NCCL_ENV = 0x80,
  NCCL_ALLOC = 0x100,
  NCCL_CALL = 0x200,
  NCCL_PROXY = 0x400,
  NCCL_NVLS = 0x800,
  NCCL_BOOTSTRAP = 0x1000,
  NCCL_REG = 0x2000,
  NCCL_PROFILE = 0x4000,
  NCCL_ALL = ~0
} ncclDebugLogSubSys;

// NCCL logger function type
typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags,
                                  const char* file, int line, const char* fmt,
                                  ...);

#endif  // NCCL_NET_H_

// UCCL-specific subsystems (use high bits to avoid conflicts)
#define UCCL_INIT NCCL_INIT
#define UCCL_NET NCCL_NET
#define UCCL_P2P NCCL_P2P
#define UCCL_ALL NCCL_ALL

namespace uccl {
namespace logging {

// Global logger function pointer - set during plugin initialization
inline ncclDebugLogger_t& getLogFunc() {
  static ncclDebugLogger_t log_func = nullptr;
  return log_func;
}

// Fallback logger that writes to stderr when NCCL logger is not available
inline void fallbackLogger(ncclDebugLogLevel level, unsigned long flags,
                           const char* file, int line, const char* fmt, ...) {
  (void)flags;
  const char* level_str = "UNKNOWN";
  switch (level) {
    case NCCL_LOG_WARN:
      level_str = "WARN";
      break;
    case NCCL_LOG_INFO:
      level_str = "INFO";
      break;
    case NCCL_LOG_TRACE:
      level_str = "TRACE";
      break;
    default:
      break;
  }

  std::fprintf(stderr, "UCCL %s %s:%d ", level_str, file, line);

  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);

  std::fprintf(stderr, "\n");

  if (level == NCCL_LOG_ABORT) {
    std::abort();
  }
}

// Initialize the logger (call from pluginInit)
inline void initLogger(ncclDebugLogger_t logFunction) {
  getLogFunc() = logFunction ? logFunction : fallbackLogger;
}

// Get the active logger function
inline ncclDebugLogger_t getLogger() {
  ncclDebugLogger_t func = getLogFunc();
  return func ? func : fallbackLogger;
}

}  // namespace logging
}  // namespace uccl

// Convenience macro to get the logger
#define UCCL_LOG_FUNC (uccl::logging::getLogger())

// NCCL-style logging macros
#define WARN(fmt, ...)                                              \
  (*UCCL_LOG_FUNC)(NCCL_LOG_WARN, NCCL_ALL, __PRETTY_FUNCTION__,    \
                   __LINE__, fmt, ##__VA_ARGS__)

#define INFO(flags, fmt, ...)                                       \
  (*UCCL_LOG_FUNC)(NCCL_LOG_INFO, (flags), __PRETTY_FUNCTION__,     \
                   __LINE__, fmt, ##__VA_ARGS__)

#define TRACE(flags, fmt, ...)                                      \
  (*UCCL_LOG_FUNC)(NCCL_LOG_TRACE, (flags), __PRETTY_FUNCTION__,    \
                   __LINE__, fmt, ##__VA_ARGS__)

// For backward compatibility with code using LOG(severity) << msg pattern
// These create a LogMessage that collects stream output and logs on destruction
#ifdef UCCL_COMPAT_GLOG_STYLE

#include <sstream>
#include <string>

namespace uccl {
namespace logging {

class LogMessage {
 public:
  LogMessage(ncclDebugLogLevel level, unsigned long flags, const char* file,
             int line)
      : level_(level), flags_(flags), file_(file), line_(line) {}

  ~LogMessage() {
    (*UCCL_LOG_FUNC)(level_, flags_, file_, line_, "%s", stream_.str().c_str());
  }

  std::ostream& stream() { return stream_; }

 private:
  ncclDebugLogLevel level_;
  unsigned long flags_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

}  // namespace logging
}  // namespace uccl

#define LOG_INFO \
  uccl::logging::LogMessage(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__).stream()
#define LOG_WARNING \
  uccl::logging::LogMessage(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__).stream()
#define LOG_ERROR \
  uccl::logging::LogMessage(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__).stream()
#define LOG_FATAL \
  uccl::logging::LogMessage(NCCL_LOG_ABORT, NCCL_ALL, __FILE__, __LINE__).stream()

#define LOG(severity) LOG_##severity

#endif  // UCCL_COMPAT_GLOG_STYLE
