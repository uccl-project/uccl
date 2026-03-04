/**
 * This was designed with the following features in mind:
 * 1. Logging and filtering across multiple levels
 * 2. Logging and filtering across multiple subsystems
 * 3. Support for stream syntax when logging
 * 4. Support for operations like CHECK and DCHECK
 * 5. Regular logging functions and log if functions
 *
 * References
 * 1. https://github.com/microsoft/mscclpp/blob/main/src/core/include/debug.h
 * 2. https://github.com/KjellKod/g3log/tree/master
 * 3. https://github.com/microsoft/mscclpp/blob/main/src/core/include/logger.hpp
 */
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>

namespace uccl {
enum UCCLLogLevel { FATAL = 0, ERROR, WARNING, INFO };

class UCCLLogger;
class UCCLLogCapture;
class UCCLVLogCapture;
class UCCLCheckCapture;
struct UCCLVoidify;

#define UCCL_LOG_INTERNAL(level)                                              \
  if (uccl::ucclLogger.shouldLog(level))                                      \
  uccl::UCCLLogCapture(uccl::ucclLogger, level, __FILE__, __LINE__, __func__) \
      .stream()

#define LOG(level) UCCL_LOG_INTERNAL(level)

#define UCCL_LOG_EVERY_N_INTERNAL(level, n)                                    \
  static std::atomic<int> log_every_n_counter_{0};                             \
  log_every_n_counter_.fetch_add(1);                                           \
  (uccl::ucclLogger.shouldLog(level) && (log_every_n_counter_ % (n) != 0))     \
      ? (void)0                                                                \
      : uccl::UCCLVoidify() &                                                  \
            (uccl::UCCLLogCapture(uccl::ucclLogger, level, __FILE__, __LINE__, \
                                  __func__)                                    \
                 .stream())

#define LOG_EVERY_N(level, n) UCCL_LOG_EVERY_N_INTERNAL(level, n)

#define UCCL_LOG_IF_INTERNAL(level, condition)                                \
  if (condition && uccl::ucclLogger.shouldLog(level))                         \
  uccl::UCCLLogCapture(uccl::ucclLogger, level, __FILE__, __LINE__, __func__) \
      .stream()

#define LOG_IF(level, condition) UCCL_LOG_IF_INTERNAL(level, condition)

#define UCCL_VLOG_INTERNAL(vlogLevel)                                      \
  if (uccl::ucclLogger.shouldVLog(vlogLevel)) {                            \
    uccl::UCCLVLogCapture(uccl::ucclLogger, vLogLevel, __FILE__, __LINE__, \
                          __func__)                                        \
        .stream()                                                          \
  }

#define VLOG(level) UCCL_VLOG_INTERNAL(level)

// here, the & operator has a lower precedence than the << operator
// hence, it the << would resolve first and the type of the : branch would also
// be void
#define UCCL_CHECK_INTERNAL(condition)                                    \
  condition                                                               \
      ? (void)0                                                           \
      : uccl::UCCLVoidify() &                                             \
            (uccl::UCCLCheckCapture(uccl::ucclLogger, __FILE__, __LINE__, \
                                    __func__, #condition, "CHECK")        \
                 .stream())

#define CHECK(condition) UCCL_CHECK_INTERNAL((condition))

#define CHECK_EQ(first, second) UCCL_CHECK_INTERNAL(((first) == (second)))

#define CHECK_NE(first, second) UCCL_CHECK_INTERNAL(((first) != (second)))

#define CHECK_LT(first, second) UCCL_CHECK_INTERNAL(((first) < (second)))

#define CHECK_LE(first, second) UCCL_CHECK_INTERNAL(((first) <= (second)))

#define CHECK_GT(first, second) UCCL_CHECK_INTERNAL(((first) > (second)))

#define CHECK_GTE(first, second) UCCL_CHECK_INTERNAL(((first) >= (second)))

#ifdef NDEBUG
#define DCHECK_EQ(first, second) \
  do {                           \
  } while (0)
#define DCHECK_NE(first, second) \
  do {                           \
  } while (0)
#define DCHECK_LT(first, second) \
  do {                           \
  } while (0)
#define DCHECK_LE(first, second) \
  do {                           \
  } while (0)
#define DCHECK_GT(first, second) \
  do {                           \
  } while (0)
#define DCHECK_GTE(first, second) \
  do {                            \
  } while (0)
#else
#define DCHECK_EQ(first, second) CHECK_EQ(first, second)
#define DCHECK_NE(first, second) CHECK_NE(first, second)
#define DCHECK_LT(first, second) CHECK_LT(first, second)
#define DCHECK_LE(first, second) CHECK_LE(first, second)
#define DCHECK_GT(first, second) CHECK_GT(first, second)
#define DCHECK_GTE(first, second) CHECK_GTE(first, second)
#endif

#define UCCL_PCHECK_INTERNAL(check)                                         \
  check ? (void)0                                                           \
        : uccl::UCCLVoidify() &                                             \
              (uccl::UCCLCheckCapture(uccl::ucclLogger, __FILE__, __LINE__, \
                                      __func__, #check, "PCHECK", errno)    \
                   .stream())

#define PCHECK(condition) UCCL_PCHECK_INTERNAL(condition)

class UCCLLogger {
 public:
  UCCLLogger(std::ostream& stream) : stream_(stream) {
    _initializeLogLevel();
    _initializeVlogLevel();
  };

  std::ostream& stream() { return stream_; }

  void log(UCCLLogLevel logLevel, std::string_view filename, int line_number,
           std::string_view function_name, std::string const& message) {
    // NOTE: also unsure if we want all threads to be stuck on this mutex
    std::lock_guard<std::mutex> lock(mu_);

    // NOTE: flush on every write, not sure if that is what we want though
    // TODO: add time?
    stream_ << "[" << logLevelToString(logLevel) << " | " << function_name
            << " | " << filename << ":" << line_number << "] " << message
            << std::endl;
  };

  bool shouldLog(UCCLLogLevel logLevel) { return logLevel <= logLevel_; }

  void vlog(int vlogLevel, std::string_view filename, int line_number,
            std::string_view function_name, std::string const& message) {
    std::lock_guard<std::mutex> lock(mu_);

    stream_ << "["
            << "VLOG " << vlogLevel << " | " << function_name << " | "
            << filename << ":" << line_number << "] " << message << std::endl;
  }

  bool shouldVLog(int vlogLevel) { return vlogLevel <= vlogLevel_; }

 private:
  std::ostream& stream_;
  std::mutex mu_;
  int logLevel_;
  int vlogLevel_{5};

  constexpr std::string_view logLevelToString(UCCLLogLevel level) {
    switch (level) {
      case UCCLLogLevel::ERROR: {
        return "ERROR";
      }
      case UCCLLogLevel::WARNING: {
        return "WARNING";
      }
      case UCCLLogLevel::INFO: {
        return "INFO";
      }
      case UCCLLogLevel::FATAL: {
        return "FATAL";
      }
    }
    return "UNKNOWN";
  }

  void _initializeLogLevel() {
    char const* loggingSubsystems = std::getenv("UCCL_LOG_LEVEL");

    std::string_view sv;

    if (loggingSubsystems) {
      sv = std::string_view{loggingSubsystems};
    } else {
      // turn on all logs by default
      logLevel_ = UCCLLogLevel::INFO;
    }

    if (sv == "FATAL") {
      logLevel_ = UCCLLogLevel::FATAL;
    } else if (sv == "ERROR") {
      logLevel_ = UCCLLogLevel::ERROR;
    } else if (sv == "WARNING") {
      logLevel_ = UCCLLogLevel::WARNING;
    } else if (sv == "INFO") {
      logLevel_ = UCCLLogLevel::INFO;
    }
  }

  void _initializeVlogLevel() {
    char const* vlog_level_str = std::getenv("UCCL_VLOG_LEVEL");
    if (vlog_level_str) {
      vlogLevel_ = std::stoi(vlog_level_str);
    }
  }

} ucclLogger(std::cout);

class UCCLLogCapture {
 public:
  UCCLLogCapture(UCCLLogger& logger, UCCLLogLevel level,
                 std::string_view fileName, int lineNumber,
                 std::string_view functionName)
      : logger_(logger),
        level_(level),
        fileName_(fileName),
        lineNumber_(lineNumber),
        functionName_(functionName){};

  ~UCCLLogCapture() {
    logger_.log(level_, fileName_, lineNumber_, functionName_, stream_.str());

    if (level_ == UCCLLogLevel::FATAL) std::abort();
  }

  std::ostringstream& stream() { return stream_; }

 private:
  UCCLLogger& logger_;
  UCCLLogLevel level_;
  std::ostringstream stream_;
  std::string_view fileName_;
  std::string_view functionName_;
  int lineNumber_;
};

class UCCLVLogCapture {
 public:
  UCCLVLogCapture(UCCLLogger& logger, int vLogLevel, std::string_view fileName,
                  int lineNumber, std::string_view functionName)
      : logger_(logger),
        vLogLevel_(vLogLevel),
        fileName_(fileName),
        lineNumber_(lineNumber),
        functionName_(functionName){};

  ~UCCLVLogCapture() {
    logger_.vlog(vLogLevel_, fileName_, lineNumber_, functionName_,
                 stream_.str());
  }

  std::ostringstream& stream() { return stream_; }

 private:
  UCCLLogger& logger_;
  int vLogLevel_;
  std::ostringstream stream_;
  std::string_view fileName_;
  std::string_view functionName_;
  int lineNumber_;
};

class UCCLCheckCapture {
 public:
  UCCLCheckCapture(UCCLLogger& logger, std::string_view fileName,
                   int lineNumber, std::string_view functionName,
                   std::string_view checkCondition, std::string_view checkType,
                   int capturedErrno = 0)
      : logger_(logger),
        fileName_(fileName),
        lineNumber_(lineNumber),
        functionName_(functionName),
        capturedErrno_(capturedErrno) {
    // add a print message to describe the failed check
    stream_ << checkType << " failed: " << checkCondition << " ";
  };

  ~UCCLCheckCapture() {
    // mimic perror behavior
    if (capturedErrno_ != 0) {
      stream_ << ": " << strerror(capturedErrno_) << " ";
    }
    logger_.log(UCCLLogLevel::FATAL, fileName_, lineNumber_, functionName_,
                stream_.str());
  }

  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  UCCLLogger& logger_;
  std::string_view fileName_;
  int lineNumber_;
  std::string_view functionName_;
  int capturedErrno_;
};

struct UCCLVoidify {
  void operator&(std::ostream&) const {}
};

}  // namespace uccl