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
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iosfwd>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <sys/syscall.h>
#include <unistd.h>

#define UCCL_DEBUG_HOSTNAME_MAX_LEN 1024

// Everything at global scope so macros always resolve via :: regardless of
// whether this header is first included from inside namespace uccl.

enum UCCLLogLevel { FATAL = 0, ERROR, WARNING, INFO };

enum UCCLLogSubsys {
  UCCL_INIT = 0,
  UCCL_AFXDP,
  UCCL_DPDK,
  UCCL_EFA,
  UCCL_RDMA,
  UCCL_EP,
  UCCL_P2P,
  UCCL_UTIL,
  UCCL_SUBSYS_COUNT
};

static std::unordered_map<std::string_view, UCCLLogSubsys>
    uccl_log_subsys_map_ = {{"INIT", UCCLLogSubsys::UCCL_INIT},
                            {"AFXDP", UCCLLogSubsys::UCCL_AFXDP},
                            {"DPDK", UCCLLogSubsys::UCCL_DPDK},
                            {"EFA", UCCLLogSubsys::UCCL_EFA},
                            {"RDMA", UCCLLogSubsys::UCCL_RDMA},
                            {"EP", UCCLLogSubsys::UCCL_EP},
                            {"P2P", UCCLLogSubsys::UCCL_P2P},
                            {"UTIL", UCCLLogSubsys::UCCL_UTIL}};

class UCCLLogger;
class UCCLLogCapture;
class UCCLVLogCapture;
class UCCLCheckCapture;
struct UCCLVoidify;
struct UCCLNullStream {};

#define UCCL_LOG_INTERNAL(level, subsys)                                  \
  (!::ucclLogger.shouldLog(level, subsys))                                \
      ? (void)0                                                           \
      : ::UCCLVoidify() & (::UCCLLogCapture(::ucclLogger, level, subsys,  \
                                            __FILE__, __LINE__, __func__) \
                               .stream())

#define UCCL_LOG(level, subsys) UCCL_LOG_INTERNAL(level, subsys)

// https://stackoverflow.com/questions/1489932/how-can-i-concatenate-twice-with-the-c-preprocessor-and-expand-a-macro-as-in-ar
#define UCCL_FUNC_NAME_CONCAT_INTERNAL(x, y) x##y

#define UCCL_LOG_EVERY_N_INTERNAL_ATOM_NAME(funcName, identifier) \
  UCCL_FUNC_NAME_CONCAT_INTERNAL(funcName, identifier)

#define UCCL_LOG_EVERY_N_INTERNAL(level, subsys, n)                       \
  static std::atomic<int> UCCL_LOG_EVERY_N_INTERNAL_ATOM_NAME(            \
      log_every_n_counter, __LINE__){1};                                  \
  (!(::ucclLogger.shouldLog(level, subsys) &&                             \
     ((UCCL_LOG_EVERY_N_INTERNAL_ATOM_NAME(log_every_n_counter, __LINE__) \
           .fetch_add(1) %                                                \
       (n)) == 0)))                                                       \
      ? (void)0                                                           \
      : ::UCCLVoidify() & ::UCCLLogCapture(::ucclLogger, level, subsys,   \
                                           __FILE__, __LINE__, __func__)  \
                              .stream()

#define UCCL_LOG_EVERY_N(level, subsys, n) \
  UCCL_LOG_EVERY_N_INTERNAL(level, subsys, n)

#define UCCL_LOG_FIRST_N_INTERNAL(level, subsys, n)                       \
  static std::atomic<int> UCCL_LOG_EVERY_N_INTERNAL_ATOM_NAME(            \
      log_first_n_counter, __LINE__){1};                                  \
  (!(::ucclLogger.shouldLog(level, subsys) &&                             \
     ((UCCL_LOG_EVERY_N_INTERNAL_ATOM_NAME(log_first_n_counter, __LINE__) \
           .fetch_add(1) <= (n)))))                                       \
      ? (void)0                                                           \
      : ::UCCLVoidify() & ::UCCLLogCapture(::ucclLogger, level, subsys,   \
                                           __FILE__, __LINE__, __func__)  \
                              .stream()

#define UCCL_LOG_FIRST_N(level, subsys, n) \
  UCCL_LOG_FIRST_N_INTERNAL(level, subsys, n)

#define UCCL_LOG_IF_INTERNAL(level, subsys, condition)                    \
  (!(condition && ::ucclLogger.shouldLog(level, subsys)))                 \
      ? (void)0                                                           \
      : ::UCCLVoidify() & (::UCCLLogCapture(::ucclLogger, level, subsys,  \
                                            __FILE__, __LINE__, __func__) \
                               .stream())

#define UCCL_LOG_IF(level, subsys, condition) \
  UCCL_LOG_IF_INTERNAL(level, subsys, condition)

#define UCCL_VLOG_IF_INTERNAL(vLogLevel, condition)                        \
  !(condition && ::ucclLogger.shouldVLog(vLogLevel))                       \
      ? (void)0                                                            \
      : ::UCCLVoidify() & (::UCCLVLogCapture(::ucclLogger, vLogLevel,      \
                                             __FILE__, __LINE__, __func__) \
                               .stream())

#define UCCL_VLOG_IF(vLogLevel, condition) \
  UCCL_VLOG_IF_INTERNAL(vLogLevel, condition)

#define UCCL_VLOG_INTERNAL(vLogLevel)                                      \
  (!::ucclLogger.shouldVLog(vLogLevel))                                    \
      ? (void)0                                                            \
      : ::UCCLVoidify() & (::UCCLVLogCapture(::ucclLogger, vLogLevel,      \
                                             __FILE__, __LINE__, __func__) \
                               .stream())

#define UCCL_VLOG(level) UCCL_VLOG_INTERNAL(level)

// here, the & operator has a lower precedence than the << operator
// hence, it the << would resolve first and the type of the : branch would
// also be void
#define UCCL_CHECK_INTERNAL(condition)                                    \
  (condition) ? (void)0                                                   \
              : ::UCCLVoidify() &                                         \
                    (::UCCLCheckCapture(::ucclLogger, __FILE__, __LINE__, \
                                        __func__, #condition, "CHECK")    \
                         .stream())

#define UCCL_CHECK(condition) UCCL_CHECK_INTERNAL((condition))

#define UCCL_CHECK_EQ(first, second) UCCL_CHECK_INTERNAL(((first) == (second)))

#define UCCL_CHECK_NE(first, second) UCCL_CHECK_INTERNAL(((first) != (second)))

#define UCCL_CHECK_LT(first, second) UCCL_CHECK_INTERNAL(((first) < (second)))

#define UCCL_CHECK_LE(first, second) UCCL_CHECK_INTERNAL(((first) <= (second)))

#define UCCL_CHECK_GT(first, second) UCCL_CHECK_INTERNAL(((first) > (second)))

#define UCCL_CHECK_GTE(first, second) UCCL_CHECK_INTERNAL(((first) >= (second)))

#ifdef NDEBUG
static UCCLNullStream uccl_nullstream;
#define UCCL_DCHECK(condition) ::uccl_nullstream
#define UCCL_DCHECK_EQ(first, second) ::uccl_nullstream
#define UCCL_DCHECK_NE(first, second) ::uccl_nullstream
#define UCCL_DCHECK_LT(first, second) ::uccl_nullstream
#define UCCL_DCHECK_LE(first, second) ::uccl_nullstream
#define UCCL_DCHECK_GT(first, second) ::uccl_nullstream
#define UCCL_DCHECK_GTE(first, second) ::uccl_nullstream
#else
#define UCCL_DCHECK(condition) UCCL_CHECK(condition)
#define UCCL_DCHECK_EQ(first, second) UCCL_CHECK_EQ(first, second)
#define UCCL_DCHECK_NE(first, second) UCCL_CHECK_NE(first, second)
#define UCCL_DCHECK_LT(first, second) UCCL_CHECK_LT(first, second)
#define UCCL_DCHECK_LE(first, second) UCCL_CHECK_LE(first, second)
#define UCCL_DCHECK_GT(first, second) UCCL_CHECK_GT(first, second)
#define UCCL_DCHECK_GTE(first, second) UCCL_CHECK_GTE(first, second)
#endif

#define UCCL_PCHECK_INTERNAL(check)                                           \
  check ? (void)0                                                             \
        : ::UCCLVoidify() &                                                   \
              (::UCCLCheckCapture(::ucclLogger, __FILE__, __LINE__, __func__, \
                                  #check, "PCHECK", errno)                    \
                   .stream())

#define UCCL_PCHECK(condition) UCCL_PCHECK_INTERNAL(condition)

template <typename T>
inline T* UCCLCheckNotNullCapture(void* ptr, char const* expr);

#define UCCL_CHECK_NOTNULL_INTERNAL(ptr) UCCLCheckNotNullCapture(ptr, #ptr)

#define UCCL_CHECK_NOTNULL(ptr) UCCL_CHECK_NOTNULL_INTERNAL(ptr)

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

constexpr std::string_view logSubsysToString(UCCLLogSubsys subsys) {
  switch (subsys) {
    case UCCLLogSubsys::UCCL_INIT:
      return "INIT";
    case UCCLLogSubsys::UCCL_AFXDP:
      return "AFXDP";
    case UCCLLogSubsys::UCCL_DPDK:
      return "DPDK";
    case UCCLLogSubsys::UCCL_EFA:
      return "EFA";
    case UCCLLogSubsys::UCCL_RDMA:
      return "RDMA";
    case UCCLLogSubsys::UCCL_EP:
      return "EP";
    case UCCLLogSubsys::UCCL_P2P:
      return "P2P";
    case UCCLLogSubsys::UCCL_UTIL:
      return "UTIL";
    default:
      return "";
  }
}

class UCCLLogger {
 public:
  UCCLLogger(std::ostream& stream) : stream_(stream) {
    _initializeLogLevel();
    _initializeLoggingSubsystems();
    _initializeVlogLevel();

    gethostname(hostname_, UCCL_DEBUG_HOSTNAME_MAX_LEN);
    pid_ = getpid();
  };

  std::ostream& stream() { return stream_; }

  void setLogLevel(UCCLLogLevel level) { logLevel_ = level; }

  void log(UCCLLogLevel logLevel, UCCLLogSubsys subsys,
           std::string_view filename, int line_number,
           std::string_view function_name, int threadId,
           std::string_view const& message) {
    // NOTE: also unsure if we want all threads to be stuck on this mutex
    std::lock_guard<std::mutex> lock(mu_);

    // NOTE: flush on every write, not sure if that is what we want though
    // TODO: add time?
    stream_ << "[" << logLevelToString(logLevel);

    if (logSubsysToString(subsys).size() > 0) {
      stream_ << " | " << logSubsysToString(subsys);
    }

    stream_ << " | " << hostname_ << " | " << pid_ << " | " << threadId << " | "
            << function_name << " | " << filename << ":" << line_number << "] "
            << message;

    if (logLevel == UCCLLogLevel::FATAL) {
      stream_ << std::endl;
      std::abort();
    } else {
      stream_ << '\n';
    }
  };

  bool shouldLog(UCCLLogLevel logLevel, UCCLLogSubsys subsys) {
    return logLevel <= logLevel_ && subsys_bitset_.test(subsys);
  }

  void vlog(int vlogLevel, std::string_view filename, int line_number,
            std::string_view function_name, int threadId,
            std::string const& message) {
    std::lock_guard<std::mutex> lock(mu_);

    stream_ << "["
            << "VLOG(" << vlogLevel << ") | " << hostname_ << " | " << pid_
            << " | " << threadId << " | " << function_name << " | " << filename
            << ":" << line_number << "] " << message << '\n';
  }

  bool shouldVLog(int vlogLevel) { return vlogLevel <= vlogLevel_; }

 private:
  std::ostream& stream_;
  std::mutex mu_;
  int logLevel_;
  int vlogLevel_{0};
  std::bitset<static_cast<std::size_t>(UCCLLogSubsys::UCCL_SUBSYS_COUNT)>
      subsys_bitset_;
  pid_t pid_;
  char hostname_[UCCL_DEBUG_HOSTNAME_MAX_LEN]{};

  void _initializeLogLevel() {
    char const* loggingLevel = std::getenv("UCCL_DEBUG");

    std::string_view sv;

    if (loggingLevel) {
      sv = std::string_view{loggingLevel};
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
    char const* vlog_level_str = std::getenv("UCCL_DEBUG_VLOG_LEVEL");
    if (vlog_level_str) {
      vlogLevel_ = std::stoi(vlog_level_str);
    }
  }

  void _initializeLoggingSubsystems() {
    char const* loggingSubsystems = std::getenv("UCCL_DEBUG_SUBSYS");

    std::string_view sv;

    if (loggingSubsystems) {
      sv = std::string_view{loggingSubsystems};
    } else {
      // if env is not set, enable all logs by default
      subsys_bitset_.set();
    }

    while (!sv.empty()) {
      auto comma = sv.find(',');
      auto token = sv.substr(0, comma);
      if (token == "ALL") {
        subsys_bitset_.set();
      } else if (uccl_log_subsys_map_.count(token)) {
        subsys_bitset_.set(uccl_log_subsys_map_[token]);
      }
      if (comma == std::string_view::npos) break;
      sv = sv.substr(comma + 1);
    }
  }
};
inline UCCLLogger ucclLogger(std::cout);

class UCCLLogCapture {
 public:
  UCCLLogCapture(UCCLLogger& logger, UCCLLogLevel level, UCCLLogSubsys subsys,
                 std::string_view fileName, int lineNumber,
                 std::string_view functionName)
      : logger_(logger),
        level_(level),
        subsys_(subsys),
        fileName_(fileName),
        lineNumber_(lineNumber),
        functionName_(functionName){};

  ~UCCLLogCapture() {
    logger_.log(level_, subsys_, fileName_, lineNumber_, functionName_,
                getThreadId(), stream_.str());
  }

  static int getThreadId() {
    thread_local int threadId_ = -1;
    if (threadId_ == -1) {
      threadId_ = static_cast<int>(syscall(SYS_gettid));
    }
    return threadId_;
  }

  std::ostringstream& stream() { return stream_; }

 private:
  UCCLLogger& logger_;
  UCCLLogLevel level_;
  UCCLLogSubsys subsys_;
  std::ostringstream stream_;
  std::string_view fileName_;
  int lineNumber_;
  std::string_view functionName_;
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
                 getThreadId(), stream_.str());
  }

  std::ostringstream& stream() { return stream_; }

  static int getThreadId() {
    thread_local int threadId_ = -1;
    if (threadId_ == -1) {
      threadId_ = static_cast<int>(syscall(SYS_gettid));
    }
    return threadId_;
  }

 private:
  UCCLLogger& logger_;
  int vLogLevel_;
  std::ostringstream stream_;
  std::string_view fileName_;
  int lineNumber_;
  std::string_view functionName_;
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
    logger_.log(UCCLLogLevel::FATAL, UCCLLogSubsys::UCCL_SUBSYS_COUNT,
                fileName_, lineNumber_, functionName_, getThreadId(),
                stream_.str());
  }

  std::ostringstream& stream() { return stream_; }

  static int getThreadId() {
    thread_local int threadId_ = -1;
    if (threadId_ == -1) {
      threadId_ = static_cast<int>(syscall(SYS_gettid));
    }
    return threadId_;
  }

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

template <typename T>
inline T* UCCLCheckNotNullCapture(T* ptr, char const* expr) {
  if (ptr == nullptr) {
    ::UCCLCheckCapture(::ucclLogger, __FILE__, __LINE__, __func__, expr,
                       "CHECK_NOTNULL", 0);
  }

  return ptr;
}

// https://stackoverflow.com/questions/8433302/null-stream-do-i-have-to-include-ostream
// Swallow all types
template <typename T>
inline UCCLNullStream& operator<<(UCCLNullStream& s, T const&) {
  return s;
}

// Swallow manipulator templates
inline UCCLNullStream& operator<<(UCCLNullStream& s,
                                  std::ostream&(std::ostream&)) {
  return s;
}

// Re-export into uccl for backward compatibility.
namespace uccl {
using ::ERROR;
using ::FATAL;
using ::INFO;
using ::logLevelToString;
using ::logSubsysToString;
using ::UCCL_AFXDP;
using ::UCCL_DPDK;
using ::UCCL_EFA;
using ::UCCL_EP;
using ::UCCL_INIT;
using ::UCCL_P2P;
using ::UCCL_RDMA;
using ::UCCL_SUBSYS_COUNT;
using ::UCCL_UTIL;
using ::UCCLCheckCapture;
using ::UCCLLogCapture;
using ::UCCLLogger;
using ::ucclLogger;
using ::UCCLLogLevel;
using ::UCCLLogSubsys;
using ::UCCLNullStream;
using ::UCCLVLogCapture;
using ::UCCLVoidify;
using ::WARNING;
}  // namespace uccl