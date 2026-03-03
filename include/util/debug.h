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

#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unistd.h>

namespace uccl {
enum UCCLLogLevel { FATAL = 0, ERROR, WARN, INFO };

enum UCCLLogSubsys { INIT = 0, AFXDP, DPDK, EFA, RDMA, EP, P2P, SUBSYS_COUNT };

static std::unordered_map<std::string_view, UCCLLogSubsys> subsysMap_ = {
    {"INIT", UCCLLogSubsys::INIT}, {"AFXDP", UCCLLogSubsys::AFXDP},
    {"DPDK", UCCLLogSubsys::DPDK}, {"EFA", UCCLLogSubsys::EFA},
    {"RDMA", UCCLLogSubsys::RDMA}, {"EP", UCCLLogSubsys::EP},
    {"P2P", UCCLLogSubsys::P2P},
};

class UCCLLogger;
class UCCLLogCapture;
class UCCLCheckCapture;
struct UCCLVoidify;

#define UCCL_LOG_INTERNAL(level, subsys)                                    \
  if (uccl::ucclLogger.shouldLog(level, subsys))                            \
  uccl::UCCLLogCapture(uccl::ucclLogger, level, subsys, __FILE__, __LINE__, \
                       __func__)                                            \
      .stream()

#define LOG(level, subsys) UCCL_LOG_INTERNAL(level, subsys)

#define UCCL_LOG_IF_INTERNAL(level, subsys, condition)                      \
  if (condition && uccl::ucclLogger.shouldLog(level, subsys))               \
  uccl::UCCLLogCapture(uccl::ucclLogger, level, subsys, __FILE__, __LINE__, \
                       __func__)                                            \
      .stream()

#define LOG_IF(level, subsys, condition) \
  UCCL_LOG_IF_INTERNAL(level, subsys, condition)

// here, the & operator has a lower precedence than the << operator
// hence, it the << would resolve first and the type of the : branch would also
// be void
#define UCCL_CHECK_INTERNAL(subsys, condition)                                \
  condition ? (void)0                                                         \
            : uccl::UCCLVoidify() &                                           \
                  (uccl::UCCLCheckCapture(uccl::ucclLogger, subsys, __FILE__, \
                                          __LINE__, __func__, #condition)     \
                       .stream())

#define CHECK_EQ(subsys, first, second) \
  UCCL_CHECK_INTERNAL(subsys, ((first) == (second)))

#define CHECK_NE(subsys, first, second) \
  UCCL_CHECK_INTERNAL(subsys, ((first) != (second)))

#define CHECK_LT(subsys, first, second) \
  UCCL_CHECK_INTERNAL(subsys, ((first) < (second)))

#define CHECK_LE(subsys, first, second) \
  UCCL_CHECK_INTERNAL(subsys, ((first) <= (second)))

#define CHECK_GT(subsys, first, second) \
  UCCL_CHECK_INTERNAL(subsys, ((first) > (second)))

#define CHECK_GTE(subsys, first, second) \
  UCCL_CHECK_INTERNAL(subsys, ((first) >= (second)))

#ifdef NDEBUG
#define DCHECK_EQ(subsys, first, second) \
  do {                                   \
  } while (0)
#define DCHECK_NE(subsys, first, second) \
  do {                                   \
  } while (0)
#define DCHECK_LT(subsys, first, second) \
  do {                                   \
  } while (0)
#define DCHECK_LE(subsys, first, second) \
  do {                                   \
  } while (0)
#define DCHECK_GT(subsys, first, second) \
  do {                                   \
  } while (0)
#define DCHECK_GTE(subsys, first, second) \
  do {                                    \
  } while (0)
#else
#define DCHECK_EQ(subsys, first, second) CHECK_EQ(subsys, first, second)
#define DCHECK_NE(subsys, first, second) CHECK_NE(subsys, first, second)
#define DCHECK_LT(subsys, first, second) CHECK_LT(subsys, first, second)
#define DCHECK_LE(subsys, first, second) CHECK_LE(subsys, first, second)
#define DCHECK_GT(subsys, first, second) CHECK_GT(subsys, first, second)
#define DCHECK_GTE(subsys, first, second) CHECK_GTE(subsys, first, second)
#endif

class UCCLLogger {
 public:
  UCCLLogger(std::ostream& stream) : stream_(stream) {
    _initializeLoggingSubsystems();
    _initializeLogLevel();
  };

  std::ostream& stream() { return stream_; }

  void log(UCCLLogLevel logLevel, UCCLLogSubsys subsys,
           std::string_view filename, int line_number,
           std::string_view function_name, std::string const& message) {
    // NOTE: also unsure if we want all threads to be stuck on this mutex
    std::lock_guard<std::mutex> lock(mu_);

    // NOTE: flush on every write, not sure if that is what we want though
    // TODO: add time?
    stream_ << "[" << logLevelToString(logLevel) << " | "
            << logSubsysToString(subsys) << " | " << function_name << " | "
            << filename << ":" << line_number << "] " << message << std::endl;
  };

  bool shouldLog(UCCLLogLevel logLevel, UCCLLogSubsys subsys) {
    return logLevel <= logLevel_ && subsys_bitset_.test(subsys);
  }

 private:
  std::ostream& stream_;
  std::mutex mu_;
  std::bitset<static_cast<std::size_t>(UCCLLogSubsys::SUBSYS_COUNT)>
      subsys_bitset_;
  int logLevel_;

  constexpr std::string_view logLevelToString(UCCLLogLevel level) {
    switch (level) {
      case UCCLLogLevel::ERROR: {
        return "ERROR";
      }
      case UCCLLogLevel::WARN: {
        return "WARN";
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
      case UCCLLogSubsys::INIT:
        return "INIT";

      case UCCLLogSubsys::AFXDP:
        return "AFXDP";

      case UCCLLogSubsys::DPDK:
        return "DPDK";

      case UCCLLogSubsys::EFA:
        return "EFA";

      case UCCLLogSubsys::RDMA:
        return "RDMA";

      case UCCLLogSubsys::EP:
        return "EP";

      case UCCLLogSubsys::P2P:
        return "P2P";
      default:
        break;
    }
    return "UNKNOWN";
  }

  void _initializeLoggingSubsystems() {
    char const* loggingSubsystems = std::getenv("UCCL_LOG_SUBSYS");

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
      } else if (subsysMap_.count(token)) {
        subsys_bitset_.set(subsysMap_[token]);
      }
      if (comma == std::string_view::npos) break;
      sv = sv.substr(comma + 1);
    }
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
    } else if (sv == "WARN") {
      logLevel_ = UCCLLogLevel::WARN;
    } else if (sv == "INFO") {
      logLevel_ = UCCLLogLevel::INFO;
    }
  }

} ucclLogger(std::cout);

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
                stream_.str());

    if (level_ == UCCLLogLevel::FATAL) std::abort();
  }

  std::ostringstream& stream() { return stream_; }

 private:
  UCCLLogger& logger_;
  UCCLLogLevel level_;
  UCCLLogSubsys subsys_;
  std::ostringstream stream_;
  std::string_view fileName_;
  std::string_view functionName_;
  int lineNumber_;
};

class UCCLCheckCapture {
 public:
  UCCLCheckCapture(UCCLLogger& logger, UCCLLogSubsys subsys,
                   std::string_view fileName, int lineNumber,
                   std::string_view functionName,
                   std::string_view checkCondition)
      : logger_(logger),
        subsys_(subsys),
        fileName_(fileName),
        lineNumber_(lineNumber),
        functionName_(functionName) {
    // add a print message to describe the failed check
    stream_ << "CHECK failed: " << checkCondition << " ";
  };

  ~UCCLCheckCapture() {
    logger_.log(UCCLLogLevel::FATAL, subsys_, fileName_, lineNumber_,
                functionName_, stream_.str());
  }

  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  UCCLLogger& logger_;
  UCCLLogSubsys subsys_;
  std::string_view fileName_;
  int lineNumber_;
  std::string_view functionName_;
};

struct UCCLVoidify {
  void operator&(std::ostream&) const {}
};

}  // namespace uccl