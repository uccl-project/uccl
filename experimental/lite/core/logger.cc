// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "logger.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <mutex>

namespace mscclpp {

static LogLevel stringToLogLevel(std::string const& levelStr) {
  // capitalize
  std::string upperStr = levelStr;
  std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);
  if (upperStr == "NONE") {
    return LogLevel::NONE;
  } else if (upperStr == "DEBUG") {
    return LogLevel::DEBUG;
  } else if (upperStr == "INFO") {
    return LogLevel::INFO;
  } else if (upperStr == "WARN") {
    return LogLevel::WARN;
  } else if (upperStr == "ERROR") {
    return LogLevel::ERROR;
  }
  return LogLevel::ERROR;  // Shouldn't reach here
}

static LogSubsysSet stringToLogSubsysSet(std::string const& subsysStr) {
  bool invert = false;
  std::string str = subsysStr;
  if (!str.empty() && str[0] == '^') {
    invert = true;
    str = str.substr(1);
  }

  auto posNextCommaOrTheEnd = [](std::string const& str, size_t start) {
    size_t pos = str.find(',', start);
    return (pos == std::string::npos) ? str.length() : pos;
  };

  std::string upperStr = str;
  std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);
  LogSubsysSet set;  // all bits start cleared
  size_t start = 0;
  size_t end = posNextCommaOrTheEnd(upperStr, start);
  while (end > start) {
    std::string token = upperStr.substr(start, end - start);
    if (token == "ENV") {
      set.set(static_cast<size_t>(LogSubsys::ENV));
    } else if (token == "GPU") {
      set.set(static_cast<size_t>(LogSubsys::GPU));
    } else if (token == "NET") {
      set.set(static_cast<size_t>(LogSubsys::NET));
    } else if (token == "CONN") {
      set.set(static_cast<size_t>(LogSubsys::CONN));
    } else if (token == "EXEC") {
      set.set(static_cast<size_t>(LogSubsys::EXEC));
    } else if (token == "NCCL") {
      set.set(static_cast<size_t>(LogSubsys::NCCL));
    } else if (token == "ALL") {
      set.set();  // all bits
    }
    start = end + 1;
    end = posNextCommaOrTheEnd(upperStr, start);
  }
  if (invert) {
    set.flip();
  }
  return set;
}

namespace detail {

std::string timestamp(char const* format) {
  // Thread-local per-second UTC timestamp cache.
  struct TimeCache {
    time_t second = 0;
    char buf[64];
    size_t len = 0;
  };
  thread_local TimeCache cache;

  auto now = std::chrono::system_clock::now();
  time_t currentTime = std::chrono::system_clock::to_time_t(now);

  // Fast path: return cached string if still in the same second
  if (cache.second == currentTime && cache.len > 0) {
    return std::string(cache.buf, cache.len);
  }

  // Slow path: format new timestamp (happens at most once per second per
  // thread)
  std::tm tmBuf;
  if (::gmtime_r(&currentTime, &tmBuf) == nullptr) {
    return "";  // Conversion failure fallback.
  }

  cache.len = ::strftime(cache.buf, sizeof(cache.buf), format, &tmBuf);
  cache.second = currentTime;
  if (cache.len == 0) {
    // Formatting failure fallback.
    return "";
  }
  return std::string(cache.buf, cache.len);
}

}  // namespace detail

static std::once_flag globalLoggerInitFlag;
static std::shared_ptr<Logger> globalLoggerPtr;

Logger::Logger(std::string const& header, const LogLevel level,
               char const delimiter)
    : header_(header), level_(level), delimiter_(delimiter) {
  subsysSet_ = stringToLogSubsysSet(env()->logSubsys);
  std::string const& path = env()->logFile;
  if (!path.empty()) {
    logFileStream_.open(path, std::ios::out | std::ios::app);
    if (!logFileStream_.is_open()) {
      // Fallback notice
      std::cerr << "MSCCLPP Logger: failed to open log file '" << path
                << "', using stdout." << std::endl;
    }
  }
}

Logger& logger(std::string const& header, std::string const& levelStr,
               char delimiter) {
  std::call_once(globalLoggerInitFlag, [&]() {
    LogLevel level = stringToLogLevel(levelStr);
    globalLoggerPtr = std::make_shared<Logger>(header, level, delimiter);
  });
  return *globalLoggerPtr;
}

}  // namespace mscclpp
