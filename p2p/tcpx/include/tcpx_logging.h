/**
 * @file tcpx_logging.h
 * @brief Logging utilities and environment variable helpers
 *
 * Provides logging macros and environment variable parsing utilities
 * for the TCPX plugin. Extracted from test_tcpx_perf_multi.cc.
 */

#pragma once

#include <cstdio>
#include <cstdlib>

namespace tcpx {

// ============================================================================
// Environment Variable Helpers
// ============================================================================

/**
 * @brief Read an integer environment variable with a default value
 * @param name Environment variable name
 * @param default_val Default value if variable is not set
 * @return Parsed integer value or default
 */
inline int getEnvInt(const char* name, int default_val) {
  const char* val = std::getenv(name);
  return val ? std::atoi(val) : default_val;
}

/**
 * @brief Read a size_t environment variable with a default value
 * @param name Environment variable name
 * @param default_val Default value if variable is not set
 * @return Parsed size_t value or default
 */
inline size_t getEnvSize(const char* name, size_t default_val) {
  const char* val = std::getenv(name);
  return val ? static_cast<size_t>(std::atoll(val)) : default_val;
}

// ============================================================================
// Logging Macros
// ============================================================================

/**
 * Debug logging (controlled by TCPX_DEBUG environment variable)
 * Usage: LOG_DEBUG("Message: %d", value);
 */
#define LOG_DEBUG(fmt, ...) \
  do { \
    if (::tcpx::getEnvInt("TCPX_DEBUG", 0)) { \
      std::fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__); \
    } \
  } while (0)

/**
 * Error logging (always enabled)
 * Usage: LOG_ERROR("Error: %s", error_msg);
 */
#define LOG_ERROR(fmt, ...) \
  std::fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

/**
 * Performance logging (controlled by TCPX_PERF environment variable)
 * Usage: LOG_PERF("Bandwidth: %.2f GB/s", bw);
 */
#define LOG_PERF(fmt, ...) \
  do { \
    if (::tcpx::getEnvInt("TCPX_PERF", 0)) { \
      std::fprintf(stderr, "[PERF] " fmt "\n", ##__VA_ARGS__); \
    } \
  } while (0)

/**
 * Info logging (always enabled)
 * Usage: LOG_INFO("Status: %s", status);
 */
#define LOG_INFO(fmt, ...) \
  std::fprintf(stderr, "[INFO] " fmt "\n", ##__VA_ARGS__)

/**
 * Warning logging (always enabled)
 * Usage: LOG_WARN("Warning: %s", warning_msg);
 */
#define LOG_WARN(fmt, ...) \
  std::fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)

}  // namespace tcpx

