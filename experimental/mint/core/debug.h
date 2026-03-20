// Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_DEBUG_H_
#define MSCCLPP_DEBUG_H_

#include <chrono>
#include <type_traits>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

// Conform to pthread and NVTX standard
#define MSCCLPP_THREAD_NAMELEN 16

#ifdef __cplusplus
extern "C" {
#endif

/* Error type */
typedef enum {
  mscclppSuccess = 0,
  mscclppUnhandledCudaError = 1,
  mscclppSystemError = 2,
  mscclppInternalError = 3,
  mscclppInvalidArgument = 4,
  mscclppInvalidUsage = 5,
  mscclppRemoteError = 6,
  mscclppInProgress = 7,
  mscclppNumResults = 8
} mscclppResult_t;

/* Return the string for the given error code.
 *
 * Output:
 *   returns the string
 *
 * Inputs:
 *   result: the error code that this function needs to translate
 */
char const* mscclppGetErrorString(mscclppResult_t result);

/* Log handler type which is a callback function for
 * however user likes to handle the log messages. Once set,
 * the logger will just call this function with msg.
 */
typedef void (*mscclppLogHandler_t)(char const* msg);

/* The default log handler.
 *
 * Inputs:
 *   msg: the log message
 */
void mscclppDefaultLogHandler(char const* msg);

/* Set a custom log handler.
 *
 * Inputs:
 *   handler: the log handler function
 */
mscclppResult_t mscclppSetLogHandler(mscclppLogHandler_t handler);

#ifdef __cplusplus
}  // end extern "C"
#endif

typedef enum {
  MSCCLPP_LOG_NONE = 0,
  MSCCLPP_LOG_VERSION = 1,
  MSCCLPP_LOG_WARN = 2,
  MSCCLPP_LOG_INFO = 3,
  MSCCLPP_LOG_ABORT = 4,
  MSCCLPP_LOG_TRACE = 5
} mscclppDebugLogLevel;
typedef enum {
  MSCCLPP_INIT = 1,
  MSCCLPP_COLL = 2,
  MSCCLPP_P2P = 4,
  MSCCLPP_SHM = 8,
  MSCCLPP_NET = 16,
  MSCCLPP_GRAPH = 32,
  MSCCLPP_TUNING = 64,
  MSCCLPP_ENV = 128,
  MSCCLPP_ALLOC = 256,
  MSCCLPP_CALL = 512,
  MSCCLPP_EXECUTOR = 1024,
  MSCCLPP_NCCL = 2048,
  MSCCLPP_ALL = ~0
} mscclppDebugLogSubSys;

extern int mscclppDebugLevel;
extern uint64_t mscclppDebugMask;
extern pthread_mutex_t mscclppDebugLock;
extern FILE* mscclppDebugFile;

void mscclppDebugDefaultLogHandler(char const* msg);
void mscclppDebugLog(mscclppDebugLogLevel level, unsigned long flags,
                     char const* filefunc, int line, char const* fmt, ...)
    __attribute__((format(printf, 5, 6)));
mscclppResult_t mscclppDebugSetLogHandler(mscclppLogHandler_t handler);

// Let code temporarily downgrade WARN into INFO
extern thread_local int mscclppDebugNoWarn;
extern char mscclppLastError[];

#define WARN(...)                                                    \
  mscclppDebugLog(MSCCLPP_LOG_WARN, MSCCLPP_ALL, __FILE__, __LINE__, \
                  __VA_ARGS__)
#define INFO(FLAGS, ...) \
  mscclppDebugLog(MSCCLPP_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define TRACE_CALL(...)                                                \
  mscclppDebugLog(MSCCLPP_LOG_TRACE, MSCCLPP_CALL, __func__, __LINE__, \
                  __VA_ARGS__)

#ifdef MSCCLPP_ENABLE_TRACE
#define TRACE(FLAGS, ...) \
  mscclppDebugLog(MSCCLPP_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::steady_clock::time_point mscclppEpoch;
#else
#define TRACE(...)
#endif

void mscclppSetThreadName(pthread_t thread, char const* fmt, ...);

#endif
