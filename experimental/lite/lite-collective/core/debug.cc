// Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "debug.h"
#include "env.hpp"
#include "gpu_utils.hpp"
#include "utils.hpp"
#include <mutex>
#include <string>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

int mscclppDebugLevel = -1;
static int pid = -1;
static std::string hostname;
thread_local int mscclppDebugNoWarn = 0;
char mscclppLastError[1024] =
    "";  // Global string for the last error in human readable form
uint64_t mscclppDebugMask =
    MSCCLPP_INIT;  // Default debug sub-system mask is INIT
FILE* mscclppDebugFile = stdout;
mscclppLogHandler_t mscclppDebugLogHandler = NULL;
std::recursive_mutex mscclppDebugMutex;
std::chrono::steady_clock::time_point mscclppEpoch;

static __thread int tid = -1;

void mscclppDebugDefaultLogHandler(char const* msg) {
  fwrite(msg, 1, strlen(msg), mscclppDebugFile);
}

void mscclppDebugInit() {
  std::lock_guard<std::recursive_mutex> lock(mscclppDebugMutex);
  if (mscclppDebugLevel != -1) {
    return;
  }
  char const* mscclpp_debug = mscclpp::env()->debug.c_str();
  int tempNcclDebugLevel = -1;
  if (mscclpp_debug == NULL) {
    tempNcclDebugLevel = MSCCLPP_LOG_NONE;
  } else if (strcasecmp(mscclpp_debug, "VERSION") == 0) {
    tempNcclDebugLevel = MSCCLPP_LOG_VERSION;
  } else if (strcasecmp(mscclpp_debug, "WARN") == 0) {
    tempNcclDebugLevel = MSCCLPP_LOG_WARN;
  } else if (strcasecmp(mscclpp_debug, "INFO") == 0) {
    tempNcclDebugLevel = MSCCLPP_LOG_INFO;
  } else if (strcasecmp(mscclpp_debug, "ABORT") == 0) {
    tempNcclDebugLevel = MSCCLPP_LOG_ABORT;
  } else if (strcasecmp(mscclpp_debug, "TRACE") == 0) {
    tempNcclDebugLevel = MSCCLPP_LOG_TRACE;
  }

  /* Parse the MSCCLPP_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  std::string mscclppDebugSubsysStr = mscclpp::env()->debugSubsys;
  char const* mscclppDebugSubsysEnv = mscclppDebugSubsysStr.c_str();
  if (mscclppDebugSubsysStr != "") {
    int invert = 0;
    if (mscclppDebugSubsysEnv[0] == '^') {
      invert = 1;
      mscclppDebugSubsysEnv++;
    }
    mscclppDebugMask = invert ? ~0ULL : 0ULL;
    char* mscclppDebugSubsys = strdup(mscclppDebugSubsysEnv);
    char* subsys = strtok(mscclppDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = MSCCLPP_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = MSCCLPP_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = MSCCLPP_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = MSCCLPP_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = MSCCLPP_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = MSCCLPP_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = MSCCLPP_TUNING;
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = MSCCLPP_ENV;
      } else if (strcasecmp(subsys, "ALLOC") == 0) {
        mask = MSCCLPP_ALLOC;
      } else if (strcasecmp(subsys, "CALL") == 0) {
        mask = MSCCLPP_CALL;
      } else if (strcasecmp(subsys, "EXECUTOR") == 0) {
        mask = MSCCLPP_EXECUTOR;
      } else if (strcasecmp(subsys, "NCCL") == 0) {
        mask = MSCCLPP_NCCL;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = MSCCLPP_ALL;
      }
      if (mask) {
        if (invert)
          mscclppDebugMask &= ~mask;
        else
          mscclppDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(mscclppDebugSubsys);
  }

  // Cache pid and hostname
  hostname = mscclpp::getHostName(1024, '.');
  pid = getpid();

  /* Parse and expand the MSCCLPP_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * MSCCLPP_DEBUG level is > VERSION
   */
  char const* mscclppDebugFileEnv = mscclpp::env()->debugFile.c_str();
  if (tempNcclDebugLevel > MSCCLPP_LOG_VERSION && mscclppDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX + 1] = "";
    char* dfn = debugFn;
    while (mscclppDebugFileEnv[c] != '\0' && c < PATH_MAX) {
      if (mscclppDebugFileEnv[c++] != '%') {
        *dfn++ = mscclppDebugFileEnv[c - 1];
        continue;
      }
      switch (mscclppDebugFileEnv[c++]) {
        case '%':  // Double %
          *dfn++ = '%';
          break;
        case 'h':  // %h = hostname
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname.c_str());
          break;
        case 'p':  // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", pid);
          break;
        default:  // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = mscclppDebugFileEnv[c - 1];
          break;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE* file = fopen(debugFn, "w");
      if (file != nullptr) {
        setbuf(file, nullptr);  // disable buffering
        mscclppDebugFile = file;
      }
    }
  }

  if (mscclppDebugLogHandler == NULL)
    mscclppDebugLogHandler = mscclppDefaultLogHandler;

  mscclppEpoch = std::chrono::steady_clock::now();
  __atomic_store_n(&mscclppDebugLevel, tempNcclDebugLevel, __ATOMIC_RELEASE);
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void mscclppDebugLog(mscclppDebugLogLevel level, unsigned long flags,
                     char const* filefunc, int line, char const* fmt, ...) {
  if (__atomic_load_n(&mscclppDebugLevel, __ATOMIC_ACQUIRE) == -1)
    mscclppDebugInit();
  if (mscclppDebugNoWarn != 0 && level == MSCCLPP_LOG_WARN) {
    level = MSCCLPP_LOG_INFO;
    flags = mscclppDebugNoWarn;
  }
  // Save the last error (WARN) as a human readable string
  if (level == MSCCLPP_LOG_WARN) {
    std::lock_guard<std::recursive_mutex> lock(mscclppDebugMutex);
    va_list vargs;
    va_start(vargs, fmt);
    (void)vsnprintf(mscclppLastError, sizeof(mscclppLastError), fmt, vargs);
    va_end(vargs);
  }
  if (mscclppDebugLevel < level || ((flags & mscclppDebugMask) == 0)) return;

  if (tid == -1) {
    tid = syscall(SYS_gettid);
  }

  int cudaDev;
  if (!(level == MSCCLPP_LOG_TRACE && flags == MSCCLPP_CALL)) {
    MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDev));
  }

  char buffer[1024];
  size_t len = 0;
  if (level == MSCCLPP_LOG_WARN) {
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] %s:%d MSCCLPP WARN ",
                   hostname.c_str(), pid, tid, cudaDev, filefunc, line);
  } else if (level == MSCCLPP_LOG_INFO) {
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] MSCCLPP INFO ",
                   hostname.c_str(), pid, tid, cudaDev);
  } else if (level == MSCCLPP_LOG_TRACE && flags == MSCCLPP_CALL) {
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d MSCCLPP CALL ",
                   hostname.c_str(), pid, tid);
  } else if (level == MSCCLPP_LOG_TRACE) {
    auto delta = std::chrono::steady_clock::now() - mscclppEpoch;
    double timestamp =
        std::chrono::duration_cast<std::chrono::duration<double>>(delta)
            .count() *
        1000;
    len = snprintf(buffer, sizeof(buffer),
                   "%s:%d:%d [%d] %f %s:%d MSCCLPP TRACE ", hostname.c_str(),
                   pid, tid, cudaDev, timestamp, filefunc, line);
  }

  if (len > 0) {
    va_list vargs;
    va_start(vargs, fmt);
    int ret = vsnprintf(buffer + len, sizeof(buffer) - len, fmt, vargs);
    va_end(vargs);
    if (ret >= 0) {
      len += ret;
      buffer[len++] = '\n';
      buffer[len] = '\0';
      mscclppDebugLogHandler(buffer);
    }
  }
}

mscclppResult_t mscclppDebugSetLogHandler(mscclppLogHandler_t handler) {
  if (__atomic_load_n(&mscclppDebugLevel, __ATOMIC_ACQUIRE) == -1)
    mscclppDebugInit();
  if (handler == NULL) return mscclppInvalidArgument;
  std::lock_guard<std::recursive_mutex> lock(mscclppDebugMutex);
  mscclppDebugLogHandler = handler;
  return mscclppSuccess;
}

void mscclppSetThreadName(pthread_t thread, char const* fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  char threadName[MSCCLPP_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, MSCCLPP_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread, threadName);
#endif
}
