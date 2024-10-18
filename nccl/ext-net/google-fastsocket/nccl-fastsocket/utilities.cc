#include "utilities.h"

static void dummyDebugLog(ncclDebugLogLevel level, uint64_t flags,
                          const char* filefunc, int line, const char* fmt,
                          ...) {}

ncclDebugLogger_t nccl_log_func = dummyDebugLog;
