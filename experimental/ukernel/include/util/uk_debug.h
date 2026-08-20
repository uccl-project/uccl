#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

// UKernel CCL diagnostics — one switch (UK_CCL_DEBUG) gates all logging,
// counters, tracing and profiling:
//   1 = EXEC  executor/ccl events, put-path counters, stall dumps
//   2 = TPT   + transport/signal events and signal traffic logs
//   3 = ALL   + verbose heartbeats, per-op completion trace, host
//              orchestration profile (printed at executor teardown)
// SIGUSR2 with level >= 1 dumps all running runs.
// UK_CCL_RUN_WATCHDOG_MS is the only separate diagnostic: a functional
// watchdog that fails runs which stop making progress.
//
// UK_DBG(lvl, fmt, ...) prints when UK_CCL_DEBUG >= lvl, with a
// [DBG<lvl>|<tid>] prefix, to stderr.

#define UK_DBG_LVL_EXEC 1
#define UK_DBG_LVL_TPT 2
#define UK_DBG_LVL_ALL 3

static inline int uk_dbg_lvl() {
  static int lvl = [] {
    char const* v = std::getenv("UK_CCL_DEBUG");
    return v ? std::atoi(v) : 0;
  }();
  return lvl;
}

static inline unsigned uk_tid() {
  static thread_local unsigned id =
      (unsigned)(uintptr_t)pthread_self() & 0xFFFFu;
  return id;
}

#define UK_DBG(lvl, fmt, ...)                                                 \
  do {                                                                        \
    if (uk_dbg_lvl() >= (lvl))                                                \
      std::fprintf(stderr, "[DBG%u|%u] " fmt "\n", (unsigned)(lvl), uk_tid(), \
                   ##__VA_ARGS__);                                            \
  } while (0)
