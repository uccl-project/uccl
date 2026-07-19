#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

// UKernel CCL debug logging.
//
// Set UK_CCL_DEBUG to enable:
//   1 = EXEC  executor / ccl-level events
//   2 = TPT   + transport-level events
//   3 = ALL   + verbose heartbeats (alive/drain-spin traces)
//
// UK_DBG(lvl, fmt, ...) prints when UK_CCL_DEBUG >= lvl, with a
// [DBG<lvl>|<tid>] prefix, to stderr.

#define UK_DBG_LVL_EXEC  1
#define UK_DBG_LVL_TPT   2
#define UK_DBG_LVL_ALL   3

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

#define UK_DBG(lvl, fmt, ...)                                        \
  do {                                                               \
    if (uk_dbg_lvl() >= (lvl))                                       \
      std::fprintf(stderr, "[DBG%u|%u] " fmt "\n",                  \
                   (unsigned)(lvl), uk_tid(), ##__VA_ARGS__);        \
  } while (0)
