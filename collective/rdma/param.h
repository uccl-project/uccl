// This file is copied from UCCL to avoid the dependency on NCCL.

#ifndef UCCL_PARAM_H_
#define UCCL_PARAM_H_

#include <stdint.h>

char const* userHomeDir();
void setEnvFile(char const* fileName);
void initEnv();
char const* ucclGetEnv(char const* name);

void ucclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized,
                   int64_t* cache);

#define UCCL_PARAM(name, env, deftVal)                                  \
  static inline int64_t ucclParam##name() {                             \
    constexpr int64_t uninitialized = INT64_MIN;                        \
    static_assert(deftVal != uninitialized,                             \
                  "default value cannot be the uninitialized value.");  \
    static int64_t cache = uninitialized;                               \
    if (__builtin_expect(                                               \
            __atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, \
            false)) {                                                   \
      ucclLoadParam("UCCL_" env, deftVal, uninitialized, &cache);       \
    }                                                                   \
    return cache;                                                       \
  }

// Mainly used by UCCL plugin to get NCCL env parameters.
#define NCCL_PARAM(name, env, deftVal)                                  \
  static inline int64_t ncclParam##name() {                             \
    constexpr int64_t uninitialized = INT64_MIN;                        \
    static_assert(deftVal != uninitialized,                             \
                  "default value cannot be the uninitialized value.");  \
    static int64_t cache = uninitialized;                               \
    if (__builtin_expect(                                               \
            __atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, \
            false)) {                                                   \
      ucclLoadParam("NCCL_" env, deftVal, uninitialized, &cache);       \
    }                                                                   \
    return cache;                                                       \
  }

#endif
