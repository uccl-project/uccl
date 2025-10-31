#pragma once

// https://github.com/ROCm/llvm-project/blob/683ed44e262553f3bde34b09d29c1aee1e8e7663/libc/src/time/gpu/nanosleep.cpp#L2
#define __nanosleep(nsecs)                                                \
  do {                                                                    \
    auto __nsecs = static_cast<uint64_t>((nsecs));                        \
    constexpr uint64_t __clock_freq = 100000000UL;                        \
    constexpr uint64_t __ticks_per_sec = 1000000000UL;                    \
    uint64_t __tick_rate = __ticks_per_sec / __clock_freq;                \
    uint64_t __start = __builtin_readsteadycounter();                     \
    uint64_t __end = __start + (__nsecs + __tick_rate - 1) / __tick_rate; \
    uint64_t __cur = __builtin_readsteadycounter();                       \
    __builtin_amdgcn_s_sleep(2);                                          \
    while (__cur < __end) {                                               \
      __builtin_amdgcn_s_sleep(15);                                       \
      __cur = __builtin_readsteadycounter();                              \
    }                                                                     \
    uint64_t __stop = __builtin_readsteadycounter();                      \
    uint64_t __elapsed = (__stop - __start) * __tick_rate;                \
    if (__elapsed < __nsecs) {                                            \
      printf("__nanosleep elapsed time less than %lu ns", __nsecs);      \
      abort();                                                            \
    }                                                                     \
  } while (0)
