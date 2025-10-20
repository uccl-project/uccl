#pragma once

// https://github.com/ROCm/llvm-project/blob/683ed44e262553f3bde34b09d29c1aee1e8e7663/libc/src/time/gpu/nanosleep.cpp#L2
#define __nanosleep(nsecs)                                                  \
  do {                                                                      \
    constexpr uint64_t clock_freq = 100000000UL;                            \
    constexpr uint64_t ticks_per_sec = 1000000000UL;                        \
    uint64_t tick_rate = ticks_per_sec / clock_freq;                        \
    uint64_t start = __builtin_readsteadycounter();                         \
    uint64_t end =                                                          \
        start + (static_cast<uint64_t>(nsecs) + tick_rate - 1) / tick_rate; \
    uint64_t cur = __builtin_readsteadycounter();                           \
    __builtin_amdgcn_s_sleep(2);                                            \
    while (cur < end) {                                                     \
      __builtin_amdgcn_s_sleep(15);                                         \
      cur = __builtin_readsteadycounter();                                  \
    }                                                                       \
    uint64_t stop = __builtin_readsteadycounter();                          \
    uint64_t elapsed = (stop - start) * tick_rate;                          \
    if (elapsed < nsecs) {                                                  \
      printf("__nanosleep elapsed time less than %ld ns", nsecs);           \
      abort();                                                              \
    }                                                                       \
  } while (0)
