#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_acle.h>
#else
static_assert(false,
              "Unsupported architecture, please add the pause intrinsic for "
              "the architecture.");
#endif

static void inline machnet_pause() {
#if defined(__x86_64__)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    __asm__ volatile("yield" ::: "memory");
#else
    static_assert(
        false,
        "Unsupported architecture, please add the pause intrinsic for "
        "the architecture.");
#endif
}

#ifdef __cplusplus
}
#endif
