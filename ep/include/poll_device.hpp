// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_POLL_DEVICE_HPP_
#define MSCCLPP_POLL_DEVICE_HPP_

#include "assert_device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)

// If a spin is stuck, print a warning and keep spinning.
#define POLL_MAYBE_JAILBREAK(__cond, __max_spin_cnt)                                          \
  do {                                                                                        \
    [[maybe_unused]] int64_t __spin_cnt = 0;                                                  \
    while (__cond) {                                                                          \
      MSCCLPP_ASSERT_DEVICE((__max_spin_cnt < 0 || __spin_cnt++ != __max_spin_cnt), #__cond); \
    }                                                                                         \
  } while (0);

// the as POLL_MAYBE_JAILBREAK except that __cond1 is checked before __cond2
// this is specially useful when __cond1 is faster to check
#define OR_POLL_MAYBE_JAILBREAK(__cond1, __cond2, __max_spin_cnt)                                       \
  do {                                                                                                  \
    [[maybe_unused]] int64_t __spin_cnt = 0;                                                            \
    while (true) {                                                                                      \
      if (!(__cond1)) {                                                                                 \
        break;                                                                                          \
      } else if (!(__cond2)) {                                                                          \
        break;                                                                                          \
      }                                                                                                 \
      MSCCLPP_ASSERT_DEVICE((__max_spin_cnt < 0 || __spin_cnt++ != __max_spin_cnt), #__cond1 #__cond2); \
    }                                                                                                   \
  } while (0);

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

#endif  // MSCCLPP_POLL_DEVICE_HPP_
