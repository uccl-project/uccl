// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ASSERT_DEVICE_HPP_
#define MSCCLPP_ASSERT_DEVICE_HPP_

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)

#include <cstdint>

#if !defined(DEBUG_BUILD)

/// Assert a condition on the device and print a message if the condition is false.
/// This macro does nothing in a release mode build (when DEBUG_BUILD is undefined).
#define MSCCLPP_ASSERT_DEVICE(__cond, __msg)

#else  // defined(DEBUG_BUILD)

#if defined(MSCCLPP_DEVICE_HIP)
extern "C" __device__ void __assert_fail(const char *__assertion, const char *__file, unsigned int __line,
                                         const char *__function);
#else   // !defined(MSCCLPP_DEVICE_HIP)
extern "C" __host__ __device__ void __assert_fail(const char *__assertion, const char *__file, unsigned int __line,
                                                  const char *__function) __THROW;
#endif  // !defined(MSCCLPP_DEVICE_HIP)

/// Assert a condition on the device and print a message if the condition is false.
/// This macro does nothing in a release mode build (when DEBUG_BUILD is undefined).
#define MSCCLPP_ASSERT_DEVICE(__cond, __msg)                         \
  do {                                                               \
    if (!(__cond)) {                                                 \
      __assert_fail(__msg, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
    }                                                                \
  } while (0)

#endif  // !defined(DEBUG_BUILD)

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

#endif  // MSCCLPP_ASSERT_DEVICE_HPP_
