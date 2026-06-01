// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ATOMIC_HPP_
#define MSCCLPP_ATOMIC_HPP_

#if defined(MSCCLPP_USE_CUDA)
#ifndef MSCCLPP_DEVICE_CUDA
#define MSCCLPP_DEVICE_CUDA
#include "atomic_device.hpp"
#undef MSCCLPP_DEVICE_CUDA
#endif  // !defined(MSCCLPP_DEVICE_CUDA)
#else   // !defined(MSCCLPP_USE_CUDA)
#ifndef MSCCLPP_DEVICE_HIP
#define MSCCLPP_DEVICE_HIP
#include "atomic_device.hpp"
#undef MSCCLPP_DEVICE_HIP
#endif  // !defined(MSCCLPP_DEVICE_HIP)
#endif  // !defined(MSCCLPP_USE_CUDA)

#endif  // MSCCLPP_ATOMIC_HPP_