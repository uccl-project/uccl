// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ATOMIC_HPP_
#define MSCCLPP_ATOMIC_HPP_

#if !defined(__HIP_DEVICE_COMPILE__)
#define MSCCLPP_DEVICE_CUDA
#include "atomic_device.hpp"
#undef MSCCLPP_DEVICE_CUDA
#else  // !defined(__HIP_DEVICE_COMPILE__)
#define MSCCLPP_DEVICE_HIP
#include "atomic_device.hpp"
#undef MSCCLPP_DEVICE_HIP
#endif  // !defined(__HIP_DEVICE_COMPILE__)

#endif  // MSCCLPP_ATOMIC_HPP_
