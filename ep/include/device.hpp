// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DEVICE_HPP_
#define MSCCLPP_DEVICE_HPP_

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif  // defined(__HIP_PLATFORM_AMD__)

#if (defined(__NVCC__) || defined(__HIP_PLATFORM_AMD__))

#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_INLINE __forceinline__
#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#if defined(__HIP_PLATFORM_AMD__)
#define MSCCLPP_DEVICE_HIP
#else  // !(defined(__HIP_PLATFORM_AMD__))
#define MSCCLPP_DEVICE_CUDA
#endif  // defined(__HIP_PLATFORM_AMD__)

#else  // !(defined(__NVCC__) || defined(__HIP_PLATFORM_AMD__))

#define MSCCLPP_HOST_COMPILE
#define MSCCLPP_INLINE inline
#define MSCCLPP_HOST_DEVICE_INLINE inline

#endif  // !(defined(__NVCC__) || defined(__HIP_PLATFORM_AMD__))

#endif  // MSCCLPP_DEVICE_HPP_