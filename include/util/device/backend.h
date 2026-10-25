#pragma once

// The build selects a config header. Legacy defines remain supported for
// EP/GIN and external consumers of util/gpu_rt.h.
#if defined(UCCL_USE_MUSA) &&                                                \
    (defined(__HIP_PLATFORM_AMD__) || defined(__CAMBRICON_PLATFORM_MLU__) || \
     defined(USE_CUDA) || defined(__CUDACC__))
#error "UCCL_USE_MUSA conflicts with another GPU backend"
#endif
#if defined(UCCL_DEVICE_BACKEND_HEADER)
#include UCCL_DEVICE_BACKEND_HEADER
#elif defined(UCCL_USE_MUSA)
#include "util/device/musa/config.h"
#elif defined(__CAMBRICON_PLATFORM_MLU__)
#include "util/device/cnrt/config.h"
#elif defined(__HIP_PLATFORM_AMD__)
#include "util/device/hip/config.h"
#else
#include "util/device/cuda/config.h"
#endif
