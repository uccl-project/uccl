#pragma once
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err__));                           \
      std::abort();                                                 \
    }                                                               \
  } while (0)
