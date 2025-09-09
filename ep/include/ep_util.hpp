#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <string>
#include <cuda_runtime.h>

class EPException : public std::exception {
 private:
  std::string message = {};

 public:
  explicit EPException(char const* name, char const* file, int const line,
                       std::string const& error) {
    message = std::string("Failed: ") + name + " error " + file + ":" +
              std::to_string(line) + " '" + error + "'";
  }

  char const* what() const noexcept override { return message.c_str(); }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                     \
  do {                                                                      \
    cudaError_t e = (cmd);                                                  \
    if (e != cudaSuccess) {                                                 \
      throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                       \
  } while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                     \
  do {                                                           \
    if (not(cond)) {                                             \
      throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    }                                                            \
  } while (0)
#endif

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond)                                               \
  do {                                                                       \
    if (not(cond)) {                                                         \
      printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, \
             #cond);                                                         \
      asm("trap;");                                                          \
    }                                                                        \
  } while (0)
#endif

__device__ __forceinline__ int ld_acquire_global(int const* ptr) {
  int ret;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_release_sys_global(int const* ptr, int val) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val)
               : "memory");
}
