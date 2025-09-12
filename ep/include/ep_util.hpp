#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <infiniband/verbs.h>
#include <chrono>
#include <string>
#include <thread>
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

inline void drain_cq(ibv_cq* cq, int empty_rounds_target = 5) {
  if (!cq) return;
  int empty_rounds = 0;
  while (empty_rounds < empty_rounds_target) {
    ibv_wc wc[64];
    int n = ibv_poll_cq(cq, (int)(sizeof(wc) / sizeof(wc[0])), wc);
    if (n < 0) {
      fprintf(stderr, "[destroy] ibv_poll_cq returned %d\n", n);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    if (n == 0) {
      ++empty_rounds;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    for (int i = 0; i < n; ++i) {
      if (wc[i].status != IBV_WC_SUCCESS) {
      }
    }
    empty_rounds = 0;
  }
}

inline void qp_to_error(ibv_qp* qp) {
  if (!qp) return;
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_ERR;
  int ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  if (ret) {
    fprintf(stderr, "[destroy] ibv_modify_qp->ERR failed (ret=%d)\n", ret);
  }
}