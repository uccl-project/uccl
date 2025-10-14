#pragma once
#include "exception.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#ifndef DISABLE_SM90_FEATURES
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                     \
  cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
  cudaLaunchAttribute attr[2];                                                \
  attr[0].id = cudaLaunchAttributeCooperative;                                \
  attr[0].val.cooperative = 1;                                                \
  attr[1].id = cudaLaunchAttributeClusterDimension;                           \
  attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);                      \
  attr[1].val.clusterDim.y = 1;                                               \
  attr[1].val.clusterDim.z = 1;                                               \
  cfg.attrs = attr;                                                           \
  cfg.numAttrs = 2
#else
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
typedef struct {
  dim3 num_sms;
  dim3 num_threads;
  unsigned int shared_mem_bytes;
  hipStream_t stream;
} hipLaunchConfig_t;
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
  hipLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream};
#else
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
  int __num_sms = (sms);                          \
  int __num_threads = (threads);                  \
  auto __stream = (stream)
#endif
#endif
#endif

#ifndef LAUNCH_KERNEL
#ifndef DISABLE_SM90_FEATURES
#define LAUNCH_KERNEL(config, kernel, ...) \
  CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#else

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
template <typename T>
void fill_kernel_args(void** f, size_t idx, T&& arg) {
  f[idx] = (void*)std::addressof(arg);
}

template <typename Head, typename... Tail>
void fill_kernel_args(void** f, size_t idx, Head&& head, Tail&&... tail) {
  f[idx] = (void*)std::addressof(head);
  fill_kernel_args(f, idx + 1, std::forward<Tail>(tail)...);
}

template <typename T, typename Kern, typename... Args>
inline void LAUNCH_KERNEL(T&& config, Kern&& kernel, Args&&... args) {
  constexpr size_t k_num_kernel_args = sizeof...(args);
  void* kernel_args[k_num_kernel_args];
  fill_kernel_args(kernel_args, 0, std::forward<Args>(args)...);
  CUDA_CHECK(hipLaunchCooperativeKernel(
      std::forward<Kern>(kernel), config->num_sms, config->num_threads,
      kernel_args, config->shared_mem_bytes, config->stream));
}
#else
#define LAUNCH_KERNEL(config, kernel, ...)                          \
  do {                                                              \
    kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__); \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
      EPException cuda_exception("CUDA", __FILE__, __LINE__,        \
                                 cudaGetErrorString(e));            \
      fprintf(stderr, "%s\n", cuda_exception.what());               \
      throw cuda_exception;                                         \
    }                                                               \
  } while (0)
#endif
#endif
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#ifndef DISABLE_SM90_FEATURES
#define SET_SHARED_MEMORY_FOR_TMA(kernel)                                 \
  EP_HOST_ASSERT(cudaFuncSetAttribute(                                    \
                     kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                     smem_size) == cudaSuccess);                          \
  cfg.dynamicSmemBytes = smem_size;
#else
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#endif
#endif

#define SWITCH_HIDDEN(case_macro)                       \
  do {                                                  \
    switch (hidden) {                                   \
      case 2048:                                        \
        case_macro(2048);                               \
      case 2560:                                        \
        case_macro(2560);                               \
      case 4096:                                        \
        case_macro(4096);                               \
      case 5120:                                        \
        case_macro(5120);                               \
      case 6144:                                        \
        case_macro(6144); /* For qwen3 coder */         \
      case 7168:                                        \
        case_macro(7168);                               \
      case 8192:                                        \
        case_macro(8192);                               \
      default:                                          \
        EP_HOST_ASSERT(false and "Unsupported hidden"); \
    }                                                   \
  } while (false)

#define SWITCH_RDMA_RANKS(case_macro)                      \
  do {                                                     \
    switch (num_ranks / NUM_MAX_NVL_PEERS) {               \
      case 2:                                              \
        case_macro(2);                                     \
      case 3:                                              \
        case_macro(3);                                     \
      case 4:                                              \
        case_macro(4);                                     \
      case 8:                                              \
        case_macro(8);                                     \
      case 16:                                             \
        case_macro(16);                                    \
      default:                                             \
        EP_HOST_ASSERT(false && "Unsupported RDMA ranks"); \
    }                                                      \
  } while (false)

#define SWITCH_RANKS(case_macro)                       \
  do {                                                 \
    switch (num_ranks) {                               \
      case 2:                                          \
        case_macro(2);                                 \
      case 4:                                          \
        case_macro(4);                                 \
      case 8:                                          \
        case_macro(8);                                 \
      default:                                         \
        EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                  \
  } while (false)

#define SWITCH_TYPES(case_macro)                      \
  do {                                                \
    switch (type) {                                   \
      case CUDA_R_16BF:                               \
        case_macro(nv_bfloat16);                      \
      default:                                        \
        EP_HOST_ASSERT(false and "Unsupported type"); \
    }                                                 \
  } while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)     \
  do {                                                 \
    switch (num_ranks) {                               \
      case 2:                                          \
        case_macro(dtype, 2);                          \
      case 4:                                          \
        case_macro(dtype, 4);                          \
      case 8:                                          \
        case_macro(dtype, 8);                          \
      default:                                         \
        EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                  \
  } while (false)
