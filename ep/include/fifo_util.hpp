// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <numa.h>

// ============================================================================
// GPU Platform Abstraction (from gpu.hpp)
// ============================================================================
#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaMemcpyKind = hipMemcpyKind;

constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaHostAllocMapped = hipHostMallocMapped;

#define cudaGetDevice(...) hipGetDevice(__VA_ARGS__)

#else

#include <cuda.h>
#include <cuda_runtime.h>

#endif

// ============================================================================
// Device Compilation Macros (from device.hpp)
// ============================================================================
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

// ============================================================================
// CUDA Atomic includes (must be outside namespace)
// ============================================================================
#if defined(MSCCLPP_DEVICE_CUDA)
#include <cuda/atomic>
#endif

namespace mscclpp {

// ============================================================================
// Error Handling (from errors.hpp)
// ============================================================================

/// Base class for all errors thrown by MSCCL++.
class BaseError : public std::runtime_error {
 public:
  BaseError(std::string const& message, int errorCode)
      : std::runtime_error(""), message_(message), errorCode_(errorCode) {}

  explicit BaseError(int errorCode)
      : std::runtime_error(""), errorCode_(errorCode) {}

  virtual ~BaseError() = default;

  int getErrorCode() const { return errorCode_; }

  char const* what() const noexcept override { return message_.c_str(); }

 protected:
  std::string message_;
  int errorCode_;
};

/// An error from a CUDA runtime library call.
class CudaError : public BaseError {
 public:
  CudaError(std::string const& message, int errorCode) : BaseError(errorCode) {
    message_ = message + " (Cuda failure: " +
               cudaGetErrorString(static_cast<cudaError_t>(errorCode)) + ")";
  }
  virtual ~CudaError() = default;
};

/// An error from a CUDA driver library call.
class CuError : public BaseError {
 public:
  CuError(std::string const& message, int errorCode) : BaseError(errorCode) {
    char const* errStr;
    if (cuGetErrorString(static_cast<CUresult>(errorCode), &errStr) !=
        CUDA_SUCCESS) {
      errStr = "failed to get error string";
    }
    message_ = message + " (Cu failure: " + errStr + ")";
  }
  virtual ~CuError() = default;
};

// ============================================================================
// CUDA Error Checking Macros (from gpu_utils.hpp)
// ============================================================================

/// Throw mscclpp::CudaError if @p cmd does not return cudaSuccess.
#define MSCCLPP_CUDATHROW(cmd)                                              \
  do {                                                                      \
    cudaError_t err = cmd;                                                  \
    if (err != cudaSuccess) {                                               \
      throw ::mscclpp::CudaError(std::string("Call to " #cmd " failed. ") + \
                                     __FILE__ + ":" +                       \
                                     std::to_string(__LINE__),              \
                                 err);                                      \
    }                                                                       \
  } while (false)

/// Throw mscclpp::CuError if @p cmd does not return CUDA_SUCCESS.
#define MSCCLPP_CUTHROW(cmd)                                                  \
  do {                                                                        \
    CUresult err = cmd;                                                       \
    if (err != CUDA_SUCCESS) {                                                \
      throw ::mscclpp::CuError(std::string("Call to " #cmd " failed. ") +     \
                                   __FILE__ + ":" + std::to_string(__LINE__), \
                               err);                                          \
    }                                                                         \
  } while (false)

// ============================================================================
// NUMA Functions (from numa.hpp)
// ============================================================================

// Convert a logical deviceId index to the NVML device minor number
static inline std::string const getBusId(int deviceId) {
  char busIdChar[] = "00000000:00:00.0";
  MSCCLPP_CUDATHROW(
      cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
  // we need the hex in lower case format
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
}

inline int getDeviceNumaNode(int deviceId) {
  std::string busId = getBusId(deviceId);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/numa_node";
  std::ifstream file(file_str);
  int numaNode = -1;
  if (file.is_open()) {
    if (!(file >> numaNode)) {
      // Failed to read - return -1
      return -1;
    }
  } else {
    // Failed to open file - return -1
    return -1;
  }
  return numaNode;
}

inline void numaBind(int node) {
  int totalNumNumaNodes = numa_num_configured_nodes();
  if (node < 0 || node >= totalNumNumaNodes) {
    // Invalid node - just return
    return;
  }
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set_compat(&mask, node);
  numa_bind_compat(&mask);
}

// ============================================================================
// Atomic Operations (from atomic_device.hpp)
// ============================================================================

#if defined(MSCCLPP_DEVICE_CUDA)

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;
constexpr cuda::memory_order memoryOrderRelease = cuda::memory_order_release;
constexpr cuda::memory_order memoryOrderAcqRel = cuda::memory_order_acq_rel;
constexpr cuda::memory_order memoryOrderSeqCst = cuda::memory_order_seq_cst;

constexpr cuda::thread_scope scopeSystem = cuda::thread_scope_system;
constexpr cuda::thread_scope scopeDevice = cuda::thread_scope_device;

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(T* ptr,
                                        cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, Scope>{*ptr}.load(memoryOrder);
}

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, T const& val,
                                            cuda::memory_order memoryOrder) {
  cuda::atomic_ref<T, Scope>{*ptr}.store(val, memoryOrder);
}

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, T const& val,
                                            cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, Scope>{*ptr}.fetch_add(val, memoryOrder);
}

#elif defined(MSCCLPP_DEVICE_HIP)

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

// HIP does not have thread scope enums like CUDA
constexpr auto scopeSystem = 0;
constexpr auto scopeDevice = 0;

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(T const* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, T const& val,
                                            int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, T const& val,
                                            int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

#else  // Host-side (non-device) compilation

// For host-side code, provide simple atomic wrappers using GCC built-ins
constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

constexpr auto scopeSystem = 0;
constexpr auto scopeDevice = 0;

template <typename T, int scope = scopeSystem>
inline T atomicLoad(T const* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T, int scope = scopeSystem>
inline void atomicStore(T* ptr, T const& val, int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T, int scope = scopeSystem>
inline T atomicFetchAdd(T* ptr, T const& val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

#endif  // defined(MSCCLPP_DEVICE_HIP)

// ============================================================================
// Device Assertions (from assert_device.hpp)
// ============================================================================

#if defined(MSCCLPP_DEVICE_COMPILE)

#if !defined(DEBUG_BUILD)

#define MSCCLPP_ASSERT_DEVICE(__cond, __msg)

#else  // defined(DEBUG_BUILD)

#if defined(MSCCLPP_DEVICE_HIP)
extern "C" __device__ void __assert_fail(char const* __assertion,
                                         char const* __file,
                                         unsigned int __line,
                                         char const* __function);
#else   // !defined(MSCCLPP_DEVICE_HIP)
extern "C" __host__ __device__ void __assert_fail(
    char const* __assertion, char const* __file, unsigned int __line,
    char const* __function) __THROW;
#endif  // !defined(MSCCLPP_DEVICE_HIP)

#define MSCCLPP_ASSERT_DEVICE(__cond, __msg)                         \
  do {                                                               \
    if (!(__cond)) {                                                 \
      __assert_fail(__msg, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
    }                                                                \
  } while (0)

#endif  // !defined(DEBUG_BUILD)

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

// ============================================================================
// Polling Macros (from poll_device.hpp)
// ============================================================================

#if defined(MSCCLPP_DEVICE_COMPILE)

#define POLL_MAYBE_JAILBREAK(__cond, __max_spin_cnt)                        \
  do {                                                                      \
    [[maybe_unused]] int64_t __spin_cnt = 0;                                \
    while (__cond) {                                                        \
      MSCCLPP_ASSERT_DEVICE(                                                \
          (__max_spin_cnt < 0 || __spin_cnt++ != __max_spin_cnt), #__cond); \
    }                                                                       \
  } while (0);

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

// ============================================================================
// GPU Memory Management (from gpu_utils.hpp)
// ============================================================================

namespace detail {

static inline bool isCudaTeardownError(cudaError_t err) {
#if defined(__HIP_PLATFORM_AMD__)
  return err == cudaErrorContextIsDestroyed || err == cudaErrorInvalidDevice;
#else
  return err == cudaErrorCudartUnloading ||
         err == cudaErrorContextIsDestroyed ||
         err == cudaErrorInitializationError || err == cudaErrorInvalidDevice ||
         err == cudaErrorLaunchFailure;
#endif
}

inline void* gpuCalloc(size_t bytes) {
  void* ptr;
  MSCCLPP_CUDATHROW(cudaMalloc(&ptr, bytes));
  MSCCLPP_CUDATHROW(cudaMemset(ptr, 0, bytes));
  return ptr;
}

inline void* gpuCallocHost(size_t bytes, unsigned int flags) {
  void* ptr;
  MSCCLPP_CUDATHROW(cudaHostAlloc(&ptr, bytes, flags));
  ::memset(ptr, 0, bytes);
  return ptr;
}

inline void gpuFree(void* ptr) {
  cudaError_t err = cudaFree(ptr);
  if (!isCudaTeardownError(err) && err != cudaSuccess) {
    throw ::mscclpp::CudaError(std::string("Call to cudaFree failed. ") +
                                   __FILE__ + ":" + std::to_string(__LINE__),
                               err);
  } else if (isCudaTeardownError(err)) {
    (void)cudaGetLastError();
  }
}

inline void gpuFreeHost(void* ptr) {
  cudaError_t err = cudaFreeHost(ptr);
  if (!isCudaTeardownError(err) && err != cudaSuccess) {
    throw ::mscclpp::CudaError(std::string("Call to cudaFreeHost failed. ") +
                                   __FILE__ + ":" + std::to_string(__LINE__),
                               err);
  } else if (isCudaTeardownError(err)) {
    (void)cudaGetLastError();
  }
}

/// A deleter that calls gpuFree for use with std::unique_ptr or
/// std::shared_ptr.
template <class T = void>
struct GpuDeleter {
  void operator()(void* ptr) { gpuFree(ptr); }
};

/// A deleter that calls gpuFreeHost for use with std::unique_ptr or
/// std::shared_ptr.
template <class T = void>
struct GpuHostDeleter {
  void operator()(void* ptr) { gpuFreeHost(ptr); }
};

template <class T>
using UniqueGpuPtr = std::unique_ptr<T, detail::GpuDeleter<T>>;

template <class T>
using UniqueGpuHostPtr = std::unique_ptr<T, detail::GpuHostDeleter<T>>;

/// A template function that allocates memory while ensuring that the memory
/// will be freed when the returned object is destroyed.
template <class T, class Deleter, class Memory, typename Alloc,
          typename... Args>
Memory safeAlloc(Alloc alloc, size_t nelems, Args&&... args) {
  T* ptr = nullptr;
  try {
    ptr = reinterpret_cast<T*>(
        alloc(nelems * sizeof(T), std::forward<Args>(args)...));
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

template <class T>
auto gpuCallocUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::GpuDeleter<T>, UniqueGpuPtr<T>>(
      detail::gpuCalloc, nelems);
}

template <class T>
auto gpuCallocHostUnique(size_t nelems = 1,
                         unsigned int flags = cudaHostAllocMapped) {
  return detail::safeAlloc<T, detail::GpuHostDeleter<T>, UniqueGpuHostPtr<T>>(
      detail::gpuCallocHost, nelems, flags);
}

}  // namespace detail

}  // namespace mscclpp
