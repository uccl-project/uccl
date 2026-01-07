/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <string>
#include <vector>

/// Wrapper to test return status of CUDA functions
#define CUDA_VERIFY(X)                                         \
  do {                                                         \
    auto err__ = (X);                                          \
    CHECK_EQ(err__, cudaSuccess)                               \
        << "CUDA error " << dietgpu::errorToName(err__) << " " \
        << dietgpu::errorToString(err__);                      \
  } while (0)

#define CURAND_VERIFY(X)                                                     \
  do {                                                                       \
    auto err__ = (X);                                                        \
    CHECK_EQ(err__, CURAND_STATUS_SUCCESS) << "cuRAND error " << (int)err__; \
  } while (0)

#ifdef __CUDA_ARCH__
#define GPU_ASSERT(X) assert(X)
#else
#define GPU_ASSERT(X) CHECK(X)
#endif // __CUDA_ARCH__

/// Wrapper to synchronously probe for CUDA errors
// #define GPU_SYNC_ERROR 1

#ifdef GPU_SYNC_ERROR
#define CUDA_TEST_ERROR()                 \
  do {                                    \
    CUDA_VERIFY(cudaDeviceSynchronize()); \
  } while (0)
#else
#define CUDA_TEST_ERROR()            \
  do {                               \
    CUDA_VERIFY(cudaGetLastError()); \
  } while (0)
#endif

namespace dietgpu {

/// std::string wrapper around cudaGetErrorString
std::string errorToString(cudaError_t err);

/// std::string wrapper around cudaGetErrorName
std::string errorToName(cudaError_t err);

/// Returns the current thread-local GPU device
int getCurrentDevice();

/// Sets the current thread-local GPU device
void setCurrentDevice(int device);

/// Returns the number of available GPU devices
int getNumDevices();

/// Starts the CUDA profiler (exposed via SWIG)
void profilerStart();

/// Stops the CUDA profiler (exposed via SWIG)
void profilerStop();

/// Synchronizes the CPU against all devices (equivalent to
/// cudaDeviceSynchronize for each device)
void synchronizeAllDevices();

/// Returns a cached cudaDeviceProp for the given device
const cudaDeviceProp& getDeviceProperties(int device);

/// Returns the cached cudaDeviceProp for the current device
const cudaDeviceProp& getCurrentDeviceProperties();

/// Returns the maximum number of threads available for the given GPU
/// device
int getMaxThreads(int device);

/// Equivalent to getMaxThreads(getCurrentDevice())
int getMaxThreadsCurrentDevice();

/// Returns the maximum smem available for the given GPU device
size_t getMaxSharedMemPerBlock(int device);

/// Equivalent to getMaxSharedMemPerBlock(getCurrentDevice())
size_t getMaxSharedMemPerBlockCurrentDevice();

/// For a given pointer, returns whether or not it is located on
/// a device (deviceId >= 0) or the host (-1).
int getDeviceForAddress(const void* p);

/// Does the given device support full unified memory sharing host
/// memory?
bool getFullUnifiedMemSupport(int device);

/// Equivalent to getFullUnifiedMemSupport(getCurrentDevice())
bool getFullUnifiedMemSupportCurrentDevice();

/// RAII object to set the current device, and restore the previous
/// device upon destruction
class DeviceScope {
 public:
  explicit DeviceScope(int device);
  ~DeviceScope();

 private:
  int prevDevice_;
};

// RAII object to manage a cudaEvent_t
class CudaEvent {
 public:
  /// Creates an event and records it in this stream
  explicit CudaEvent(cudaStream_t stream, bool timer = false);
  CudaEvent(const CudaEvent& event) = delete;
  CudaEvent(CudaEvent&& event) noexcept;
  ~CudaEvent();

  CudaEvent& operator=(CudaEvent&& event) noexcept;
  CudaEvent& operator=(CudaEvent& event) = delete;

  inline cudaEvent_t get() {
    return event_;
  }

  /// Wait on this event in this stream
  void streamWaitOnEvent(cudaStream_t stream);

  /// Have the CPU wait for the completion of this event
  void cpuWaitOnEvent();

  /// Returns the elapsed time from the other event
  float timeFrom(CudaEvent& from);

 private:
  cudaEvent_t event_;
};

// RAII object to manage a cudaStream_t
class CudaStream {
 public:
  /// Creates a stream on the current device
  CudaStream(int flags = cudaStreamDefault);
  CudaStream(const CudaStream& stream) = delete;
  CudaStream(CudaStream&& stream) noexcept;
  ~CudaStream();

  CudaStream& operator=(CudaStream&& stream) noexcept;
  CudaStream& operator=(CudaStream& stream) = delete;

  inline cudaStream_t get() {
    return stream_;
  }

  operator cudaStream_t() {
    return stream_;
  }

  static CudaStream make();
  static CudaStream makeNonBlocking();

 private:
  cudaStream_t stream_;
};

/// Call for a collection of streams to wait on
template <typename L1, typename L2>
void streamWaitBase(const L1& listWaiting, const L2& listWaitOn) {
  // For all the streams we are waiting on, create an event
  std::vector<cudaEvent_t> events;
  for (auto& stream : listWaitOn) {
    cudaEvent_t event;
    CUDA_VERIFY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_VERIFY(cudaEventRecord(event, stream));
    events.push_back(event);
  }

  // For all the streams that are waiting, issue a wait
  for (auto& stream : listWaiting) {
    for (auto& event : events) {
      CUDA_VERIFY(cudaStreamWaitEvent(stream, event, 0));
    }
  }

  for (auto& event : events) {
    CUDA_VERIFY(cudaEventDestroy(event));
  }
}

/// These versions allow usage of initializer_list as arguments, since
/// otherwise {...} doesn't have a type
template <typename L1>
void streamWait(const L1& a, const std::initializer_list<cudaStream_t>& b) {
  streamWaitBase(a, b);
}

template <typename L2>
void streamWait(const std::initializer_list<cudaStream_t>& a, const L2& b) {
  streamWaitBase(a, b);
}

inline void streamWait(
    const std::initializer_list<cudaStream_t>& a,
    const std::initializer_list<cudaStream_t>& b) {
  streamWaitBase(a, b);
}

} // namespace dietgpu
