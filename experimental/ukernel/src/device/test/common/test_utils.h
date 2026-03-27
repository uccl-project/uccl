#pragma once

#include "gpu_rt.h"
#include "task.h"
#include "worker.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace UKernel {
namespace Device {
namespace TestUtil {

[[noreturn]] inline void fail(std::string const& msg) {
  throw std::runtime_error(msg);
}

inline void require(bool cond, std::string const& msg) {
  if (!cond) fail(msg);
}

template <typename Fn>
inline void run_case(char const* suite, char const* name, Fn&& fn) {
  std::cout << "[test] " << suite << " " << name << "..." << std::endl;
  fn();
}

struct TaskManagerScope {
  explicit TaskManagerScope(uint32_t capacity) { TaskManager::instance().init(capacity); }
  ~TaskManagerScope() { TaskManager::instance().release(); }

  TaskManagerScope(TaskManagerScope const&) = delete;
  TaskManagerScope& operator=(TaskManagerScope const&) = delete;
};

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t nbytes) : bytes(nbytes) {
    GPU_RT_CHECK(gpuMalloc(&ptr, bytes));
  }
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      gpuFree(ptr);
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr;
    other.bytes = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (ptr != nullptr) {
      gpuFree(ptr);
    }
    ptr = other.ptr;
    bytes = other.bytes;
    other.ptr = nullptr;
    other.bytes = 0;
    return *this;
  }
};

inline WorkerPool::Config default_config(uint32_t max_workers = 4,
                                         uint32_t fifo_capacity = 16,
                                         uint32_t threads_per_block = 256) {
  WorkerPool::Config cfg;
  cfg.numMaxWorkers = max_workers;
  cfg.threadsPerBlock = threads_per_block;
  cfg.fifoCapacity = fifo_capacity;
  return cfg;
}

template <typename T>
inline void upload_vector(void* dst, std::vector<T> const& values) {
  GPU_RT_CHECK(gpuMemcpy(dst, values.data(), values.size() * sizeof(T),
                         gpuMemcpyHostToDevice));
}

inline void zero_buffer(void* dst, size_t bytes) {
  GPU_RT_CHECK(gpuMemset(dst, 0, bytes));
}

template <typename T>
inline std::vector<T> download_vector(void const* src, size_t bytes) {
  require(bytes % sizeof(T) == 0, "buffer size must align with element size");
  std::vector<T> out(bytes / sizeof(T));
  GPU_RT_CHECK(gpuMemcpy(out.data(), src, bytes, gpuMemcpyDeviceToHost));
  return out;
}

inline void wait_until_done(WorkerPool& pool, uint64_t task_id, uint32_t fifo_id,
                            std::chrono::milliseconds timeout =
                                std::chrono::seconds(5)) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pool.is_done(task_id, fifo_id)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  require(pool.is_done(task_id, fifo_id), "device task timed out");
}

inline void verify_bytes(std::vector<char> const& actual,
                         std::vector<char> const& expected,
                         std::string const& label) {
  require(actual.size() == expected.size(), label + " size mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    require(actual[i] == expected[i],
            label + " mismatch at byte " + std::to_string(i));
  }
}

inline void verify_floats(std::vector<float> const& actual,
                          std::vector<float> const& expected,
                          std::string const& label, float tol = 1e-5f) {
  require(actual.size() == expected.size(), label + " size mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    require(std::fabs(actual[i] - expected[i]) < tol,
            label + " mismatch at index " + std::to_string(i) +
                ", got=" + std::to_string(actual[i]) +
                ", expected=" + std::to_string(expected[i]));
  }
}

}  // namespace TestUtil
}  // namespace Device
}  // namespace UKernel
