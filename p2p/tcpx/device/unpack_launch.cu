/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "unpack_launch.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

// Forward declare CUDA kernels
extern "C" {
__global__ void tcpxUnpackKernel(
    tcpx::rx::UnpackDescriptorBlock const* desc_block);
__global__ void tcpxUnpackKernelSmall(
    tcpx::rx::UnpackDescriptorBlock const* desc_block);
__global__ void tcpxUnpackKernelProbeByte(
    tcpx::rx::UnpackDescriptorBlock const* desc_block);
}

// Forward declare utility function from unpack_kernels.cu
extern "C" {
tcpx::device::KernelLaunchParams calculateLaunchParams(
    tcpx::rx::UnpackDescriptorBlock const& desc_block);
}

namespace tcpx {
namespace device {

UnpackLauncher::UnpackLauncher(UnpackLaunchConfig const& config)
    : config_(config),
      d_desc_block_(nullptr),
      d_desc_block_size_(0),
      d_staging_buffer_(nullptr),
      d_staging_buffer_size_(0),
      start_event_(nullptr),
      stop_event_(nullptr),
      events_created_(false) {
  if (config_.enable_profiling) {
    cudaError_t err1 = cudaEventCreate(&start_event_);
    cudaError_t err2 = cudaEventCreate(&stop_event_);
    events_created_ = (err1 == cudaSuccess && err2 == cudaSuccess);

    if (!events_created_) {
      std::cerr << "Warning: Failed to create CUDA events for profiling\n";
    }
  }
}

UnpackLauncher::~UnpackLauncher() {
  if (d_desc_block_) {
    cudaFree(d_desc_block_);
  }
  if (d_staging_buffer_) {
    cudaFree(d_staging_buffer_);
  }

  if (events_created_) {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
  }
}

int UnpackLauncher::launch(tcpx::rx::UnpackDescriptorBlock const& desc_block) {
  return launch(desc_block, config_.stream);
}

int UnpackLauncher::launch(tcpx::rx::UnpackDescriptorBlock const& desc_block,
                           cudaStream_t stream) {
  if (desc_block.count == 0) {
    return 0;  // Nothing to do
  }

  if (desc_block.count > config_.max_descriptors) {
    stats_.kernel_errors++;
    return -1;
  }

  bool const dbg = (std::getenv("UCCL_TCPX_LAUNCH_DEBUG") &&
                    std::string(std::getenv("UCCL_TCPX_LAUNCH_DEBUG")) == "1");
  if (dbg) {
    std::cout << "[Debug Kernel] [launch] begin, stream=" << stream
              << " desc.count=" << desc_block.count
              << " total_bytes=" << desc_block.total_bytes
              << " bounce_buf=" << desc_block.bounce_buffer
              << " dst_buf=" << desc_block.dst_buffer
              << " ready_flag=" << desc_block.ready_flag << std::endl;
    std::cout.flush();
  }

  // Copy descriptor block to device
  if (dbg) {
    std::cout << "[Debug Kernel] [launch] copyDescriptorBlockToDevice..."
              << std::endl;
    std::cout.flush();
  }

  int ret = copyDescriptorBlockToDevice(desc_block);

  if (dbg) {
    std::cout << "[Debug Kernel] [launch] copyDescriptorBlockToDevice rc="
              << ret << std::endl;
    std::cout.flush();
  }

  if (ret < 0) {
    stats_.kernel_errors++;
    return ret;
  }

  // Calculate launch parameters
  if (dbg) {
    std::cout << "[Debug Kernel] [launch] calculateLaunchParams..."
              << std::endl;
    std::cout.flush();
  }

  KernelLaunchParams params = calculateLaunchParams(desc_block);

  if (dbg) {
    std::cout << "[Debug Kernel] [launch] params: grid=" << params.grid_size.x
              << " block=" << params.block_size.x << std::endl;
    std::cout.flush();
  }

  if (!launch_utils::validateLaunchParams(params)) {
    stats_.kernel_errors++;
    return -3;
  }

  // Check if staging buffer is needed
  char const* staging_env = std::getenv("UCCL_TCPX_USE_STAGING");
  bool use_staging = (staging_env && std::string(staging_env) == "1");

  if (use_staging) {
    if (dbg) {
      std::cout << "[Debug Kernel] [staging] DISABLED - staging buffer causes "
                   "offset issues"
                << std::endl;
      std::cout << "[Debug Kernel] [staging] Reason: descriptors have src_off "
                   "relative to original bounce_buffer"
                << std::endl;
      std::cout << "[Debug Kernel] [staging] Copying entire bounce buffer "
                   "would require recalculating all offsets"
                << std::endl;
      std::cout
          << "[Debug Kernel] [staging] Proceeding without staging buffer..."
          << std::endl;
      std::cout.flush();
    }
    // DISABLED: Staging buffer approach is flawed because:
    // 1. We copy total_bytes from bounce_buffer to staging
    // 2. But descriptors have src_off relative to original bounce_buffer base
    // 3. If we change bounce_buffer pointer, we'd need to:
    //    a) Find the minimum src_off across all descriptors
    //    b) Copy from (bounce_buffer + min_offset) for (max_offset -
    //    min_offset) bytes c) Adjust all src_off values by subtracting
    //    min_offset
    // 4. This is complex and error-prone
    // 5. Better to let kernel read directly from bounce_buffer (which is
    // already on GPU)
  }

  // Start profiling if enabled
  float execution_time = 0.0f;
  if (dbg) {
    std::cout << "[Debug Kernel] [launch] launchKernel..." << std::endl;
    std::cout.flush();
  }

  if (config_.enable_profiling && events_created_) {
    cudaEventRecord(start_event_, stream);
  }

  // Launch kernel
  ret = launchKernel(params);
  if (dbg) {
    std::cout << "[Debug Kernel] [launch] launchKernel rc=" << ret << std::endl;
    std::cout.flush();
  }

  if (ret < 0) {
    stats_.kernel_errors++;
    return ret;
  }

  // Stop profiling if enabled
  if (config_.enable_profiling && events_created_) {
    cudaEventRecord(stop_event_, stream);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&execution_time, start_event_, stop_event_);
  }

  // Update statistics
  updateStats(desc_block, execution_time);

  return 0;
}

int UnpackLauncher::launchSync(
    tcpx::rx::UnpackDescriptorBlock const& desc_block) {
  bool const dbg = (std::getenv("UCCL_TCPX_LAUNCH_DEBUG") &&
                    std::string(std::getenv("UCCL_TCPX_LAUNCH_DEBUG")) == "1");
  int ret = launch(desc_block);
  if (ret < 0) {
    if (dbg) {
      std::cout << "[Debug Kernel] [launchSync] launch rc=" << ret << std::endl;
      std::cout.flush();
    }
    return ret;
  }

  if (dbg) {
    std::cout << "[Debug Kernel] [launchSync] waitForCompletion (stream sync)"
              << std::endl;
    std::cout.flush();
  }

  if (dbg && config_.stream) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
      cudaError_t q = cudaStreamQuery(config_.stream);
      if (q == cudaSuccess) break;
      if (q != cudaErrorNotReady) {
        std::cout << "[Debug Kernel] [launchSync] streamQuery error: "
                  << cudaGetErrorString(q) << std::endl;
        std::cout.flush();
        break;
      }
      auto now = std::chrono::steady_clock::now();
      auto ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
              .count();
      if (ms % 1000 < 50) {
        std::cout << "[Debug Kernel] [launchSync] still waiting... elapsed_ms="
                  << ms << std::endl;
        std::cout.flush();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  cudaError_t err = waitForCompletion();
  if (dbg) {
    std::cout << "[Debug Kernel] [launchSync] waitForCompletion rc=" << (int)err
              << " (" << cudaGetErrorString(err) << ")" << std::endl;
    std::cout.flush();
  }
  return (err == cudaSuccess) ? 0 : -4;
}

bool UnpackLauncher::isComplete() const {
  if (!config_.stream) {
    return true;  // Default stream is always synchronous
  }

  cudaError_t err = cudaStreamQuery(config_.stream);
  return (err == cudaSuccess);
}

cudaError_t UnpackLauncher::waitForCompletion() {
  if (config_.stream) {
    return cudaStreamSynchronize(config_.stream);
  } else {
    return cudaDeviceSynchronize();
  }
}

int UnpackLauncher::allocateDeviceMemory(size_t size) {
  if (d_desc_block_size_ >= size) {
    return 0;  // Already have enough memory
  }

  // Free existing memory
  if (d_desc_block_) {
    cudaFree(d_desc_block_);
    d_desc_block_ = nullptr;
    d_desc_block_size_ = 0;
  }

  // Allocate new memory with some padding
  size_t padded_size = size + (size / 4);  // 25% padding
  cudaError_t err = cudaMalloc(&d_desc_block_, padded_size);
  if (err != cudaSuccess) {
    return -1;
  }

  d_desc_block_size_ = padded_size;
  return 0;
}

int UnpackLauncher::copyDescriptorBlockToDevice(
    tcpx::rx::UnpackDescriptorBlock const& desc_block) {
  constexpr size_t kHeaderSize =
      offsetof(tcpx::rx::UnpackDescriptorBlock, descriptors);
  const size_t descriptor_bytes =
      static_cast<size_t>(desc_block.count) *
      sizeof(desc_block.descriptors[0]);
  size_t required_size = kHeaderSize + descriptor_bytes;
  if (required_size < kHeaderSize) {
    required_size = kHeaderSize;
  }

  int ret = allocateDeviceMemory(required_size);
  if (ret < 0) {
    return ret;
  }

  bool const dbg = (std::getenv("UCCL_TCPX_LAUNCH_DEBUG") &&
                    std::string(std::getenv("UCCL_TCPX_LAUNCH_DEBUG")) == "1");
  if (dbg) {
    std::cout << "[Debug Kernel] [copy] H2D desc_block required_size="
              << required_size << " stream=" << config_.stream << std::endl;
    std::cout.flush();
  }

  cudaError_t err;
  if (config_.stream) {
    err = cudaMemcpyAsync(d_desc_block_, &desc_block, required_size,
                          cudaMemcpyHostToDevice, config_.stream);
  } else {
    err = cudaMemcpy(d_desc_block_, &desc_block, required_size,
                     cudaMemcpyHostToDevice);
  }

  if (dbg) {
    std::cout << "[Debug Kernel] [copy] H2D rc=" << (int)err << " ("
              << cudaGetErrorString(err) << ")" << std::endl;
    std::cout.flush();
  }

  if (err != cudaSuccess) {
    return -2;
  }

  return 0;
}

KernelLaunchParams UnpackLauncher::calculateLaunchParams(
    tcpx::rx::UnpackDescriptorBlock const& desc_block) const {
  // Call the global function declared in extern "C" block
  return ::calculateLaunchParams(desc_block);
}

int UnpackLauncher::launchKernel(KernelLaunchParams const& params) {
  tcpx::rx::UnpackDescriptorBlock const* d_desc_ptr =
      static_cast<tcpx::rx::UnpackDescriptorBlock const*>(d_desc_block_);

  cudaError_t err;

  // Optional: probe mode. If env UCCL_TCPX_PROBE_BYTE=1 we launch a minimal
  // kernel first
  char const* probe_env = std::getenv("UCCL_TCPX_PROBE_BYTE");
  if (probe_env && std::string(probe_env) == "1") {
    std::cout << "[Debug Kernel] ProbeByte kernel launch..." << std::endl;
    tcpxUnpackKernelProbeByte<<<1, 1, 0, config_.stream>>>(d_desc_ptr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "[Debug Kernel] ProbeByte launch failed: "
                << cudaGetErrorString(err) << std::endl;
      return -1;
    }
  }

  if (config_.use_small_kernel) {
    std::cout << "[Debug Kernel] Launch Small: grid=" << params.grid_size.x
              << " block=" << params.block_size.x << std::endl;
    tcpxUnpackKernelSmall<<<params.grid_size, params.block_size,
                            params.shared_mem_size, config_.stream>>>(
        d_desc_ptr);
  } else {
    std::cout << "[Debug Kernel] Launch Main: grid=" << params.grid_size.x
              << " block=" << params.block_size.x << std::endl;
    tcpxUnpackKernel<<<params.grid_size, params.block_size,
                       params.shared_mem_size, config_.stream>>>(d_desc_ptr);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }

  return 0;
}

void UnpackLauncher::updateStats(
    tcpx::rx::UnpackDescriptorBlock const& desc_block,
    float execution_time_ms) {
  stats_.launches++;
  stats_.descriptors_processed += desc_block.count;
  stats_.bytes_unpacked += desc_block.total_bytes;

  if (execution_time_ms > 0.0f) {
    stats_.total_time_ms += execution_time_ms;

    // Calculate bandwidth in GB/s
    float bandwidth_gbps =
        (desc_block.total_bytes / (1024.0f * 1024.0f * 1024.0f)) /
        (execution_time_ms / 1000.0f);

    // Update average bandwidth
    stats_.avg_bandwidth_gbps =
        (stats_.avg_bandwidth_gbps * (stats_.launches - 1) + bandwidth_gbps) /
        stats_.launches;
  }
}

void UnpackLauncher::updateConfig(UnpackLaunchConfig const& config) {
  config_ = config;

  // Recreate events if profiling setting changed
  if (config_.enable_profiling && !events_created_) {
    cudaError_t err1 = cudaEventCreate(&start_event_);
    cudaError_t err2 = cudaEventCreate(&stop_event_);
    events_created_ = (err1 == cudaSuccess && err2 == cudaSuccess);
  } else if (!config_.enable_profiling && events_created_) {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
    events_created_ = false;
  }
}

namespace launch_utils {

int calculateOptimalBlockSize(uint32_t descriptor_count,
                              uint32_t avg_descriptor_size) {
  // For small descriptors, use more threads per block
  if (avg_descriptor_size < 256) {
    return 256;
  } else if (avg_descriptor_size < 1024) {
    return 512;
  } else {
    return 1024;
  }
}

float estimateExecutionTime(tcpx::rx::UnpackDescriptorBlock const& desc_block) {
  // Simple estimation based on data size and theoretical bandwidth
  float theoretical_bandwidth = calculateTheoreticalBandwidth();
  float data_gb = desc_block.total_bytes / (1024.0f * 1024.0f * 1024.0f);

  // Add some overhead for kernel launch and small transfers
  float base_overhead_ms = 0.01f;  // 10 microseconds
  float transfer_time_ms = (data_gb / theoretical_bandwidth) * 1000.0f;

  return base_overhead_ms + transfer_time_ms;
}

bool validateLaunchParams(KernelLaunchParams const& params) {
  if (params.grid_size.x == 0 || params.block_size.x == 0) {
    return false;
  }

  if (params.block_size.x > 1024) {
    return false;  // Exceeds maximum block size
  }

  return true;
}

cudaDeviceProp getDeviceProperties() {
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  return prop;
}

float calculateTheoreticalBandwidth() {
  cudaDeviceProp prop = getDeviceProperties();

  // Calculate memory bandwidth in GB/s
  float memory_clock_khz = prop.memoryClockRate;
  int memory_bus_width = prop.memoryBusWidth;

  // Bandwidth = (memory_clock * 2) * (bus_width / 8) / 1e6
  float bandwidth_gbps =
      (memory_clock_khz * 2.0f * memory_bus_width / 8.0f) / 1e6f;

  // Apply efficiency factor (typically 80-90% for memory copy)
  return bandwidth_gbps * 0.85f;
}

}  // namespace launch_utils
}  // namespace device
}  // namespace tcpx
