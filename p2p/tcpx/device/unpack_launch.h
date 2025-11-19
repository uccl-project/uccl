#ifndef TCPX_DEVICE_UNPACK_LAUNCH_H_
#define TCPX_DEVICE_UNPACK_LAUNCH_H_

#include "../include/unpack_descriptor.h"
#include <cuda_runtime.h>

namespace tcpx {
namespace device {

// Kernel launch parameters structure
struct KernelLaunchParams {
  dim3 grid_size;
  dim3 block_size;
  size_t shared_mem_size;
  bool use_small_kernel;

  KernelLaunchParams()
      : grid_size(1, 1, 1),
        block_size(1, 1, 1),
        shared_mem_size(0),
        use_small_kernel(true) {}
};

// Unpack launcher configuration
struct UnpackLaunchConfig {
  cudaStream_t stream;       // CUDA stream for execution
  bool use_small_kernel;     // Use optimized kernel for small transfers
  bool enable_profiling;     // Enable CUDA events for profiling
  uint32_t max_descriptors;  // Maximum descriptors per launch

  UnpackLaunchConfig()
      : stream(nullptr),
        use_small_kernel(true),
        enable_profiling(false),
        max_descriptors(2048) {}
};

// Unpack execution statistics
struct UnpackStats {
  uint64_t launches;               // Number of kernel launches
  uint64_t descriptors_processed;  // Total descriptors processed
  uint64_t bytes_unpacked;         // Total bytes unpacked
  uint64_t kernel_errors;          // Kernel execution errors
  float total_time_ms;       // Total execution time (if profiling enabled)
  float avg_bandwidth_gbps;  // Average bandwidth (if profiling enabled)

  UnpackStats() { reset(); }

  void reset() {
    launches = descriptors_processed = bytes_unpacked = kernel_errors = 0;
    total_time_ms = avg_bandwidth_gbps = 0.0f;
  }
};

// Main unpack launcher class
class UnpackLauncher {
 public:
  explicit UnpackLauncher(UnpackLaunchConfig const& config);
  ~UnpackLauncher();

  // Launch unpack kernel for descriptor block
  // Returns 0 on success, negative error code on failure
  int launch(tcpx::rx::UnpackDescriptorBlock const& desc_block);

  // Launch with custom stream
  int launch(tcpx::rx::UnpackDescriptorBlock const& desc_block,
             cudaStream_t stream);

  // Synchronous launch (waits for completion)
  int launchSync(tcpx::rx::UnpackDescriptorBlock const& desc_block);

  // Check if last launch completed
  bool isComplete() const;

  // Wait for completion of last launch
  cudaError_t waitForCompletion();

  // Get execution statistics
  UnpackStats const& getStats() const { return stats_; }
  void resetStats() { stats_.reset(); }

  // Update configuration
  void updateConfig(UnpackLaunchConfig const& config);

 private:
  // Allocate device memory for descriptor block
  int allocateDeviceMemory(size_t size);

  // Copy descriptor block to device
  int copyDescriptorBlockToDevice(
      tcpx::rx::UnpackDescriptorBlock const& desc_block);

  // Calculate optimal launch parameters
  KernelLaunchParams calculateLaunchParams(
      tcpx::rx::UnpackDescriptorBlock const& desc_block) const;

  // Launch kernel with parameters
  int launchKernel(KernelLaunchParams const& params);

  // Update statistics after launch
  void updateStats(tcpx::rx::UnpackDescriptorBlock const& desc_block,
                   float execution_time_ms = 0.0f);

  UnpackLaunchConfig config_;
  UnpackStats stats_;

  // Device memory management
  void* d_desc_block_;            // Device descriptor block
  size_t d_desc_block_size_;      // Allocated size
  void* d_staging_buffer_;        // Staging buffer for devmem-tcp workaround
  size_t d_staging_buffer_size_;  // Staging buffer allocated size

  // Profiling events
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  bool events_created_;
};

// Utility functions
namespace launch_utils {

// Calculate optimal block size for given descriptor count
int calculateOptimalBlockSize(uint32_t descriptor_count,
                              uint32_t avg_descriptor_size);

// Estimate kernel execution time
float estimateExecutionTime(tcpx::rx::UnpackDescriptorBlock const& desc_block);

// Validate launch parameters
bool validateLaunchParams(KernelLaunchParams const& params);

// Get device properties for optimization
cudaDeviceProp getDeviceProperties();

// Calculate theoretical bandwidth
float calculateTheoreticalBandwidth();

}  // namespace launch_utils
}  // namespace device
}  // namespace tcpx

#endif  // TCPX_DEVICE_UNPACK_LAUNCH_H_
