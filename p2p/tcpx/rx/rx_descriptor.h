/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TCPX_RX_DESCRIPTOR_H_
#define TCPX_RX_DESCRIPTOR_H_

#include <stdint.h>
#include <vector>
#include "rx_cmsg_parser.h"

// CUDA headers - let the build system handle this
#include <cuda_runtime.h>

namespace tcpx {
namespace rx {

// GPU-visible unpack descriptor (16-byte aligned, matches NCCL loadMeta)
union alignas(16) UnpackDescriptor {
  uint64_t r64[2];
  struct {
    uint32_t src_off;    // Source offset in bounce buffer
    uint32_t len;        // Data length
    uint64_t dst_off;    // Destination offset in user buffer
  };
  
  UnpackDescriptor() : r64{0, 0} {}
  
  UnpackDescriptor(uint32_t src_offset, uint32_t length, uint64_t dst_offset)
    : src_off(src_offset), len(length), dst_off(dst_offset) {}
};
static_assert(sizeof(UnpackDescriptor) == 16, "Must be 16-byte aligned");

// Maximum descriptors per unpack operation
#define MAX_UNPACK_DESCRIPTORS 2048

// Descriptor block for GPU kernel
struct UnpackDescriptorBlock {
  UnpackDescriptor descriptors[MAX_UNPACK_DESCRIPTORS];
  uint32_t count;           // Number of valid descriptors
  uint32_t total_bytes;     // Total bytes to unpack
  void* bounce_buffer;      // Source bounce buffer base
  void* dst_buffer;         // Destination buffer base
  // Optional device-side readiness flag (e.g., pointer to cnt in meta ring)
  // If non-null, kernels will issue a device load from this address before copying
  void* ready_flag;         // Device pointer to a 64-bit counter/flag (optional)
  uint64_t ready_threshold; // Optional: expected minimal value to consider ready

  UnpackDescriptorBlock()
    : count(0), total_bytes(0), bounce_buffer(nullptr), dst_buffer(nullptr)
    , ready_flag(nullptr), ready_threshold(0) {}
};

// Descriptor builder configuration
struct DescriptorConfig {
  void* bounce_buffer;      // Bounce buffer base address
  void* dst_buffer;         // Destination buffer base address
  size_t max_descriptors;   // Maximum descriptors per block
  
  DescriptorConfig()
    : bounce_buffer(nullptr), dst_buffer(nullptr)
    , max_descriptors(MAX_UNPACK_DESCRIPTORS) {}
};

// Descriptor builder statistics
struct DescriptorStats {
  uint64_t blocks_built;
  uint64_t descriptors_created;
  uint64_t bytes_processed;
  uint64_t build_errors;
  
  DescriptorStats() { reset(); }
  
  void reset() {
    blocks_built = descriptors_created = bytes_processed = build_errors = 0;
  }
};

// Descriptor builder class
class DescriptorBuilder {
public:
  explicit DescriptorBuilder(const DescriptorConfig& config);
  ~DescriptorBuilder() = default;

  // Build descriptor block from scatter list
  // Returns 0 on success, negative error code on failure
  int buildDescriptors(const ScatterList& scatter_list, 
                       UnpackDescriptorBlock& desc_block);
  
  // Validate descriptor block
  bool validateDescriptors(const UnpackDescriptorBlock& desc_block) const;
  
  // Get builder statistics
  const DescriptorStats& getStats() const { return stats_; }
  void resetStats() { stats_.reset(); }
  
  // Update configuration
  void updateConfig(const DescriptorConfig& config) { config_ = config; }

private:
  // Convert scatter entry to descriptor
  UnpackDescriptor convertToDescriptor(const ScatterEntry& entry) const;
  
  // Validate individual descriptor
  bool validateDescriptor(const UnpackDescriptor& desc) const;
  
  DescriptorConfig config_;
  DescriptorStats stats_;
};

// Utility functions
namespace descriptor_utils {

// Calculate alignment for optimal GPU access
uint32_t calculateOptimalAlignment(uint32_t offset, uint32_t length);

// Merge adjacent descriptors for efficiency
int mergeDescriptors(UnpackDescriptorBlock& desc_block);

// Split large descriptors for better parallelism
int splitDescriptors(UnpackDescriptorBlock& desc_block, uint32_t max_chunk_size);

// Debug: dump descriptor block to string
std::string dumpDescriptorBlock(const UnpackDescriptorBlock& desc_block);

// Validate descriptor block consistency
bool validateDescriptorBlock(const UnpackDescriptorBlock& desc_block);

} // namespace descriptor_utils
} // namespace rx
} // namespace tcpx

#endif // TCPX_RX_DESCRIPTOR_H_
