/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "rx_descriptor.h"
#include <algorithm>
#include <sstream>
#include <iostream>

namespace tcpx {
namespace rx {

DescriptorBuilder::DescriptorBuilder(const DescriptorConfig& config)
  : config_(config) {
}

int DescriptorBuilder::buildDescriptors(const ScatterList& scatter_list,
                                        UnpackDescriptorBlock& desc_block) {
  // Reset descriptor block
  desc_block.count = 0;
  desc_block.total_bytes = 0;
  desc_block.bounce_buffer = config_.bounce_buffer;
  desc_block.dst_buffer = config_.dst_buffer;
  
  if (scatter_list.entries.empty()) {
    return 0; // Empty scatter list is valid
  }
  
  // Check if we have too many entries
  if (scatter_list.entries.size() > config_.max_descriptors) {
    stats_.build_errors++;
    return -1;
  }
  
  // Convert each scatter entry to descriptor
  for (const auto& entry : scatter_list.entries) {
    // Skip linear entries for now - they should be handled by CPU copy
    if (!entry.is_devmem) {
      continue;
    }
    
    UnpackDescriptor desc = convertToDescriptor(entry);
    
    if (!validateDescriptor(desc)) {
      stats_.build_errors++;
      return -2;
    }
    
    desc_block.descriptors[desc_block.count++] = desc;
    desc_block.total_bytes += desc.len;
  }
  
  // Optimize descriptor block
  descriptor_utils::mergeDescriptors(desc_block);
  
  stats_.blocks_built++;
  stats_.descriptors_created += desc_block.count;
  stats_.bytes_processed += desc_block.total_bytes;
  
  return 0;
}

UnpackDescriptor DescriptorBuilder::convertToDescriptor(const ScatterEntry& entry) const {
  return UnpackDescriptor(
    entry.src_offset,    // Source offset in bounce buffer
    entry.length,        // Data length
    entry.dst_offset     // Destination offset
  );
}

bool DescriptorBuilder::validateDescriptor(const UnpackDescriptor& desc) const {
  // Check for zero length
  if (desc.len == 0) {
    return false;
  }
  
  // Check for reasonable bounds (avoid overflow)
  if (desc.src_off > (1ULL << 32) || desc.dst_off > (1ULL << 48)) {
    return false;
  }
  
  // Check alignment (optional optimization)
  if (desc.src_off % 4 != 0 || desc.dst_off % 4 != 0) {
    // Warning: unaligned access may be slower
  }
  
  return true;
}

bool DescriptorBuilder::validateDescriptors(const UnpackDescriptorBlock& desc_block) const {
  if (desc_block.count > MAX_UNPACK_DESCRIPTORS) {
    return false;
  }
  
  uint32_t calculated_total = 0;
  for (uint32_t i = 0; i < desc_block.count; ++i) {
    if (!validateDescriptor(desc_block.descriptors[i])) {
      return false;
    }
    calculated_total += desc_block.descriptors[i].len;
  }
  
  return calculated_total == desc_block.total_bytes;
}

namespace descriptor_utils {

uint32_t calculateOptimalAlignment(uint32_t offset, uint32_t length) {
  // Find the largest power of 2 that divides both offset and length
  uint32_t combined = offset | length;
  uint32_t alignment = combined & (~combined + 1); // Extract lowest set bit
  
  // Clamp to reasonable values (1, 2, 4, 8, 16 bytes)
  if (alignment >= 16) return 16;
  if (alignment >= 8) return 8;
  if (alignment >= 4) return 4;
  if (alignment >= 2) return 2;
  return 1;
}

int mergeDescriptors(UnpackDescriptorBlock& desc_block) {
  if (desc_block.count <= 1) {
    return 0; // Nothing to merge
  }
  
  uint32_t write_idx = 0;
  
  for (uint32_t read_idx = 0; read_idx < desc_block.count; ++read_idx) {
    UnpackDescriptor& current = desc_block.descriptors[read_idx];
    
    // Try to merge with previous descriptor
    if (write_idx > 0) {
      UnpackDescriptor& prev = desc_block.descriptors[write_idx - 1];
      
      // Check if descriptors are adjacent
      bool src_adjacent = (prev.src_off + prev.len == current.src_off);
      bool dst_adjacent = (prev.dst_off + prev.len == current.dst_off);
      
      if (src_adjacent && dst_adjacent) {
        // Merge into previous descriptor
        prev.len += current.len;
        continue;
      }
    }
    
    // Cannot merge, copy to write position
    if (write_idx != read_idx) {
      desc_block.descriptors[write_idx] = current;
    }
    write_idx++;
  }
  
  uint32_t merged_count = desc_block.count - write_idx;
  desc_block.count = write_idx;
  
  return merged_count;
}

int splitDescriptors(UnpackDescriptorBlock& desc_block, uint32_t max_chunk_size) {
  if (max_chunk_size == 0) {
    return -1;
  }
  
  std::vector<UnpackDescriptor> new_descriptors;
  
  for (uint32_t i = 0; i < desc_block.count; ++i) {
    const UnpackDescriptor& desc = desc_block.descriptors[i];
    
    if (desc.len <= max_chunk_size) {
      // No need to split
      new_descriptors.push_back(desc);
    } else {
      // Split into chunks
      uint32_t remaining = desc.len;
      uint32_t src_offset = desc.src_off;
      uint64_t dst_offset = desc.dst_off;
      
      while (remaining > 0) {
        uint32_t chunk_size = std::min(remaining, max_chunk_size);
        
        new_descriptors.emplace_back(src_offset, chunk_size, dst_offset);
        
        src_offset += chunk_size;
        dst_offset += chunk_size;
        remaining -= chunk_size;
      }
    }
  }
  
  // Check if we exceed maximum descriptors
  if (new_descriptors.size() > MAX_UNPACK_DESCRIPTORS) {
    return -2;
  }
  
  // Copy back to descriptor block
  desc_block.count = new_descriptors.size();
  for (size_t i = 0; i < new_descriptors.size(); ++i) {
    desc_block.descriptors[i] = new_descriptors[i];
  }
  
  return new_descriptors.size() - desc_block.count;
}

std::string dumpDescriptorBlock(const UnpackDescriptorBlock& desc_block) {
  std::ostringstream oss;
  oss << "UnpackDescriptorBlock: " << desc_block.count << " descriptors, "
      << desc_block.total_bytes << " total bytes\n";
  oss << "  Bounce buffer: " << desc_block.bounce_buffer << "\n";
  oss << "  Dst buffer: " << desc_block.dst_buffer << "\n";
  
  for (uint32_t i = 0; i < desc_block.count; ++i) {
    const auto& desc = desc_block.descriptors[i];
    oss << "  [" << i << "] src_off=" << desc.src_off
        << " dst_off=" << desc.dst_off
        << " len=" << desc.len
        << " align=" << calculateOptimalAlignment(desc.src_off, desc.len)
        << "\n";
  }
  
  return oss.str();
}

bool validateDescriptorBlock(const UnpackDescriptorBlock& desc_block) {
  if (desc_block.count > MAX_UNPACK_DESCRIPTORS) {
    return false;
  }
  
  if (!desc_block.bounce_buffer || !desc_block.dst_buffer) {
    return false;
  }
  
  uint32_t calculated_total = 0;
  for (uint32_t i = 0; i < desc_block.count; ++i) {
    const auto& desc = desc_block.descriptors[i];
    
    if (desc.len == 0) {
      return false;
    }
    
    calculated_total += desc.len;
  }
  
  return calculated_total == desc_block.total_bytes;
}

} // namespace descriptor_utils
} // namespace rx
} // namespace tcpx
