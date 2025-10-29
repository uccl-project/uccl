/**
 * @file unpack_descriptor.h
 * @brief RX descriptor construction utilities (header-only)
 *
 * Builds GPU unpack descriptors from TCPX receive metadata.
 * After a GPU receive completes, the plugin provides scatter-gather metadata
 * (loadMeta array) describing how to unpack fragmented data from bounce
 * buffers.
 *
 * Design: Header-only, uses tcpx::plugin::loadMeta directly (no duplication).
 */

#ifndef TCPX_RX_DESCRIPTOR_H_
#define TCPX_RX_DESCRIPTOR_H_

#include <cstddef>
#include <cstdint>

namespace tcpx {
namespace plugin {

struct DevmemToken {
  std::uint32_t token_start;
  std::uint32_t token_count;
};

struct unpackSlot {
  bool active;
  std::uint64_t idx;
  void* mem;
  std::uint64_t* cnt;
  std::uint64_t cnt_cache;
  std::size_t* fds_cnt;
  std::size_t* pgtok_cnts;
  int* fds;
  DevmemToken* pgtoks;
};

struct tcpxRequest {
  void* comm;
  void* data;
  int op;
  int mem_type;
  int next_sock_id;
  int next_size;
  int offset;
  int size;
  int size_pending;
  int gpu_mem_fd;
  int gpu_mem_off;
  unpackSlot unpack_slot;
};

struct NcclNetDeviceHandle {
  int netDeviceType;
  int netDeviceVersion;
  void* handle;
  std::size_t size;
  int needsProxyProgress;
};

struct unpackNetDeviceHandle {
  void* meta;
  void* bounce_buf;
  std::uint64_t head;
};

struct loadMeta {
  std::uint32_t src_off;
  std::uint32_t len;
  std::uint64_t dst_off;
};
static_assert(sizeof(loadMeta) == 16, "loadMeta must be 16 bytes");

constexpr int kNetDeviceUnpackMaxQueueDepth = 16;
constexpr int kNetUnpackMaxSliceSize = 4 * 1024 * 1024;
constexpr int kSlicePageSize = 4096;
constexpr int kNetUnpackMaxSlicePages =
    (kNetUnpackMaxSliceSize / kSlicePageSize) * 2;

}  // namespace plugin

namespace rx {

// Use TCPX plugin's loadMeta as the descriptor type (avoids duplication)
using UnpackDescriptor = tcpx::plugin::loadMeta;

// Maximum descriptors per unpack operation
#define MAX_UNPACK_DESCRIPTORS 2048

// Descriptor block for GPU kernel
struct UnpackDescriptorBlock {
  // Keep scalar header fields before the descriptor array so we can copy
  // only the populated descriptors to device memory.
  uint32_t count;        // Number of valid descriptors
  uint32_t total_bytes;  // Total bytes to unpack
  void* bounce_buffer;   // Source bounce buffer base
  void* dst_buffer;      // Destination buffer base
  void* ready_flag;      // Device pointer to a 64-bit counter/flag (optional)
  uint64_t
      ready_threshold;  // Optional: expected minimal value to consider ready
  UnpackDescriptor descriptors[MAX_UNPACK_DESCRIPTORS];

  UnpackDescriptorBlock()
      : count(0),
        total_bytes(0),
        bounce_buffer(nullptr),
        dst_buffer(nullptr),
        ready_flag(nullptr),
        ready_threshold(0) {}
};

/**
 * @brief Build descriptor block from TCPX receive metadata
 * @param meta_entries Array of loadMeta from rx_req->unpack_slot.mem
 * @param count Number of descriptors
 * @param bounce_buffer Bounce buffer base address (from device handle)
 * @param dst_buffer Destination buffer base address
 * @param desc_block Output descriptor block
 *
 * Copies descriptors and calculates total_bytes. Can be used for:
 * - D2D unpack: Loop and call cuMemcpyDtoD per fragment
 * - Host unpack: DtoH + gather + HtoD
 * - Kernel unpack: Copy to device and launch GPU kernel
 */
inline void buildDescriptorBlock(tcpx::plugin::loadMeta const* meta_entries,
                                 uint32_t count, void* bounce_buffer,
                                 void* dst_buffer,
                                 UnpackDescriptorBlock& desc_block) {
  desc_block.count = count;
  desc_block.total_bytes = 0;
  desc_block.bounce_buffer = bounce_buffer;
  desc_block.dst_buffer = dst_buffer;

  for (uint32_t i = 0; i < count && i < MAX_UNPACK_DESCRIPTORS; ++i) {
    desc_block.descriptors[i] = meta_entries[i];
    desc_block.total_bytes += meta_entries[i].len;
  }
}

}  // namespace rx
}  // namespace tcpx

#endif  // TCPX_RX_DESCRIPTOR_H_
