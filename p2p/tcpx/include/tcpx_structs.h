/**
 * @file tcpx_structs.h
 * @brief TCPX plugin internal structure definitions
 *
 * These structures mirror the TCPX plugin's internal layout.
 * WARNING: Must match plugin implementation - changes to plugin may require
 * updates. Reference: nccl-plugin-gpudirecttcpx/src/work_queue.h,
 * devcomm/unpack_defs1.h
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace tcpx {
namespace plugin {

// Devmem token for dmabuf page ranges
struct DevmemToken {
  std::uint32_t token_start;
  std::uint32_t token_count;
};

// Unpack slot containing scatter-gather metadata for GPU receives
struct unpackSlot {
  bool active;
  std::uint64_t idx;
  void* mem;           // Pointer to loadMeta array (scatter-gather list)
  std::uint64_t* cnt;  // Device-visible counter (incremented when data ready)
  std::uint64_t cnt_cache;  // Host-cached value of cnt
  std::size_t* fds_cnt;
  std::size_t* pgtok_cnts;
  int* fds;  // Array of dmabuf file descriptors
  DevmemToken* pgtoks;
};

// TCPX request for tracking async send/recv operations
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
  unpackSlot unpack_slot;  // Unpack metadata (for GPU receives)
};

// NCCL network device handle (generic)
struct NcclNetDeviceHandle {
  int netDeviceType;
  int netDeviceVersion;
  void* handle;
  std::size_t size;
  int needsProxyProgress;
};

// TCPX device handle for GPU unpack operations
struct unpackNetDeviceHandle {
  void* meta;          // Metadata queue
  void* bounce_buf;    // Bounce buffer base address
  std::uint64_t head;  // Queue head index
};

// Scatter-gather descriptor (one fragment to copy from bounce buffer to
// destination) Layout must be exactly 16 bytes for GPU kernel efficiency
struct loadMeta {
  std::uint32_t src_off;  // Offset in bounce buffer
  std::uint32_t len;      // Fragment length
  std::uint64_t dst_off;  // Offset in destination buffer
};
static_assert(sizeof(loadMeta) == 16,
              "loadMeta must remain 16 bytes for GPU alignment");

// TCPX plugin constants
constexpr int kNetDeviceUnpackMaxQueueDepth = 16;
constexpr int kNetUnpackMaxSliceSize = 4 * 1024 * 1024;
constexpr int kSlicePageSize = 4096;
constexpr int kNetUnpackMaxSlicePages =
    (kNetUnpackMaxSliceSize / kSlicePageSize) * 2;

}  // namespace plugin
}  // namespace tcpx
