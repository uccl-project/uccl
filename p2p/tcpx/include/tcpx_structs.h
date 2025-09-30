#pragma once

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
static_assert(sizeof(loadMeta) == 16, "loadMeta must remain 16 bytes");

constexpr int kNetDeviceUnpackMaxQueueDepth = 16;
constexpr int kNetUnpackMaxSliceSize = 4 * 1024 * 1024;
constexpr int kSlicePageSize = 4096;
constexpr int kNetUnpackMaxSlicePages = (kNetUnpackMaxSliceSize / kSlicePageSize) * 2;

}  // namespace plugin
}  // namespace tcpx
