#pragma once

#include "p2p/rdma/define.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace Transport {

struct LocalSlice {
  uint32_t buffer_id = 0;
  size_t offset = 0;
  size_t bytes = 0;
};

struct RemoteWriteHint {
  // One-sided remote write hint shared by UCCL and RDMA.
  // UCCL uses `key`; RDMA uses `rdma_keys`.
  uint64_t addr = 0;
  uint32_t key = 0;
  uint32_t capacity = 0;
  std::array<uint32_t, kNICContextNumber> rdma_keys{};
  uint32_t memory_type = 0;
  uint32_t rid = 0;
  uint32_t engine_offset = 0;

  bool usable_for_uccl() const { return addr != 0 && key != 0; }
  bool usable_for_rdma() const {
    if (addr == 0) return false;
    for (uint32_t key_item : rdma_keys) {
      if (key_item != 0) return true;
    }
    return false;
  }
  bool usable() const { return usable_for_uccl() || usable_for_rdma(); }
};

struct RemoteSlice {
  // Cross-transport destination hint contract:
  // 1) Common semantic: destination is [remote MR `buffer_id`] + `offset`.
  // 2) IPC/TCP/RDMA/UCCL all support this common hint (`buffer_id`, `offset`).
  // 3) UCCL/RDMA may additionally use `write` for one-sided write fast path.
  //    IPC/TCP ignore `write`.
  uint32_t buffer_id = 0;
  size_t offset = 0;
  RemoteWriteHint write{};

  bool has_write_hint() const { return write.usable(); }
  bool has_uccl_write_hint() const { return write.usable_for_uccl(); }
  bool has_rdma_write_hint() const { return write.usable_for_rdma(); }
};

}  // namespace Transport
}  // namespace UKernel
