#pragma once

#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace Transport {

struct LocalSlice {
  uint32_t mem_id = 0;
  size_t offset = 0;
  size_t bytes = 0;
};

struct RemoteWriteHint {
  // One-sided remote write hint used by UCCL only.
  // `addr`/`key` identify remote target memory, other fields are transport-
  // specific FIFO metadata.
  uint64_t addr = 0;
  uint32_t key = 0;
  uint32_t capacity = 0;
  uint32_t rid = 0;
  uint32_t engine_offset = 0;

  bool usable() const { return addr != 0 && key != 0; }
};

struct RemoteSlice {
  // Cross-transport destination hint contract:
  // 1) Common semantic: destination is [remote MR `mem_id`] + `offset`.
  // 2) IPC/TCP/UCCL all support this common hint (`mem_id`, `offset`).
  // 3) UCCL may additionally use `write` for one-sided write fast path.
  //    IPC/TCP ignore `write`.
  // 4) `binding_version` is IPC metadata epoch/version. Non-zero means sender
  //    must match the same published IPC buffer version before direct access.
  //    Zero means "version unspecified" and should fall back to safe handshake.
  uint32_t mem_id = 0;
  size_t offset = 0;
  RemoteWriteHint write{};
  uint64_t binding_version = 0;

  bool has_write_hint() const { return write.usable(); }
};

}  // namespace Transport
}  // namespace UKernel