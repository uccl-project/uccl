#pragma once

#include <atomic>
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

enum class RequestType : uint8_t { Send, Recv };
enum class RequestState : uint8_t {
  Created,
  Queued,
  Running,
  Completed,
  Failed
};

struct Request {
  unsigned id = 0;
  uint64_t match_seq = 0;
  void* buffer = nullptr;
  size_t size_bytes = 0;
  RemoteSlice remote_slice{};
  RequestType type = RequestType::Send;
  std::atomic<RequestState> state{RequestState::Created};
  std::atomic<uint32_t> remaining_completions{0};

  Request(unsigned id, uint64_t match_seq, void* buffer, size_t size_bytes,
          RemoteSlice remote_slice, RequestType type)
      : id(id),
        match_seq(match_seq),
        buffer(buffer),
        size_bytes(size_bytes),
        remote_slice(remote_slice),
        type(type) {}

  void mark_queued(uint32_t completion_count = 1);
  void mark_running();
  void mark_failed();
  void complete_one();

  RequestState load_state(
      std::memory_order order = std::memory_order_acquire) const {
    return state.load(order);
  }

  bool is_finished(std::memory_order order = std::memory_order_acquire) const {
    RequestState current = load_state(order);
    return current == RequestState::Completed ||
           current == RequestState::Failed;
  }

  bool has_failed(std::memory_order order = std::memory_order_acquire) const {
    return load_state(order) == RequestState::Failed;
  }

  void* data() const { return buffer; }
};

}  // namespace Transport
}  // namespace UKernel
