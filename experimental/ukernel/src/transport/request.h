#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace Transport {

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
  size_t offset_bytes = 0;
  size_t size_bytes = 0;
  uint32_t local_mr_id = 0;
  uint32_t remote_mr_id = 0;
  RequestType type = RequestType::Send;
  std::atomic<RequestState> state{RequestState::Created};
  std::atomic<uint32_t> remaining_completions{0};

  Request(unsigned id, uint64_t match_seq, void* buffer, size_t offset_bytes,
          size_t size_bytes, uint32_t local_mr_id, uint32_t remote_mr_id,
          RequestType type)
      : id(id),
        match_seq(match_seq),
        buffer(buffer),
        offset_bytes(offset_bytes),
        size_bytes(size_bytes),
        local_mr_id(local_mr_id),
        remote_mr_id(remote_mr_id),
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

  void* data() const {
    return static_cast<void*>(static_cast<char*>(buffer) + offset_bytes);
  }
};

}  // namespace Transport
}  // namespace UKernel
