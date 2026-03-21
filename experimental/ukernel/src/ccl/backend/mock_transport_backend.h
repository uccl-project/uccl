#pragma once

#include "../backend.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>
#include <unordered_map>

namespace UKernel {
namespace CCL {

struct MockCommunicator {
  int world_size() const { return 2; }
  bool poll(unsigned) { return true; }
  void release(unsigned) {}
  unsigned isend(int, void*, int, size_t, uint32_t, uint32_t, bool) { return 1; }
  unsigned irecv(int, void*, int, size_t, bool) { return 1; }
  int peer_transport_kind(int) const { return 0; }
  void* get_local_mr(uint32_t) { return nullptr; }
  bool notify_mr(int, void*) { return true; }
  bool wait_mr_notify(int, void*&) { return true; }
  void* reg_mr(void*, size_t) { return nullptr; }
  void dereg_mr(void*) {}
};

// Simplified transport backend for testing that doesn't require full transport layer
class MockTransportBackend final : public Backend {
 public:
  explicit MockTransportBackend(MockCommunicator& comm, CollectiveMemory memory)
      : comm_(comm), memory_(std::move(memory)) {}

  char const* name() const override { return "mock-transport"; }

  bool supports(ExecutionOpKind kind) const override {
    return kind == ExecutionOpKind::Send ||
           kind == ExecutionOpKind::Recv;
  }

  BackendToken submit(ExecutionOp const& op) override {
    BackendToken token{next_token_++};
    completed_tokens_[token.value] = true;
    completed_queue_.push_back(token.value);
    return token;
  }

  bool poll(BackendToken token) override {
    auto it = completed_tokens_.find(token.value);
    return it != completed_tokens_.end() && it->second;
  }

  bool try_pop_completed(BackendToken& token) override {
    if (completed_queue_.empty()) return false;
    token.value = completed_queue_.front();
    completed_queue_.pop_front();
    return true;
  }

  void release(BackendToken token) override {
    completed_tokens_.erase(token.value);
  }

 private:
  MockCommunicator& comm_;
  CollectiveMemory memory_{};
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, bool> completed_tokens_;
  std::deque<uint64_t> completed_queue_;
};

}  // namespace CCL
}  // namespace UKernel
