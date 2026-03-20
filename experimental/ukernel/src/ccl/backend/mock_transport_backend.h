#pragma once

#include "../backend.h"
#include <cstddef>
#include <cstdint>
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
  explicit MockTransportBackend(MockCommunicator& comm, int peer_rank,
                               CollectiveBuffers buffers)
      : comm_(comm), peer_rank_(peer_rank), buffers_(buffers) {}

  char const* name() const override { return "mock-transport"; }

  bool supports(ExecutionOpKind kind) const override {
    return kind == ExecutionOpKind::RdmaSend ||
           kind == ExecutionOpKind::RdmaRecv;
  }

  BackendToken submit(ExecutionOp const& op) override {
    BackendToken token{next_token_++};
    completed_tokens_[token.value] = true;
    return token;
  }

  bool poll(BackendToken token) override {
    auto it = completed_tokens_.find(token.value);
    return it != completed_tokens_.end() && it->second;
  }

  void release(BackendToken token) override {
    completed_tokens_.erase(token.value);
  }

 private:
  MockCommunicator& comm_;
  int peer_rank_ = -1;
  CollectiveBuffers buffers_{};
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, bool> completed_tokens_;
};

}  // namespace CCL
}  // namespace UKernel
