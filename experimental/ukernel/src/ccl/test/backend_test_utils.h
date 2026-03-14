#pragma once

#include "../src/ccl/backend.h"
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace UKernel {
namespace CCL {

namespace Testing {

inline BackendToken submit_token(uint64_t& next_token, uint64_t& submissions,
                                 uint32_t polls_before_ready,
                                 std::unordered_map<uint64_t, uint32_t>& pending_polls) {
  BackendToken token{next_token++};
  ++submissions;
  pending_polls.emplace(token.value, polls_before_ready);
  return token;
}

inline bool poll_token(BackendToken token,
                       std::unordered_map<uint64_t, uint32_t>& pending_polls) {
  auto it = pending_polls.find(token.value);
  if (it == pending_polls.end()) return true;
  if (it->second == 0) return true;
  --it->second;
  return it->second == 0;
}

inline void release_token(BackendToken token,
                          std::unordered_map<uint64_t, uint32_t>& pending_polls) {
  pending_polls.erase(token.value);
}

class MockBackend final : public Backend {
 public:
  explicit MockBackend(uint32_t polls_before_ready = 1)
      : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

  char const* name() const override { return "mock"; }
  bool supports(ExecutionOpKind) const override { return true; }
  BackendToken submit(ExecutionOp const&) override {
    return submit_token(next_token_, submissions_, polls_before_ready_,
                        pending_polls_);
  }
  bool poll(BackendToken token) override {
    return poll_token(token, pending_polls_);
  }
  void release(BackendToken token) override {
    release_token(token, pending_polls_);
  }

  uint64_t submissions() const { return submissions_; }

 private:
  uint32_t polls_before_ready_ = 1;
  uint64_t next_token_ = 1;
  uint64_t submissions_ = 0;
  std::unordered_map<uint64_t, uint32_t> pending_polls_;
};

class MockPersistentKernelBackend final : public Backend {
 public:
  explicit MockPersistentKernelBackend(uint32_t polls_before_ready = 1)
      : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

  char const* name() const override { return "persistent"; }
  bool supports(ExecutionOpKind kind) const override {
    switch (kind) {
      case ExecutionOpKind::PkCopy:
      case ExecutionOpKind::PkReduce:
      case ExecutionOpKind::EventWait:
      case ExecutionOpKind::Barrier:
        return true;
      case ExecutionOpKind::RdmaSend:
      case ExecutionOpKind::RdmaRecv:
      case ExecutionOpKind::CeCopy:
        return false;
    }
    return false;
  }
  BackendToken submit(ExecutionOp const& op) override {
    if (!supports(op.kind)) {
      throw std::invalid_argument("persistent backend does not support this op");
    }
    return submit_token(next_token_, submissions_, polls_before_ready_,
                        pending_polls_);
  }
  bool poll(BackendToken token) override {
    return poll_token(token, pending_polls_);
  }
  void release(BackendToken token) override {
    release_token(token, pending_polls_);
  }

  uint64_t submissions() const { return submissions_; }

 private:
  uint32_t polls_before_ready_ = 1;
  uint64_t next_token_ = 1;
  uint64_t submissions_ = 0;
  std::unordered_map<uint64_t, uint32_t> pending_polls_;
};

}  // namespace Testing

}  // namespace CCL
}  // namespace UKernel
