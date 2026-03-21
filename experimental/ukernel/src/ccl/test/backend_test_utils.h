#pragma once

#include "../executor.h"
#include "../backend.h"
#include <cassert>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace CCL {

namespace Testing {

inline CollectiveMemory make_test_memory(int rank, int nranks, size_t bytes) {
  CollectiveMemory memory;
  memory.tensor.local_rank = rank;
  memory.tensor.bytes = bytes;
  memory.tensor.local_ptr = nullptr;
  memory.tensor.local_mr_id = 1;
  memory.tensor.peers.resize(static_cast<size_t>(nranks));
  for (int peer = 0; peer < nranks; ++peer) {
    auto& peer_view = memory.tensor.peers[static_cast<size_t>(peer)];
    peer_view.rank = peer;
    peer_view.mr_id = static_cast<uint32_t>(peer + 1);
    peer_view.same_node = true;
    peer_view.peer_accessible = true;
  }
  memory.recv_staging = nullptr;
  memory.recv_staging_bytes = bytes;
  return memory;
}

inline CollectiveConfig make_ring_config(int nranks, int rank, size_t bytes_per_rank,
                                         size_t chunk_bytes, uint32_t channels = 1) {
  CollectiveConfig config{};
  config.nranks = nranks;
  config.rank = rank;
  config.channels = channels;
  config.bytes_per_rank = bytes_per_rank;
  config.chunk_bytes = chunk_bytes;
  config.algorithm = AlgorithmKind::Ring;
  return config;
}

inline void validate_basic_plan(CollectivePlan const& plan) {
  assert(plan.nranks >= 2);
  assert(plan.rank >= 0 && plan.rank < plan.nranks);
  assert(plan.channels >= 1);
  assert(plan.bytes_per_rank > 0);
  assert(plan.chunk_bytes > 0);
  assert(!plan.steps.empty());

  std::unordered_set<uint32_t> op_ids;
  for (auto const& step : plan.steps) {
    assert(!step.ops.empty());
    assert(step.chunk.channel_id < plan.channels);
    assert(step.chunk.size_bytes > 0);
    assert(step.chunk.offset_bytes + step.chunk.size_bytes <= plan.bytes_per_rank);
    for (uint32_t pred : step.predecessors) {
      assert(pred < plan.steps.size());
    }
    for (auto const& op : step.ops) {
      assert(op_ids.insert(op.op_id).second);
      assert(op.chunk.size_bytes > 0);
      assert(op.chunk.channel_id < plan.channels);
      if (op.peer_rank >= 0) {
        assert(op.peer_rank < plan.nranks);
        assert(op.peer_rank != plan.rank);
      }
      if (op.src.slot == MemorySlot::RecvStaging) {
        assert(op.src.rank == -1);
      }
      if (op.dst.slot == MemorySlot::RecvStaging) {
        assert(op.dst.rank == -1);
      }
      if (op.src.slot == MemorySlot::SymmetricTensor && op.src.rank >= 0) {
        assert(op.src.rank < plan.nranks);
      }
      if (op.dst.slot == MemorySlot::SymmetricTensor && op.dst.rank >= 0) {
        assert(op.dst.rank < plan.nranks);
      }
    }
  }

  for (auto const& step : plan.steps) {
    for (auto const& op : step.ops) {
      for (uint32_t dep : op.deps) {
        assert(op_ids.count(dep) == 1);
      }
    }
  }
}

inline size_t count_ops(CollectivePlan const& plan, ExecutionOpKind kind) {
  size_t total = 0;
  for (auto const& step : plan.steps) {
    for (auto const& op : step.ops) {
      if (op.kind == kind) {
        ++total;
      }
    }
  }
  return total;
}

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
    BackendToken token = submit_token(next_token_, submissions_, polls_before_ready_,
                                      pending_polls_);
    return token;
  }
  bool poll(BackendToken token) override {
    return poll_token(token, pending_polls_);
  }
  bool try_pop_completed(BackendToken&) override { return false; }
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

class MockDeviceBackend final : public Backend {
 public:
  explicit MockDeviceBackend(uint32_t polls_before_ready = 1)
      : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

  char const* name() const override { return "device"; }
  bool supports(ExecutionOpKind kind) const override {
    switch (kind) {
      case ExecutionOpKind::Copy:
      case ExecutionOpKind::Reduce:
      case ExecutionOpKind::EventWait:
      case ExecutionOpKind::Barrier:
        return true;
      case ExecutionOpKind::Send:
      case ExecutionOpKind::Recv:
        return false;
    }
    return false;
  }
  BackendToken submit(ExecutionOp const& op) override {
    if (!supports(op.kind)) {
      throw std::invalid_argument("device backend does not support this op");
    }
    return submit_token(next_token_, submissions_, polls_before_ready_,
                        pending_polls_);
  }
  bool poll(BackendToken token) override {
    return poll_token(token, pending_polls_);
  }
  bool try_pop_completed(BackendToken&) override { return false; }
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
