#pragma once

#include "../backend/backend.h"
#include "../executor.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace UKernel {
namespace CCL {

namespace Testing {

inline CollectiveMemory make_test_memory(int rank, int nranks, size_t bytes) {
  CollectiveMemory memory;
  memory.tensor.local_rank = rank;
  memory.tensor.bytes = bytes;
  memory.tensor.layout.sizes = {static_cast<int64_t>(bytes)};
  memory.tensor.layout.strides = {1};
  memory.tensor.local_ptr = nullptr;
  memory.tensor.local_mr_id = 1;
  memory.tensor.peer_views.resize(static_cast<size_t>(nranks));
  for (int peer = 0; peer < nranks; ++peer) {
    auto& peer_view = memory.tensor.peer_views[static_cast<size_t>(peer)];
    peer_view.mr_id = static_cast<uint32_t>(peer + 1);
    peer_view.same_node = true;
    peer_view.peer_accessible = true;
  }
  memory.staging.local_ptr = nullptr;
  memory.staging.bytes = bytes;
  memory.staging.layout.sizes = {static_cast<int64_t>(bytes)};
  memory.staging.layout.strides = {1};
  return memory;
}

inline CollectiveConfig make_test_config(int nranks, int rank,
                                         size_t tensor_bytes, size_t tile_bytes,
                                         uint32_t num_flows = 1) {
  CollectiveConfig config{};
  config.nranks = nranks;
  config.rank = rank;
  config.num_flows = num_flows;
  config.tensor_bytes = tensor_bytes;
  config.tile_bytes = tile_bytes;
  config.staging_bytes =
      std::max(static_cast<size_t>(num_flows),
               static_cast<size_t>(nranks > 0 ? nranks - 1 : 0)) *
      tile_bytes;
  config.algorithm = AlgorithmKind::Ring;
  return config;
}

inline void validate_basic_plan(CollectivePlan const& plan) {
  assert(plan.nranks >= 2);
  assert(plan.rank >= 0 && plan.rank < plan.nranks);
  assert(plan.num_flows >= 1);
  assert(plan.tensor_bytes > 0);
  assert(plan.tile_bytes > 0);
  assert(!plan.ops.empty());

  std::unordered_set<uint32_t> op_ids;
  for (size_t index = 0; index < plan.ops.size(); ++index) {
    auto const& op = plan.ops[index];
    assert(op_ids.insert(op.op_id).second);
    assert(op.op_id == index);
    assert(op.tile.size_bytes > 0);
    assert(op.tile.flow_index < plan.num_flows);
    assert(op.tile.offset_bytes + op.tile.size_bytes <= plan.tensor_bytes);
    if (op.peer_rank >= 0) {
      assert(op.peer_rank < plan.nranks);
      assert(op.peer_rank != plan.rank);
    }
    if (op.kind == PrimitiveOpKind::Send || op.kind == PrimitiveOpKind::Copy ||
        op.kind == PrimitiveOpKind::Reduce) {
      assert(op.src.kind == BufferKind::Tensor ||
             op.src.kind == BufferKind::Staging);
    }
    if (op.kind == PrimitiveOpKind::Recv || op.kind == PrimitiveOpKind::Copy ||
        op.kind == PrimitiveOpKind::Reduce) {
      assert(op.dst.kind == BufferKind::Tensor ||
             op.dst.kind == BufferKind::Staging);
    }
    for (uint32_t dep : op.deps) {
      assert(dep < plan.ops.size());
      assert(dep < op.op_id);
    }
  }
}

inline size_t count_ops(CollectivePlan const& plan, PrimitiveOpKind kind) {
  size_t total = 0;
  for (auto const& op : plan.ops) {
    if (op.kind == kind) ++total;
  }
  return total;
}

inline BackendToken submit_token(
    uint64_t& next_token, uint64_t& submissions, uint32_t polls_before_ready,
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

inline void release_token(
    BackendToken token, std::unordered_map<uint64_t, uint32_t>& pending_polls) {
  pending_polls.erase(token.value);
}

class MockBackend final : public Backend {
 public:
  explicit MockBackend(uint32_t polls_before_ready = 1)
      : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

  char const* name() const override { return "mock"; }
  void validate(ExecutionPlan const&) const override {}
  bool supports(ExecOpKind) const override { return true; }
  BackendToken submit(ExecOp const&) override {
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

class MockDeviceBackend final : public Backend {
 public:
  explicit MockDeviceBackend(uint32_t polls_before_ready = 1)
      : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

  char const* name() const override { return "device"; }
  void validate(ExecutionPlan const&) const override {}
  bool supports(ExecOpKind kind) const override {
    return kind == ExecOpKind::DeviceCopy || kind == ExecOpKind::DeviceReduce;
  }
  BackendToken submit(ExecOp const& op) override {
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
