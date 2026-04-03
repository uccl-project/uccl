#pragma once

#include "../backend/backend.h"
#include "../executor.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace UKernel {
namespace CCL {

namespace Testing {

inline void* allocate_test_storage(size_t bytes) {
  static std::vector<std::unique_ptr<uint8_t[]>> pool;
  if (bytes == 0) bytes = 1;
  pool.emplace_back(std::make_unique<uint8_t[]>(bytes));
  std::fill_n(pool.back().get(), bytes, uint8_t{0});
  return pool.back().get();
}

inline void init_registered_buffer(RegisteredBuffer& buffer, size_t bytes,
                                   int nranks, void* local_ptr) {
  buffer.bytes = bytes;
  buffer.layout.sizes = {static_cast<int64_t>(bytes)};
  buffer.layout.strides = {1};
  buffer.local_ptr = local_ptr;
  buffer.local_mr_id = 1;
  buffer.peer_views.resize(static_cast<size_t>(nranks));
  for (int peer = 0; peer < nranks; ++peer) {
    auto& peer_view = buffer.peer_views[static_cast<size_t>(peer)];
    peer_view.mr_id = static_cast<uint32_t>(peer + 1);
    peer_view.same_node = true;
  }
}

inline CollectiveBinding make_test_memory(int rank, int nranks, size_t bytes,
                                          CollectiveBufferRoles roles = {}) {
  CollectiveBinding binding;
  binding.registry = std::make_shared<BufferRegistry>();
  binding.registry->local_rank = rank;
  binding.roles = roles;
  binding.roles.validate();

  RegisteredBuffer& input =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Input));
  init_registered_buffer(input, bytes, nranks, allocate_test_storage(bytes));

  if (binding.buffer_id(CollectiveBufferRole::Output) !=
      binding.buffer_id(CollectiveBufferRole::Input)) {
    RegisteredBuffer& output =
        binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Output));
    init_registered_buffer(output, bytes, nranks, allocate_test_storage(bytes));
  }

  RegisteredBuffer& staging =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Scratch));
  init_registered_buffer(staging, bytes, nranks, allocate_test_storage(bytes));
  return binding;
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
    auto validate_ref = [&](BufferRef const& ref) {
      assert(ref.buffer_id != kInvalidBufferId);
      if (ref.kind == BufferKind::Remote) {
        assert(ref.rank >= 0);
        assert(ref.rank < plan.nranks);
        assert(ref.rank != plan.rank);
      } else {
        assert(ref.rank < 0);
      }
    };
    validate_ref(op.src);
    validate_ref(op.dst);
    if (op.kind == PrimitiveOpKind::Send) {
      assert(op.src.kind == BufferKind::Local);
      assert(op.dst.kind == BufferKind::Remote);
    }
    if (op.kind == PrimitiveOpKind::Recv) {
      assert(op.src.kind == BufferKind::Remote);
      assert(op.dst.kind == BufferKind::Local);
    }
    if (op.kind == PrimitiveOpKind::Copy || op.kind == PrimitiveOpKind::Reduce) {
      assert(op.src.kind == BufferKind::Local ||
             op.src.kind == BufferKind::Remote);
      assert(op.dst.kind == BufferKind::Local ||
             op.dst.kind == BufferKind::Remote);
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

inline void validate_basic_exec_plan(ExecutionPlan const& plan) {
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
    auto validate_ref = [&](BufferRef const& ref) {
      assert(ref.buffer_id != kInvalidBufferId);
      if (ref.kind == BufferKind::Remote) {
        assert(ref.rank >= 0);
        assert(ref.rank < plan.nranks);
        assert(ref.rank != plan.rank);
      } else {
        assert(ref.rank < 0);
      }
    };
    validate_ref(op.src);
    validate_ref(op.dst);
    for (uint32_t dep : op.deps) {
      assert(dep < plan.ops.size());
      assert(dep < op.op_id);
    }
  }
}

inline size_t count_exec_ops(ExecutionPlan const& plan, ExecOpKind kind) {
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
  void validate(ExecutionPlan const&, CollectiveBinding&) const override {}
  bool supports(ExecOpKind) const override { return true; }
  BackendToken submit(ExecOp const&, CollectiveBinding&) override {
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
  void validate(ExecutionPlan const&, CollectiveBinding&) const override {}
  bool supports(ExecOpKind kind) const override {
    return kind == ExecOpKind::DeviceCopy || kind == ExecOpKind::DeviceReduce;
  }
  BackendToken submit(ExecOp const& op, CollectiveBinding&) override {
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

class ThrowingBackend final : public Backend {
 public:
  explicit ThrowingBackend(std::string message)
      : message_(std::move(message)) {}

  char const* name() const override { return "throwing"; }
  void validate(ExecutionPlan const&, CollectiveBinding&) const override {}
  bool supports(ExecOpKind) const override { return true; }
  BackendToken submit(ExecOp const&, CollectiveBinding&) override {
    throw std::runtime_error(message_);
  }
  bool poll(BackendToken) override { return false; }
  bool try_pop_completed(BackendToken&) override { return false; }
  void release(BackendToken) override {}

 private:
  std::string message_;
};

}  // namespace Testing

}  // namespace CCL
}  // namespace UKernel
