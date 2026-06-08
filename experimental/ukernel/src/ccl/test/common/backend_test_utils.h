#pragma once

#include "../backend/backend.h"
#include "../executor.h"
#include "../utils.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

struct TestMemory {
  void* input_ptr = nullptr;
  void* output_ptr = nullptr;
  void* scratch_ptr = nullptr;
};

inline TestMemory make_test_memory(int rank, int nranks, size_t bytes) {
  (void)rank;
  (void)nranks;
  TestMemory mem;
  mem.input_ptr = allocate_test_storage(bytes);
  mem.output_ptr = allocate_test_storage(bytes);
  mem.scratch_ptr = allocate_test_storage(bytes);
  return mem;
}

inline CollectiveConfig make_test_config(int nranks, int rank,
                                         size_t tensor_bytes,
                                         size_t tile_bytes) {
  CollectiveConfig config{};
  config.nranks = nranks;
  config.rank = rank;
  config.input_bytes = tensor_bytes;
  config.output_bytes = tensor_bytes;
  config.tile_bytes = tile_bytes;
  config.kind = CollKind::AllReduceRing;
  return config;
}

inline void validate_basic_tiled(TiledResult const& result) {
  assert(result.input_bytes > 0);
  assert(!result.ops.empty());

  for (size_t index = 0; index < result.ops.size(); ++index) {
    auto const& op = result.ops[index];
    assert(op.bytes > 0);
    if (op.kind == OpKind::Send) {
      assert(op.src_peer == ~0u);
      assert(op.dst_peer != ~0u);
    }
    if (op.kind == OpKind::Recv ||
        op.kind == OpKind::RecvReduce) {
      assert(op.src_peer != ~0u);
      assert(op.dst_peer == ~0u);
    }
    if (op.kind == OpKind::Copy || op.kind == OpKind::Reduce) {
      assert(op.src_peer == ~0u || op.dst_peer == ~0u);
    }
    for (uint32_t dep : op.deps) {
      assert(dep < result.ops.size());
      assert(dep < index);
    }
  }
}

inline size_t count_ops(TiledResult const& result, OpKind kind) {
  size_t total = 0;
  for (auto const& op : result.ops)
    if (op.kind == kind) ++total;
  return total;
}

inline void validate_basic_exec_tiled(TiledResult const& result) {
  validate_basic_tiled(result);
}

inline size_t count_exec_ops(TiledResult const& result, OpKind kind) {
  return count_ops(result, kind);
}

// ── Mock backends for unit tests ──────────────────────────────────────

class MockBackend final : public Backend {
 public:
  explicit MockBackend(uint32_t polls_before_ready = 1)
      : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

  char const* name() const override { return "mock"; }
  void validate(TiledResult const&, void*, void*, void*) override {}
  bool supports(OpKind kind) const override {
    return kind == OpKind::Send || kind == OpKind::Recv;
  }

  BackendToken submit(Op const&, OpBindings const&,
                      void* input_ptr, void* output_ptr, void* scratch_ptr) override {
    BackendToken t{next_token_++};
    ++submissions_;
    pending_polls_[t.value] = polls_before_ready_;
    return t;
  }

  size_t drain(BackendToken* out, size_t max_count) override {
    size_t n = 0;
    auto it = pending_polls_.begin();
    while (it != pending_polls_.end() && n < max_count) {
      if (it->second > 1) {
        --it->second;
        ++it;
      } else {
        out[n].value = it->first;
        out[n].failed = false;
        ++n;
        it = pending_polls_.erase(it);
      }
    }
    return n;
  }

  size_t submissions() const { return submissions_; }

 private:
  uint32_t polls_before_ready_ = 1;
  uint64_t next_token_ = 1;
  size_t submissions_ = 0;
  std::unordered_map<uint64_t, uint32_t> pending_polls_;
};

class MockDeviceBackend final : public Backend {
 public:
  explicit MockDeviceBackend(uint32_t polls_before_ready = 1)
      : polls_before_ready_(polls_before_ready == 0 ? 1 : polls_before_ready) {}

  char const* name() const override { return "device"; }
  void validate(TiledResult const&, void*, void*, void*) override {}
  bool supports(OpKind kind) const override {
    return kind == OpKind::Copy || kind == OpKind::Reduce ||
           kind == OpKind::Send || kind == OpKind::RecvReduce ||
           kind == OpKind::Recv;
  }

  BackendToken submit(Op const& op, OpBindings const&,
                      void* input_ptr, void* output_ptr, void* scratch_ptr) override {
    if (!supports(op.kind))
      throw std::invalid_argument("device backend does not support this op");
    BackendToken t{next_token_++};
    ++submissions_;
    pending_polls_[t.value] = polls_before_ready_;
    return t;
  }

  size_t drain(BackendToken* out, size_t max_count) override {
    size_t n = 0;
    auto it = pending_polls_.begin();
    while (it != pending_polls_.end() && n < max_count) {
      if (it->second > 1) {
        --it->second;
        ++it;
      } else {
        out[n].value = it->first;
        out[n].failed = false;
        ++n;
        it = pending_polls_.erase(it);
      }
    }
    return n;
  }

  size_t submissions() const { return submissions_; }

 private:
  uint32_t polls_before_ready_ = 1;
  uint64_t next_token_ = 1;
  size_t submissions_ = 0;
  std::unordered_map<uint64_t, uint32_t> pending_polls_;
};

class ThrowingBackend final : public Backend {
 public:
  explicit ThrowingBackend(std::string message)
      : message_(std::move(message)) {}

  char const* name() const override { return "throwing"; }
  void validate(TiledResult const&, void*, void*, void*) override {}
  bool supports(OpKind) const override { return true; }
  BackendToken submit(Op const&, OpBindings const&,
                      void* input_ptr, void* output_ptr, void* scratch_ptr) override {
    throw std::runtime_error(message_);
  }
  size_t drain(BackendToken*, size_t) override { return 0; }

 private:
  std::string message_;
};

}  // namespace Testing

}  // namespace CCL
}  // namespace UKernel
