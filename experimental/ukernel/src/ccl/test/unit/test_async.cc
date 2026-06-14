#include "algo/chunk_graph.h"
#include "backend/async_backend.h"
#include "backend/backend.h"
#include "coll_config.h"
#include "executor.h"
#include "lower.h"
#include "test_config.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace UKernel {
namespace CCL {
namespace {

// ── Mock backend ─────────────────────────────────────────────────────────

class MockBackend final : public BatchBackend {
 public:
  MockBackend(bool auto_complete = false) : auto_complete_(auto_complete) {}

  char const* name() const override { return "mock"; }
  bool supports(OpKind) const override { return true; }

  void init(BufSpec[3]) override { inited_ = true; }

  size_t enqueue(Cmd const* cmds, size_t n,
                 uint32_t* out_indices = nullptr) override {
    std::lock_guard lock(mtx_);
    size_t accepted = 0;
    while (accepted < n && in_flight_ < capacity()) {
      enqueued_.push_back(cmds[accepted]);
      uint32_t idx = cmd_next_++;
      if (out_indices) out_indices[accepted] = idx;
      if (auto_complete_) completed_.push_back(idx);
      ++in_flight_;
      ++accepted;
    }
    return accepted;
  }

  size_t drain(uint32_t* out, size_t max) override {
    std::lock_guard lock(mtx_);
    size_t n = std::min(completed_.size(), max);
    for (size_t i = 0; i < n; ++i) {
      out[i] = completed_.front();
      completed_.pop_front();
      --in_flight_;
    }
    return n;
  }

  size_t capacity() const override { return 256; }

  void complete_last_n(size_t n) {
    std::lock_guard lock(mtx_);
    uint32_t first = cmd_next_ - n;
    for (uint32_t i = 0; i < n; ++i) completed_.push_back(first + i);
  }

  size_t enqueued_count() const {
    std::lock_guard lock(mtx_);
    return enqueued_.size();
  }

 private:
  mutable std::mutex mtx_;
  bool auto_complete_;
  std::vector<Cmd> enqueued_;
  std::deque<uint32_t> completed_;
  uint32_t cmd_next_ = 0;
  size_t in_flight_ = 0;
  bool inited_ = false;
};

// ── AsyncBackend tests ───────────────────────────────────────────────────

void test_async_basic_enqueue_drain() {
  printf("[test] async backend: basic enqueue → drain...\n");

  MockBackend mock(false);  // explicit completion mode
  AsyncBackend async(&mock, 256, 256);
  async.start();

  // Build 5 commands
  CmdWithId cmds[5];
  for (int i = 0; i < 5; ++i) {
    cmds[i].cmd.kind = OpKind::Copy;
    cmds[i].cmd.bytes = 128;
    cmds[i].cmd.src_buf = 1;
    cmds[i].cmd.dst_buf = 2;
    cmds[i].cmd.src_peer = ~0u;
    cmds[i].cmd.dst_peer = ~0u;
    cmds[i].caller_id = 100 + i;
  }

  size_t n = async.try_enqueue(cmds, 5);
  assert(n == 5);

  // Wait for submit thread to pick up, then simulate completion
  uint32_t out[5];
  size_t total = 0;
  for (int retry = 0; retry < 500 && mock.enqueued_count() < 5; ++retry)
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  assert(mock.enqueued_count() == 5);
  mock.complete_last_n(5);

  // Drain thread should now push caller_ids to done_ring
  total = 0;
  for (int retry = 0; retry < 1000 && total < 5; ++retry) {
    size_t d = async.try_drain(out + total, 5 - total);
    total += d;
    if (total < 5) std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  assert(total == 5);

  // Verify caller_ids match
  for (int i = 0; i < 5; ++i) assert(out[i] >= 100 && out[i] <= 104);

  assert(mock.enqueued_count() == 5);

  async.stop();
  assert(true);
}

void test_async_capacity_backpressure() {
  printf("[test] async backend: capacity backpressure...\n");

  MockBackend mock(false);
  // cmd_ring: 4 slots → usable 3
  AsyncBackend async(&mock, 4, 64);

  CmdWithId cmds[8];
  for (int i = 0; i < 8; ++i) {
    cmds[i].cmd.kind = OpKind::Copy;
    cmds[i].cmd.bytes = 64;
    cmds[i].cmd.src_buf = 1;
    cmds[i].cmd.dst_buf = 2;
    cmds[i].cmd.src_peer = ~0u;
    cmds[i].cmd.dst_peer = ~0u;
    cmds[i].caller_id = i;
  }

  // Before start, cmd_ring is empty, so we can enqueue up to capacity
  size_t nfree = async.cmd_free();
  assert(nfree == 3);  // 4 slots → 3 usable

  size_t n = async.try_enqueue(cmds, 8);
  assert(n == 3);  // only 3 fit

  async.start();

  // Wait for submit thread to drain the ring
  for (int retry = 0; retry < 500; ++retry) {
    if (async.cmd_free() >= 3) break;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  assert(async.cmd_free() >= 3);

  n = async.try_enqueue(cmds + 3, 5);
  assert(n >= 3);

  async.stop();
}

void test_async_done_ring_multiple_drain() {
  printf("[test] async backend: done_ring multiple drain batches...\n");

  MockBackend mock;
  AsyncBackend async(&mock, 512, 512);
  async.start();

  constexpr int N = 100;
  CmdWithId cmds[N];
  for (int i = 0; i < N; ++i) {
    cmds[i].cmd.kind = OpKind::Copy;
    cmds[i].cmd.bytes = 8;
    cmds[i].cmd.src_buf = 1;
    cmds[i].cmd.dst_buf = 2;
    cmds[i].cmd.src_peer = ~0u;
    cmds[i].cmd.dst_peer = ~0u;
    cmds[i].caller_id = 1000 + i;
  }

  size_t n = async.try_enqueue(cmds, N);
  assert(n == N);

  // Wait for submit thread and complete all
  for (int retry = 0; retry < 500 && mock.enqueued_count() < N; ++retry)
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  assert(mock.enqueued_count() == N);
  mock.complete_last_n(N);

  uint32_t out[N];
  size_t total = 0;
  for (int retry = 0; retry < 500 && total < N; ++retry) {
    size_t d = async.try_drain(out + total, 16);  // drain in small batches
    total += d;
    if (total < N) std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  assert(total == N);

  // All caller_ids should be in range
  for (size_t i = 0; i < N; ++i) assert(out[i] >= 1000 && out[i] < 1000 + N);

  async.stop();
}

// ── SprayExecutor integration test ───────────────────────────────────────

void test_executor_allreduce_async() {
  printf("[test] executor: async allreduce via mock backends...\n");

  MockBackend dev_mock(true), tpt_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock);

  CollectiveConfig cfg = Testing::make_test_config(4, 0, 1024, 256);
  std::vector<uint8_t> in(1024, 0xAA);
  std::vector<uint8_t> out(1024, 0);
  std::vector<uint8_t> scratch(1024, 0);

  auto h = ex->submit_allreduce(cfg, in.data(), out.data(), scratch.data());

  bool done = ex->wait(h, std::chrono::milliseconds(5000));
  assert(done);
  assert(ex->status(h) == CollectiveOpStatus::Completed);

  size_t dev_cmds = dev_mock.enqueued_count();
  size_t tpt_cmds = tpt_mock.enqueued_count();
  printf("  dev enqueued: %zu, tpt enqueued: %zu\n", dev_cmds, tpt_cmds);
  assert(dev_cmds + tpt_cmds > 0);

  ex->release(h);
  printf("  PASSED\n");
}

void test_executor_alltoall_async() {
  printf("[test] executor: async alltoall via mock backends...\n");

  MockBackend dev_mock(true), tpt_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock);

  CollectiveConfig cfg;
  cfg.nranks = 4;
  cfg.rank = 0;
  cfg.input_bytes = 512;
  cfg.output_bytes = 512;
  cfg.tile_bytes = 128;
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = false;

  std::vector<uint8_t> in(512, 0xBB);
  std::vector<uint8_t> out(512, 0);
  std::vector<uint8_t> scratch(1024, 0);

  auto h = ex->submit_alltoall(cfg, in.data(), out.data(), scratch.data());

  bool done = ex->wait(h, std::chrono::milliseconds(5000));
  assert(done);
  assert(ex->status(h) == CollectiveOpStatus::Completed);

  size_t dev_cmds = dev_mock.enqueued_count();
  size_t tpt_cmds = tpt_mock.enqueued_count();
  fprintf(stderr, "  dev enqueued: %zu, tpt enqueued: %zu\n", dev_cmds,
          tpt_cmds);
  assert(dev_cmds + tpt_cmds > 0);

  ex->release(h);
}

void test_executor_multiple_submits() {
  printf("[test] executor: multiple concurrent submits...\n");

  MockBackend dev_mock(true), tpt_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock);

  CollectiveConfig cfg = Testing::make_test_config(2, 0, 256, 64);
  std::vector<uint8_t> in(256, 0xCC);
  std::vector<uint8_t> out(256, 0);
  std::vector<uint8_t> scratch(256, 0);

  auto h1 = ex->submit_allreduce(cfg, in.data(), out.data(), scratch.data());
  auto h2 = ex->submit_allreduce(cfg, in.data(), out.data(), scratch.data());
  auto h3 = ex->submit_allreduce(cfg, in.data(), out.data(), scratch.data());

  // With auto-complete, runs may finish very fast; just verify all complete
  bool d1 = ex->wait(h1, std::chrono::milliseconds(5000));
  bool d2 = ex->wait(h2, std::chrono::milliseconds(5000));
  bool d3 = ex->wait(h3, std::chrono::milliseconds(5000));
  assert(d1 && d2 && d3);

  ex->release(h1);
  ex->release(h2);
  ex->release(h3);

  printf("  PASSED\n");
}

void test_executor_run_tiled_sync() {
  printf("[test] executor: run_tiled synchronous path...\n");

  MockBackend dev_mock(true), tpt_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock);

  CollectiveConfig cfg = Testing::make_test_config(2, 0, 512, 128);

  std::vector<uint8_t> in(cfg.input_bytes, 0xDD);
  std::vector<uint8_t> out(cfg.output_bytes, 0);
  std::vector<uint8_t> scratch(1024, 0);

  auto h = ex->submit_allreduce(cfg, in.data(), out.data(), scratch.data());
  bool done = ex->wait(h, std::chrono::milliseconds(5000));
  assert(done);

  size_t total = dev_mock.enqueued_count() + tpt_mock.enqueued_count();
  assert(total > 0);
  fprintf(stderr, "  submit_allreduce processed %zu commands\n", total);
}

void test_executor_error_message() {
  printf("[test] executor: error_message on fresh handle...\n");

  MockBackend dev_mock(true), tpt_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock);

  assert(ex->error_message(999) == "");  // non-existent handle
}

void test_executor_active_count() {
  printf("[test] executor: active_count...\n");

  MockBackend dev_mock(true), tpt_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock);

  assert(ex->active_count() == 0);

  CollectiveConfig cfg = Testing::make_test_config(2, 0, 256, 64);
  std::vector<uint8_t> in(256), out(256), scratch(256);
  auto h = ex->submit_allreduce(cfg, in.data(), out.data(), scratch.data());
  // With auto-complete, run may already be done; just verify wait succeeds
  ex->wait(h, std::chrono::milliseconds(5000));
  assert(ex->active_count() == 0);
  ex->release(h);
}

}  // namespace
}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;

  printf("=== AsyncBackend Tests ===\n");
  test_async_basic_enqueue_drain();
  fprintf(stderr, "  PASSED\n");
  test_async_capacity_backpressure();
  fprintf(stderr, "  PASSED\n");
  test_async_done_ring_multiple_drain();
  fprintf(stderr, "  PASSED\n");

  printf("\n=== SprayExecutor Integration Tests ===\n");
  test_executor_allreduce_async();
  test_executor_alltoall_async();
  test_executor_multiple_submits();
  test_executor_run_tiled_sync();
  test_executor_error_message();
  test_executor_active_count();
  fprintf(stderr, "  PASSED\n");

  printf("\n=== All async tests PASSED ===\n");
  return 0;
}
