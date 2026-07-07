#include "algo/chunk_graph.h"
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

// Mock backend

class MockBackend final : public BatchBackend {
 public:
  MockBackend(bool auto_complete = false) : auto_complete_(auto_complete) {}

  char const* name() const override { return "mock"; }
  bool supports(ExecOpKind) const override { return true; }

  size_t do_enqueue(Cmd const* cmds, size_t n,
                    uint32_t* out_indices = nullptr) override {
    std::lock_guard lock(mtx_);
    size_t accepted = 0;
    while (accepted < n && in_flight_ < capacity()) {
      enqueued_.push_back(cmds[accepted]);
      uint32_t be = next_be_++;
      if (out_indices) out_indices[accepted] = be;
      completed_.push_back(be);
      ++in_flight_;
      ++accepted;
    }
    return accepted;
  }

  size_t do_drain(uint32_t* out, size_t max) override {
    std::lock_guard lock(mtx_);
    size_t n = std::min(completed_.size(), max);
    for (size_t i = 0; i < n; ++i) {
      out[i] = completed_.front();
      completed_.pop_front();
      --in_flight_;
    }
    return n;
  }

  size_t capacity() const override { return 4096; }

  size_t enqueued_count() const {
    std::lock_guard lock(mtx_);
    return enqueued_.size();
  }

 private:
  mutable std::mutex mtx_;
  bool auto_complete_;
  std::vector<Cmd> enqueued_;
  std::deque<uint32_t> completed_;
  size_t in_flight_ = 0;
  uint32_t next_be_ = 1;
};

// SprayExecutor integration tests

void test_executor_allreduce_async() {
  printf("[test] executor: async allreduce via mock backends...\n");

  MockBackend dev_mock(true), tpt_mock(true), signal_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock, &signal_mock);

  CollectiveConfig cfg = Testing::make_test_config(4, 0, 1024, 256);
  std::vector<uint8_t> in(1024, 0xAA);
  std::vector<uint8_t> out(1024, 0);
  std::vector<uint8_t> scratch(1024, 0);

  auto h = ex->submit(cfg, in.data(), out.data(), scratch.data());

  bool done = ex->wait(h, std::chrono::milliseconds(5000));
  assert(done);
  // Spin until drain threads finish processing all completions
  for (int retry = 0; retry < 100; ++retry) {
    if (ex->status(h) == CollectiveOpStatus::Completed) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
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

  MockBackend dev_mock(true), tpt_mock(true), signal_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock, &signal_mock);

  CollectiveConfig cfg;
  cfg.nranks = 4;
  cfg.rank = 0;
  cfg.input_bytes = 512;
  cfg.output_bytes = 512;
  cfg.tile_bytes = 128;
  cfg.kind = CollKind::AllToAllPairwise;

  std::vector<uint8_t> in(512, 0xBB);
  std::vector<uint8_t> out(512, 0);
  std::vector<uint8_t> scratch(1024, 0);

  auto h = ex->submit(cfg, in.data(), out.data(), scratch.data());

  bool done = ex->wait(h, std::chrono::milliseconds(5000));
  assert(done);
  // Spin until drain threads finish processing all completions
  for (int retry = 0; retry < 100; ++retry) {
    if (ex->status(h) == CollectiveOpStatus::Completed) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
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

  MockBackend dev_mock(true), tpt_mock(true), signal_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock, &signal_mock);

  CollectiveConfig cfg = Testing::make_test_config(2, 0, 256, 64);
  std::vector<uint8_t> in(256, 0xCC);
  std::vector<uint8_t> out(256, 0);
  std::vector<uint8_t> scratch(256, 0);

  auto h1 = ex->submit(cfg, in.data(), out.data(), scratch.data());
  auto h2 = ex->submit(cfg, in.data(), out.data(), scratch.data());
  auto h3 = ex->submit(cfg, in.data(), out.data(), scratch.data());

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

  MockBackend dev_mock(true), tpt_mock(true), signal_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock, &signal_mock);

  CollectiveConfig cfg = Testing::make_test_config(2, 0, 512, 128);

  std::vector<uint8_t> in(cfg.input_bytes, 0xDD);
  std::vector<uint8_t> out(cfg.output_bytes, 0);
  std::vector<uint8_t> scratch(1024, 0);

  auto h = ex->submit(cfg, in.data(), out.data(), scratch.data());
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

  MockBackend dev_mock(true), tpt_mock(true), signal_mock(true);
  auto ex = std::make_unique<SprayExecutor>(&dev_mock, &tpt_mock, &signal_mock);

  CollectiveConfig cfg = Testing::make_test_config(2, 0, 256, 64);
  std::vector<uint8_t> in(256), out(256), scratch(256);
  auto h = ex->submit(cfg, in.data(), out.data(), scratch.data());
  // With auto-complete, run may already be done; just verify wait succeeds
  ex->wait(h, std::chrono::milliseconds(5000));
  // Let drain threads finish processing so active_runs_ settles to 0
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  assert(ex->active_count() == 0);
  ex->release(h);
}

}  // namespace
}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;

  printf("\nSprayExecutor Integration Tests\n");
  test_executor_allreduce_async();
  test_executor_alltoall_async();
  test_executor_multiple_submits();
  test_executor_run_tiled_sync();
  test_executor_error_message();
  test_executor_active_count();
  fprintf(stderr, "  PASSED\n");

  printf("\nAll async tests PASSED\n");
  return 0;
}
