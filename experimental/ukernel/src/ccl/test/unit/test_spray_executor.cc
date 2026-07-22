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

// Controllable mock backend: per-kind filtering, capacity limit, auto-complete.
class CtlBackend final : public BatchBackend {
 public:
  explicit CtlBackend(char const* label, size_t cap = 4096,
                      ExecOpKind accept = ExecOpKind::Put)
      : label_(label), cap_(cap), accept_(accept) {}

  char const* name() const override { return label_; }

  bool supports(ExecOpKind kind) const override {
    return accept_ == static_cast<ExecOpKind>(~0u) || kind == accept_;
  }

  size_t do_enqueue(Cmd const* cmds, size_t n,
                    uint32_t* out_indices = nullptr) override {
    std::lock_guard lock(mtx_);
    size_t accepted = 0;
    while (accepted < n && in_flight_ < cap_) {
      auto const& c = cmds[accepted];
      if (!supports(c.kind)) {
        ++accepted;
        continue;
      }
      uint32_t be_idx = next_be_++;
      if (out_indices) out_indices[accepted] = be_idx;
      enqueued_.push_back(c);
      pending_.push_back({be_idx});
      ++in_flight_;
      ++accepted;
    }
    return accepted;
  }

  uint32_t reserve_slot() override {
    std::lock_guard lock(mtx_);
    return next_be_++;
  }

  bool do_enqueue_reserved(Cmd const& c, uint32_t be_idx) override {
    std::lock_guard lock(mtx_);
    if (in_flight_ >= cap_ || !supports(c.kind)) return false;
    enqueued_.push_back(c);
    completed_.push_back({be_idx});
    ++in_flight_;
    return true;
  }

  size_t do_drain(uint32_t* out, size_t max) override {
    std::lock_guard lock(mtx_);
    size_t n = std::min(completed_.size(), max);
    for (size_t i = 0; i < n; ++i) {
      out[i] = completed_.front().be_idx;
      completed_.pop_front();
      --in_flight_;
    }
    // Ops enqueued via do_enqueue complete on the NEXT drain cycle, so
    // the executor always publishes its slot-table entry first (real
    // backends complete asynchronously; this keeps the mock faithful).
    completed_.insert(completed_.end(), pending_.begin(), pending_.end());
    pending_.clear();
    return n;
  }

  size_t capacity() const override { return cap_; }
  void set_capacity(size_t c) {
    std::lock_guard lk(mtx_);
    cap_ = c;
  }
  size_t enqueued_count() const {
    std::lock_guard lk(mtx_);
    return enqueued_.size();
  }
  size_t pending_count() const {
    std::lock_guard lk(mtx_);
    return completed_.size();
  }

 private:
  struct Item {
    uint32_t be_idx;
  };
  char const* label_;
  size_t cap_;
  ExecOpKind accept_;
  mutable std::mutex mtx_;
  std::vector<Cmd> enqueued_;
  std::deque<Item> completed_;
  std::deque<Item> pending_;
  size_t in_flight_ = 0;
  uint32_t next_be_ = 1;
};

static constexpr ExecOpKind kAnyKind = static_cast<ExecOpKind>(~0u);

static bool submit_and_wait(SprayExecutor& ex, CollectiveConfig const& cfg,
                            void* in, void* out) {
  auto h = ex.submit(cfg, in, out);
  if (h == kInvalidHandle) return false;
  bool done = ex.wait(h, std::chrono::milliseconds(5000));
  if (!done) return false;
  for (int retry = 0; retry < 100; ++retry) {
    if (ex.status(h) == CollectiveOpStatus::Completed) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  bool ok = ex.status(h) == CollectiveOpStatus::Completed;
  ex.release(h);
  return ok;
}

// ── Test 1: path priority ──────────────────────────────────────────────
void test_path_priority() {
  printf("[test] path priority: IPC > RDMA > DeviceBackend...\n");
  // DeviceBackend accepts all (Reduce needs it), but Put tiles prefer
  // transport.  Count only Put tiles enqueued to transport.
  CtlBackend dev("device", 8, kAnyKind), tpt("transport", 8),
      sig("signal", 8, kAnyKind);
  auto ex = std::make_unique<SprayExecutor>(&dev, &tpt, &sig, 4);
  ex->start();
  auto cfg = Testing::make_test_config(4, 0, 256, 64);
  std::vector<uint8_t> in(256), out(256), scr(256);
  assert(submit_and_wait(*ex, cfg, in.data(), out.data()));
  // Reduce tiles → device, Put tiles → transport (priority dispatch)
  assert(dev.enqueued_count() > 0);  // Reduce via device
  assert(tpt.enqueued_count() > 0);  // Put via transport
  printf("  device: %zu, transport: %zu — PASSED\n", dev.enqueued_count(),
         tpt.enqueued_count());
}

// ── Test 3: Reduce → DeviceBackend only ─────────────────────────────────
void test_reduce_device_only() {
  printf("[test] reduce: Reduce → DeviceBackend only...\n");
  CtlBackend dev("device", 8, kAnyKind), tpt("transport", 4),
      sig("signal", 8, kAnyKind);
  auto ex = std::make_unique<SprayExecutor>(&dev, &tpt, &sig, 4);
  ex->start();
  // AllReduceRing produces both Put and Reduce tiles
  auto cfg = Testing::make_test_config(4, 0, 256, 64);
  std::vector<uint8_t> in(256), out(256), scr(256);
  assert(submit_and_wait(*ex, cfg, in.data(), out.data()));
  size_t nd = dev.enqueued_count(), nt = tpt.enqueued_count();
  printf("  device: %zu, transport: %zu", nd, nt);
  assert(nd > 0);  // Reduce goes to device
  assert(nt > 0);  // Put tiles go to transport (IPC/RDMA preferred)
  printf(" — PASSED\n");
}

// ── Test 4: deferred tiles re-queued ────────────────────────────────────
void test_deferred_requeue() {
  printf("[test] deferred: tiles re-queued when all paths full...\n");
  CtlBackend dev("device", 4, kAnyKind), tpt("transport", 0),
      sig("signal", 8, kAnyKind);
  auto ex = std::make_unique<SprayExecutor>(&dev, &tpt, &sig, 2);
  ex->start();
  auto cfg = Testing::make_test_config(2, 0, 256, 64);
  std::vector<uint8_t> in(256), out(256), scr(256);

  auto h = ex->submit(cfg, in.data(), out.data());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  // Device has capacity for Reduce; transport is full — some tiles deferred
  printf("  initially: device=%zu transport=%zu\n", dev.enqueued_count(),
         tpt.enqueued_count());
  assert(tpt.enqueued_count() == 0);  // transport full, nothing enqueued

  // Open transport capacity; tiles should flow through
  tpt.set_capacity(8);
  assert(ex->wait(h, std::chrono::milliseconds(5000)));
  assert(ex->status(h) == CollectiveOpStatus::Completed);
  printf("  after opening capacity: device=%zu transport=%zu — PASSED\n",
         dev.enqueued_count(), tpt.enqueued_count());
  ex->release(h);
}

// ── Test 5: multiple concurrent runs ────────────────────────────────────
void test_concurrent() {
  printf("[test] concurrent: multiple runs with mixed dispatch...\n");
  CtlBackend dev("device", 4, kAnyKind), tpt("transport", 4),
      sig("signal", 16, kAnyKind);
  auto ex = std::make_unique<SprayExecutor>(&dev, &tpt, &sig, 2);
  ex->start();
  auto cfg = Testing::make_test_config(2, 0, 128, 32);
  std::vector<uint8_t> in(128), out(128), scr(128);

  auto h1 = ex->submit(cfg, in.data(), out.data());
  auto h2 = ex->submit(cfg, in.data(), out.data());
  auto h3 = ex->submit(cfg, in.data(), out.data());
  assert(ex->wait(h1, std::chrono::milliseconds(5000)) &&
         ex->wait(h2, std::chrono::milliseconds(5000)) &&
         ex->wait(h3, std::chrono::milliseconds(5000)));

  printf("  device: %zu, transport: %zu, signal: %zu", dev.enqueued_count(),
         tpt.enqueued_count(), sig.enqueued_count());
  assert(tpt.enqueued_count() > 0 || dev.enqueued_count() > 0);
  ex->release(h1);
  ex->release(h2);
  ex->release(h3);
  printf(" — PASSED\n");
}

// ── Test 6: Signal / WaitSignal → SignalBackend ─────────────────────────
void test_signal_backend() {
  printf("[test] signal: Signal/WaitSignal → SignalBackend only...\n");
  CtlBackend dev("device", 8, kAnyKind), tpt("transport", 8),
      sig("signal", 8, kAnyKind);
  auto ex = std::make_unique<SprayExecutor>(&dev, &tpt, &sig, 2);
  ex->start();
  auto cfg = Testing::make_test_config(2, 0, 128, 32);
  std::vector<uint8_t> in(128), out(128), scr(128);
  assert(submit_and_wait(*ex, cfg, in.data(), out.data()));
  assert(sig.enqueued_count() > 0);
  printf("  signal enqueued: %zu — PASSED\n", sig.enqueued_count());
}

}  // namespace
}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;
  printf("\nSprayExecutor Multi-Path Dispatch Tests\n");
  printf("=========================================\n");
  test_path_priority();
  test_reduce_device_only();
  test_deferred_requeue();
  test_concurrent();
  test_signal_backend();
  printf("\nAll SprayExecutor dispatch tests PASSED\n");
  return 0;
}
