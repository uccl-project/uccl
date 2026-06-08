#include "backend_test_utils.h"
#include "test_utils.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

using TestUtil::throws;
using TestUtil::require;

// ── Mock Backend 正确性测试 ───────────────────────────────────────────

void test_mock_backend_submit_drain_single() {
  printf("[test] mock backend submit + drain single...\n");

  Testing::MockBackend backend(0);  // polls_before_ready=0 → immediate
  void* dummy_ptr = nullptr;
  Op dummy_op;
  dummy_op.kind = OpKind::Send;
  dummy_op.dst_peer = 1;
  dummy_op.bytes = 256;

  BackendToken t0 = backend.submit(dummy_op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr);
  assert(t0.value != 0);

  BackendToken out[4];
  size_t n = backend.drain(out, 4);
  assert(n == 1);
  assert(out[0].value == t0.value);
  assert(!out[0].failed);

  n = backend.drain(out, 4);
  assert(n == 0);
}

void test_mock_backend_submit_drain_many() {
  printf("[test] mock backend submit + drain many...\n");

  Testing::MockBackend backend(5);
  void* dummy_ptr = nullptr;
  Op op;
  op.kind = OpKind::Send;
  op.dst_peer = 1;
  op.bytes = 256;

  constexpr int kN = 100;
  uint64_t tokens[kN];
  for (int i = 0; i < kN; ++i)
    tokens[i] = backend.submit(op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr).value;

  BackendToken out[200];
  size_t total_drained = 0;
  // First 4 drains → should harvest 0 (need 5 decrements)
  for (int cycle = 0; cycle < 5; ++cycle) {
    size_t n = backend.drain(out, 200);
    assert(n == (cycle == 4 ? kN : 0));
    total_drained += n;
  }
  assert(total_drained == kN);

  // Verify no duplicates
  bool seen[kN] = {false};
  for (size_t i = 0; i < total_drained; ++i) {
    int idx = -1;
    for (int j = 0; j < kN; ++j)
      if (tokens[j] == out[i].value) { idx = j; break; }
    assert(idx >= 0 && !seen[idx]);
    seen[idx] = true;
  }
  // No more tokens
  assert(backend.drain(out, 200) == 0);
}

void test_mock_backend_max_count_clamping() {
  printf("[test] mock backend drain max_count clamping...\n");

  Testing::MockBackend backend(0);
  void* dummy_ptr = nullptr;
  Op op;
  op.kind = OpKind::Copy;
  op.bytes = 256;

  for (int i = 0; i < 10; ++i)
    backend.submit(op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr);

  // First drain: max_count=3 → only 3 returned
  BackendToken out[10];
  size_t n = backend.drain(out, 3);
  assert(n == 3);

  n = backend.drain(out, 3);
  assert(n == 3);

  n = backend.drain(out, 10);
  assert(n == 4);  // remaining

  n = backend.drain(out, 10);
  assert(n == 0);
}

void test_device_mock_submit_drain() {
  printf("[test] device mock backend submit + drain...\n");

  Testing::MockDeviceBackend backend(2);
  void* dummy_ptr = nullptr;
  Op op;
  op.kind = OpKind::Copy;
  op.bytes = 256;

  BackendToken t = backend.submit(op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr);
  assert(t.value != 0);

  // Reject unsupported op kind (use invalid enum value)
  Op bad_op = op;
  bad_op.kind = static_cast<OpKind>(99);
  assert(throws([&] { backend.submit(bad_op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr); }));

  // 2 polls needed → first drain returns 0, second harvests
  BackendToken out[4];
  assert(backend.drain(out, 4) == 0);
  assert(backend.drain(out, 4) == 1);
  assert(out[0].value == t.value);

  // No more
  assert(backend.drain(out, 4) == 0);
}

void test_throwing_backend_drain_empty() {
  printf("[test] throwing backend drain returns 0...\n");

  Testing::ThrowingBackend backend("test error");
  void* dummy_ptr = nullptr;
  Op op;

  assert(throws([&] { backend.submit(op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr); }));
  BackendToken out[4];
  assert(backend.drain(out, 4) == 0);
}

// ── Drain 性能微基准 ──────────────────────────────────────────────────

void bench_mock_drain_throughput() {
  printf("[bench] mock drain throughput...\n");

  Testing::MockBackend backend(0);
  void* dummy_ptr = nullptr;
  Op op;
  op.kind = OpKind::Send;
  op.dst_peer = 1;
  op.bytes = 256;

  constexpr int kTotal = 50000;
  constexpr int kBatch = 1024;

  BackendToken out[kBatch];

  // Submit phase
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kTotal; ++i)
    backend.submit(op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr);
  auto t1 = std::chrono::high_resolution_clock::now();
  double submit_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

  // Drain phase
  auto t2 = std::chrono::high_resolution_clock::now();
  size_t total = 0;
  while (total < kTotal) {
    size_t n = backend.drain(out, kBatch);
    total += n;
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  double drain_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

  printf("  submit: %d ops in %.1f ms  (%.0f ops/ms)\n",
         kTotal, submit_us / 1000.0, kTotal / (submit_us / 1000.0));
  printf("  drain:  %d ops in %.1f ms  (%.0f ops/ms, batch=%d)\n",
         kTotal, drain_us / 1000.0, kTotal / (drain_us / 1000.0), kBatch);
}

void bench_mock_drain_with_delay() {
  printf("[bench] mock drain with delay...\n");

  constexpr int kPolls = 3;
  Testing::MockBackend backend(kPolls);
  void* dummy_ptr = nullptr;
  Op op;
  op.kind = OpKind::Copy;
  op.bytes = 256;

  constexpr int kTotal = 10000;
  constexpr int kBatch = 1024;
  BackendToken out[kBatch];

  for (int i = 0; i < kTotal; ++i)
    backend.submit(op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr);

  auto t0 = std::chrono::high_resolution_clock::now();
  size_t total = 0;
  size_t cycles = 0;
  while (total < kTotal) {
    size_t n = backend.drain(out, kBatch);
    total += n;
    ++cycles;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  printf("  %d ops, polls_before_ready=%d, batch=%d\n",
         kTotal, kPolls, kBatch);
  printf("  total cycles=%zu  elapsed=%.1f ms  (%.0f ops/ms)\n",
         cycles, elapsed_ms, kTotal / elapsed_ms);
}

// ── Drain 并发压力测试 ────────────────────────────────────────────────

void test_drain_no_duplicates_or_loss() {
  printf("[test] drain correctness — no duplicates or loss...\n");

  Testing::MockBackend backend(3);
  void* dummy_ptr = nullptr;
  Op op;
  op.kind = OpKind::Send;
  op.dst_peer = 1;
  op.bytes = 256;

  constexpr int kN = 2000;
  std::vector<uint64_t> submitted;
  submitted.reserve(kN);
  for (int i = 0; i < kN; ++i)
    submitted.push_back(backend.submit(op, OpBindings{}, dummy_ptr, dummy_ptr, dummy_ptr).value);

  // Collect all drained tokens across many drain calls
  std::vector<uint64_t> drained;
  BackendToken out[kN];
  while (drained.size() < kN) {
    size_t n = backend.drain(out, kN);
    for (size_t i = 0; i < n; ++i) {
      assert(!out[i].failed);
      drained.push_back(out[i].value);
    }
  }

  // Must be exactly kN unique tokens matching submitted set
  std::sort(submitted.begin(), submitted.end());
  std::sort(drained.begin(), drained.end());
  assert(submitted == drained);

  // Extra drain → empty
  assert(backend.drain(out, 1) == 0);
}

// ── BackendToken 值语义验证 ────────────────────────────────────────────

void test_backend_token_defaults() {
  printf("[test] BackendToken defaults...\n");
  BackendToken t{};
  assert(t.value == 0);
  assert(!t.failed);
}

}  // namespace

}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;

  printf("=== Backend Component Tests ===\n\n");

  test_backend_token_defaults();
  test_mock_backend_submit_drain_single();
  test_mock_backend_submit_drain_many();
  test_mock_backend_max_count_clamping();
  test_device_mock_submit_drain();
  test_throwing_backend_drain_empty();
  test_drain_no_duplicates_or_loss();

  printf("\n=== Backend Benchmarks ===\n\n");
  bench_mock_drain_throughput();
  bench_mock_drain_with_delay();

  printf("\n=== Backend tests PASSED ===\n");
  return 0;
}
