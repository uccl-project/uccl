#include "rdma/cc_send_tracker.h"
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

static void test_out_of_order_completions() {
  CcSendTracker tracker;
  auto small = tracker.record(4096);
  auto large = tracker.record(128 * 1024);
  auto tail = tracker.record(17);
  assert(tracker.inflight_bytes() == 4096 + 128 * 1024 + 17);
  assert(tracker.complete(large) == 128 * 1024);
  assert(tracker.inflight_bytes() == 4096 + 17);
  assert(
      !tracker.complete(large));  // Duplicate CQE cannot return credits twice.
  assert(!tracker.complete(12345));
  assert(tracker.complete(tail) == 17);
  assert(tracker.complete(small) == 4096);
  assert(tracker.inflight_bytes() == 0);
}

static void test_rejected_send_and_zero_length() {
  CcSendTracker tracker;
  auto pending = tracker.record(128 * 1024);
  auto rejected = tracker.record(4096);
  assert(tracker.complete(rejected) == 4096);  // Roll back a rejected WR.
  assert(tracker.inflight_bytes() == 128 * 1024);
  assert(!tracker.complete(rejected));
  auto empty = tracker.record(0);
  auto bytes = tracker.complete(empty);
  assert(bytes.has_value() && *bytes == 0);
  assert(tracker.complete(pending) == 128 * 1024);
  assert(tracker.inflight_bytes() == 0);
}

static void test_id_wrap() {
  CcSendTracker tracker(std::numeric_limits<uint32_t>::max() - 1);
  auto a = tracker.record(1);
  auto b = tracker.record(2);
  auto c = tracker.record(3);
  assert(a == std::numeric_limits<uint32_t>::max() - 1);
  assert(b == std::numeric_limits<uint32_t>::max());
  assert(c == 0);
  assert(tracker.complete(c) == 3);
  assert(tracker.complete(a) == 1);
  assert(tracker.complete(b) == 2);
  assert(tracker.inflight_bytes() == 0);
}

static void test_concurrent_posts_and_completions() {
  CcSendTracker tracker;
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::pair<uint32_t, size_t>> completions;
  constexpr size_t kProducers = 4;
  constexpr size_t kSends = 10000;
  std::thread poller([&] {
    for (size_t i = 0; i < kProducers * kSends; ++i) {
      std::unique_lock<std::mutex> lock(mutex);
      ready.wait(lock, [&] { return !completions.empty(); });
      auto entry = completions.back();
      completions.pop_back();
      lock.unlock();
      assert(tracker.complete(entry.first) == entry.second);
    }
  });
  std::vector<std::thread> producers;
  for (size_t p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p] {
      for (size_t i = 0; i < kSends; ++i) {
        size_t bytes = 1 + ((p * kSends + i) % (128 * 1024));
        uint32_t id = tracker.record(bytes);
        {
          std::lock_guard<std::mutex> lock(mutex);
          completions.emplace_back(id, bytes);
        }
        ready.notify_one();
        // The completion may run before this simulated post returns.
        if (i % 16 == 0) std::this_thread::yield();
      }
    });
  }
  for (auto& producer : producers) producer.join();
  poller.join();
  assert(tracker.inflight_bytes() == 0);
}

int main() {
  test_out_of_order_completions();
  test_rejected_send_and_zero_length();
  test_id_wrap();
  test_concurrent_posts_and_completions();
  std::cout << "CC send accounting tests passed\n";
}
