#include "util/proxy_scheduler.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition)                                                 \
  do {                                                                   \
    if (!(condition))                                                    \
      throw std::runtime_error(std::string(__func__) + ": " #condition); \
  } while (false)

struct Task {
  int id;
  int lane;
  size_t count;
  int release = 0;
  int latency = 2;
  size_t group_size = 1;
  int retries = 0;
  size_t fail_after = static_cast<size_t>(-1);
  bool fail_completion = false;
  bool overlaps = false;
  size_t posted = 0;
  size_t outstanding = 0;
  int first_post = -1;
  int finished = -1;
  bool success = false;
};
using Handle = std::shared_ptr<Task>;
using Scheduler = uccl::ProxyScheduler<Handle>;

// The double controls only the transport boundary. Selection, offsets,
// credits, completion retirement, barriers and finish decisions run in the
// same scheduler header used by Endpoint's production proxy threads.
struct Transport {
  struct Pending {
    Handle task;
    size_t count;
    int due;
  };
  int round = 0;
  int64_t next_wr = 1;
  size_t peak_outstanding = 0;
  std::map<int64_t, Pending> pending;
  std::vector<int> finish_order;
  std::vector<int> post_order;
  std::vector<int> exclusive_order;

  void drive() { ++round; }

  uccl::ProxyPostResult post(Handle& task, size_t offset, size_t budget) {
    CHECK(offset == task->posted);
    CHECK(task->finished == -1);
    CHECK(budget > 0);
    if (task->retries > 0) {
      --task->retries;
      return {uccl::ProxyPostStatus::Retry, 0, {}};
    }
    if (offset >= task->fail_after)
      return {uccl::ProxyPostStatus::Failed, 0, {}};
    size_t const count =
        std::min({budget, task->count - offset, task->fail_after - offset});
    uccl::ProxyPostResult result{uccl::ProxyPostStatus::Posted, count, {}};
    for (size_t left = count; left > 0;) {
      size_t const n = std::min(left, task->group_size);
      int64_t const wr = next_wr++;
      result.waits.push_back({wr, n});
      pending.emplace(
          wr, Pending{task, n, std::max(round + task->latency, task->release)});
      left -= n;
    }
    if (task->first_post == -1) task->first_post = round;
    task->posted += count;
    task->outstanding += count;
    post_order.push_back(task->id);
    if (task->posted == task->fail_after)
      result.status = uccl::ProxyPostStatus::Failed;
    size_t live = 0;
    for (auto const& entry : pending) live += entry.second.count;
    peak_outstanding = std::max(peak_outstanding, live);
    return result;
  }

  uccl::ProxyCompletion poll(Handle& task, uccl::ProxyWait const& wait) {
    auto it = pending.find(wait.wr_id);
    CHECK(it != pending.end());
    CHECK(it->second.task == task);
    CHECK(it->second.count == wait.iov_count);
    if (round < it->second.due) return uccl::ProxyCompletion::Pending;
    CHECK(task->outstanding >= wait.iov_count);
    task->outstanding -= wait.iov_count;
    pending.erase(it);
    return task->fail_completion ? uccl::ProxyCompletion::Failed
                                 : uccl::ProxyCompletion::Complete;
  }

  bool conflicts(Handle const& a, Handle const& b) const {
    return a->lane == b->lane || (a->overlaps && b->overlaps);
  }

  bool run_exclusive(Handle& task) {
    CHECK(pending.empty());
    exclusive_order.push_back(task->id);
    return task->fail_after != 0;
  }

  void finish(Handle& task, bool success) {
    CHECK(task->finished == -1);
    CHECK(task->outstanding == 0);
    task->finished = round;
    task->success = success;
    finish_order.push_back(task->id);
  }
};

static Handle make_task(int id, int lane, size_t count) {
  return std::make_shared<Task>(Task{id, lane, count});
}

static void drain(Scheduler& scheduler, Transport& transport, int limit = 500) {
  for (int i = 0; i < limit && !scheduler.empty(); ++i)
    scheduler.progress(transport);
  CHECK(scheduler.empty());
  CHECK(scheduler.inflight() == 0);
  CHECK(transport.pending.empty());
}

// Replacing the production round-robin with run-one-task-to-completion must
// fail this test. No elapsed-time threshold or simulated microsecond claim.
static void independent_small_request_passes_unfinished_large_request() {
  Scheduler scheduler({8, 4, 2, 8});
  Transport transport;
  auto large = make_task(1, 1, 12);
  large->release = 20;
  CHECK(scheduler.enqueue(large, large->count));
  scheduler.progress(transport);
  scheduler.progress(transport);
  auto small = make_task(2, 2, 1);
  CHECK(scheduler.enqueue(small, small->count));
  for (int i = 0; i < 5; ++i) scheduler.progress(transport);
  CHECK(small->first_post >= 0);
  CHECK(small->finished >= 0);
  CHECK(large->finished == -1);
  CHECK(large->outstanding > 0);
  drain(scheduler, transport);
  CHECK(large->success && small->success);
  CHECK(transport.peak_outstanding <= 8);
}

static void same_connection_and_conflicting_requests_keep_fifo() {
  for (bool same_lane : {false, true}) {
    Scheduler scheduler({8, 4, 2, 8});
    Transport transport;
    auto first = make_task(1, 1, 6);
    first->release = 15;
    auto second = make_task(2, same_lane ? 1 : 2, 1);
    first->overlaps = second->overlaps = !same_lane;
    CHECK(scheduler.enqueue(first, first->count));
    CHECK(scheduler.enqueue(second, second->count));
    for (int i = 0; i < 8; ++i) scheduler.progress(transport);
    CHECK(second->first_post == -1);
    drain(scheduler, transport);
    CHECK(second->first_post >= first->finished);
    CHECK(transport.finish_order == std::vector<int>({1, 2}));
  }
}

static void grouped_and_out_of_order_completions_return_iov_credits() {
  for (size_t group : {size_t{1}, size_t{3}}) {
    Scheduler scheduler({7, 4, 3, 8});
    Transport transport;
    auto slow = make_task(1, 1, 17);
    auto fast = make_task(2, 2, 5);
    slow->group_size = fast->group_size = group;
    slow->latency = 7;
    fast->latency = 1;
    CHECK(scheduler.enqueue(slow, slow->count));
    CHECK(scheduler.enqueue(fast, fast->count));
    for (int i = 0; i < 300 && !scheduler.empty(); ++i) {
      scheduler.progress(transport);
      CHECK(scheduler.inflight() <= 7);
      CHECK(slow->outstanding <= 4);
      CHECK(fast->outstanding <= 4);
      CHECK(scheduler.inflight() == slow->outstanding + fast->outstanding);
    }
    CHECK(scheduler.empty());
    CHECK(slow->posted == 17 && fast->posted == 5);
    CHECK(slow->success && fast->success);
    CHECK(fast->finished < slow->finished);
  }
}

static void retry_does_not_consume_credits_or_block_other_lanes() {
  Scheduler scheduler({4, 2, 2, 8});
  Transport transport;
  auto retrying = make_task(1, 1, 5);
  retrying->retries = 12;
  auto ready = make_task(2, 2, 1);
  CHECK(scheduler.enqueue(retrying, retrying->count));
  CHECK(scheduler.enqueue(ready, ready->count));
  for (int i = 0; i < 5; ++i) scheduler.progress(transport);
  CHECK(ready->success);
  CHECK(retrying->posted == 0);
  CHECK(scheduler.inflight() == 0);
  drain(scheduler, transport);
  CHECK(retrying->success);
}

static void submission_failure_drains_accepted_work_before_finish() {
  Scheduler scheduler({8, 4, 2, 8});
  Transport transport;
  auto failing = make_task(1, 1, 10);
  failing->fail_after = 3;  // Includes a partially accepted second batch.
  failing->release = 20;
  auto later = make_task(2, 1, 1);
  CHECK(scheduler.enqueue(failing, failing->count));
  CHECK(scheduler.enqueue(later, later->count));
  for (int i = 0; i < 8; ++i) scheduler.progress(transport);
  CHECK(failing->posted == 3);
  CHECK(failing->finished == -1);
  CHECK(later->first_post == -1);
  drain(scheduler, transport);
  CHECK(!failing->success && later->success);
  CHECK(failing->finished >= 20);
  CHECK(later->first_post >= failing->finished);
}

static void failed_completion_stops_new_posts_and_drains_other_waits() {
  Scheduler scheduler({8, 4, 2, 8});
  Transport transport;
  auto task = make_task(1, 1, 20);
  task->fail_completion = true;
  CHECK(scheduler.enqueue(task, task->count));
  drain(scheduler, transport);
  CHECK(!task->success);
  CHECK(task->posted < task->count);
  CHECK(task->outstanding == 0);
}

static void fallback_is_a_global_barrier() {
  Scheduler scheduler({8, 4, 2, 8});
  Transport transport;
  auto first = make_task(1, 1, 7);
  first->release = 12;
  auto fallback = make_task(2, 2, 1);
  auto last = make_task(3, 3, 1);
  CHECK(scheduler.enqueue(first, first->count));
  CHECK(scheduler.enqueue(fallback, fallback->count, true));
  CHECK(scheduler.enqueue(last, last->count));
  for (int i = 0; i < 5; ++i) scheduler.progress(transport);
  CHECK(transport.exclusive_order.empty());
  CHECK(last->first_post == -1);
  drain(scheduler, transport);
  CHECK(transport.exclusive_order == std::vector<int>({2}));
  CHECK(transport.finish_order == std::vector<int>({1, 2, 3}));
}

static void one_credit_is_fair_across_ready_lanes() {
  Scheduler scheduler({1, 1, 1, 8});
  Transport transport;
  auto large = make_task(1, 1, 30);
  auto small = make_task(2, 2, 1);
  large->latency = 3;
  CHECK(scheduler.enqueue(large, large->count));
  scheduler.progress(transport);
  CHECK(scheduler.enqueue(small, small->count));
  drain(scheduler, transport);
  CHECK(small->finished < large->finished);
  CHECK(transport.post_order.size() >= 2);
  CHECK(transport.post_order[0] == 1 && transport.post_order[1] == 2);
  CHECK(transport.peak_outstanding == 1);
}

static void stopping_admission_drains_owned_requests() {
  Scheduler scheduler({4, 2, 1, 2});
  Transport transport;
  auto first = make_task(1, 1, 8);
  auto second = make_task(2, 2, 3);
  auto rejected = make_task(3, 3, 1);
  CHECK(scheduler.enqueue(first, first->count));
  CHECK(scheduler.enqueue(second, second->count));
  CHECK(!scheduler.enqueue(rejected, rejected->count));
  scheduler.progress(transport);
  scheduler.stop_admission();
  CHECK(!scheduler.accepting());
  CHECK(!scheduler.enqueue(rejected, rejected->count));
  drain(scheduler, transport);
  CHECK(first->success && second->success);
  CHECK(rejected->first_post == -1 && rejected->finished == -1);
}

static void zero_iov_and_immediate_failure_finish_once() {
  Scheduler scheduler({4, 2, 1, 4});
  Transport transport;
  auto empty = make_task(1, 1, 0);
  auto failed = make_task(2, 2, 1);
  failed->fail_after = 0;
  CHECK(scheduler.enqueue(empty, 0));
  CHECK(scheduler.enqueue(failed, 1));
  drain(scheduler, transport);
  CHECK(empty->success && !failed->success);
  CHECK(empty->posted == 0 && failed->posted == 0);
  CHECK(transport.finish_order.size() == 2);
}

static void empty_request_preserves_dependencies() {
  for (bool barrier : {false, true}) {
    Scheduler scheduler({4, 2, 1, 4});
    Transport transport;
    auto first = make_task(1, 1, 3);
    first->release = 12;
    auto middle = make_task(2, 2, 1);
    auto empty = make_task(3, barrier ? 3 : 1, 0);
    CHECK(scheduler.enqueue(first, first->count));
    if (barrier) CHECK(scheduler.enqueue(middle, middle->count, true));
    CHECK(scheduler.enqueue(empty, 0));
    for (int i = 0; i < 5; ++i) scheduler.progress(transport);
    CHECK(empty->finished == -1);
    drain(scheduler, transport);
    CHECK(empty->finished >= first->finished);
    if (barrier) CHECK(empty->finished >= middle->finished);
    CHECK(empty->success && empty->posted == 0);
  }
}

int main() {
  using Test = void (*)();
  std::vector<std::pair<char const*, Test>> tests = {
      {"independent small request",
       independent_small_request_passes_unfinished_large_request},
      {"ordering and conflicts",
       same_connection_and_conflicting_requests_keep_fifo},
      {"grouped completions",
       grouped_and_out_of_order_completions_return_iov_credits},
      {"retry backpressure",
       retry_does_not_consume_credits_or_block_other_lanes},
      {"submission failure drain",
       submission_failure_drains_accepted_work_before_finish},
      {"completion failure drain",
       failed_completion_stops_new_posts_and_drains_other_waits},
      {"fallback barrier", fallback_is_a_global_barrier},
      {"one-credit fairness", one_credit_is_fair_across_ready_lanes},
      {"shutdown drain", stopping_admission_drains_owned_requests},
      {"empty and failed requests", zero_iov_and_immediate_failure_finish_once},
      {"empty request ordering", empty_request_preserves_dependencies},
  };
  size_t failures = 0;
  for (auto const& test : tests) {
    try {
      test.second();
      std::cout << "PASS: " << test.first << '\n';
    } catch (std::exception const& error) {
      ++failures;
      std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
    }
  }
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
