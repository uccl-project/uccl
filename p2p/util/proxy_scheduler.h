#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uccl {

// A completion may cover multiple IOVs (one signaled tail per RDMA QP).
// Credits are counted in IOVs, never in completion records.
struct ProxyWait {
  int64_t wr_id;
  size_t iov_count;
};

enum class ProxyPostStatus { Posted, Retry, Failed };

struct ProxyPostResult {
  ProxyPostStatus status;
  size_t posted;
  std::vector<ProxyWait> waits;
};

// Failed is a terminal completion: the covered buffers are no longer in use.
// A transport error that does not establish that must NOT retire its wait.
enum class ProxyCompletion { Pending, Complete, Failed };

struct ProxySchedulerOptions {
  size_t max_inflight = 256;
  size_t per_request_inflight = 64;
  size_t quantum = 16;
  size_t max_requests = 32;
};

// Single-owner cooperative scheduling. Task contains the transport-specific
// request and its lifetime guards; this class owns scheduling and accounting.
//
// Backend contract:
//   drive()                         progress the underlying transport
//   post(task, offset, budget)      bounded, nonblocking submission
//   poll(task, wait)                nonblocking completion query
//   conflicts(older, newer)         preserve a dependency in admission order
//   run_exclusive(task)             existing blocking fallback, after draining
//   finish(task, success)           publish completion exactly once
//
// post() reports EVERY accepted IOV, even on failure, via waits whose counts
// sum to posted. Retry must not hide accepted work. Callbacks must not reenter
// the scheduler; the production raw transport keeps its fail-stop policy for
// unrecoverable partial submissions rather than falsely releasing buffers.
template <class Task>
class ProxyScheduler {
 public:
  explicit ProxyScheduler(ProxySchedulerOptions options = {})
      : options_(options) {
    if (!options_.max_inflight || !options_.per_request_inflight ||
        !options_.quantum || !options_.max_requests) {
      throw std::invalid_argument("proxy scheduler limits must be positive");
    }
    options_.per_request_inflight =
        std::min(options_.per_request_inflight, options_.max_inflight);
  }

  bool enqueue(Task task, size_t num_iovs, bool exclusive = false) {
    if (!accepting_ || requests_.size() >= options_.max_requests) return false;
    requests_.push_back(
        State{std::move(task), next_id_++, num_iovs, exclusive});
    return true;
  }

  bool empty() const { return requests_.empty(); }
  size_t size() const { return requests_.size(); }
  size_t inflight() const { return inflight_; }
  bool accepting() const { return accepting_; }

  // Stop taking ownership of new work; drain all work already accepted,
  // including unsent IOVs. Never abandon live DMA on a stop request.
  void stop_admission() { accepting_ = false; }

  template <class Backend>
  void progress(Backend& backend) {
    if (requests_.empty()) return;
    backend.drive();

    for (auto& state : requests_) {
      size_t retained = 0;
      for (size_t i = 0; i < state.waits.size(); ++i) {
        ProxyWait const wait = state.waits[i];
        auto const result = backend.poll(state.task, wait);
        if (result == ProxyCompletion::Pending) {
          state.waits[retained++] = wait;
          continue;
        }
        if (result == ProxyCompletion::Failed) state.failed = true;
        state.inflight -= wait.iov_count;
        inflight_ -= wait.iov_count;
      }
      state.waits.resize(retained);
    }
    finish_drained(backend);
    if (requests_.empty()) return;

    // Keep admission order in the deque for dependency checks. Rotate only
    // the service cursor, and only after an actual submission. Rotating on
    // empty CQ polls can starve a lane when completions align with polling.
    size_t start = 0;
    while (start < requests_.size() && requests_[start].id < service_id_)
      ++start;
    if (start == requests_.size()) start = 0;
    size_t const count = requests_.size();
    for (size_t visit = 0; visit < count; ++visit) {
      size_t const index = (start + visit) % count;
      auto& state = requests_[index];
      if (state.failed || state.exclusive_finished) continue;
      if (blocked(index, backend)) continue;

      if (state.exclusive) {
        // An exclusive request blocks every successor. All predecessors
        // must have been finished and removed before executing it.
        if (index != 0 || inflight_ != 0) continue;
        state.failed = !backend.run_exclusive(state.task);
        state.exclusive_finished = true;
        continue;
      }
      if (state.next == state.total ||
          state.inflight >= options_.per_request_inflight ||
          inflight_ >= options_.max_inflight)
        continue;
      size_t const budget =
          std::min({options_.quantum, state.total - state.next,
                    options_.per_request_inflight - state.inflight,
                    options_.max_inflight - inflight_});
      auto result = backend.post(state.task, state.next, budget);

      size_t covered = 0;
      for (auto const& wait : result.waits) {
        if (!wait.iov_count || covered > budget ||
            wait.iov_count > budget - covered)
          throw std::logic_error("invalid proxy completion credit count");
        covered += wait.iov_count;
      }
      if (result.posted > budget || covered != result.posted)
        throw std::logic_error("proxy backend lost submitted IOVs");

      state.next += result.posted;
      state.inflight += result.posted;
      inflight_ += result.posted;
      state.waits.insert(state.waits.end(), result.waits.begin(),
                         result.waits.end());
      if (result.status == ProxyPostStatus::Failed) state.failed = true;
      if (result.posted) service_id_ = state.id + 1;
    }
    finish_drained(backend);
  }

 private:
  struct State {
    Task task;
    uint64_t id;
    size_t total;
    bool exclusive;
    size_t next = 0;
    size_t inflight = 0;
    bool failed = false;
    bool exclusive_finished = false;
    std::vector<ProxyWait> waits;
  };

  template <class Backend>
  bool blocked(size_t index, Backend& backend) const {
    for (size_t earlier = 0; earlier < index; ++earlier) {
      auto const& older = requests_[earlier];
      auto const& newer = requests_[index];
      if (older.exclusive || newer.exclusive ||
          backend.conflicts(older.task, newer.task))
        return true;
    }
    return false;
  }

  template <class Backend>
  void finish_drained(Backend& backend) {
    for (size_t index = 0; index < requests_.size();) {
      auto it = requests_.begin() + index;
      bool const finished =
          it->exclusive
              ? it->exclusive_finished
              : (it->failed || it->next == it->total) && it->inflight == 0;
      // Even a zero-IOV request must not publish completion ahead of an
      // earlier request on its connection, or across an exclusive barrier.
      if (!finished || blocked(index, backend)) {
        ++index;
        continue;
      }
      backend.finish(it->task, !it->failed);
      requests_.erase(it);
    }
  }

  ProxySchedulerOptions options_;
  std::deque<State> requests_;
  size_t inflight_ = 0;
  uint64_t next_id_ = 0;
  uint64_t service_id_ = 0;
  bool accepting_ = true;
};

}  // namespace uccl
