/**
 * @file timer.h
 * @brief Helper functions for timers
 */

#pragma once

#include "glog/logging.h"
#include "transport_config.h"
#include <chrono>
#include <unordered_map>
#include <stdint.h>
#include <stdlib.h>

namespace uccl {

/// Return the TSC
static inline size_t rdtsc() {
  uint64_t rax;
  uint64_t rdx;
  asm volatile("rdtsc" : "=a"(rax), "=d"(rdx));
  return static_cast<size_t>((rdx << 32) | rax);
}

/// An alias for rdtsc() to distinguish calls on the critical path
static auto const& dpath_rdtsc = rdtsc;

static void nano_sleep(size_t ns, double freq_ghz) {
  size_t start = rdtsc();
  size_t end = start;
  size_t upp = static_cast<size_t>(freq_ghz * ns);
  while (end - start < upp) end = rdtsc();
}

/// Simple time that uses std::chrono
class ChronoTimer {
 public:
  ChronoTimer() { reset(); }
  void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }

  /// Return seconds elapsed since this timer was created or last reset
  double get_sec() const { return get_ns() / 1e9; }

  /// Return milliseconds elapsed since this timer was created or last reset
  double get_ms() const { return get_ns() / 1e6; }

  /// Return microseconds elapsed since this timer was created or last reset
  double get_us() const { return get_ns() / 1e3; }

  /// Return nanoseconds elapsed since this timer was created or last reset
  size_t get_ns() const {
    return static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start_time_)
            .count());
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

static double measure_rdtsc_freq() {
  ChronoTimer chrono_timer;
  const uint64_t rdtsc_start = rdtsc();

  // Do not change this loop! The hardcoded value below depends on this loop
  // and prevents it from being optimized out.
  uint64_t sum = 5;
  for (uint64_t i = 0; i < 1000000; i++) {
    sum += i + (sum + i) * (i % sum);
  }
  CHECK(sum == 13580802877818827968ull) << "Error in RDTSC freq measurement";

  const uint64_t rdtsc_cycles = rdtsc() - rdtsc_start;
  double const freq_ghz = rdtsc_cycles * 1.0 / chrono_timer.get_ns();
  CHECK(freq_ghz >= 0.5 && freq_ghz <= 5.0) << "Invalid RDTSC frequency";

  return freq_ghz;
}

static double freq_ghz = measure_rdtsc_freq();

/// Convert cycles measured by rdtsc with frequence \p freq_ghz to seconds
static double to_sec(size_t cycles, double freq_ghz) {
  return (cycles / (freq_ghz * 1000000000));
}

/// Convert cycles measured by rdtsc with frequence \p freq_ghz to msec
static double to_msec(size_t cycles, double freq_ghz) {
  return (cycles / (freq_ghz * 1000000));
}

/// Convert cycles measured by rdtsc with frequence \p freq_ghz to usec
static double to_usec(size_t cycles, double freq_ghz) {
  return (cycles / (freq_ghz * 1000));
}

static size_t ms_to_cycles(double ms, double freq_ghz) {
  return static_cast<size_t>(ms * 1000 * 1000 * freq_ghz);
}

static size_t us_to_cycles(double us, double freq_ghz) {
  return static_cast<size_t>(us * 1000 * freq_ghz);
}

static size_t ns_to_cycles(double ns, double freq_ghz) {
  return static_cast<size_t>(ns * freq_ghz);
}

/// Convert cycles measured by rdtsc with frequence \p freq_ghz to nsec
static double to_nsec(size_t cycles, double freq_ghz) {
  return (cycles / freq_ghz);
}

/// Simple time that uses RDTSC
class TscTimer {
 public:
  size_t start_tsc_ = 0;
  size_t tsc_sum_ = 0;
  size_t num_calls_ = 0;

  inline void start() { start_tsc_ = rdtsc(); }
  inline void stop() {
    tsc_sum_ += (rdtsc() - start_tsc_);
    num_calls_++;
  }

  void reset() {
    start_tsc_ = 0;
    tsc_sum_ = 0;
    num_calls_ = 0;
  }

  size_t avg_cycles() const { return tsc_sum_ / num_calls_; }
  double avg_sec(double freq_ghz) const {
    return to_sec(avg_cycles(), freq_ghz);
  }

  double avg_usec(double freq_ghz) const {
    return to_usec(avg_cycles(), freq_ghz);
  }

  double avg_nsec(double freq_ghz) const {
    return to_nsec(avg_cycles(), freq_ghz);
  }
};

struct TimerData {
  void* rdma_ctx;
  void* qpw;
};

class TimerManager {
 public:
  using CycleCount = uint64_t;

  explicit TimerManager(double timeout_us = 1000)
      : timeout_(us_to_cycles(timeout_us, freq_ghz)) {
    heap_.reserve(kMaxPath * 64);
    qpw_map_.reserve(kMaxPath * 64);
  }

  void arm_timer(struct TimerData data) {
    if (auto it = qpw_map_.find(data.qpw); it != qpw_map_.end()) {
      // Already being armed, do nothing.
    } else {
      // Add new timer.
      const CycleCount new_expire = rdtsc() + timeout_;
      heap_.push_back({new_expire, data});
      qpw_map_[data.qpw] = heap_.size() - 1;
      heapify_up(heap_.size() - 1);
    }
  }

  void arm_timer(struct TimerData data, double timeout_us) {
    auto timeout = us_to_cycles(timeout_us, freq_ghz);
    if (auto it = qpw_map_.find(data.qpw); it != qpw_map_.end()) {
      // Already being armed, do nothing.
    } else {
      // Add new timer.
      const CycleCount new_expire = rdtsc() + timeout;
      heap_.push_back({new_expire, data});
      qpw_map_[data.qpw] = heap_.size() - 1;
      heapify_up(heap_.size() - 1);
    }
  }

  void rearm_timer(struct TimerData data) {
    const CycleCount new_expire = rdtsc() + timeout_;
    if (auto it = qpw_map_.find(data.qpw); it != qpw_map_.end()) {
      // Update existing timer.
      const size_t index = it->second;
      heap_[index].expire = new_expire;
      adjust_heap_node(index);
    } else {
      // Add new timer.
      heap_.push_back({new_expire, data});
      qpw_map_[data.qpw] = heap_.size() - 1;
      heapify_up(heap_.size() - 1);
    }
  }

  void rearm_timer(struct TimerData data, double timeout_us) {
    auto timeout = us_to_cycles(timeout_us, freq_ghz);
    const CycleCount new_expire = rdtsc() + timeout;
    if (auto it = qpw_map_.find(data.qpw); it != qpw_map_.end()) {
      // Update existing timer.
      const size_t index = it->second;
      heap_[index].expire = new_expire;
      adjust_heap_node(index);
    } else {
      // Add new timer.
      heap_.push_back({new_expire, data});
      qpw_map_[data.qpw] = heap_.size() - 1;
      heapify_up(heap_.size() - 1);
    }
  }

  std::vector<struct TimerData> check_expired() {
    std::vector<struct TimerData> expired;
    const CycleCount now = rdtsc();

    while (!heap_.empty() && heap_[0].expire <= now) {
      expired.push_back(heap_[0].data);
      qpw_map_.erase(heap_[0].data.qpw);

      if (heap_.size() > 1) {
        heap_[0] = heap_.back();
        qpw_map_[heap_[0].data.qpw] = 0;
      }
      heap_.pop_back();

      if (!heap_.empty()) {
        heapify_down(0);
      }
    }
    return expired;
  }

  void disarm_timer(struct TimerData data) {
    auto it = qpw_map_.find(data.qpw);
    if (it == qpw_map_.end()) return;

    const size_t index = it->second;
    qpw_map_.erase(data.qpw);

    if (index == heap_.size() - 1) {
      heap_.pop_back();
    } else {
      heap_[index] = heap_.back();
      qpw_map_[heap_[index].data.qpw] = index;
      heap_.pop_back();

      adjust_heap_node(index);
    }
  }

 private:
  struct TimerNode {
    CycleCount expire;
    struct TimerData data;

    bool operator<(TimerNode const& rhs) const { return expire < rhs.expire; }
  };

  std::vector<TimerNode> heap_;
  std::unordered_map<void*, size_t> qpw_map_;
  const CycleCount timeout_;

  void heapify_up(size_t index) {
    while (index > 0) {
      const size_t parent = (index - 1) / 2;
      if (heap_[index] < heap_[parent]) {
        swap_nodes(index, parent);
        index = parent;
      } else {
        break;
      }
    }
  }

  void heapify_down(size_t index) {
    const size_t size = heap_.size();
    while (true) {
      size_t smallest = index;
      const size_t left = 2 * index + 1;
      const size_t right = 2 * index + 2;

      if (left < size && heap_[left] < heap_[smallest]) {
        smallest = left;
      }
      if (right < size && heap_[right] < heap_[smallest]) {
        smallest = right;
      }

      if (smallest != index) {
        swap_nodes(index, smallest);
        index = smallest;
      } else {
        break;
      }
    }
  }

  void adjust_heap_node(size_t index) {
    if (index > 0 && heap_[index] < heap_[(index - 1) / 2]) {
      heapify_up(index);
    } else {
      heapify_down(index);
    }
  }

  void swap_nodes(size_t a, size_t b) {
    std::swap(heap_[a], heap_[b]);
    qpw_map_[heap_[a].data.qpw] = a;
    qpw_map_[heap_[b].data.qpw] = b;
  }
};

}  // namespace uccl
