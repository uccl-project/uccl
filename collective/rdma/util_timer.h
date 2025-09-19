/**
 * @file timer.h
 * @brief Helper functions for timers
 */

#pragma once

#include "glog/logging.h"
#include "transport_config.h"
#include "util/timer.h"
#include <chrono>
#include <unordered_map>
#include <stdint.h>
#include <stdlib.h>

namespace uccl {

struct TimerData {
  void* rdma_ctx;
  void* flow;
};

class TimerManager {
 public:
  using CycleCount = uint64_t;

  explicit TimerManager(double timeout_us = 1000)
      : timeout_(us_to_cycles(timeout_us, freq_ghz)) {
    heap_.reserve(ucclParamPORT_ENTROPY() * 64);
    flow_map_.reserve(ucclParamPORT_ENTROPY() * 64);
  }

  uint32_t size() const { return heap_.size(); }

  void arm_timer(struct TimerData data) {
    if (auto it = flow_map_.find(data.flow); it != flow_map_.end()) {
      // Already being armed, do nothing.
    } else {
      // Add new timer.
      const CycleCount new_expire = rdtsc() + timeout_;
      heap_.push_back({new_expire, data});
      flow_map_[data.flow] = heap_.size() - 1;
      heapify_up(heap_.size() - 1);
    }
  }

  void arm_timer(struct TimerData data, double timeout_us) {
    auto timeout = us_to_cycles(timeout_us, freq_ghz);
    if (auto it = flow_map_.find(data.flow); it != flow_map_.end()) {
      // Already being armed, do nothing.
    } else {
      // Add new timer.
      const CycleCount new_expire = rdtsc() + timeout;
      heap_.push_back({new_expire, data});
      flow_map_[data.flow] = heap_.size() - 1;
      heapify_up(heap_.size() - 1);
    }
  }

  void rearm_timer(struct TimerData data) {
    const CycleCount new_expire = rdtsc() + timeout_;
    if (auto it = flow_map_.find(data.flow); it != flow_map_.end()) {
      // Update existing timer.
      const size_t index = it->second;
      heap_[index].expire = new_expire;
      adjust_heap_node(index);
    } else {
      // Add new timer.
      heap_.push_back({new_expire, data});
      flow_map_[data.flow] = heap_.size() - 1;
      heapify_up(heap_.size() - 1);
    }
  }

  void rearm_timer(struct TimerData data, double timeout_us) {
    auto timeout = us_to_cycles(timeout_us, freq_ghz);
    const CycleCount new_expire = rdtsc() + timeout;
    if (auto it = flow_map_.find(data.flow); it != flow_map_.end()) {
      // Update existing timer.
      const size_t index = it->second;
      heap_[index].expire = new_expire;
      adjust_heap_node(index);
    } else {
      // Add new timer.
      heap_.push_back({new_expire, data});
      flow_map_[data.flow] = heap_.size() - 1;
      heapify_up(heap_.size() - 1);
    }
  }

  std::vector<struct TimerData> check_expired() {
    std::vector<struct TimerData> expired;
    const CycleCount now = rdtsc();

    while (!heap_.empty() && heap_[0].expire <= now) {
      expired.push_back(heap_[0].data);
      flow_map_.erase(heap_[0].data.flow);

      if (heap_.size() > 1) {
        heap_[0] = heap_.back();
        flow_map_[heap_[0].data.flow] = 0;
      }
      heap_.pop_back();

      if (!heap_.empty()) {
        heapify_down(0);
      }
    }
    return expired;
  }

  void disarm_timer(struct TimerData data) {
    auto it = flow_map_.find(data.flow);
    if (it == flow_map_.end()) return;

    const size_t index = it->second;
    flow_map_.erase(data.flow);

    if (index == heap_.size() - 1) {
      heap_.pop_back();
    } else {
      heap_[index] = heap_.back();
      flow_map_[heap_[index].data.flow] = index;
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
  std::unordered_map<void*, size_t> flow_map_;
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
    flow_map_[heap_[a].data.flow] = a;
    flow_map_[heap_[b].data.flow] = b;
  }
};

}  // namespace uccl
