#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <vector>

#ifndef COPY_RING_CAP
#define COPY_RING_CAP 4096
#endif

struct CopyTask {
  uint64_t wr_id;
  int dst_dev;
  void* src_ptr;
  void* dst_ptr;
  size_t bytes;
};

struct CopyRing {
  CopyTask buf[COPY_RING_CAP];
  std::atomic<uint32_t> head{0};
  std::atomic<uint32_t> tail{0};
  std::atomic<uint32_t> emplace_count{0};
  std::atomic<uint32_t> pop_count{0};

  /** push: returns false when the ring is full */
  bool emplace(CopyTask const& t) {
    emplace_count.fetch_add(1, std::memory_order_relaxed);
    uint32_t h = head.load(std::memory_order_relaxed);
    uint32_t n = (h + 1) & (COPY_RING_CAP - 1);
    if (n == tail.load(std::memory_order_acquire)) return false;  // full
    buf[h] = t;
    head.store(n, std::memory_order_release);
    return true;
  }

  bool emplace(std::vector<CopyTask> const& tasks) {
    if (tasks.empty()) return true;

    emplace_count.fetch_add(static_cast<uint32_t>(tasks.size()),
                            std::memory_order_relaxed);

    uint32_t h = head.load(std::memory_order_relaxed);
    uint32_t t = tail.load(std::memory_order_acquire);
    uint32_t cap = COPY_RING_CAP;

    // Compute free space in the ring
    uint32_t free_slots = (t + cap - h - 1) & (cap - 1);
    if (tasks.size() > free_slots) return false;

    // Insert tasks with wraparound
    uint32_t idx = h;
    for (CopyTask const& task : tasks) {
      buf[idx] = task;
      idx = (idx + 1) & (cap - 1);
    }

    head.store(idx, std::memory_order_release);
    return true;
  }

  /** pop: returns nullptr when the ring is empty */
  CopyTask* pop() {
    pop_count.fetch_add(1, std::memory_order_relaxed);
    uint32_t t = tail.load(std::memory_order_relaxed);
    if (t == head.load(std::memory_order_acquire)) return nullptr;  // empty
    CopyTask* ret = &buf[t];
    tail.store((t + 1) & (COPY_RING_CAP - 1), std::memory_order_release);
    return ret;
  }
};
