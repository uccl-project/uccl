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

  bool emplace(CopyTask const& t) {
    uint32_t h = head.load(std::memory_order_relaxed);
    uint32_t n = h + 1;
    if (n == tail.load(std::memory_order_acquire)) return false;  // full
    buf[h & (COPY_RING_CAP - 1)] = t;
    head.store(n, std::memory_order_release);
    emplace_count.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  bool emplace(std::vector<CopyTask> const& tasks) {
    if (tasks.empty()) return true;

    uint32_t h = head.load(std::memory_order_relaxed);
    uint32_t t = tail.load(std::memory_order_acquire);
    uint32_t free_slots = (t + COPY_RING_CAP - h - 1) & (COPY_RING_CAP - 1);
    if (tasks.size() > free_slots) return false;

    uint32_t idx = h;
    for (CopyTask const& task : tasks) {
      buf[idx & (COPY_RING_CAP - 1)] = task;
      idx++;
    }

    head.store(idx, std::memory_order_release);
    emplace_count.fetch_add(static_cast<uint32_t>(tasks.size()),
                            std::memory_order_relaxed);
    return true;
  }

  CopyTask* pop() {
    uint32_t t = tail.load(std::memory_order_relaxed);
    if (t == head.load(std::memory_order_acquire)) return nullptr;
    CopyTask* ret = &buf[t & (COPY_RING_CAP - 1)];
    tail.store(t + 1, std::memory_order_release);
    pop_count.fetch_add(1, std::memory_order_relaxed);
    return ret;
  }

  size_t popN(std::vector<CopyTask>& tasks, size_t n) {
    tasks.clear();
    uint32_t t = tail.load(std::memory_order_relaxed);
    uint32_t h = head.load(std::memory_order_acquire);

    if (t == h) return 0;

    size_t available = (h - t) & (COPY_RING_CAP - 1);
    size_t count = std::min(n, available);

    for (size_t i = 0; i < count; ++i) {
      tasks.push_back(buf[t & (COPY_RING_CAP - 1)]);
      t = t + 1;
    }
    pop_count.fetch_add(static_cast<uint32_t>(count),
                        std::memory_order_relaxed);
    tail.store(t, std::memory_order_release);
    return count;
  }

  bool has_completed(uint32_t idx) {
    return tail.load(std::memory_order_acquire) >= idx;
  }

  uint32_t get_head() const { return head.load(std::memory_order_acquire); }

  uint32_t get_tail() const { return tail.load(std::memory_order_acquire); }
};
