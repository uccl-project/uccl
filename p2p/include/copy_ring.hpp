#pragma once
#include "common.hpp"
#include <infiniband/verbs.h>
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

template <class T>
struct alignas(64) PaddedAtomic {
  std::atomic<T> v;
  char pad[64 - sizeof(std::atomic<T>)];
};

struct CopyRing {
  CopyTask buf[COPY_RING_CAP];
  PaddedAtomic<uint32_t> head{0};
  PaddedAtomic<uint32_t> tail{0};
  PaddedAtomic<uint32_t> emplace_count{0};
  PaddedAtomic<uint32_t> pop_count{0};
  struct ibv_qp* ack_qp;
  ibv_mr* ack_mr = nullptr;
  uint64_t ack_buf[RECEIVER_BATCH_SIZE];

  bool emplace(CopyTask const& t) {
    uint32_t h = head.v.load(std::memory_order_relaxed);
    uint32_t n = (h + 1) & (COPY_RING_CAP - 1);
    if (n == tail.v.load(std::memory_order_acquire)) return false;
    buf[h] = t;
    head.v.store(n, std::memory_order_release);
    emplace_count.v.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  bool emplace(std::vector<CopyTask> const& tasks) {
    if (tasks.empty()) return true;

    uint32_t h = head.v.load(std::memory_order_relaxed);
    uint32_t t = tail.v.load(std::memory_order_acquire);
    uint32_t cap = COPY_RING_CAP;
    uint32_t free_slots = (t + cap - h - 1) & (cap - 1);
    if (tasks.size() > free_slots) {
      fprintf(stderr,
              "Not enough space in CopyRing: %zu tasks, %u free slots\n",
              tasks.size(), free_slots);
      std::abort();
    }

    uint32_t idx = h;
    for (CopyTask const& task : tasks) {
      buf[idx] = task;
      idx = (idx + 1) & (cap - 1);
    }

    head.v.store(idx, std::memory_order_release);
    emplace_count.v.fetch_add(static_cast<uint32_t>(tasks.size()),
                              std::memory_order_relaxed);
    return true;
  }

  CopyTask* pop() {
    uint32_t t = tail.v.load(std::memory_order_relaxed);
    if (t == head.v.load(std::memory_order_acquire)) return nullptr;  // empty
    CopyTask* ret = &buf[t];
    tail.v.store((t + 1) & (COPY_RING_CAP - 1), std::memory_order_release);
    pop_count.v.fetch_add(1, std::memory_order_relaxed);
    return ret;
  }

  size_t popN(CopyTask* tasks, size_t n) {
    uint32_t t = tail.v.load(std::memory_order_relaxed);
    uint32_t h = head.v.load(std::memory_order_acquire);

    if (t == h) return 0;

    size_t available = (h - t) & (COPY_RING_CAP - 1);
    size_t count = std::min(n, available);

    for (size_t i = 0; i < count; ++i) {
      tasks[i] = buf[t];
      t = (t + 1) & (COPY_RING_CAP - 1);  // Wraparound
    }
    pop_count.v.fetch_add(static_cast<uint32_t>(count),
                          std::memory_order_relaxed);
    tail.v.store(t, std::memory_order_release);
    return count;
  }
};
