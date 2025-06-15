#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>

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

struct alignas(64) CopyRingMPMC {
  struct Slot {
    std::atomic<uint64_t> seq;
    CopyTask task;
  };

  Slot buf[COPY_RING_CAP];

  alignas(64) std::atomic<uint64_t> head{0};
  alignas(64) std::atomic<uint64_t> tail{0};

  CopyRingMPMC() {
    for (uint64_t i = 0; i < COPY_RING_CAP; ++i)
      buf[i].seq.store(i, std::memory_order_relaxed);
  }

  bool emplace(CopyTask const& t) {
    uint64_t pos = head.load(std::memory_order_relaxed);

    for (;;) {
      Slot& s = buf[pos & (COPY_RING_CAP - 1)];
      uint64_t seq = s.seq.load(std::memory_order_acquire);

      intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
      if (diff == 0) {
        if (head.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
          s.task = t;
          s.seq.store(pos + 1, std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = head.load(std::memory_order_relaxed);
      }
    }
  }

  CopyTask* pop() {
    uint64_t pos = tail.load(std::memory_order_relaxed);

    for (;;) {
      Slot& s = buf[pos & (COPY_RING_CAP - 1)];
      uint64_t seq = s.seq.load(std::memory_order_acquire);

      intptr_t diff =
          static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
      if (diff == 0) {
        if (tail.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
          s.seq.store(pos + COPY_RING_CAP, std::memory_order_release);
          return &s.task;
        }
      } else if (diff < 0) {
        return nullptr;
      } else {
        pos = tail.load(std::memory_order_relaxed);
      }
    }
  }
};