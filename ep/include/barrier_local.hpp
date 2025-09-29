#pragma once
#include <atomic>
#include <cstdint>

#ifndef UCCL_MAX_LOCAL_RANKS
#define UCCL_MAX_LOCAL_RANKS 64
#endif

struct alignas(64) LocalBarrier {
  std::atomic<uint16_t> seq{0};           // last barrier seq noticed
  std::atomic<uint64_t> arrived_mask{0};  // bits set by local ranks
  uint64_t full_mask{0};                  // (1ULL<<num_local)-1
  alignas(64) std::atomic<
      uint16_t> release_seq[UCCL_MAX_LOCAL_RANKS];  // per local_rank
  LocalBarrier() {
    for (auto& a : release_seq) a.store(0, std::memory_order_relaxed);
  }
};