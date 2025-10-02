#pragma once
#include <atomic>
#include <cstdint>

#ifndef UCCL_MAX_LOCAL_RANKS
#define UCCL_MAX_LOCAL_RANKS 8
#endif

struct LocalBarrier {
  std::atomic<uint32_t> arrive_seq[UCCL_MAX_LOCAL_RANKS];
  std::atomic<uint16_t> release_seq[UCCL_MAX_LOCAL_RANKS];
  std::atomic<uint16_t> seq;
  std::atomic<uint64_t> full_mask;     // unchanged; still used for size/info
  std::atomic<uint64_t> arrived_mask;  // optional: keep only for debug prints
};