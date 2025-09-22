#pragma once

#include "eqds.h"
#include "swift.h"
#include "timely.h"
#include "timing_wheel.h"
#include "util/util.h"
#include <glog/logging.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace uccl {

/**
 * @brief Protocol Control Block.
 */
struct PCB {
  static constexpr std::size_t kSackBitmapBucketSize = sizeof(uint64_t) * 8;

  PCB() {}

  PCB(double link_bandwidth)
      : timely_cc(freq_ghz, link_bandwidth),
        swift_cc(freq_ghz, link_bandwidth) {}

  timely::TimelyCC timely_cc;

  swift::SwiftCC swift_cc;

  eqds::EQDSCC eqds_cc;

  // Next sequence number to be sent.
  UINT_CSN snd_nxt{0};
  // Oldest unacknowledged sequence number.
  UINT_CSN snd_una{0};
  // Next expected sequence number to be received.
  UINT_CSN rcv_nxt{0};

  // SACK bitmap at the receiver side.
  uint64_t sack_bitmap[kSackBitmapSize / kSackBitmapBucketSize]{};
  uint8_t sack_bitmap_count{0};
  // SACK bitmap at the sender side.
  uint64_t tx_sack_bitmap[kSackBitmapSize / kSackBitmapBucketSize]{};
  uint8_t tx_sack_bitmap_count{0};
  // The starting CSN of the copy of SACK bitmap.
  uint32_t tx_sack_bitmap_base{0};

  // Timestamp of the last received data.
  uint64_t t_remote_nic_rx{0};

  // Incremented when a bitmap is shifted left by 1.
  // Even if increment every one microsecond, it will take 584542 years to
  // overflow.
  uint64_t shift_count;
  uint16_t duplicate_acks{0};
  uint16_t rto_rexmits_consectutive{0};
  UINT_CSN snd_ooo_acks{0};

  // Stats
  uint32_t stats_fast_rexmits{0};
  uint32_t stats_rto_rexmits{0};
  uint32_t stats_accept_retr{0};
  uint32_t stats_accept_barrier{0};
  uint32_t stats_chunk_drop{0};
  uint32_t stats_barrier_drop{0};
  uint32_t stats_retr_chunk_drop{0};
  uint32_t stats_ooo{0};
  uint32_t stats_real_ooo{0};
  uint32_t stats_maxooo{0};

  UINT_CSN seqno() const { return snd_nxt; }
  UINT_CSN get_snd_nxt() {
    UINT_CSN seqno = snd_nxt;
    snd_nxt += 1;
    return seqno;
  }

  UINT_CSN ackno() const { return rcv_nxt; }
  UINT_CSN get_rcv_nxt() const { return rcv_nxt; }
  void advance_rcv_nxt(UINT_CSN n) { rcv_nxt += n; }
  void advance_rcv_nxt() { rcv_nxt += 1; }

  void sack_bitmap_shift_left_one() {
    constexpr size_t sack_bitmap_bucket_max_idx =
        kSackBitmapSize / kSackBitmapBucketSize - 1;

    for (size_t i = 0; i < sack_bitmap_bucket_max_idx; i++) {
      // Shift the current each bucket to the left by 1 and take the most
      // significant bit from the next bucket
      uint64_t& sack_bitmap_left_bucket = sack_bitmap[i];
      const uint64_t sack_bitmap_right_bucket = sack_bitmap[i + 1];

      sack_bitmap_left_bucket =
          (sack_bitmap_left_bucket >> 1) | (sack_bitmap_right_bucket << 63);
    }

    // Special handling for the right most bucket
    uint64_t& sack_bitmap_right_most_bucket =
        sack_bitmap[sack_bitmap_bucket_max_idx];
    sack_bitmap_right_most_bucket >>= 1;

    sack_bitmap_count--;
  }

  // Check if the bit at the given index is set.
  bool sack_bitmap_bit_is_set(const size_t index) const {
    const size_t sack_bitmap_bucket_idx = index / kSackBitmapBucketSize;
    const size_t sack_bitmap_idx_in_bucket = index % kSackBitmapBucketSize;
    return sack_bitmap[sack_bitmap_bucket_idx] &
           (1ULL << sack_bitmap_idx_in_bucket);
  }
  // Set the bit at the given index.
  void sack_bitmap_bit_set(const size_t index) {
    const size_t sack_bitmap_bucket_idx = index / kSackBitmapBucketSize;
    const size_t sack_bitmap_idx_in_bucket = index % kSackBitmapBucketSize;

    LOG_IF(FATAL, index >= kSackBitmapSize) << "Index out of bounds: " << index;

    sack_bitmap[sack_bitmap_bucket_idx] |= (1ULL << sack_bitmap_idx_in_bucket);

    sack_bitmap_count++;
  }
};

}  // namespace uccl
