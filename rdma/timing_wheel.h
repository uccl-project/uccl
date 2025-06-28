/**
 * @file timing_wheel.h
 * @brief Timing wheel implementation from Carousel [SIGCOMM 17]
 * Units: Microseconds or TSC for time, bytes/sec for throughput
 *
 * SSlots scheduled for transmission in a wheel slot are stored in a chain
 * of associative buckets. Deletes do not compact the chain, so some buckets in
 * a chain may be empty or partially full.
 */

#pragma once

#include "timely.h"
#include "transport_config.h"
#include "util/cb.h"
#include <queue>

namespace uccl {

#pragma once

#include "util_timer.h"

struct alignas(64) wr_ex {
  struct ibv_send_wr wr;
  struct ibv_sge sge;

  struct ucclRequest* ureq;
  uint32_t qpidx;
};
static_assert(sizeof(wr_ex) == 192, "wr_ex size mismatch");

/// Used for fast recording of wheel actions for debugging
struct wheel_record_t {
  size_t record_tsc_;  ///< Timestamp at which this record was created
  bool insert_;        ///< Is this a record for a wheel insertion?
  size_t pkt_num_;     ///< The request number of the wheel entry's sslot
  size_t abs_tx_tsc_;  ///< For inserts, the requested TX timestamp

  /// Record an wheel insertion entry
  wheel_record_t(size_t pkt_num, size_t abs_tx_tsc)
      : record_tsc_(rdtsc()),
        insert_(true),
        pkt_num_(pkt_num),
        abs_tx_tsc_(abs_tx_tsc) {}

  /// Record a wheel reap entry
  wheel_record_t(size_t pkt_num)
      : record_tsc_(rdtsc()), insert_(false), pkt_num_(pkt_num) {}

  std::string to_string(size_t console_ref_tsc, double freq_ghz) {
    std::ostringstream ret;
    size_t record_us = to_usec(record_tsc_ - console_ref_tsc, freq_ghz);
    size_t abs_tx_us = to_usec(abs_tx_tsc_ - console_ref_tsc, freq_ghz);
    if (insert_) {
      ret << "[Insert for pkt" << pkt_num_ << ", at " << record_us
          << " us, abs TX " << abs_tx_us << " us]";
    } else {
      ret << "[Reap for req " << pkt_num_ << ", at " << record_us << " us]";
    }

    return ret.str();
  }
};

static constexpr double kWheelSlotWidthUs = .5;  ///< Duration per wheel slot
static uint32_t const MAX_TIMING_WHEEL_PKTS = 1024;
static constexpr size_t kSessionCredits = MAX_TIMING_WHEEL_PKTS;
static double kWheelHorizonUs =
    1000000 * (kSessionCredits * (ucclParamCHUNK_SIZE_KB() << 10)) /
    timely::TimelyCC::kMinRate;

// This ensures that packets for an sslot undergoing retransmission are rarely
// in the wheel. This is recommended but not required.
// static_assert(kWheelHorizonUs <= kRpcRTOUs, "");

static size_t kWheelNumWslots = 1 + DIVUP(kWheelHorizonUs, kWheelSlotWidthUs);

static constexpr bool kWheelRecord = false;  ///< Fast-record wheel actions

/// One entry in a timing wheel bucket
struct wheel_ent_t {
  uint64_t sslot_;
  wheel_ent_t(void* sslot) : sslot_(reinterpret_cast<uint64_t>(sslot)) {}
};
static_assert(sizeof(wheel_ent_t) == 8, "");

// Handle the fact that we're not using all 64 TSC bits
static constexpr size_t kWheelBucketCap = 5;  ///< Wheel entries per bucket
static constexpr size_t kNumBktEntriesBits = 3;
static_assert((1ull << kNumBktEntriesBits) > kWheelBucketCap, "");

static size_t kBktPoolSize = kWheelNumWslots * kWheelBucketCap;

// TSC ticks per day = ~3 billion per second * 86400 seconds per day
// Require that rollback happens only after server lifetime
static constexpr size_t kKTscTicks = 1ull << (64 - kNumBktEntriesBits);
static_assert(kKTscTicks / (3000000000ull * 86400) > 2000, "");

struct wheel_bkt_t {
  size_t num_entries_ : kNumBktEntriesBits;  ///< Valid entries in this bucket

  /// Timestamp at which it is safe to transmit packets in this bucket's chain
  size_t tx_tsc_ : 64 - kNumBktEntriesBits;

  wheel_bkt_t* last_;  ///< Last bucket in chain. Used only at the first bucket.
  wheel_bkt_t* next_;  ///< Next bucket in chain
  wheel_ent_t entry_[kWheelBucketCap];  ///< Space for wheel entries
};
static_assert(sizeof(wheel_bkt_t) == 64, "");

struct timing_wheel_args_t {
  double freq_ghz_;
  uint64_t wslot_width_tsc_;
  uint64_t horizon_tsc_;
  size_t bkt_pool_;
  double link_bw_;
};

class TimingWheel {
 public:
  TimingWheel(timing_wheel_args_t args)
      : freq_ghz_(args.freq_ghz_),
        wslot_width_tsc_(args.wslot_width_tsc_),
        horizon_tsc_(args.horizon_tsc_),
        bkt_pool_(args.bkt_pool_),
        link_bw_(args.link_bw_) {
    wheel_buffer_ = new uint8_t[kWheelNumWslots * sizeof(wheel_bkt_t)];

    size_t base_tsc = rdtsc();
    wheel_ = reinterpret_cast<wheel_bkt_t*>(wheel_buffer_);
    for (size_t ws_i = 0; ws_i < kWheelNumWslots; ws_i++) {
      reset_bkt(&wheel_[ws_i]);
      wheel_[ws_i].tx_tsc_ = base_tsc + (ws_i + 1) * wslot_width_tsc_;
      wheel_[ws_i].last_ = &wheel_[ws_i];
    }

    bkt_pool_buf_ = new uint8_t[sizeof(wheel_bkt_t) * kBktPoolSize];
    for (int i = 0; i < kBktPoolSize; i++) {
      CHECK(bkt_pool_.push_front((wheel_bkt_t*)bkt_pool_buf_ + i));
    }
  }

  ~TimingWheel() {
    delete[] wheel_buffer_;
    delete[] bkt_pool_buf_;
  }

  // Queue a work request (i.e., one chunk) on the timing wheel.
  // Returns true if the work request was queued on the wheel.
  // Otherwise, the timing wheel was bypassed and the caller can transmit
  // directly.
  inline bool queue_on_timing_wheel(double target_rate,
                                    size_t* prev_desired_tx_tsc, size_t ref_tsc,
                                    void* wr, size_t chunk_size,
                                    bool allow_bypass) {
    if constexpr (kTestConstantRate) target_rate = gbps_to_rate(link_bw_);

    if (chunk_size < kBypassTimingWheelThres && allow_bypass) return false;

    double ns_delta = 1000000000 * (chunk_size / target_rate);
    double cycle_delta = ns_to_cycles(ns_delta, freq_ghz);

    size_t desired_tx_tsc = *prev_desired_tx_tsc + cycle_delta;
    desired_tx_tsc = (std::max)(desired_tx_tsc, ref_tsc);

    *prev_desired_tx_tsc = desired_tx_tsc;

    if (desired_tx_tsc == ref_tsc && allow_bypass) {
      return false;
    }

    insert(wheel_ent_t{wr}, ref_tsc, desired_tx_tsc);
    return true;
  }

  /// Return a dummy wheel entry
  static wheel_ent_t get_dummy_ent() {
    return wheel_ent_t(reinterpret_cast<void*>(0xdeadbeef));
  }

  /// Roll the wheel forward until it catches up with current time. Hopefully
  /// this is needed only during initialization.
  void catchup() {
    while (wheel_[cur_wslot_].tx_tsc_ < rdtsc()) reap(rdtsc());
  }

  /// Move entries from all wheel slots older than reap_tsc to the ready
  /// queue. The max_tsc of all these wheel slots is advanced. This function
  /// must be called with non-decreasing values of reap_tsc
  void reap(size_t reap_tsc) {
    while (wheel_[cur_wslot_].tx_tsc_ <= reap_tsc) {
      reap_wslot(cur_wslot_);
      wheel_[cur_wslot_].tx_tsc_ += (wslot_width_tsc_ * kWheelNumWslots);

      cur_wslot_++;
      if (cur_wslot_ == kWheelNumWslots) cur_wslot_ = 0;
    }
  }

  /**
   * @brief Add an entry to the wheel. This may move existing entries to the
   * wheel's ready queue.
   *
   * Even if the entry falls in the "current" wheel slot, we must not place it
   * entry directly in the ready queue. Doing so can reorder this entry before
   * prior entries in the current wheel slot that have not been reaped.
   *
   * @param ent The wheel entry to add
   * @param ref_tsc A recent timestamp
   * @param desired_tx_tsc The desired time for packet transmission
   */
  inline void insert(wheel_ent_t const& ent, size_t ref_tsc,
                     size_t desired_tx_tsc) {
    CHECK(desired_tx_tsc >= ref_tsc);
    CHECK(desired_tx_tsc - ref_tsc <= horizon_tsc_)
        << desired_tx_tsc - ref_tsc << " vs "
        << horizon_tsc_;  // Horizon definition

    reap(ref_tsc);  // Advance the wheel to a recent time
    assert(wheel_[cur_wslot_].tx_tsc_ > ref_tsc);

    size_t dst_wslot;
    if (desired_tx_tsc <= wheel_[cur_wslot_].tx_tsc_) {
      dst_wslot = cur_wslot_;
    } else {
      size_t wslot_delta =
          1 + (desired_tx_tsc - wheel_[cur_wslot_].tx_tsc_) / wslot_width_tsc_;
      CHECK(wslot_delta < kWheelNumWslots)
          << wslot_delta << " vs " << kWheelNumWslots;

      dst_wslot = cur_wslot_ + wslot_delta;
      if (dst_wslot >= kWheelNumWslots) dst_wslot -= kWheelNumWslots;
    }

    if (kWheelRecord) {
      auto wr = reinterpret_cast<struct wr_ex*>(ent.sslot_);
      record_vec_.emplace_back(wr->sge.length, desired_tx_tsc);
    }

    insert_into_wslot(dst_wslot, ent);
  }

 private:
  void insert_into_wslot(size_t ws_i, wheel_ent_t const& ent) {
    wheel_bkt_t* last_bkt = wheel_[ws_i].last_;
    assert(last_bkt->next_ == nullptr);

    assert(last_bkt->num_entries_ < kWheelBucketCap);
    last_bkt->entry_[last_bkt->num_entries_] = ent;
    last_bkt->num_entries_++;

    // If last_bkt is full, allocate a new one and make it the last
    if (last_bkt->num_entries_ == kWheelBucketCap) {
      wheel_bkt_t* new_bkt = alloc_bkt();
      last_bkt->next_ = new_bkt;
      wheel_[ws_i].last_ = new_bkt;
    }
  }

  /// Transfer all entries from a wheel slot to the ready queue. The wheel
  /// slot is reset and its chained buckets are returned to the pool.
  void reap_wslot(size_t ws_i) {
    wheel_bkt_t* bkt = &wheel_[ws_i];
    while (bkt != nullptr) {
      for (size_t i = 0; i < bkt->num_entries_; i++) {
        ready_entries_++;
        ready_queue_.push_back(bkt->entry_[i]);
        if (kWheelRecord) {
          auto wr = reinterpret_cast<struct wr_ex*>(bkt->entry_[i].sslot_);
          record_vec_.push_back(wheel_record_t(wr->sge.length));
        }
      }

      wheel_bkt_t* tmp_next = bkt->next_;

      reset_bkt(bkt);
      if (bkt != &wheel_[ws_i]) CHECK(bkt_pool_.push_front(bkt));
      bkt = tmp_next;
    }

    wheel_[ws_i].last_ = &wheel_[ws_i];  // Reset last pointer
  }

  inline void reset_bkt(wheel_bkt_t* bkt) {
    bkt->next_ = nullptr;
    bkt->num_entries_ = 0;
  }

  wheel_bkt_t* alloc_bkt() {
    wheel_bkt_t* bkt;
    CHECK(bkt_pool_.pop_front(&bkt));  // Exception if allocation fails
    reset_bkt(bkt);
    return bkt;
  }

  double const freq_ghz_;         ///< TSC freq, used only for us/tsc conversion
  size_t const wslot_width_tsc_;  ///< Time-granularity in TSC units
  size_t const horizon_tsc_;      ///< Horizon in TSC units
  uint8_t* wheel_buffer_;

  wheel_bkt_t* wheel_;
  size_t cur_wslot_ = 0;
  CircularBuffer<wheel_bkt_t*, /*sync=*/false> bkt_pool_;
  uint8_t* bkt_pool_buf_;

  double link_bw_;

 public:
  std::vector<wheel_record_t> record_vec_;  ///< Used only with kWheelRecord
  uint64_t ready_entries_ = 0;
  std::deque<wheel_ent_t> ready_queue_;
};
}  // namespace uccl