#pragma once

#include <glog/logging.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "timely.h"
#include "timing_wheel.h"

namespace uccl {
namespace swift {

constexpr bool seqno_lt(uint32_t a, uint32_t b) {
    return static_cast<int32_t>(a - b) < 0;
}
constexpr bool seqno_le(uint32_t a, uint32_t b) {
    return static_cast<int32_t>(a - b) <= 0;
}
constexpr bool seqno_eq(uint32_t a, uint32_t b) {
    return static_cast<int32_t>(a - b) == 0;
}
constexpr bool seqno_ge(uint32_t a, uint32_t b) {
    return static_cast<int32_t>(a - b) >= 0;
}
constexpr bool seqno_gt(uint32_t a, uint32_t b) {
    return static_cast<int32_t>(a - b) > 0;
}

/**
 * @brief Swift Congestion Control (SWCC) protocol control block.
 */
struct Pcb {
    static constexpr std::size_t kInitialCwnd = 256;
    static constexpr std::size_t kSackBitmapSize = 1024;
    static constexpr std::size_t kSackBitmapBucketSize = sizeof(uint64_t) * 8;
    static constexpr std::size_t kFastRexmitDupAckThres = 3;
    static constexpr std::size_t kRtoMaxRexmitConsectutiveAllowed = 102400;
    static constexpr std::size_t kRttProbingIntervalUs = 50;
    static constexpr int kRtoExpireThresInTicks = 3;  // in slow timer ticks.
    static constexpr int kRtoDisabled = -1;
    Pcb()
        : timely(ghz, kLinkBandwidth),
          wheel_({ghz}),
          prev_desired_tx_tsc_(rdtsc()) {
        wheel_.catchup();
    }

    Timely timely;
    TimingWheel wheel_;
    size_t prev_desired_tx_tsc_;

    inline void update_rate(size_t _rdtsc, size_t sample_rtt_tsc) {
        timely.update_rate(_rdtsc, sample_rtt_tsc);
    }

    inline void queue_on_timing_wheel(size_t ref_tsc, size_t pkt_size) {
        // double ns_delta = 1000000000 * (pkt_size / timely.rate_);
        // double ns_delta = 1000000000 * (pkt_size / timely.link_bandwidth_);
        const double test_link_bw = 40.0 * 1000 * 1000 * 1000 / 8;
        double ns_delta = 1000000000 * (pkt_size / test_link_bw);
        double cycle_delta = ns_to_cycles(ns_delta, ghz);

        size_t desired_tx_tsc = prev_desired_tx_tsc_ + cycle_delta;
        desired_tx_tsc = (std::max)(desired_tx_tsc, ref_tsc);

        prev_desired_tx_tsc_ = desired_tx_tsc;

        wheel_.insert(TimingWheel::get_dummy_ent(), ref_tsc, desired_tx_tsc);
    }

    inline uint32_t get_num_ready_tx_pkt() {
        size_t cur_tsc = rdtsc();
        wheel_.reap(cur_tsc);

        size_t num_ready = wheel_.ready_queue_.size();
        for (size_t i = 0; i < num_ready; i++) {
            wheel_.ready_queue_.pop();
        }

        return num_ready;
    }

    // Return the sender effective window in # of packets.
    uint32_t effective_wnd() const {
        if (snd_nxt < snd_una + snd_ooo_acks) return 1;
        uint32_t inflight = snd_nxt - snd_una - snd_ooo_acks;
        if (cwnd <= inflight) return 1;
        uint32_t effective_wnd = std::ceil((cwnd - inflight) * ecn_alpha);
        return effective_wnd == 0 ? 1 : effective_wnd;
    }

    void mutliplicative_decrease() { ecn_alpha /= 2; }
    void additive_increase() {
        ecn_alpha += 0.1;
        if (ecn_alpha > 1.0) ecn_alpha = 1.0;
    }

    uint32_t seqno() const { return snd_nxt; }
    uint32_t get_snd_nxt() {
        uint32_t seqno = snd_nxt;
        snd_nxt++;
        return seqno;
    }

    std::string to_string() const {
        std::stringstream ecn_alpha_stream;
        ecn_alpha_stream << std::fixed << std::setprecision(2) << ecn_alpha;

        std::string s;
        s += "[CC] snd_nxt: " + std::to_string(snd_nxt) +
             " snd_una: " + std::to_string(snd_una) +
             " rcv_nxt: " + std::to_string(rcv_nxt) +
             " cwnd: " + std::to_string(cwnd) +
             " fast_rexmits: " + std::to_string(fast_rexmits) +
             " rto_rexmits: " + std::to_string(rto_rexmits) +
             " effective_wnd: " + std::to_string(effective_wnd()) +
             " ecn_alpha: " + ecn_alpha_stream.str();
        return s;
    }

    uint32_t ackno() const { return rcv_nxt; }
    bool max_rto_rexmits_consectutive_reached() const {
        return rto_rexmits_consectutive >= kRtoMaxRexmitConsectutiveAllowed;
    }
    bool rto_disabled() const { return rto_timer == kRtoDisabled; }
    bool rto_expired() const { return rto_timer >= kRtoExpireThresInTicks; }

    uint32_t get_rcv_nxt() const { return rcv_nxt; }
    void advance_rcv_nxt() { rcv_nxt++; }
    void rto_enable() { rto_timer = 0; }
    void rto_disable() { rto_timer = kRtoDisabled; }
    void rto_reset() { rto_enable(); }
    void rto_maybe_reset() {
        if (snd_una == snd_nxt)
            rto_disable();
        else
            rto_reset();
    }
    void rto_advance() { rto_timer++; }

    void sack_bitmap_shift_left_one() {
        constexpr size_t sack_bitmap_bucket_max_idx =
            kSackBitmapSize / kSackBitmapBucketSize - 1;

        for (size_t i = 0; i < sack_bitmap_bucket_max_idx; i++) {
            // Shift the current each bucket to the left by 1 and take the most
            // significant bit from the next bucket
            uint64_t &sack_bitmap_left_bucket = sack_bitmap[i];
            const uint64_t sack_bitmap_right_bucket = sack_bitmap[i + 1];

            sack_bitmap_left_bucket = (sack_bitmap_left_bucket >> 1) |
                                      (sack_bitmap_right_bucket << 63);
        }

        // Special handling for the right most bucket
        uint64_t &sack_bitmap_right_most_bucket =
            sack_bitmap[sack_bitmap_bucket_max_idx];
        sack_bitmap_right_most_bucket >>= 1;

        sack_bitmap_count--;
    }

    void sack_bitmap_bit_set(const size_t index) {
        const size_t sack_bitmap_bucket_idx = index / kSackBitmapBucketSize;
        const size_t sack_bitmap_idx_in_bucket = index % kSackBitmapBucketSize;

        LOG_IF(FATAL, index >= kSackBitmapSize)
            << "Index out of bounds: " << index;

        sack_bitmap[sack_bitmap_bucket_idx] |=
            (1ULL << sack_bitmap_idx_in_bucket);

        sack_bitmap_count++;
    }

    uint32_t target_delay{0};
    uint32_t snd_nxt{0};
    uint32_t snd_una{0};
    uint32_t snd_ooo_acks{0};
    uint32_t rcv_nxt{0};
    uint64_t sack_bitmap[kSackBitmapSize / kSackBitmapBucketSize]{0};
    uint8_t sack_bitmap_count{0};
    uint16_t cwnd{kInitialCwnd};
    uint16_t duplicate_acks{0};
    int rto_timer{kRtoDisabled};
    uint32_t fast_rexmits{0};
    uint32_t rto_rexmits{0};
    uint16_t rto_rexmits_consectutive{0};
    double ecn_alpha{1.0};
};

}  // namespace swift
}  // namespace uccl
