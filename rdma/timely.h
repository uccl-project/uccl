/**
 * @file timely.h
 * @brief TIMELY congestion control [SIGCOMM 15]
 * From: http://people.eecs.berkeley.edu/~radhika/timely-code-snippet.cc
 * Units: Microseconds or TSC for time, bytes/sec for throughput
 */

#pragma once

#include "util/latency.h"
#include "util/util.h"
#include "util_timer.h"
#include <infiniband/verbs.h>
#include <iomanip>

/**
 * @file timely-sweep-params.h
 * @brief Timely parameters that need to be sweeped
 */
static constexpr bool kPatched = true;  ///< Patch from ECN-vs-delay
// EWMA alpha used for global CC state.
static constexpr double kEwmaAlpha = 0.125;
static constexpr double kBeta = 0.008;

// EWMA alpha used for per-path CC states.
static constexpr double kPPEwmaAlpha = 0.125;

namespace uccl {
namespace timely {

struct timely_record_t {
  double rtt_;
  double rate_;

  timely_record_t() : rtt_(0.0), rate_(0.0) {}
  timely_record_t(double rtt, double rate) : rtt_(rtt), rate_(rate) {}

  std::string to_string() {
    std::ostringstream ret;
    ret << "[RTT " << std::setprecision(5) << rtt_ << " us"
        << ", rate " << std::setprecision(4)
        << (rate_ / (1000 * 1000 * 1000)) * 8 << "]";
    return ret.str();
  }
};

/// Implementation of the Timely congestion control protocol from SIGCOMM 15
class TimelyCC {
 public:
  // Debugging
  static constexpr bool kVerbose = false;
  static constexpr bool kRecord = false;        ///< Fast-record Timely steps
  static constexpr bool kLatencyStats = false;  ///< Track per-packet RTT stats

  // Config
  static constexpr double kMinRate = 0.1 * 1000 * 1000 * 1000;
  static constexpr double kAddRate = 0.5 * 1000 * 1000 * 1000;
  static constexpr double kTLow = 35;
  static constexpr double kTHigh = 350;

  static constexpr double kMinRTT = 2;
  static constexpr size_t kHaiThresh = 5;

  double rate_ = 0.0;  ///< The current sending rate
  size_t neg_gradient_count_ = 0;
  double prev_rtt_ = kMinRTT;
  double avg_rtt_diff_ = 0.0;
  size_t last_update_tsc_ = 0;

  // Const
  double min_rtt_tsc_ = 0.0;
  double t_low_tsc_ = 0.0;
  double freq_ghz_ = 0.0;
  double link_bandwidth_ = 0.0;

  // For latency stats
  Latency latency_;

  // For recording, used only with kRecord
  size_t create_tsc_;
  std::vector<timely_record_t> record_vec_;

  TimelyCC() {}
  TimelyCC(double freq_ghz, double link_bandwidth)
      : last_update_tsc_(rdtsc()),
        min_rtt_tsc_(kMinRTT * freq_ghz * 1000),
        t_low_tsc_(kTLow * freq_ghz * 1000),
        freq_ghz_(freq_ghz),
        link_bandwidth_(link_bandwidth),
        create_tsc_(rdtsc()) {
    rate_ = link_bandwidth;  // Start sending at the max rate
    if (kRecord) record_vec_.reserve(1000000);
  }

  /// The w() function from the ECN-vs-delay paper by Zhu et al. (CoNEXT 16)
  static double w_func(double g) {
    assert(kPatched);
    if (g <= -0.25) return 0;
    if (g >= 0.25) return 1;
    return (2 * g + 0.5);
  }

  // Last desired tx timestamp for timing wheel.
  size_t prev_desired_tx_tsc_ = 0;

  /**
   * @brief Perform a rate update
   *
   * @param _rdtsc A recently sampled RDTSC. This can reduce calls to rdtsc()
   * when the caller can reuse a sampled RDTSC.
   * @param sample_rtt_tsc The RTT sample in RDTSC cycles
   */
  void update_rate(size_t _rdtsc, size_t sample_rtt_tsc, double ewma_alpha) {
    assert(_rdtsc >= 1000000000 && _rdtsc >= last_update_tsc_);  // Sanity check
    static constexpr bool kCcOptTimelyBypass = true;
    if (kCcOptTimelyBypass &&
        (rate_ == link_bandwidth_ && sample_rtt_tsc <= t_low_tsc_)) {
      // Bypass expensive computation, but include the latency sample in
      // stats.
      if (kLatencyStats) {
        latency_.update(
            static_cast<size_t>(to_usec(sample_rtt_tsc, freq_ghz_)));
      }
      return;
    }

    // Sample RTT can be lower than min RTT during retransmissions
    if (unlikely(sample_rtt_tsc < min_rtt_tsc_)) return;

    // Convert the sample RTT to usec, and don't use _sample_rtt_tsc from
    // now
    double sample_rtt = to_usec(sample_rtt_tsc, freq_ghz_);

    double rtt_diff = sample_rtt - prev_rtt_;
    neg_gradient_count_ = (rtt_diff < 0) ? neg_gradient_count_ + 1 : 0;
    avg_rtt_diff_ =
        ((1 - ewma_alpha) * avg_rtt_diff_) + (ewma_alpha * rtt_diff);

    double delta_factor = (_rdtsc - last_update_tsc_) / min_rtt_tsc_;  // fdiv
    delta_factor = (std::min)(delta_factor, 1.0);

    double ai_factor = kAddRate * delta_factor;

    double new_rate;
    if (sample_rtt < kTLow) {
      // Additive increase
      new_rate = rate_ + ai_factor;
    } else {
      double md_factor = delta_factor * kBeta;     // Scaled factor for decrease
      double norm_grad = avg_rtt_diff_ / kMinRTT;  // Normalized gradient

      if (likely(sample_rtt <= kTHigh)) {
        if (kPatched) {
          double wght = w_func(norm_grad);
          double err = (sample_rtt - kTLow) / kTLow;
          if (kVerbose) {
            printf(
                "wght = %.4f, err = %.4f, md = x%.3f, ai = %.3f "
                "Gbps\n",
                wght, err, (1 - md_factor * wght * err),
                rate_to_gbps(ai_factor * (1 - wght)));
          }

          new_rate =
              rate_ * (1 - md_factor * wght * err) + ai_factor * (1 - wght);
        } else {
          // Original logic
          if (norm_grad <= 0) {
            size_t n = neg_gradient_count_ >= kHaiThresh ? 5 : 1;
            new_rate = rate_ + n * ai_factor;
          } else {
            new_rate = rate_ * (1.0 - md_factor * norm_grad);
          }
        }
      } else {
        // Multiplicative decrease based on current RTT sample, not
        // average
        new_rate = rate_ * (1 - md_factor * (1 - kTHigh / sample_rtt));
      }
    }

    rate_ = (std::max)(new_rate, rate_ * 0.5);
    rate_ = (std::min)(rate_, link_bandwidth_);
    rate_ = (std::max)(rate_, double(kMinRate));

    prev_rtt_ = sample_rtt;
    last_update_tsc_ = _rdtsc;

    // Debug/stats code goes here
    if (kLatencyStats) latency_.update(static_cast<size_t>(sample_rtt));
    if (kRecord && rate_ != link_bandwidth_) {
      record_vec_.emplace_back(sample_rtt, rate_);
    }

    if (kRecord && rate_ == kMinRate) {
      // If we reach min rate after steady state, print a log and exit
      double sec_since_creation = to_sec(rdtsc() - create_tsc_, freq_ghz_);
      if (sec_since_creation >= 0.0) {
        for (auto& r : record_vec_) printf("%s\n", r.to_string().c_str());
        exit(-1);
      }
    }
  }

  uint32_t get_wnd() const {
    // Rate is in bytes/sec, RTT is in usec.
    // window = rate * RTT
    return static_cast<uint32_t>(rate_ * prev_rtt_ / 1e6);
  }

  /// Get RTT percentile if latency stats are enabled, and reset latency stats
  double get_rtt_perc(double perc) const {
    if (!kLatencyStats || latency_.count() == 0) return -1.0;
    double ret = latency_.perc(perc);
    return ret;
  }

  void reset_rtt_stats() { latency_.reset(); }

  double get_avg_rtt() const { return prev_rtt_; }
  double get_avg_rtt_diff() const { return avg_rtt_diff_; }
  double get_rate_gbps() const { return rate_to_gbps(rate_); }
};

}  // namespace timely
}  // namespace uccl