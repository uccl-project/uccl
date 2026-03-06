/**
 * @file swift.h
 * @brief Swift congestion control [SIGCOMM 20]
 */

#pragma once

#include "util/latency.h"
#include "util/util.h"
#include "util_timer.h"
#include <infiniband/verbs.h>
#include <iomanip>

namespace uccl {
namespace swift {

struct swift_record_t {
  double rtt_;
  double rate_;

  swift_record_t() : rtt_(0.0), rate_(0.0) {}
  swift_record_t(double rtt, double rate) : rtt_(rtt), rate_(rate) {}

  std::string to_string() {
    std::ostringstream ret;
    ret << "[RTT " << std::setprecision(5) << rtt_ << " us"
        << ", rate " << std::setprecision(4)
        << (rate_ / (1000 * 1000 * 1000)) * 8 << "]";
    return ret.str();
  }
};
/// Implementation of the Swift congestion control protocol from SIGCOMM 20
/// TODO: Implement pacing when cwnd < CHUNK_SIZE
class SwiftCC {
 public:
  // Debugging
  static constexpr bool kVerbose = false;
  static constexpr bool kRecord = false;        ///< Fast-record Swift steps
  static constexpr bool kLatencyStats = false;  ///< Track per-packet RTT stats

  // Config
  static constexpr uint32_t kMSS = 4096;
  static constexpr uint32_t kDefaultWnd = 1024 * 1024;
  static constexpr uint32_t kMinCwnd = kMSS;         // in Bytes
  static constexpr uint32_t kMaxCwnd = 1024 * 1024;  // in Bytes
  static constexpr double kAI = 8;
  static constexpr double kBeta = 0.8;
  static constexpr double kMaxDF = 0.5;
  static constexpr double kBaseDelay = 50;  // in microseconds

  // flow scaling
  static constexpr double kFSRange = 5 * kBaseDelay;
  static constexpr double kFSMinCwnd = 32;   // in MTU-sized packets
  static constexpr double kFSMaxCwnd = 100;  // in MTU-sized packets
  static constexpr double kFSAlpha = kFSRange / ((1.0 / std::sqrt(kFSMinCwnd)) -
                                                 (1.0 / std::sqrt(kFSMaxCwnd)));
  static constexpr double kFSBeta = -kFSAlpha / std::sqrt(kFSMaxCwnd);

  double rate_ = 0.0;  ///< The current sending rate
  size_t last_decrease_tsc_ = 0;
  double rtt_ = kBaseDelay;
  double min_rtt_ = kBaseDelay;

  uint32_t prev_cwnd_;
  uint32_t swift_cwnd_ = kDefaultWnd;

  // Const
  double min_rtt_tsc_ = 0.0;
  double t_low_tsc_ = 0.0;
  double freq_ghz_ = 0.0;
  double link_bandwidth_ = 0.0;

  // For latency stats
  Latency latency_;

  // For recording, used only with kRecord
  size_t create_tsc_;
  std::vector<swift_record_t> record_vec_;

  SwiftCC() {}
  SwiftCC(double freq_ghz, double link_bandwidth)
      : last_decrease_tsc_(rdtsc()),
        min_rtt_tsc_(kBaseDelay * freq_ghz * 1000),
        freq_ghz_(freq_ghz),
        link_bandwidth_(link_bandwidth),
        create_tsc_(rdtsc()) {
    rate_ = link_bandwidth;  // Start sending at the max rate
    if (kRecord) record_vec_.reserve(1000000);
  }

  // Last desired tx timestamp for timing wheel.
  size_t prev_desired_tx_tsc_ = 0;

  void update_rtt(double delay) {
    rtt_ = rtt_ * 7 / 8 + delay / 8;
    min_rtt_ = std::min(min_rtt_, delay);
  }

  void adjust_wnd(double delay, uint32_t acked_bytes) {
    prev_cwnd_ = swift_cwnd_;
    bool cand = can_decrease();

    update_rtt(delay);

    // LOG_EVERY_N(ERROR, 100) << "Average RTT: " << rtt_ << " us, min:" <<
    // min_rtt << " us";

    double target_delay = get_target_delay();

    if (delay < target_delay) {
      swift_cwnd_ = swift_cwnd_ + (acked_bytes * kAI) / swift_cwnd_;
    } else if (cand) {
      swift_cwnd_ =
          swift_cwnd_ *
          std::max(1 - kBeta * (delay - target_delay) / delay, 1 - kMaxDF);
    }

    if (swift_cwnd_ < kMinCwnd) swift_cwnd_ = kMinCwnd;
    if (swift_cwnd_ > kMaxCwnd) swift_cwnd_ = kMaxCwnd;
  }

  uint32_t get_wnd() const { return swift_cwnd_; }

  bool can_decrease() const {
    return to_usec((rdtsc() - last_decrease_tsc_), freq_ghz) >= rtt_;
  }

  double get_target_delay() const {
    double fs_delay = kFSAlpha / std::sqrt(prev_cwnd_ / kMSS) + kFSBeta;

    if (fs_delay > kFSRange) {
      fs_delay = kFSRange;
    }

    if (fs_delay < 0.0) {
      fs_delay = 0.0;
    }

    if (prev_cwnd_ == 0) {
      fs_delay = 0.0;
    }

    // FIXME: fs_delay should add hop delay.
    return kBaseDelay + fs_delay;
  }

  /// Get RTT percentile if latency stats are enabled, and reset latency stats
  double get_rtt_perc(double perc) const {
    if (!kLatencyStats || latency_.count() == 0) return -1.0;
    double ret = latency_.perc(perc);
    return ret;
  }

  void reset_rtt_stats() { latency_.reset(); }

  double get_avg_rtt() const { return rtt_; }
  double get_rate_gbps() const { return rate_to_gbps(rate_); }
};

}  // namespace swift
}  // namespace uccl