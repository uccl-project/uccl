/**
 * @file cc_state.h
 * @brief Shared congestion control state for RoCE RDMA transports.
 *        Used by both P2P and EP subsystems.
 */

#pragma once

#include "cc/swift.h"
#include "cc/timely.h"
#include "util/timer.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>

namespace uccl {
namespace cc {

class CongestionControlState {
 public:
  enum class Mode : uint8_t { kNone, kTimely, kSwift };

  CongestionControlState() = default;

  /// Construct with known parameters (e.g. P2P path where link BW is known).
  CongestionControlState(Mode mode, double freq_ghz, double link_bw_bps)
      : mode_(mode),
        timely_(freq_ghz, link_bw_bps),
        swift_(freq_ghz, link_bw_bps) {}

  /// Deferred init (e.g. EP path — after RDMA context is available).
  void init(Mode mode, double freq_ghz, double link_bw_bps) {
    mode_ = mode;
    timely_ = timely::TimelyCC(freq_ghz, link_bw_bps);
    swift_ = swift::SwiftCC(freq_ghz, link_bw_bps);
  }

  /// Parse CC mode from an environment variable (case-insensitive).
  static Mode parseMode(char const* env_var) {
    auto* env = std::getenv(env_var);
    if (env == nullptr) return Mode::kNone;
    std::string mode(env);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (mode == "timely") return Mode::kTimely;
    if (mode == "swift") return Mode::kSwift;
    return Mode::kNone;
  }

  bool enabled() const { return mode_ != Mode::kNone; }
  Mode mode() const { return mode_; }

  /// Record send timestamp for a given wr_id.
  void recordSendTsc(uint64_t wr_id) {
    if (mode_ == Mode::kNone) return;
    send_tsc_[wr_id % kTscWindowSize].store(rdtsc(), std::memory_order_release);
  }

  /// Update CC state on ACK.  Call once per completed WR.
  void onAck(uint64_t wr_id, size_t acked_bytes) {
    if (mode_ == Mode::kNone) return;

    uint64_t send_tsc =
        send_tsc_[wr_id % kTscWindowSize].load(std::memory_order_acquire);
    if (send_tsc == 0) return;

    uint64_t now = rdtsc();
    size_t sample_rtt_tsc = now - send_tsc;
    if (sample_rtt_tsc == 0) return;

    if (mode_ == Mode::kTimely) {
      timely_.update_rate(now, sample_rtt_tsc, ::kEwmaAlpha);
    } else if (mode_ == Mode::kSwift) {
      double delay_us = to_usec(sample_rtt_tsc, freq_ghz);
      uint32_t bytes = acked_bytes > 0
                           ? static_cast<uint32_t>(acked_bytes)
                           : static_cast<uint32_t>(swift::SwiftCC::kMSS);
      swift_.adjust_wnd(delay_us, bytes);
    }
    send_tsc_[wr_id % kTscWindowSize].store(0, std::memory_order_relaxed);
  }

  /// Returns CC-controlled window in bytes, or 0 if CC is disabled.
  size_t getWindowBytes() const {
    if (mode_ == Mode::kTimely) return timely_.get_wnd();
    if (mode_ == Mode::kSwift) return swift_.get_wnd();
    return 0;
  }

 private:
  static constexpr size_t kTscWindowSize = 65536;
  std::unique_ptr<std::atomic<uint64_t>[]> send_tsc_ {
    new std::atomic<uint64_t>[kTscWindowSize] {}
  };
  Mode mode_ = Mode::kNone;
  timely::TimelyCC timely_;
  swift::SwiftCC swift_;
};

}  // namespace cc
}  // namespace uccl
