#ifndef ADAPTIVE_SLEEPER_HPP
#define ADAPTIVE_SLEEPER_HPP

#include "common.hpp"
#include "proxy_ctx.hpp"
#include "util/debug.h"
#include <chrono>
#include <ctime>
#include <string_view>
#include <sys/eventfd.h>
#include <sys/poll.h>

// to enable Adaptive Sleeper, set UCCL_RDMA_ADAPTIVE_SLEEP=1
// handles level of sleep on the proxy thread, based on RDMA request volume
// Adaptive sleeper states:
// 1. POLL = no delay at all
// 2. SLEEP = put the CPU to sleep, while letting it poll on GPU initiated
// events and the completion events queue. This happens when there has been no
// work for >= kNoActivityDuration (120 seconds)
class AdaptiveSleeper {
 public:
  enum SleepState { POLL = 0, SLEEP };

  AdaptiveSleeper();

  ~AdaptiveSleeper();

  // decide whether or not to put the CPU to sleep based on its current
  void maybe_sleep(ProxyCtx& proxy_ctx);

  void maybe_wake_proxy_thread();

  // this function kickk starts the inactivity timer, and is guarded by the
  // UCCL_RDMA_ADAPTIVE_SLEEP flag
  void init_timer();

 private:
  static constexpr auto kNoActivityThreshold = std::chrono::seconds(120);
  static constexpr int kNumActivitiesToPoll = 2;
  static constexpr struct timespec kPollSleepDuration = {
      .tv_sec = 5,
      .tv_nsec = 0,
  };
  static constexpr int kWakeEventConst = 0x42;

  SleepState state_;
  int work_eventfd_;
  std::chrono::steady_clock::time_point last_event_time_;
};

#endif