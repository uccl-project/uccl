#ifndef ADAPTIVE_SLEEPER_H
#define ADAPTIVE_SLEEPER_H

#include "util/debug.h"
#include <chrono>
#include <cstring>
#include <ctime>
#include <sys/eventfd.h>
#include <sys/poll.h>

// to enable Adaptive Sleeper, set UCCL_RDMA_ADAPTIVE_SLEEP=1
// handles level of sleep on the proxy thread, based on async / RDMA request
// volume Adaptive sleeper states:
class P2PAdaptiveSleeper {
 public:
  enum SleepState { POLL = 0, SLEEP };

  P2PAdaptiveSleeper();

  ~P2PAdaptiveSleeper();

  void maybe_sleep();

  void maybe_wake_proxy_thread();

  // this function kickk starts the inactivity timer, and is guarded by the
  // UCCL_RDMA_ADAPTIVE_SLEEP flag
  void update_timer();

 private:
  // TODO: change after testing
  static constexpr auto kNoActivityThreshold = std::chrono::seconds(120);
  static constexpr int kNumActivitiesToPoll = 1;
  static constexpr struct timespec kPollSleepDuration = {
      .tv_sec = 5,
      .tv_nsec = 0,
  };
  static constexpr int kWakeEventConst = 0x1;

  SleepState state_;
  std::chrono::steady_clock::time_point last_event_time_;
  int work_eventfd_;
  bool is_adaptive_sleep_;
};

#endif