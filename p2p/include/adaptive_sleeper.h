#ifndef ADAPTIVE_SLEEPER_H
#define ADAPTIVE_SLEEPER_H

#include "util/debug.h"
#include <chrono>
#include <ctime>
#include <sys/eventfd.h>
#include <sys/poll.h>

class AdaptiveSleeper {
 public:
  enum SleepState { POLL = 0, SLEEP };

  AdaptiveSleeper()
      : state_(POLL),
        last_event_time_(std::chrono::steady_clock::time_point::min()) {
    work_eventfd_ = eventfd(0, EFD_NONBLOCK);
  }

  ~AdaptiveSleeper() { close(work_eventfd_); };

  void maybe_sleep() {
    int ret;
    if (std::chrono::steady_clock::now() - last_event_time_ >=
            kNoActivityThreshold &&
        last_event_time_ != std::chrono::steady_clock::time_point::min()) {
      UCCL_LOG(INFO, UCCL_P2P) << "No activity detected for the last 120 "
                                  "seconds, putting proxy to sleep";

      state_ = SLEEP;
    }

    if (state_ == POLL) {
      return;
    } else {
      UCCL_LOG(INFO, UCCL_P2P) << "Sleeping proxy thread";

      // continuously sleep while there are no new work entries
      struct pollfd events_to_poll[kNumActivitiesToPoll] = {
          {.fd = work_eventfd_, .events = POLLIN},
      };

      int n = ppoll(events_to_poll, kNumActivitiesToPoll, &kPollSleepDuration,
                    NULL);
      UCCL_PCHECK(n != -1);

      if (n > 0) {
        if (events_to_poll[0].revents == POLLIN) {
          UCCL_LOG(INFO, UCCL_P2P) << "Waking up becuase of work signal";
          eventfd_t work_val;
          ret = eventfd_read(work_eventfd_, &work_val);
          UCCL_PCHECK(ret == 0);

          last_event_time_ = std::chrono::steady_clock::now();
          state_ = POLL;
        }
      }
    }
  }

  void maybe_wake_proxy_thread() {
    int ret = eventfd_write(work_eventfd_, kWakeEventConst);
    UCCL_CHECK(ret == 0);
  }

  void update_timer() {
    char const* is_adaptive_sleep = std::getenv("UCCL_RDMA_ADAPTIVE_SLEEP");
    if (is_adaptive_sleep && strcmp(is_adaptive_sleep, "1") == 0) {
      UCCL_LOG_FRIST_N(INFO, UCCL_P2P, 1) << "Adaptive sleeper configured";
      last_event_time_ = std::chrono::steady_clock::now();
    }
  }

 private:
  // TODO: change after testing
  static constexpr auto kNoActivityThreshold = std::chrono::seconds(3);
  static constexpr int kNumActivitiesToPoll = 1;
  static constexpr struct timespec kPollSleepDuration = {
      .tv_sec = 5,
      .tv_nsec = 0,
  };
  static constexpr int kWakeEventConst = 0x42;

  SleepState state_;
  std::chrono::steady_clock::time_point last_event_time_;
  int work_eventfd_;
};

#endif