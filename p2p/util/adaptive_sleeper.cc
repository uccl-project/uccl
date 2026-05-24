#include "adaptive_sleeper.h"
#include <cstdlib>
#include <unistd.h>

P2PAdaptiveSleeper::P2PAdaptiveSleeper()
    : state_(POLL),
      last_event_time_(std::chrono::steady_clock::time_point::min()) {
  work_eventfd_ = eventfd(0, EFD_NONBLOCK);
  char const* adative_sleep_env = std::getenv("UCCL_RDMA_ADAPTIVE_SLEEP");
  is_adaptive_sleep_ = adative_sleep_env && strcmp(adative_sleep_env, "1") == 0;
}

P2PAdaptiveSleeper::~P2PAdaptiveSleeper() { close(work_eventfd_); }

void P2PAdaptiveSleeper::maybe_sleep() {
  if (!is_adaptive_sleep_) {
    return;
  }

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

    int n =
        ppoll(events_to_poll, kNumActivitiesToPoll, &kPollSleepDuration, NULL);
    UCCL_PCHECK(n != -1);

    if (n > 0) {
      if (events_to_poll[0].revents == POLLIN) {
        eventfd_t work_val;
        ret = eventfd_read(work_eventfd_, &work_val);
        UCCL_PCHECK(ret == 0);

        last_event_time_ = std::chrono::steady_clock::now();
        state_ = POLL;
      }
    } else {
      UCCL_LOG(INFO, UCCL_P2P) << "Going back to sleep...";
    }
  }
}

void P2PAdaptiveSleeper::maybe_wake_proxy_thread() {
  if (!is_adaptive_sleep_) {
    return;
  }
  UCCL_LOG(INFO, UCCL_P2P) << "Sending wake signal to thread";
  int ret = eventfd_write(work_eventfd_, kWakeEventConst);
  UCCL_CHECK(ret == 0);
}

void P2PAdaptiveSleeper::update_timer() {
  if (!is_adaptive_sleep_) {
    return;
  }

  UCCL_LOG_FIRST_N(INFO, UCCL_P2P, 1) << "Adaptive sleeper configured";
  last_event_time_ = std::chrono::steady_clock::now();
}
