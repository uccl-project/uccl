#include "adaptive_sleeper.hpp"
#include "util/debug.h"
#include <cstring>

EPAdaptiveSleeper::EPAdaptiveSleeper()
    : state_(POLL),
      last_event_time_(std::chrono::steady_clock::time_point::min()) {
  work_eventfd_ = eventfd(0, EFD_NONBLOCK);
  char const* adative_sleep_env = std::getenv("UCCL_RDMA_ADAPTIVE_SLEEP");
  is_adaptive_sleep_ = adative_sleep_env && strcmp(adative_sleep_env, "1") == 0;
}

EPAdaptiveSleeper::~EPAdaptiveSleeper() { close(work_eventfd_); }

void EPAdaptiveSleeper::maybe_sleep(ProxyCtx& proxy_ctx) {
  int ret;

  if (std::chrono::steady_clock::now() - last_event_time_ >=
          kNoActivityThreshold &&
      last_event_time_ != std::chrono::steady_clock::time_point::min()) {
    UCCL_LOG(INFO, UCCL_EP) << "No activity detected for the last 120 "
                               "seconds, putting proxy to sleep";

    state_ = SLEEP;

    ret = ibv_req_notify_cq(proxy_ctx.cq, 0);
    UCCL_PCHECK(ret == 0);
  }

  if (state_ == POLL || proxy_ctx.comp_channel == nullptr) {
    return;
  } else if (state_ == SLEEP) {
    UCCL_LOG(INFO, UCCL_EP) << "Sleeping proxy thread";

    // continuously sleep while there are no new work entries
    struct pollfd events_to_poll[kNumActivitiesToPoll] = {
        {.fd = work_eventfd_, .events = POLLIN},
        {.fd = proxy_ctx.comp_channel->fd, .events = POLLIN},
    };

    int n =
        ppoll(events_to_poll, kNumActivitiesToPoll, &kPollSleepDuration, NULL);
    UCCL_PCHECK(n != -1);

    if (n > 0) {
      if (events_to_poll[0].revents == POLLIN) {
        UCCL_LOG(INFO, UCCL_EP)
            << "Waking up because of dispatch/combine trigger";

        eventfd_t work_value;
        ret = eventfd_read(work_eventfd_, &work_value);
        // we check for % since cumulative unread writes will add on each
        // other
        UCCL_CHECK(ret == 0 && work_value % kWakeEventConst == 0);
      }

      if (events_to_poll[1].revents == POLLIN) {
        UCCL_LOG(INFO, UCCL_EP) << "Waking up because of RDMA event";
        void* ctx;
        ibv_get_cq_event(proxy_ctx.comp_channel, &proxy_ctx.cq, &ctx);
        // acknowledge the event to clear the notification
        ibv_ack_cq_events(proxy_ctx.cq, 1);
      }

      last_event_time_ = std::chrono::steady_clock::now();
      state_ = POLL;
    } else if (n == 0) {
      UCCL_LOG(INFO, UCCL_EP)
          << "Proxy thread woke due to poll timeout, sleeping again";
    }
  }
}

void EPAdaptiveSleeper::maybe_wake_proxy_thread() {
  if (!is_adaptive_sleep_) {
    return;
  }
  int ret = eventfd_write(work_eventfd_, kWakeEventConst);
  UCCL_CHECK(ret == 0);
}

void EPAdaptiveSleeper::update_timer() {
  if (!is_adaptive_sleep_) {
    return;
  }
  UCCL_LOG(INFO, UCCL_EP) << "Sleep timer updated";
  last_event_time_ = std::chrono::steady_clock::now();
}