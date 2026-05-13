#ifndef ADAPTIVE_SLEEPER_HPP
#define ADAPTIVE_SLEEPER_HPP

#include "proxy_ctx.hpp"
#include "util/debug.h"
#include <chrono>
#include <ctime>
#include <string_view>
#include <sys/eventfd.h>
#include <sys/poll.h>

// handles level of sleep on the proxy thread, based on RDMA request volume
// Adaptive sleeper states:
// 1. POLL = no delay at all
// 2. SLEEP = put the CPU to sleep, while letting it poll on GPU initiated
// events and the completion events queue. This happens when there has been no
// work for >= kNoActivityDuration
class AdaptiveSleeper {
 public:
  enum SleepState { POLL = 0, SLEEP };

  AdaptiveSleeper()
      : state_(POLL),
        last_event_time_(std::chrono::steady_clock::time_point::min()) {
    work_eventfd_ = eventfd(0, EFD_NONBLOCK);
  }

  ~AdaptiveSleeper() { close(work_eventfd_); }

  // decide whether or not to put the CPU to sleep based on its current
  void maybe_sleep(ProxyCtx& proxy_ctx) {
    int ret;

    if (std::chrono::steady_clock::now() - last_event_time_ >=
            kNoActivityThreshold &&
        last_event_time_ != std::chrono::steady_clock::time_point::min()) {
      UCCL_LOG(INFO, UCCL_EP) << "No activity detected for the last 30 "
                                 "seconds, putting proxy to sleep";

      state_ = SLEEP;

      // TODO: do I have to clear anything from the completion channel if I
      // did not ask for anything?
      // TODO: should I check for all types of notifications (solicited only)?
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

      int n = ppoll(events_to_poll, kNumActivitiesToPoll, &kPollSleepDuration,
                    NULL);
      UCCL_PCHECK(n != -1);

      if (n > 0) {
        // handle the necessary acknowledgements for the file descriptors
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
          // TODO: need to dig into what each of these events does
          void* ctx;
          ibv_get_cq_event(proxy_ctx.comp_channel, &proxy_ctx.cq, &ctx);
          // acknowledge the event to clear the notification
          // TODO: what happens If I dont ack
          // TODO: what happens if I ack more than I get?
          ibv_ack_cq_events(proxy_ctx.cq, 1);
        }

        // TODO: there is some small bug that stops the second pass of bench
        // kineto from running as quickly as it normally would does not affect
        // correctness though
        last_event_time_ = std::chrono::steady_clock::now();
        state_ = POLL;
      } else if (n == 0) {
        UCCL_LOG(INFO, UCCL_EP)
            << "Proxy thread woke due to poll timeout, sleeping again";
      }
    }
  }

  void maybe_wake_proxy_thread() {
    int ret = eventfd_write(work_eventfd_, kWakeEventConst);
    UCCL_CHECK(ret == 0);
  }

  void init_timer() { last_event_time_ = std::chrono::steady_clock::now(); }

  std::string_view get_sleep_state() {
    switch (state_) {
      case (POLL):
        return "POLL";
      case (SLEEP):
        return "SLEEP";
    }
  }

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