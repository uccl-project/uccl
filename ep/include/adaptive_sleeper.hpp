#ifndef ADAPTIVE_SLEEPER_HPP
#define ADAPTIVE_SLEEPER_HPP

#include <sys/eventfd.h>
#include <sys/poll.h>
#include <chrono>
#include <ctime>
#include <string_view>
#include "util/debug.h"
#include "proxy_ctx.hpp"

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

  // decide whether or not to put the CPU to sleep based on its current
  void maybe_sleep(ProxyCtx& proxy_ctx) {
    int ret;

    if (state_ == POLL &&
        std::chrono::steady_clock::now() - last_event_time_ >=
            kNoActivityThreshold &&
        last_event_time_ != std::chrono::steady_clock::time_point::min()) {
      UCCL_LOG(INFO, UCCL_EP) << "No activity detected for the last 30 "
                                 "seconds, putting proxy to sleep";

      state_ = SLEEP;

      // clear all pending notifications from the notification channels and arm
      // them
      eventfd_t event_fd_buffer;
      ret = eventfd_read(work_eventfd_, &event_fd_buffer);
      UCCL_CHECK(ret == 0);

      // TODO: do I have to clear anything from the completion channel if I did
      // not ask for anything?
      // TODO: should I check for all types of notifications (solicited only)?
      ret = ibv_req_notify_cq(proxy_ctx.cq, 0);
      UCCL_CHECK(ret == 0);
    }

    if (state_ == POLL) {
      return;
    } else if (state_ == SLEEP) {
      UCCL_LOG(INFO, UCCL_EP) << "Sleeping proxy thread";

      // continuously sleep while there are no new work entries
      while (true) {
        struct pollfd events_to_poll[kNumActivitiesToPoll] = {
            {.fd = work_eventfd_, .events = POLLIN},
            {.fd = proxy_ctx.comp_channel->fd, .events = POLLIN}};

        int n = ppoll(events_to_poll, kNumActivitiesToPoll, &kPollSleepDuration,
                      NULL);
        if (n > 0) {
          UCCL_LOG(INFO, UCCL_EP) << "Detected new event, waking proxy thread";

          // handle the necessary acknowledgements for the file descriptors
          if (events_to_poll[0].revents == POLLIN) {
            eventfd_t work_value;
            ret = eventfd_read(work_eventfd_, &work_value);
            UCCL_CHECK(ret == 0 && work_value == kWakeEventConst);
          } else if (events_to_poll[1].revents == POLLIN) {
            // TODO: need to dig into what each of these events does
            void* ctx;
            ibv_get_cq_event(proxy_ctx.comp_channel, &proxy_ctx.cq, &ctx);
            // acknowledge the event to clear the notification
            // TODO: what happens If I dont ack
            // TODO: what happens if I ack more than I get?
            ibv_ack_cq_events(proxy_ctx.cq, 1);
          }

          state_ = POLL;
          return;
        } else if (n == 0) {
          UCCL_LOG(INFO, UCCL_EP)
              << "Proxy thread woke due to poll timeout, sleeping again";
        } else {
          UCCL_LOG(FATAL) << "Encountered ppoll error";
        }
      }
    }
  }

  void maybe_wake_proxy_thread() {
    // TODO: what happens if you write multiple times to the same eventfd: will
    // the value just overwrite?
    int ret = eventfd_write(work_eventfd_, kWakeEventConst);
    UCCL_CHECK(ret == 0);

    last_event_time_ = std::chrono::steady_clock::now();
  }

  std::string_view get_sleep_state() {
    switch (state_) {
      case (POLL):
        return "POLL";
      case (SLEEP):
        return "SLEEP";
    }
  }

 private:
  static constexpr auto kNoActivityThreshold = std::chrono::seconds(30);
  static constexpr int kNumActivitiesToPoll = 2;
  static constexpr struct timespec kPollSleepDuration = {.tv_sec = 5};
  static constexpr int kWakeEventConst = 0x42;

  SleepState state_;
  // used by python API to inform proxy threads of new work entry
  int work_eventfd_;
  std::chrono::steady_clock::time_point last_event_time_;
};

#endif