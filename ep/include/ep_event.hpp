#pragma once

#include "ep_util.hpp"
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <cuda_runtime.h>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// On ROCm, destroying a hipEvent immediately after hipStreamWaitEvent can
// release the underlying HSA signal before the hardware queue consumes the
// wait, silently dropping the cross-queue dependency.  This only manifests
// when streams land on different HW queues (e.g. GPU_MAX_HW_QUEUES > 2).
//
// Mitigation: keep a bounded FIFO of cudaEvent_t handles so destruction is
// deferred until many subsequent stream_wait calls have been issued, giving
// the hardware time to consume the waits.  A UCCL_EP_STREAM_WAIT_MODE=sync
// fallback that uses hipStreamSynchronize is also provided for A/B testing.
struct DeferredEventQueue {
  static constexpr size_t kCapacity = 128;
  std::mutex mu;
  std::deque<cudaEvent_t> q;

  void push(cudaEvent_t ev) {
    cudaEvent_t to_destroy = nullptr;
    {
      std::lock_guard<std::mutex> lk(mu);
      q.push_back(ev);
      if (q.size() > kCapacity) {
        to_destroy = q.front();
        q.pop_front();
      }
    }
    if (to_destroy != nullptr) {
      (void)cudaEventDestroy(to_destroy);
    }
  }

  ~DeferredEventQueue() {
    std::lock_guard<std::mutex> lk(mu);
    for (auto ev : q) {
      (void)cudaEventDestroy(ev);
    }
    q.clear();
  }
};

inline DeferredEventQueue& deferred_events() {
  static DeferredEventQueue inst;
  return inst;
}

inline int stream_wait_mode() {
  // 0 = deferred-event (default), 1 = stream-synchronize
  static int mode = -1;
  if (mode < 0) {
    auto* env = std::getenv("UCCL_EP_STREAM_WAIT_MODE");
    if (env) {
      auto s = std::string(env);
      if (s == "sync")
        mode = 1;
      else
        mode = 0;
    } else {
      mode = 0;
    }
  }
  return mode;
}
#endif

struct EventHandle {
  std::shared_ptr<cudaEvent_t> event;

  static std::shared_ptr<cudaEvent_t> make_recorded(cudaStream_t stream) {
    auto deleter = [](cudaEvent_t* ev_ptr) {
      if (ev_ptr == nullptr) return;
      if (*ev_ptr != nullptr) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        // Defer destruction so any in-flight hipStreamWaitEvent that
        // references this event has time to be processed by the hardware
        // queue before the HSA signal is released.  See comment above.
        deferred_events().push(*ev_ptr);
#else
        cudaEventDestroy(*ev_ptr);
#endif
      }
      delete ev_ptr;
    };

    auto ev_ptr = std::shared_ptr<cudaEvent_t>(new cudaEvent_t{}, deleter);
    CUDA_CHECK(cudaEventCreateWithFlags(ev_ptr.get(), cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(*ev_ptr, stream));
    return ev_ptr;
  }

  EventHandle() : event(make_recorded(nullptr)) {}
  explicit EventHandle(cudaStream_t stream) : event(make_recorded(stream)) {}

  EventHandle(EventHandle const& other) = default;

  void current_stream_wait(std::uintptr_t stream_ptr) const {
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    CUDA_CHECK(cudaStreamWaitEvent(stream, *event, 0));
  }
};

inline void stream_wait(cudaStream_t dst, cudaStream_t src) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (stream_wait_mode() == 1) {
    CUDA_CHECK(cudaStreamSynchronize(src));
    return;
  }
  // Deferred-event path: create, record, wait, then park the event in a
  // FIFO so cudaEventDestroy is deferred until ~kCapacity subsequent calls
  // have been issued.  This is required on ROCm when GPU_MAX_HW_QUEUES > 2
  // causes `dst` and `src` to land on different HW queues.
  cudaEvent_t ev{};
  CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(ev, src));
  CUDA_CHECK(cudaStreamWaitEvent(dst, ev, 0));
  deferred_events().push(ev);
#else
  cudaEvent_t ev{};
  CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(ev, src));
  CUDA_CHECK(cudaStreamWaitEvent(dst, ev, 0));
  CUDA_CHECK(cudaEventDestroy(ev));
#endif
}

inline void stream_wait(cudaStream_t stream, EventHandle const& event) {
  if (event.event) CUDA_CHECK(cudaStreamWaitEvent(stream, *event.event, 0));
}
