#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>

struct EventHandle {
  std::shared_ptr<torch::Event> event;

  EventHandle() {
    event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(at::cuda::getCurrentCUDAStream());
  }

  explicit EventHandle(at::cuda::CUDAStream const& stream) {
    event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(stream);
  }

  EventHandle(EventHandle const& other) = default;

  void current_stream_wait() const {
    at::cuda::getCurrentCUDAStream().unwrap().wait(*event);
  }
};

inline torch::Event create_event(at::cuda::CUDAStream const& s) {
  auto event = torch::Event(torch::kCUDA);
  event.record(s);
  return event;
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// On ROCm, destroying a hipEvent immediately after hipStreamWaitEvent can
// release the underlying HSA signal before the hardware queue consumes the
// wait, silently dropping the cross-queue dependency.  This only manifests
// when streams are mapped to different HW queues (GPU_MAX_HW_QUEUES > 2).
//
// We keep a bounded FIFO of torch::Event objects so that events are not
// destroyed until many subsequent stream_wait calls have been issued,
// giving the hardware time to process the waits.

struct DeferredEventQueue {
  static constexpr size_t kCapacity = 128;
  std::mutex mu;
  std::deque<torch::Event> q;

  void push(torch::Event ev) {
    std::lock_guard<std::mutex> lk(mu);
    q.push_back(std::move(ev));
    while (q.size() > kCapacity) q.pop_front();
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
      if (s == "sync") mode = 1;
      else             mode = 0;
    } else {
      mode = 0;
    }
  }
  return mode;
}
#endif

inline void stream_wait(at::cuda::CUDAStream const& s_0,
                        at::cuda::CUDAStream const& s_1) {
  EP_HOST_ASSERT(s_0.id() != s_1.id());
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (stream_wait_mode() == 1) {
    CUDA_CHECK(hipStreamSynchronize(s_1.stream()));
    return;
  }
  // Deferred-event path: create, record, wait, then park the event in a
  // FIFO so hipEventDestroy is deferred for ~128 subsequent calls.
  auto ev = create_event(s_1);
  s_0.unwrap().wait(ev);
  deferred_events().push(std::move(ev));
#else
  s_0.unwrap().wait(create_event(s_1));
#endif
}

inline void stream_wait(at::cuda::CUDAStream const& s,
                        EventHandle const& event) {
  s.unwrap().wait(*event.event);
}
