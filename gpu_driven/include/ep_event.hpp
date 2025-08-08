#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

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

inline void stream_wait(at::cuda::CUDAStream const& s_0,
                        at::cuda::CUDAStream const& s_1) {
  EP_HOST_ASSERT(s_0.id() != s_1.id());
  s_0.unwrap().wait(create_event(s_1));
}

inline void stream_wait(at::cuda::CUDAStream const& s,
                        EventHandle const& event) {
  s.unwrap().wait(*event.event);
}
