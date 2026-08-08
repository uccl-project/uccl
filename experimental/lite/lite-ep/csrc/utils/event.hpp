#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <deep_ep/common/exception.cuh>
#include <memory>

namespace deep_ep {

struct EventHandle {
  std::shared_ptr<torch::Event> event;
  std::vector<std::optional<torch::Tensor>> tensors_to_record;

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

static torch::Event create_event(at::cuda::CUDAStream const& s) {
  auto event = torch::Event(torch::kCUDA);
  event.record(s);
  return event;
}

static void stream_wait(at::cuda::CUDAStream const& s_0,
                        at::cuda::CUDAStream const& s_1) {
  EP_HOST_ASSERT(s_0.id() != s_1.id());
  s_0.unwrap().wait(create_event(s_1));
}

static void stream_wait(at::cuda::CUDAStream const& s,
                        EventHandle const& event) {
  s.unwrap().wait(*event.event);
}

}  // namespace deep_ep
