#pragma once

#include "ep_util.hpp"
#include <cstdint>
#include <memory>
#include <cuda_runtime.h>

struct EventHandle {
  std::shared_ptr<cudaEvent_t> event;

  static std::shared_ptr<cudaEvent_t> make_recorded(cudaStream_t stream) {
    auto deleter = [](cudaEvent_t* ev_ptr) {
      if (ev_ptr == nullptr) return;
      if (*ev_ptr != nullptr) cudaEventDestroy(*ev_ptr);
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
  cudaEvent_t ev{};
  CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(ev, src));
  CUDA_CHECK(cudaStreamWaitEvent(dst, ev, 0));
  CUDA_CHECK(cudaEventDestroy(ev));
}

inline void stream_wait(cudaStream_t stream, EventHandle const& event) {
  CUDA_CHECK(cudaStreamWaitEvent(stream, *event.event, 0));
}
