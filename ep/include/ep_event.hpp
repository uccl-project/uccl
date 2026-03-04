#pragma once

#include "ep_util.hpp"
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <memory>
#include <cuda_runtime.h>

struct EventHandle {
  std::shared_ptr<cudaEvent_t> event;

  static std::shared_ptr<cudaEvent_t> make_recorded(
      c10::cuda::CUDAStream const& stream) {
    auto deleter = [](cudaEvent_t* ev_ptr) {
      if (ev_ptr == nullptr) return;
      if (*ev_ptr != nullptr) cudaEventDestroy(*ev_ptr);
      delete ev_ptr;
    };

    auto ev_ptr = std::shared_ptr<cudaEvent_t>(new cudaEvent_t{}, deleter);
    CUDA_CHECK(cudaEventCreateWithFlags(ev_ptr.get(), cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(*ev_ptr, stream.stream()));
    return ev_ptr;
  }

  EventHandle() : event(make_recorded(c10::cuda::getCurrentCUDAStream())) {}

  explicit EventHandle(c10::cuda::CUDAStream const& stream)
      : event(make_recorded(stream)) {}
  explicit EventHandle(cudaStream_t stream) {
    auto deleter = [](cudaEvent_t* ev_ptr) {
      if (ev_ptr == nullptr) return;
      if (*ev_ptr != nullptr) cudaEventDestroy(*ev_ptr);
      delete ev_ptr;
    };

    auto ev_ptr = std::shared_ptr<cudaEvent_t>(new cudaEvent_t{}, deleter);
    CUDA_CHECK(cudaEventCreateWithFlags(ev_ptr.get(), cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(*ev_ptr, stream));
    event = ev_ptr;
  }

  EventHandle(EventHandle const& other) = default;

  void current_stream_wait() const {
    CUDA_CHECK(cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream().stream(),
                                   *event, 0));
  }
};

inline void stream_wait(c10::cuda::CUDAStream const& s_0,
                        c10::cuda::CUDAStream const& s_1) {
  EP_HOST_ASSERT(s_0.id() != s_1.id());
  cudaEvent_t ev{};
  CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(ev, s_1.stream()));
  CUDA_CHECK(cudaStreamWaitEvent(s_0.stream(), ev, 0));
  CUDA_CHECK(cudaEventDestroy(ev));
}

inline void stream_wait(c10::cuda::CUDAStream const& s,
                        EventHandle const& event) {
  CUDA_CHECK(cudaStreamWaitEvent(s.stream(), *event.event, 0));
}

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
