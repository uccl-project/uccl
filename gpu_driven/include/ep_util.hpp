#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <string>
#include <cuda_runtime.h>

class EPException : public std::exception {
 private:
  std::string message = {};

 public:
  explicit EPException(char const* name, char const* file, int const line,
                       std::string const& error) {
    message = std::string("Failed: ") + name + " error " + file + ":" +
              std::to_string(line) + " '" + error + "'";
  }

  char const* what() const noexcept override { return message.c_str(); }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                     \
  do {                                                                      \
    cudaError_t e = (cmd);                                                  \
    if (e != cudaSuccess) {                                                 \
      throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                       \
  } while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                     \
  do {                                                           \
    if (not(cond)) {                                             \
      throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    }                                                            \
  } while (0)
#endif

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
