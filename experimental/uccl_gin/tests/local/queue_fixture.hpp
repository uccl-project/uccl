#pragma once

#include "transport/d2h_queue_host.hpp"
#include "uccl_gin/uccl_gin.cuh"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace local_test {

inline void require(bool condition, char const* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    auto const error = (call);                                           \
    if (error != cudaSuccess) {                                          \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(error));                           \
      std::exit(1);                                                      \
    }                                                                    \
  } while (false)

inline char const* option(int argc, char** argv, char const* name,
                          char const* fallback) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      require(i + 1 < argc, "missing option value");
      return argv[i + 1];
    }
  }
  return fallback;
}

inline int integer(int argc, char** argv, char const* name, int fallback) {
  char const* text = option(argc, argv, name, nullptr);
  if (text == nullptr) return fallback;
  char* end;
  long const value = std::strtol(text, &end, 10);
  require(*end == '\0' && value >= 0 && value <= INT_MAX,
          "expected a nonnegative integer");
  return static_cast<int>(value);
}

// Uses the production FIFO and decoder. Only command completion is supplied
// by the test consumer; no Context, communicator, or network is initialized.
class QueueFixture {
 public:
  int const device;
  std::vector<std::unique_ptr<mscclpp::Fifo>> queues;
  uccl_gin::UCCLGinResources resources;
  cudaStream_t stream;
  std::vector<uint64_t> counts;

  QueueFixture(int device, int queue_count, int capacity) : device(device) {
    require(queue_count > 0 && capacity > 0, "empty queue configuration");
    CUDA_CHECK(cudaSetDevice(device));
    std::vector<d2hq::D2HHandle> handles(queue_count);
    for (int i = 0; i < queue_count; ++i) {
      queues.push_back(std::make_unique<mscclpp::Fifo>(capacity));
      handles[i].init_from_host_value(queues.back()->deviceHandle());
    }
    device_handles =
        mscclpp::detail::gpuCallocUnique<d2hq::D2HHandle>(queue_count);
    device_pointers =
        mscclpp::detail::gpuCallocUnique<d2hq::D2HHandle*>(queue_count);
    std::vector<d2hq::D2HHandle*> pointers(queue_count);
    for (int i = 0; i < queue_count; ++i)
      pointers[i] = device_handles.get() + i;
    CUDA_CHECK(cudaMemcpy(device_handles.get(), handles.data(),
                          handles.size() * sizeof(handles[0]),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_pointers.get(), pointers.data(),
                          pointers.size() * sizeof(pointers[0]),
                          cudaMemcpyHostToDevice));
    resources.d2h_queues = device_pointers.get();
    resources.num_queues = queue_count;
    counts.resize(queue_count);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  template <typename Consumer>
  void start(Consumer consume) {
    done.store(false, std::memory_order_relaxed);
    ready.store(false, std::memory_order_relaxed);
    worker = std::thread([this, consume] {
      CUDA_CHECK(cudaSetDevice(device));
      auto const deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(60);
      ready.store(true, std::memory_order_release);
      for (;;) {
        bool progress = false;
        for (size_t i = 0; i < queues.size(); ++i) {
          auto const trigger = queues[i]->poll();
          if (trigger.fst == 0) continue;
          consume(i, d2hq::decode_from_trigger(trigger));
          queues[i]->pop();
          ++counts[i];
          progress = true;
        }
        if (!progress && done.load(std::memory_order_acquire)) break;
        if (std::chrono::steady_clock::now() > deadline) {
          std::fprintf(stderr, "FAIL: device=%d consumer deadline exceeded\n",
                       device);
          std::exit(1);
        }
        if (!progress) std::this_thread::yield();
      }
    });
    while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
  }

  void finish() {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    done.store(true, std::memory_order_release);
    worker.join();
  }

  ~QueueFixture() { CUDA_CHECK(cudaStreamDestroy(stream)); }

 private:
  mscclpp::detail::UniqueGpuPtr<d2hq::D2HHandle> device_handles;
  mscclpp::detail::UniqueGpuPtr<d2hq::D2HHandle*> device_pointers;
  std::atomic<bool> done{false}, ready{false};
  std::thread worker;
};

}  // namespace local_test
