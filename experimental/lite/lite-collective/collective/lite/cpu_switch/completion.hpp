// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "cuda_utils.hpp"

#include <memory>
#include <utility>

namespace mscclpp::lite {

class Completion {
 public:
  Completion() = default;

  static Completion record(cudaStream_t stream) {
    cudaEvent_t rawEvent = nullptr;
    throwCudaError(
      cudaEventCreateWithFlags(&rawEvent, cudaEventDisableTiming),
      "CpuSwitch event creation");

    auto event = std::make_shared<Event>(rawEvent);
    throwCudaError(
      cudaEventRecord(event->value, stream),
      "CpuSwitch event record");
    return Completion(std::move(event));
  }

  bool immediate() const { return event_ == nullptr; }

  void wait() const {
    if (event_ != nullptr) {
      throwCudaError(cudaEventSynchronize(event_->value),
                     "CpuSwitch completion wait");
    }
  }

  bool ready() const {
    if (event_ == nullptr) return true;
    cudaError_t result = cudaEventQuery(event_->value);
    if (result == cudaSuccess) return true;
    if (result == cudaErrorNotReady) return false;
    throwCudaError(result, "CpuSwitch completion query");
    return false;
  }

 private:
  struct Event {
    explicit Event(cudaEvent_t event) : value(event) {}
    ~Event() {
      if (value != nullptr) cudaEventDestroy(value);
    }
    cudaEvent_t value = nullptr;
  };

  explicit Completion(std::shared_ptr<Event> event) : event_(std::move(event)) {}
  std::shared_ptr<Event> event_;
};

}  // namespace mscclpp::lite
