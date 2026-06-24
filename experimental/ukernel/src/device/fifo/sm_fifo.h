
#pragma once

#include "sm_fifo_device.h"
#include "task.h"

namespace mscclpp {

template <typename T>
class SmFifo {
 public:
  explicit SmFifo(int capacity = 512);
  ~SmFifo() = default;

  /// Return device handle
  SmDeviceHandle<T> deviceHandle() const;

  int size() const { return pimpl_->capacity; }

 private:
  struct Impl {
    detail::UniqueGpuPtr<T> buffer;  // device ring
    detail::UniqueGpuPtr<uint64_t> head_reserve;
    detail::UniqueGpuPtr<uint64_t> head_publish;
    detail::UniqueGpuPtr<uint64_t> tail;

    int const capacity;

    Impl(int cap)
        : buffer(detail::gpuCallocUnique<T>(cap)),
          head_reserve(detail::gpuCallocUnique<uint64_t>()),
          head_publish(detail::gpuCallocUnique<uint64_t>()),
          tail(detail::gpuCallocUnique<uint64_t>()),
          capacity(cap) {}
  };

  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp