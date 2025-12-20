#pragma once

#include <memory>
#include <cstddef>
#include <iterator>
#include "c2d_fifo_device.h"
#include "task.h"

namespace mscclpp {

template <typename T>
class CpuToGpuFifo {
 public:
  explicit CpuToGpuFifo(int size = 512);
  ~CpuToGpuFifo() = default;

  /// Push a single task from CPU, return task id
  uint64_t push(const T& task);

  /// Push a range of tasks [first, last) from CPU.
  template <typename InputIt>
  uint64_t push(InputIt first, InputIt last);

  /// Poll whether a specific Task is popped from the FIFO.
  bool poll(uint64_t taskId) const;

  /// Wait until a specific Task is popped from the FIFO.
  void sync(uint64_t taskId) const;

  /// Get device handle for GPU kernels.
  C2DDeviceHandle<T> deviceHandle() const;

  /// Get FIFO capacity (number of entries).
  int size() const { return pimpl_->size; }

 private:
  struct Impl {
    detail::UniqueGpuPtr<T> buffer;              // device
    detail::UniqueGpuHostPtr<uint64_t> head;     // host-pinned
    detail::UniqueGpuPtr<uint64_t> tail;         // device，会被很多线程读，所以放到显存好。
    int const size;

    Impl(int size)
        : buffer(detail::gpuCallocUnique<T>(size)),
          head(detail::gpuCallocHostUnique<uint64_t>()),
          tail(detail::gpuCallocUnique<uint64_t>()),
          size(size) {}
  };
  std::unique_ptr<Impl> pimpl_;
};
template class CpuToGpuFifo<eccl::OpTask>;

}  // namespace mscclpp
