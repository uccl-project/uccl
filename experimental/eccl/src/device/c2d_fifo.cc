#include "c2d_fifo.h"
#include "fifo_util.hpp"
#include "gpu_rt.h"
#include <iostream>
#include <thread>
#include <numaif.h>

namespace mscclpp {

template class CpuToGpuFifo<eccl::Task>;

template <typename T>
CpuToGpuFifo<T>::CpuToGpuFifo(int size) {
  int device;
  MSCCLPP_CUDATHROW(gpuGetDevice(&device));
  MSCCLPP_CUDATHROW(gpuFree(0));  // force init context for current GPU
  gpuDeviceProp deviceProp;
  MSCCLPP_CUDATHROW(gpuGetDeviceProperties(&deviceProp, device));
  std::cout << "Init CpuToGpuFifo at device " << deviceProp.name << " !"
            << std::endl;
  int numaNode = getDeviceNumaNode(device);
  unsigned long nodemask = 1UL << numaNode;
  if (set_mempolicy(MPOL_PREFERRED, &nodemask, 8 * sizeof(nodemask)) != 0) {
    throw std::runtime_error(
        "Failed to set mempolicy device: " + std::to_string(device) +
        " numaNode: " + std::to_string(numaNode));
  }
  pimpl_ = std::make_unique<Impl>(size);

  if (pimpl_->buffer.get() == nullptr) {
    std::cerr << "Error: Buffer allocation failed!" << std::endl;
    exit(1);
  }
  if (pimpl_->head.get() == nullptr) {
    std::cerr << "Error: Head allocation failed!" << std::endl;
    exit(1);
  }
  if (pimpl_->tail.get() == nullptr) {
    std::cerr << "Error: Tail allocation failed!" << std::endl;
    exit(1);
  }
}

template <typename T>
uint64_t CpuToGpuFifo<T>::push(const T& task) {
  uint64_t curHead = *(pimpl_->head);
  T* devBuffer = pimpl_->buffer.get();

  // Copy single task to device
  MSCCLPP_CUDATHROW(gpuMemcpy(&devBuffer[curHead % pimpl_->size], &task,
                              sizeof(T), gpuMemcpyHostToDevice));

  // Ensure data is visible before publishing head
  std::atomic_thread_fence(std::memory_order_release);

  // Publish the task
  atomicStore(pimpl_->head.get(), curHead + 1, memoryOrderRelease);

  return curHead;
}

template <typename T>
template <typename InputIt>
uint64_t CpuToGpuFifo<T>::push(InputIt first, InputIt last) {
  using VT = typename std::iterator_traits<InputIt>::value_type;
  static_assert(std::is_same_v<VT, T>, "Iterator value_type must be T");

  if (first == last) return *pimpl_->head;

  size_t count = std::distance(first, last);
  if (count > static_cast<size_t>(pimpl_->size)) {
    throw std::length_error("Batch exceeds FIFO capacity");
  }

  uint64_t curHead = atomicLoad(pimpl_->head.get(), memoryOrderRelaxed);
  T* devBuf = pimpl_->buffer.get();
  int size = pimpl_->size;

  size_t start = curHead % size;
  size_t firstPart = std::min(count, size - start);
  size_t secondPart = count - firstPart;

  if (firstPart > 0) {
    MSCCLPP_CUDATHROW(gpuMemcpy(devBuf + start, &*first, sizeof(T) * firstPart,
                                gpuMemcpyHostToDevice));
  }
  if (secondPart > 0) {
    MSCCLPP_CUDATHROW(gpuMemcpy(devBuf, &*(first + firstPart),
                                sizeof(T) * secondPart, gpuMemcpyHostToDevice));
  }

  //   __sync_synchronize();
  std::atomic_thread_fence(std::memory_order_release);
  atomicStore(pimpl_->head.get(), curHead + count, memoryOrderRelease);

  return curHead;
}

template <typename T>
uint64_t CpuToGpuFifo<T>::currentId() const {
  // Load tail with acquire to see latest GPU updates
  uint64_t* tail_host = detail::getGdrHostPtr(pimpl_->tail);
  uint64_t currentTail = atomicLoad(tail_host, memoryOrderAcquire);

  // Check if the task with taskId has been consumed, return current Tail
  return currentTail;
}

template <typename T>
void CpuToGpuFifo<T>::sync(uint64_t taskId) const {
  while (currentId() <= taskId) {
    std::this_thread::yield();
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

template <typename T>
C2DDeviceHandle<T> CpuToGpuFifo<T>::deviceHandle() const {
  C2DDeviceHandle<T> h;
  h.buffer = pimpl_->buffer.get();
  h.head = pimpl_->head.get();
  h.tail = pimpl_->tail.get();
  h.size = pimpl_->size;
  return h;
}

}  // namespace mscclpp