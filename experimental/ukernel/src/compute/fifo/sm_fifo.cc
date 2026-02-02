#include "sm_fifo.h"
#include "gpu_rt.h"
#include <iostream>
#include <numaif.h>

namespace mscclpp {

template class SmFifo<UKernel::Compute::Task>;

template <typename T>
SmFifo<T>::SmFifo(int capacity) {
  int device;
  MSCCLPP_CUDATHROW(gpuGetDevice(&device));
  MSCCLPP_CUDATHROW(gpuFree(0));  // force init context for current GPU
  gpuDeviceProp deviceProp;
  MSCCLPP_CUDATHROW(gpuGetDeviceProperties(&deviceProp, device));
  std::cout << "Init SmFifo at device " << deviceProp.name << " !" << std::endl;
  int numaNode = getDeviceNumaNode(device);
  unsigned long nodemask = 1UL << numaNode;
  if (set_mempolicy(MPOL_PREFERRED, &nodemask, 8 * sizeof(nodemask)) != 0) {
    throw std::runtime_error(
        "Failed to set mempolicy device: " + std::to_string(device) +
        " numaNode: " + std::to_string(numaNode));
  }
  pimpl_ = std::make_unique<Impl>(capacity);

  if (pimpl_->buffer.get() == nullptr) {
    std::cerr << "Error: Buffer allocation failed!" << std::endl;
    exit(1);
  }
  if (pimpl_->head_reserve.get() == nullptr ||
      pimpl_->head_publish.get() == nullptr) {
    std::cerr << "Error: Head allocation failed!" << std::endl;
    exit(1);
  }
  if (pimpl_->tail.get() == nullptr) {
    std::cerr << "Error: Tail allocation failed!" << std::endl;
    exit(1);
  }
}

template <typename T>
SmDeviceHandle<T> SmFifo<T>::deviceHandle() const {
  SmDeviceHandle<T> h;

  h.buffer = pimpl_->buffer.get();
  h.head_reserve = pimpl_->head_reserve.get();
  h.head_publish = pimpl_->head_publish.get();
  h.tail = pimpl_->tail.get();
  h.size = pimpl_->capacity;

  return h;
}

}  // namespace mscclpp