// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "fifo.hpp"
#include "fifo_util.hpp"
#include <numaif.h>

namespace mscclpp {

struct Fifo::Impl {
  detail::UniqueGpuHostPtr<ProxyTrigger> triggers;
  detail::UniqueGpuPtr<uint64_t> head;
  detail::UniqueGpuHostPtr<uint64_t> tail;
  detail::UniqueGpuPtr<uint64_t> tailCache;
  int const size;

  Impl(int size)
      : triggers(detail::gpuCallocHostUnique<ProxyTrigger>(size)),
        head(detail::gpuCallocUnique<uint64_t>()),
        tail(detail::gpuCallocHostUnique<uint64_t>()),
        tailCache(detail::gpuCallocUnique<uint64_t>()),
        size(size) {}
};

Fifo::Fifo(int size) {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  int numaNode = getDeviceNumaNode(device);
  unsigned long nodemask = 1UL << numaNode;
  if (set_mempolicy(MPOL_PREFERRED, &nodemask, 8 * sizeof(nodemask)) != 0) {
    throw std::runtime_error(
        "Failed to set mempolicy device: " + std::to_string(device) +
        " numaNode: " + std::to_string(numaNode));
  }
  pimpl_ = std::make_unique<Impl>(size);
}

Fifo::~Fifo() = default;

ProxyTrigger Fifo::poll() {
  ProxyTrigger trigger;
  ProxyTrigger* ptr = &pimpl_->triggers.get()[*(pimpl_->tail) % pimpl_->size];
  // we are loading fst first. if fst is non-zero then snd is also valid
  trigger.fst = atomicLoad(&(ptr->fst), memoryOrderAcquire);
  trigger.snd = ptr->snd;
  return trigger;
}

void Fifo::pop() {
  uint64_t curTail = *(pimpl_->tail);
  pimpl_->triggers.get()[curTail % pimpl_->size].fst = 0;
  atomicStore(pimpl_->tail.get(), curTail + 1, memoryOrderRelease);
}

int Fifo::size() const { return pimpl_->size; }

FifoDeviceHandle Fifo::deviceHandle() const {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl_->triggers.get();
  deviceHandle.head = pimpl_->head.get();
  deviceHandle.tail = pimpl_->tail.get();
  deviceHandle.tailCache = pimpl_->tailCache.get();
  deviceHandle.size = pimpl_->size;
  return deviceHandle;
}

}  // namespace mscclpp
