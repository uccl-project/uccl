#include "rdma_fused_proxy.h"

#include "../../device/fifo/fifo_util.hpp"
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

// Device-visible ring control block. The GPU writes index then releases
// ready=1. Host consumes in FIFO order by clearing ready and advancing tail.
struct RdmaFusedRing::DeviceHandle {
  uint64_t* indices;
  uint64_t* ready;
  uint64_t* head;  // device-owned producer counter
  int size;
};

struct RdmaFusedRing::Impl {
  std::mutex mu;
  mscclpp::detail::UniqueGpuHostPtr<uint64_t> indices;
  mscclpp::detail::UniqueGpuHostPtr<uint64_t> ready;
  mscclpp::detail::UniqueGpuPtr<uint64_t> head;
  // The handle struct itself must be host-pinned/mapped so the GPU can
  // dereference it to reach indices/ready/head.
  mscclpp::detail::UniqueGpuHostPtr<DeviceHandle> dev_ptr;
  size_t producer = 0;
  size_t tail = 0;
  size_t capacity = 0;

  explicit Impl(size_t cap)
      : indices(mscclpp::detail::gpuCallocHostUnique<uint64_t>(cap)),
        ready(mscclpp::detail::gpuCallocHostUnique<uint64_t>(cap)),
        head(mscclpp::detail::gpuCallocUnique<uint64_t>()),
        dev_ptr(mscclpp::detail::gpuCallocHostUnique<DeviceHandle>()),
        capacity(cap) {
    std::memset(indices.get(), 0, cap * sizeof(uint64_t));
    std::memset(ready.get(), 0, cap * sizeof(uint64_t));
    dev_ptr->indices = indices.get();
    dev_ptr->ready = ready.get();
    dev_ptr->head = head.get();
    dev_ptr->size = static_cast<int>(cap);
  }
};

RdmaFusedRing::RdmaFusedRing(size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}

RdmaFusedRing::~RdmaFusedRing() = default;

void* RdmaFusedRing::device_handle() {
  return impl_->dev_ptr.get();
}

bool RdmaFusedRing::pop(uint64_t& index) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  uint64_t* rp = impl_->ready.get() + (impl_->tail % impl_->capacity);
  if (*rp == 0) return false;
  index = impl_->indices.get()[impl_->tail % impl_->capacity];
  *rp = 0;
  impl_->tail++;
  return true;
}

void RdmaFusedRing::push_from_host(uint64_t index) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  if (impl_->producer - impl_->tail >= impl_->capacity) return;
  uint64_t* rp = impl_->ready.get() + (impl_->producer % impl_->capacity);
  if (*rp != 0) return;  // ring full: producer must wait
  impl_->indices.get()[impl_->producer % impl_->capacity] = index;
  *rp = 1;
  impl_->producer++;
}

struct RdmaFusedCmdPool::Impl {
  std::mutex mu;
  std::vector<FusedCmdSlot> slots;
  std::vector<bool> used;
  std::vector<uint64_t> free_list;

  explicit Impl(size_t cap) : slots(cap), used(cap, false) {
    for (uint64_t i = 0; i < cap; ++i) free_list.push_back(cap - 1 - i);
  }
};

RdmaFusedCmdPool::RdmaFusedCmdPool(size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}

RdmaFusedCmdPool::~RdmaFusedCmdPool() = default;

uint64_t RdmaFusedCmdPool::alloc(Cmd const& cmd, void* run, uint32_t op_idx,
                                 PutPath put_path) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  if (impl_->free_list.empty()) return std::numeric_limits<uint64_t>::max();
  uint64_t idx = impl_->free_list.back();
  impl_->free_list.pop_back();
  impl_->slots[idx].cmd = cmd;
  impl_->slots[idx].run = run;
  impl_->slots[idx].op_idx = op_idx;
  impl_->slots[idx].put_path = put_path;
  impl_->used[idx] = true;
  return idx;
}

FusedCmdSlot const& RdmaFusedCmdPool::get(uint64_t index) const {
  return impl_->slots[index];
}

void RdmaFusedCmdPool::release(uint64_t index) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  if (!impl_->used[index]) return;
  impl_->used[index] = false;
  impl_->free_list.push_back(index);
}

RdmaFusedProxy::RdmaFusedProxy(PostFn post_fn, size_t ring_capacity,
                               size_t pool_capacity)
    : ring_(ring_capacity), pool_(pool_capacity), post_fn_(std::move(post_fn)) {}

RdmaFusedProxy::~RdmaFusedProxy() = default;

size_t RdmaFusedProxy::progress() {
  size_t done = 0;
  uint64_t index;
  while (ring_.pop(index)) {
    if (post_fn_(index)) {
      pool_.release(index);
      ++done;
    }
  }
  return done;
}

}  // namespace CCL
}  // namespace UKernel
