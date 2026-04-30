#include "bounce_buffer_pool.h"
#include <iostream>

namespace UKernel {
namespace Transport {

BounceBufferPool::BounceBufferPool(SHMManager& shm, MRManager& mr,
                                   bool needs_uccl_mr,
                                   std::atomic<uint32_t>& ephemeral_id_gen)
    : shm_(shm),
      mr_(mr),
      needs_uccl_mr_(needs_uccl_mr),
      ephemeral_id_gen_(ephemeral_id_gen) {}

BounceBufferPool::~BounceBufferPool() { shutdown(); }

BounceBufferPool::Slot* BounceBufferPool::acquire(size_t min_bytes) {
  std::lock_guard<std::mutex> lk(mu_);

  // Best-fit: smallest free buffer ≥ min_bytes.
  auto it = free_slots_.lower_bound(min_bytes);
  if (it != free_slots_.end()) {
    auto slot = std::move(it->second);
    free_slots_.erase(it);
    Slot* ptr = slot.get();
    all_slots_.push_back(std::move(slot));
    return ptr;
  }

  // No suitable free buffer — create one.
  auto slot = std::make_unique<Slot>();
  SHMItem item = shm_.create_local_shm(min_bytes, /*shareable=*/false);
  if (!item.valid) {
    std::cerr << "[ERROR] BounceBufferPool: failed to allocate SHM buffer of "
              << min_bytes << " bytes" << std::endl;
    return nullptr;
  }

  slot->ptr = item.ptr;
  slot->bytes = item.bytes;
  slot->shm_id = item.shm_id;
  slot->shm_name = item.shm_name;

  if (needs_uccl_mr_) {
    uint32_t bid =
        ephemeral_id_gen_.fetch_add(1, std::memory_order_relaxed);
    if (bid == 0)
      bid = ephemeral_id_gen_.fetch_add(1, std::memory_order_relaxed);
    mr_.create_local_mr(bid, slot->ptr, slot->bytes);
    slot->buffer_id = bid;
  }

  Slot* ptr = slot.get();
  all_slots_.push_back(std::move(slot));
  return ptr;
}

void BounceBufferPool::release(Slot* slot) {
  if (slot == nullptr) return;
  std::lock_guard<std::mutex> lk(mu_);
  // Find the slot in all_slots_ and move it to free_slots_.
  for (auto it = all_slots_.begin(); it != all_slots_.end(); ++it) {
    if (it->get() == slot) {
      size_t sz = slot->bytes;
      auto owned = std::move(*it);
      all_slots_.erase(it);
      free_slots_.emplace(sz, std::move(owned));
      return;
    }
  }
}

void BounceBufferPool::shutdown() {
  std::lock_guard<std::mutex> lk(mu_);
  // Move free slots back into all_slots_ so we can clean them up uniformly.
  for (auto& [sz, slot] : free_slots_) {
    all_slots_.push_back(std::move(slot));
  }
  free_slots_.clear();

  for (auto& slot : all_slots_) {
    if (slot->ptr == nullptr) continue;
    if (slot->buffer_id != 0) mr_.delete_mr(slot->ptr);
    if (slot->shm_id != 0) shm_.delete_local_shm(slot->shm_id);
    slot->ptr = nullptr;
  }
  all_slots_.clear();
}

}  // namespace Transport
}  // namespace UKernel
