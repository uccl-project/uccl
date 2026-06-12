#include "request_tracker.h"
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

namespace UKernel {
namespace Transport {

unsigned RequestTracker::make_request_id(uint32_t slot_idx, uint32_t generation) {
  uint32_t gen = generation & kGenerationMask;
  if (gen == 0) gen = 1;
  return (gen << kSlotBits) | (slot_idx & (kSlotCount - 1u));
}

uint32_t RequestTracker::request_slot_index(unsigned req_id) {
  return req_id & (kSlotCount - 1u);
}

uint32_t RequestTracker::request_generation(unsigned req_id) {
  return (req_id >> kSlotBits) & kGenerationMask;
}

RequestTracker::RequestTracker(CompleteBounceFn complete_bounce,
                               CleanupFn cleanup)
    : complete_bounce_(std::move(complete_bounce)),
      cleanup_(std::move(cleanup)) {
  slots_ = std::make_unique<TrackedRequest[]>(kSlotCount);
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    slots_[i].state.store(TrackedRequest::SlotState::Free,
                          std::memory_order_relaxed);
  }
}

RequestTracker::~RequestTracker() {
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    auto* slot = &slots_[i];
    if (slot->state.load(std::memory_order_acquire) ==
        TrackedRequest::SlotState::Free)
      continue;
    cleanup_(*slot);
    finish_release(slot->request_id);
  }
}

TrackedRequest* RequestTracker::allocate(unsigned* out_req_id) {
  uint32_t start =
      alloc_cursor_.fetch_add(1, std::memory_order_relaxed) & (kSlotCount - 1u);
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    uint32_t idx = (start + i) & (kSlotCount - 1u);
    TrackedRequest* slot = &slots_[idx];
    auto expected = TrackedRequest::SlotState::Free;
    if (!slot->state.compare_exchange_strong(
            expected, TrackedRequest::SlotState::Reserved,
            std::memory_order_acq_rel, std::memory_order_acquire))
      continue;
    uint32_t gen = (slot->generation.fetch_add(1, std::memory_order_acq_rel) + 1u) &
                    kGenerationMask;
    if (gen == 0) { gen = 1; slot->generation.store(gen, std::memory_order_release); }
    unsigned rid = make_request_id(idx, gen);
    slot->request_id = rid;
    slot->adapter_request_id = 0;
    slot->peer_rank = -1;
    slot->kind = PeerTransportKind::Unknown;
    *out_req_id = rid;
    return slot;
  }
  return nullptr;
}

TrackedRequest* RequestTracker::resolve(unsigned req_id) const {
  if (req_id == 0) return nullptr;
  uint32_t idx = request_slot_index(req_id);
  if (idx >= kSlotCount) return nullptr;
  TrackedRequest* slot = &slots_[idx];
  uint32_t gen = slot->generation.load(std::memory_order_acquire) & kGenerationMask;
  if (gen == 0 || gen != request_generation(req_id)) return nullptr;
  if (slot->state.load(std::memory_order_acquire) == TrackedRequest::SlotState::Free)
    return nullptr;
  return slot;
}

bool RequestTracker::activate(unsigned req_id, unsigned adapter_req_id,
                              int peer_rank, PeerTransportKind kind) {
  TrackedRequest* slot = resolve(req_id);
  if (!slot) return false;
  slot->adapter_request_id = adapter_req_id;
  slot->peer_rank = peer_rank;
  slot->kind = kind;
  slot->state.store(TrackedRequest::SlotState::InFlight,
                    std::memory_order_release);
  return true;
}

bool RequestTracker::poll(unsigned req) {
  TrackedRequest* slot = resolve(req);
  if (!slot) return true;
  auto state = slot->state.load(std::memory_order_acquire);
  return state == TrackedRequest::SlotState::Completed ||
         state == TrackedRequest::SlotState::Failed;
}

void RequestTracker::release(unsigned req) {
  TrackedRequest* slot = resolve(req);
  if (!slot) return;
  cleanup_(*slot);
  finish_release(req);
}

void RequestTracker::finish_release(unsigned req_id) {
  TrackedRequest* slot = resolve(req_id);
  if (!slot) return;
  slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
}

}  // namespace Transport
}  // namespace UKernel
