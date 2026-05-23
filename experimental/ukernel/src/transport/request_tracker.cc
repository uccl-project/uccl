#include "request_tracker.h"
#include "adapter/ipc_adapter.h"
#include "adapter/rdma_adapter.h"
#include "adapter/tcp_adapter.h"
#include "adapter/uccl_adapter.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace Transport {

// ── static helpers ──────────────────────────────────────────────────────────

unsigned RequestTracker::make_request_id(uint32_t slot_idx,
                                         uint32_t generation) {
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

// ── lifecycle ───────────────────────────────────────────────────────────────

RequestTracker::RequestTracker(UcclTransportAdapter* uccl,
                               TcpTransportAdapter* tcp, IpcAdapter* ipc,
                               RdmaTransportAdapter* rdma,
                               CompleteBounceFn complete_bounce,
                               CleanupFn cleanup)
    : uccl_(uccl),
      tcp_(tcp),
      ipc_(ipc),
      rdma_(rdma),
      complete_bounce_(std::move(complete_bounce)),
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
    auto state = slot->state.load(std::memory_order_acquire);
    if (state == TrackedRequest::SlotState::Free) continue;
    if (state == TrackedRequest::SlotState::InFlight) {
      (void)poll_one(slot->request_id, true);
    }
    TrackedRequest snapshot{};
    if (try_release(slot->request_id, &snapshot)) {
      cleanup_(snapshot);
      finish_release(snapshot.request_id);
    } else {
      slot->state.store(TrackedRequest::SlotState::Free,
                        std::memory_order_release);
    }
  }
}

// ── slot lifecycle ──────────────────────────────────────────────────────────

TrackedRequest* RequestTracker::allocate(unsigned* out_req_id) {
  if (out_req_id == nullptr) return nullptr;
  uint32_t start =
      alloc_cursor_.fetch_add(1, std::memory_order_relaxed) & (kSlotCount - 1u);
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    uint32_t idx = (start + i) & (kSlotCount - 1u);
    TrackedRequest* slot = &slots_[idx];
    auto expected = TrackedRequest::SlotState::Free;
    if (!slot->state.compare_exchange_strong(
            expected, TrackedRequest::SlotState::Reserved,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      continue;
    }
    uint32_t gen =
        (slot->generation.fetch_add(1, std::memory_order_acq_rel) + 1u) &
        kGenerationMask;
    if (gen == 0) {
      gen = 1;
      slot->generation.store(gen, std::memory_order_release);
    }
    unsigned rid = make_request_id(idx, gen);
    slot->request_id = rid;
    slot->adapter_request_id = 0;
    slot->peer_rank = -1;
    slot->kind = PeerTransportKind::Unknown;
    slot->needs_host_to_device_copy = false;
    slot->host_copy_submitted = false;
    slot->completion_buffer = nullptr;
    slot->completion_offset = 0;
    slot->completion_bytes = 0;
    slot->host_copy_event = nullptr;
    slot->bounce_ptr = nullptr;
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
  uint32_t gen =
      slot->generation.load(std::memory_order_acquire) & kGenerationMask;
  if (gen == 0 || gen != request_generation(req_id)) return nullptr;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state == TrackedRequest::SlotState::Free) return nullptr;
  return slot;
}

bool RequestTracker::activate(unsigned req_id, unsigned adapter_req_id,
                              int peer_rank, PeerTransportKind kind) {
  TrackedRequest* slot = resolve(req_id);
  if (slot == nullptr) return false;
  slot->adapter_request_id = adapter_req_id;
  slot->peer_rank = peer_rank;
  slot->kind = kind;
  slot->state.store(TrackedRequest::SlotState::InFlight,
                    std::memory_order_release);
  return true;
}

bool RequestTracker::try_release(unsigned req_id, TrackedRequest* snapshot) {
  TrackedRequest* slot = resolve(req_id);
  if (slot == nullptr) return false;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state != TrackedRequest::SlotState::Completed &&
      state != TrackedRequest::SlotState::Failed) {
    return false;
  }
  if (!slot->state.compare_exchange_strong(
          state, TrackedRequest::SlotState::Releasing,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return false;
  }
  if (snapshot) {
    snapshot->state.store(state, std::memory_order_relaxed);
    snapshot->generation.store(slot->generation.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
    snapshot->request_id = slot->request_id;
    snapshot->adapter_request_id = slot->adapter_request_id;
    snapshot->peer_rank = slot->peer_rank;
    snapshot->kind = slot->kind;
    snapshot->needs_host_to_device_copy = slot->needs_host_to_device_copy;
    snapshot->host_copy_submitted = slot->host_copy_submitted;
    snapshot->completion_buffer = slot->completion_buffer;
    snapshot->completion_offset = slot->completion_offset;
    snapshot->completion_bytes = slot->completion_bytes;
    snapshot->host_copy_event = slot->host_copy_event;
    snapshot->bounce_ptr = slot->bounce_ptr;
  }
  return true;
}

void RequestTracker::finish_release(unsigned req_id) {
  TrackedRequest* slot = resolve(req_id);
  if (slot == nullptr) return;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state != TrackedRequest::SlotState::Releasing) return;
  slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
}

// ── adapter dispatch ────────────────────────────────────────────────────────

bool RequestTracker::poll_one(unsigned id, bool blocking) {
  TrackedRequest* slot = resolve(id);
  if (slot == nullptr) return true;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state == TrackedRequest::SlotState::Completed ||
      state == TrackedRequest::SlotState::Failed) {
    return true;
  }
  if (state != TrackedRequest::SlotState::InFlight) return false;

  bool done = false;
  bool failed = false;
  unsigned adapter_id = slot->adapter_request_id;
  if (slot->kind == PeerTransportKind::Uccl) {
    if (uccl_ == nullptr) return false;
    done = blocking ? uccl_->wait_completion(adapter_id)
                    : uccl_->poll_completion(adapter_id);
    failed = done && uccl_->request_failed(adapter_id);
  } else if (slot->kind == PeerTransportKind::Tcp) {
    if (tcp_ == nullptr) return false;
    done = blocking ? tcp_->wait_completion(adapter_id)
                    : tcp_->poll_completion(adapter_id);
    failed = done && tcp_->request_failed(adapter_id);
  } else if (slot->kind == PeerTransportKind::Ipc) {
    if (ipc_ == nullptr) return false;
    done = blocking ? ipc_->wait_completion(adapter_id)
                    : ipc_->poll_completion(adapter_id);
    failed = done && ipc_->request_failed(adapter_id);
  } else if (slot->kind == PeerTransportKind::Rdma) {
    if (rdma_ == nullptr) return false;
    done = blocking ? rdma_->wait_completion(adapter_id)
                    : rdma_->poll_completion(adapter_id);
    failed = done && rdma_->request_failed(adapter_id);
  }

  if (!done) return false;

  if (!failed) {
    bool copy_done = complete_bounce_(*slot, blocking);
    if (!copy_done) return false;
  }
  slot->state.store(failed ? TrackedRequest::SlotState::Failed
                           : TrackedRequest::SlotState::Completed,
                    std::memory_order_release);
  return true;
}

// ── user API ────────────────────────────────────────────────────────────────

bool RequestTracker::poll(unsigned req) {
  if (req == 0) return false;
  auto* slot = resolve(req);
  if (slot == nullptr) return true;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state == TrackedRequest::SlotState::InFlight) {
    if (!poll_one(req, false)) return false;
    state = slot->state.load(std::memory_order_acquire);
  }
  if (state == TrackedRequest::SlotState::Failed) {
    throw std::runtime_error("transport request failed");
  }
  return state == TrackedRequest::SlotState::Completed;
}

void RequestTracker::release(unsigned req) {
  TrackedRequest* slot = resolve(req);
  if (slot == nullptr) return;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state != TrackedRequest::SlotState::Completed &&
      state != TrackedRequest::SlotState::Failed) {
    throw std::runtime_error("cannot release an in-flight transport request");
  }
  TrackedRequest snapshot{};
  if (try_release(req, &snapshot)) {
    cleanup_(snapshot);
    finish_release(snapshot.request_id);
  }
}

bool RequestTracker::wait_finish(unsigned req) {
  return wait_finish(std::vector<unsigned>{req});
}

bool RequestTracker::wait_finish(std::vector<unsigned> const& reqs) {
  std::vector<unsigned> remaining(reqs.begin(), reqs.end());
  if (remaining.empty()) {
    for (uint32_t i = 0; i < kSlotCount; ++i) {
      auto* slot = &slots_[i];
      auto state = slot->state.load(std::memory_order_acquire);
      if (state == TrackedRequest::SlotState::InFlight ||
          state == TrackedRequest::SlotState::Completed ||
          state == TrackedRequest::SlotState::Failed) {
        remaining.push_back(slot->request_id);
      }
    }
  }

  bool any_failed = false;

  while (!remaining.empty()) {
    bool made_progress = false;
    for (auto id : remaining) {
      if (id == 0) return false;
      (void)poll_one(id, false);
    }
    std::vector<unsigned> finished;
    for (auto id : remaining) {
      TrackedRequest* slot = resolve(id);
      if (slot == nullptr) {
        finished.push_back(id);
        continue;
      }
      auto state = slot->state.load(std::memory_order_acquire);
      if (state == TrackedRequest::SlotState::Completed ||
          state == TrackedRequest::SlotState::Failed) {
        TrackedRequest snapshot{};
        if (try_release(id, &snapshot)) {
          any_failed =
              any_failed || (state == TrackedRequest::SlotState::Failed);
          cleanup_(snapshot);
          finish_release(snapshot.request_id);
          finished.push_back(id);
          made_progress = true;
        }
      }
    }
    for (auto id : finished) {
      auto it = std::find(remaining.begin(), remaining.end(), id);
      if (it != remaining.end()) remaining.erase(it);
    }
    if (remaining.empty()) break;
    if (!made_progress) {
      std::this_thread::yield();
    }
  }

  return !any_failed;
}

std::vector<std::pair<unsigned, bool>> RequestTracker::progress_all() {
  std::vector<std::pair<unsigned, bool>> completed;
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    auto* slot = &slots_[i];
    auto state = slot->state.load(std::memory_order_acquire);
    if (state != TrackedRequest::SlotState::InFlight) continue;
    unsigned req_id = slot->request_id;
    if (poll_one(req_id, false)) {
      auto final_state = slot->state.load(std::memory_order_acquire);
      completed.push_back(
          {req_id, final_state == TrackedRequest::SlotState::Failed});
    }
  }
  return completed;
}

}  // namespace Transport
}  // namespace UKernel
