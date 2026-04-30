#include "request_tracker.h"
#include "adapter/ipc_adapter.h"
#include "adapter/tcp_adapter.h"
#include "adapter/uccl_adapter.h"
#include "util/utils.h"
#include <sys/eventfd.h>
#include <poll.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <stdexcept>

namespace UKernel {
namespace Transport {

namespace {

constexpr uint32_t kActiveRingCount = 32768;

void drain_eventfd(int fd) {
  if (fd < 0) return;
  uint64_t cnt = 0;
  while (::read(fd, &cnt, sizeof(cnt)) == static_cast<ssize_t>(sizeof(cnt))) {
  }
}

}  // namespace

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

void RequestTracker::signal_eventfd(int fd) {
  if (fd < 0) return;
  uint64_t one = 1;
  while (true) {
    ssize_t n = ::write(fd, &one, sizeof(one));
    if (n == static_cast<ssize_t>(sizeof(one))) return;
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && errno == EAGAIN) return;
    return;
  }
}

// ── lifecycle ───────────────────────────────────────────────────────────────

RequestTracker::RequestTracker(UcclTransportAdapter* uccl,
                               TcpTransportAdapter* tcp, IpcAdapter* ipc,
                               CompleteBounceFn complete_bounce,
                               CleanupFn cleanup)
    : uccl_(uccl),
      tcp_(tcp),
      ipc_(ipc),
      complete_bounce_(std::move(complete_bounce)),
      cleanup_(std::move(cleanup)) {
  active_jring_ = create_ring(sizeof(unsigned), kActiveRingCount);
  if (active_jring_ == nullptr) {
    throw std::runtime_error("RequestTracker: failed to create active jring");
  }

  slots_ = std::make_unique<TrackedRequest[]>(kSlotCount);
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    slots_[i].state.store(TrackedRequest::SlotState::Free,
                          std::memory_order_relaxed);
  }

  event_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (event_fd_ < 0) {
    throw std::runtime_error("RequestTracker: failed to create eventfd");
  }
}

RequestTracker::~RequestTracker() {
  if (started_.load(std::memory_order_acquire)) {
    running_.store(false, std::memory_order_release);
    cv_.notify_all();
    signal_eventfd(event_fd_);
    if (progress_thread_.joinable()) {
      progress_thread_.join();
    }
  }

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
      slot->state.store(TrackedRequest::SlotState::Free,
                        std::memory_order_release);
    } else {
      slot->state.store(TrackedRequest::SlotState::Free,
                        std::memory_order_release);
    }
  }

  if (active_jring_ != nullptr) {
    free(active_jring_);
    active_jring_ = nullptr;
  }

  if (event_fd_ >= 0) {
    ::close(event_fd_);
    event_fd_ = -1;
  }
}

void RequestTracker::start_progress() {
  running_.store(true, std::memory_order_release);
  started_.store(true, std::memory_order_release);
  progress_thread_ = std::thread(&RequestTracker::progress_loop, this);
}

// ── slot lifecycle ──────────────────────────────────────────────────────────

TrackedRequest* RequestTracker::allocate(unsigned* out_req_id) {
  if (out_req_id == nullptr) return nullptr;
  uint32_t start = alloc_cursor_.fetch_add(1, std::memory_order_relaxed) &
                   (kSlotCount - 1u);
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
    slot->notified = false;
    slot->needs_host_to_device_copy = false;
    slot->host_copy_submitted = false;
    slot->completion_buffer = nullptr;
    slot->completion_offset = 0;
    slot->completion_bytes = 0;
    slot->host_copy_event = nullptr;
    slot->bounce_ptr = nullptr;
    slot->pool_slot = nullptr;
    slot->signal_payload = 0;
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
  inflight_count_.fetch_add(1, std::memory_order_release);
  unsigned slot_val = req_id;
  jring_mp_enqueue_bulk(active_jring_, &slot_val, 1, nullptr);
  cv_.notify_all();
  return true;
}

bool RequestTracker::try_release(unsigned req_id,
                                 TrackedRequest* snapshot) {
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
    snapshot->generation.store(
        slot->generation.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    snapshot->request_id = slot->request_id;
    snapshot->adapter_request_id = slot->adapter_request_id;
    snapshot->peer_rank = slot->peer_rank;
    snapshot->kind = slot->kind;
    snapshot->notified = slot->notified;
    snapshot->needs_host_to_device_copy = slot->needs_host_to_device_copy;
    snapshot->host_copy_submitted = slot->host_copy_submitted;
    snapshot->completion_buffer = slot->completion_buffer;
    snapshot->completion_offset = slot->completion_offset;
    snapshot->completion_bytes = slot->completion_bytes;
    snapshot->host_copy_event = slot->host_copy_event;
    snapshot->bounce_ptr = slot->bounce_ptr;
    snapshot->pool_slot = slot->pool_slot;
    snapshot->signal_payload = slot->signal_payload;
  }
  slot->state.store(TrackedRequest::SlotState::Free,
                    std::memory_order_release);
  return true;
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
  }

  if (!done) return false;

  if (done && !failed && slot->kind == PeerTransportKind::Ipc) {
    slot->signal_payload = ipc_->completion_payload(adapter_id);
  }

  if (!failed) {
    bool copy_done = complete_bounce_(*slot, blocking);
    if (!copy_done) return false;
  }
  slot->state.store(failed ? TrackedRequest::SlotState::Failed
                           : TrackedRequest::SlotState::Completed,
                    std::memory_order_release);
  inflight_count_.fetch_sub(1, std::memory_order_acq_rel);
  notify_completion();
  return true;
}

// ── user API ────────────────────────────────────────────────────────────────

bool RequestTracker::poll(unsigned req) {
  if (req == 0) return false;
  auto* slot = resolve(req);
  if (slot == nullptr) return true;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state == TrackedRequest::SlotState::InFlight) {
    if (!started_.load(std::memory_order_acquire)) {
      if (!poll_one(req, false)) return false;
      state = slot->state.load(std::memory_order_acquire);
    } else {
      return false;
    }
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
  uint64_t seen_seq = completion_seq_.load(std::memory_order_acquire);
  bool should_scan = true;

  while (!remaining.empty()) {
    if (should_scan) {
      std::vector<unsigned> finished;
      bool rescan_immediately = false;
      for (auto id : remaining) {
        if (id == 0) return false;
        if (!started_.load(std::memory_order_acquire)) {
          (void)poll_one(id, false);
        }
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
            finished.push_back(id);
          } else {
            rescan_immediately = true;
            if (resolve(id) == nullptr) finished.push_back(id);
          }
        } else if (state == TrackedRequest::SlotState::Releasing) {
          rescan_immediately = true;
        }
      }
      for (auto id : finished) {
        auto it = std::find(remaining.begin(), remaining.end(), id);
        if (it != remaining.end()) remaining.erase(it);
      }
      should_scan = rescan_immediately;
    }

    if (remaining.empty()) return !any_failed;
    if (should_scan) continue;

    if (event_fd_ >= 0) {
      pollfd pfd{};
      pfd.fd = event_fd_;
      pfd.events = POLLIN;
      int rc = ::poll(&pfd, 1, -1);
      if (rc > 0 && (pfd.revents & POLLIN)) {
        drain_eventfd(event_fd_);
      }
      uint64_t now_seq = completion_seq_.load(std::memory_order_acquire);
      if (now_seq != seen_seq) {
        seen_seq = now_seq;
        should_scan = true;
      }
    } else {
      std::this_thread::yield();
      should_scan = true;
    }
  }

  return !any_failed;
}

// ── notification ────────────────────────────────────────────────────────────

void RequestTracker::notify_completion() {
  completion_seq_.fetch_add(1, std::memory_order_release);
  signal_eventfd(event_fd_);
}

std::shared_ptr<void> RequestTracker::register_notifier(NotifyCb cb) {
  auto target = std::make_shared<NotifyTarget>();
  target->emit = std::move(cb);

  {
    std::lock_guard<std::mutex> lk(mu_);
    notifiers_.push_back(target);
  }

  cv_.notify_all();
  return std::static_pointer_cast<void>(target);
}

// ── progress loop ───────────────────────────────────────────────────────────

void RequestTracker::progress_loop() {
  constexpr size_t kBatchSize = 128;
  std::array<unsigned, kBatchSize> batch{};

  while (running_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] {
        if (!running_.load(std::memory_order_acquire)) return true;
        return inflight_count_.load(std::memory_order_acquire) > 0;
      });
    }

    if (!running_.load(std::memory_order_acquire)) break;

    bool progress = false;

    // Snapshot live notifiers.
    std::vector<std::shared_ptr<NotifyTarget>> targets_snapshot;
    {
      std::lock_guard<std::mutex> lk(mu_);
      notifiers_.erase(
          std::remove_if(notifiers_.begin(), notifiers_.end(),
                         [](std::weak_ptr<NotifyTarget> const& t) {
                           return t.expired();
                         }),
          notifiers_.end());
      targets_snapshot.reserve(notifiers_.size());
      for (auto const& weak_t : notifiers_) {
        if (auto t = weak_t.lock()) {
          targets_snapshot.push_back(std::move(t));
        }
      }
    }

    // Bulk-dequeue from jring (single CAS for up to kBatchSize items).
    unsigned n =
        jring_sc_dequeue_burst(active_jring_, batch.data(), kBatchSize, nullptr);

    auto now = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i) {
      unsigned id = batch[i];
      TrackedRequest* slot = resolve(id);
      if (slot == nullptr) continue;
      auto state = slot->state.load(std::memory_order_acquire);
      if (state != TrackedRequest::SlotState::InFlight) continue;
      if (!poll_one(id, false)) {
        unsigned id_val = id;
        jring_mp_enqueue_bulk(active_jring_, &id_val, 1, nullptr);
        continue;
      }

      bool should_emit = false;
      state = slot->state.load(std::memory_order_acquire);
      if ((state == TrackedRequest::SlotState::Completed ||
           state == TrackedRequest::SlotState::Failed) &&
          !slot->notified) {
        slot->notified = true;
        should_emit = true;
      }
      if (!should_emit) continue;
      for (auto& tgt : targets_snapshot) {
        if (!tgt) continue;
        tgt->emit(id, now);
      }
      progress = true;
    }

    // Safety net: if no progress but inflight > 0, re-enqueue missed requests.
    if (!progress) {
      if (inflight_count_.load(std::memory_order_acquire) > 0) {
        for (uint32_t i = 0; i < kSlotCount; ++i) {
          TrackedRequest* slot = &slots_[i];
          auto state = slot->state.load(std::memory_order_acquire);
          if (state != TrackedRequest::SlotState::InFlight) continue;
          unsigned id_val = slot->request_id;
          jring_mp_enqueue_bulk(active_jring_, &id_val, 1, nullptr);
        }
      } else {
        std::this_thread::yield();
      }
    }
  }
}

}  // namespace Transport
}  // namespace UKernel
