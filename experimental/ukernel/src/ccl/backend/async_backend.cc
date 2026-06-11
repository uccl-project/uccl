#include "async_backend.h"

#include <atomic>
#include <cstdlib>
#include <thread>

namespace UKernel {
namespace CCL {

AsyncBackend::AsyncBackend(BatchBackend* be, uint32_t cmd_ring_slots,
                           uint32_t done_ring_slots)
    : be_(be) {
  // Allocate and init cmd_ring (SPSC: single producer, single consumer)
  size_t cmd_sz = jring_get_buf_ring_size(sizeof(CmdWithId), cmd_ring_slots);
  cmd_ring_ = static_cast<jring_t*>(calloc(1, cmd_sz));
  jring_init(cmd_ring_, cmd_ring_slots, sizeof(CmdWithId), 0, 0);

  // Allocate and init done_ring (SPSC)
  size_t done_sz = jring_get_buf_ring_size(sizeof(uint32_t), done_ring_slots);
  done_ring_ = static_cast<jring_t*>(calloc(1, done_sz));
  jring_init(done_ring_, done_ring_slots, sizeof(uint32_t), 0, 0);

  for (size_t i = 0; i < kPendingSlots; ++i)
    pending_[i].store(~0u, std::memory_order_relaxed);
}

AsyncBackend::~AsyncBackend() {
  stop();
  free(done_ring_);
  free(cmd_ring_);
}

void AsyncBackend::start() {
  submit_th_ = std::thread(&AsyncBackend::submit_loop, this);
  drain_th_ = std::thread(&AsyncBackend::drain_loop, this);
}

void AsyncBackend::stop() {
  stop_ = true;
  if (submit_th_.joinable()) submit_th_.join();
  if (drain_th_.joinable()) drain_th_.join();
}

// ── Non-blocking API ────────────────────────────────────────────────────

size_t AsyncBackend::try_enqueue(CmdWithId const* cmds, size_t n) {
  return jring_sp_enqueue_burst(cmd_ring_, cmds,
                                static_cast<unsigned>(n), nullptr);
}

size_t AsyncBackend::try_drain(uint32_t* caller_ids, size_t max) {
  return jring_sc_dequeue_burst(done_ring_, caller_ids,
                                static_cast<unsigned>(max), nullptr);
}

size_t AsyncBackend::cmd_free() const {
  return jring_free_count(cmd_ring_);
}

size_t AsyncBackend::done_count() const {
  return jring_count(done_ring_);
}

// ── Internal threads ─────────────────────────────────────────────────────

void AsyncBackend::submit_loop() {
  CmdWithId cwi;
  while (!stop_) {
    unsigned n = jring_sc_dequeue_burst(cmd_ring_, &cwi, 1, nullptr);
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }

    uint32_t be_idx = 0;
    while (be_->enqueue(&cwi.cmd, 1, &be_idx) == 0)
      std::this_thread::yield();

    pending_[be_idx & (kPendingSlots - 1)].store(
        cwi.caller_id + 1, std::memory_order_release);
  }
}

void AsyncBackend::drain_loop() {
  uint32_t done_buf[256];
  uint32_t out_buf[256];
  while (!stop_) {
    size_t n = be_->drain(done_buf, 256);
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }

    for (size_t i = 0; i < n; ++i) {
      uint32_t val;
      while ((val = pending_[done_buf[i] & (kPendingSlots - 1)].load(
                  std::memory_order_acquire)) == ~0u)
        std::this_thread::yield();
      out_buf[i] = val - 1;
    }

    size_t written = 0;
    while (written < n) {
      written += jring_sp_enqueue_burst(done_ring_, out_buf + written,
                                        static_cast<unsigned>(n - written),
                                        nullptr);
      if (written < n) std::this_thread::yield();
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
