#include "backend.h"
extern "C" {
#include "../../transport/util/jring.h"
}
#include <cstdlib>
#include <thread>

namespace UKernel {
namespace CCL {

BatchBackend::~BatchBackend() { stop(); }

void BatchBackend::start(uint32_t cmd_slots, uint32_t done_slots) {
  if (cmd_ring_) return;

  size_t cmd_sz = jring_get_buf_ring_size(sizeof(CmdWithId), cmd_slots);
  if (cmd_sz == (size_t)-1) cmd_sz = 0;
  cmd_ring_ = static_cast<jring_t*>(calloc(1, cmd_sz));
  if (cmd_ring_) jring_init(cmd_ring_, cmd_slots, sizeof(CmdWithId), 0, 0);

  size_t done_sz = jring_get_buf_ring_size(sizeof(uint32_t), done_slots);
  if (done_sz == (size_t)-1) done_sz = 0;
  done_ring_ = static_cast<jring_t*>(calloc(1, done_sz));
  if (done_ring_) jring_init(done_ring_, done_slots, sizeof(uint32_t), 0, 0);

  pending_ = new std::atomic<uint32_t>[kPendingSlots];
  for (size_t i = 0; i < kPendingSlots; ++i)
    pending_[i].store(~0u, std::memory_order_relaxed);

  submit_th_ = std::thread(&BatchBackend::submit_loop_, this);
  drain_th_ = std::thread(&BatchBackend::drain_loop_, this);
}

void BatchBackend::stop() {
  stop_ = true;
  if (submit_th_.joinable()) submit_th_.join();
  if (drain_th_.joinable()) drain_th_.join();
  free(done_ring_);
  done_ring_ = nullptr;
  free(cmd_ring_);
  cmd_ring_ = nullptr;
  delete[] pending_;
  pending_ = nullptr;
}

size_t BatchBackend::try_enqueue(CmdWithId const* cmds, size_t n) {
  if (!cmd_ring_) {
    size_t accepted = 0;
    for (size_t i = 0; i < n; ++i) {
      uint32_t be_idx = 0;
      if (do_enqueue(&cmds[i].cmd, 1, &be_idx) == 0) return accepted;
      ++accepted;
    }
    return accepted;
  }
  return jring_sp_enqueue_burst(cmd_ring_, cmds, static_cast<unsigned>(n),
                                nullptr);
}

size_t BatchBackend::try_drain(uint32_t* caller_ids, size_t max) {
  if (!done_ring_) {
    return do_drain(caller_ids, max);
  }
  return jring_sc_dequeue_burst(done_ring_, caller_ids,
                                static_cast<unsigned>(max), nullptr);
}

void BatchBackend::submit_loop_() {
  CmdWithId cwi;
  while (!stop_) {
    unsigned n = jring_sc_dequeue_burst(cmd_ring_, &cwi, 1, nullptr);
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }
    uint32_t be_idx = 0;
    while (do_enqueue(&cwi.cmd, 1, &be_idx) == 0)
      std::this_thread::yield();
    pending_[be_idx & (kPendingSlots - 1)].store(cwi.caller_id + 1,
                                                  std::memory_order_release);
  }
}

void BatchBackend::drain_loop_() {
  uint32_t done_buf[256];
  uint32_t out_buf[256];
  while (!stop_) {
    size_t n = do_drain(done_buf, 256);
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
      written += jring_sp_enqueue_burst(
          done_ring_, out_buf + written,
          static_cast<unsigned>(n - written), nullptr);
      if (written < n) std::this_thread::yield();
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
