#pragma once

#include "backend.h"
#include <atomic>
#include <cstdint>
#include <thread>

extern "C" {
#include "util/jring.h"
}

namespace UKernel {
namespace CCL {

class AsyncBackend {
 public:
  AsyncBackend(BatchBackend* be, uint32_t cmd_ring_slots,
               uint32_t done_ring_slots);
  ~AsyncBackend();

  AsyncBackend(AsyncBackend const&) = delete;
  AsyncBackend& operator=(AsyncBackend const&) = delete;

  // ── Non-blocking, called by SprayExecutor threads ──

  // Write commands to cmd_ring. Returns number actually written (0..n).
  size_t try_enqueue(CmdWithId const* cmds, size_t n);

  // Read completed caller_ids from done_ring. Returns number read (0..max).
  size_t try_drain(uint32_t* caller_ids, size_t max);

  size_t cmd_free() const;
  size_t done_count() const;

  void start();
  void stop();

 private:
  void submit_loop();
  void drain_loop();

  BatchBackend* be_;
  jring_t* cmd_ring_;
  jring_t* done_ring_;

  static constexpr size_t kPendingSlots = 65536;
  uint32_t pending_[kPendingSlots];

  std::thread submit_th_;
  std::thread drain_th_;
  std::atomic<bool> stop_{false};
};

}  // namespace CCL
}  // namespace UKernel
