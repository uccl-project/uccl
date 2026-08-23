#pragma once

#include "backend.h"
#include <atomic>
#include <cstdint>
#include <mutex>

namespace UKernel {
namespace Transport {
class Communicator;
}
namespace CCL {

class SignalBackend final : public BatchBackend {
 public:
  SignalBackend() = default;
  ~SignalBackend() override = default;

  char const* name() const override { return "signal"; }
  bool supports(LogicalOpKind kind) const override;

  // Reserve a be_idx and communicator rid without submitting.
  // Caller must write the slot table entry BEFORE calling
  // do_enqueue_reserved so the drain thread sees it when the
  // completion arrives (signal send is synchronous for IPC).
  uint32_t reserve_slot() override;

  size_t do_enqueue(Cmd const* cmds, size_t n,
                    uint32_t* out_indices = nullptr) override;
  bool do_enqueue_reserved(Cmd const& cmd, uint32_t be_idx) override;
  size_t do_drain(uint32_t* completed, size_t max) override;
  // In-flight Signal+WaitSignal slots. Must exceed the total signal ops
  // of the largest plan (~2×tiles+2 at G=1): WaitSignals hold slots
  // until peer arrivals, and Signal ops starve deadlocked once the table
  // fills with uncompletable waits (seen at 128M in-place: 2048 waits =
  // old cap). 65536 covers runs up to ~2GB at 64KB tiles; beyond that,
  // WaitSignal submission needs throttling.
  size_t capacity() const override { return 65536; }

 private:
  // Lock-free be_idx allocator. Values stay in [0, 2^30) so they fit in
  // the tagged rid's low bits; failed enqueues leave harmless gaps (the
  // executor's slot table only ever tracks live slots).
  std::atomic<uint32_t> cmd_next_{0};
  // Serializes do_drain: the communicator's signal completion rings are
  // single-consumer, but do_drain may be called concurrently by the
  // background drain thread and by user threads in SprayExecutor::wait().
  std::mutex drain_mu_;
};

}  // namespace CCL
}  // namespace UKernel
