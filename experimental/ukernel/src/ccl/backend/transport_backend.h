#pragma once

#include "backend.h"
#include <atomic>
#include <cstdint>

namespace UKernel {
namespace Transport {
class Communicator;
}
namespace CCL {

class TransportBackend final : public BatchBackend {
 public:
  explicit TransportBackend(UKernel::Transport::Communicator* comm);
  ~TransportBackend() override = default;

  char const* name() const override { return "transport"; }
  bool supports(ExecOpKind kind) const override;

  size_t do_enqueue(Cmd const* cmds, size_t n,
                    uint32_t* out_indices = nullptr) override;
  uint32_t reserve_slot() override;
  bool do_enqueue_reserved(Cmd const& cmd, uint32_t be_idx) override;
  size_t do_drain(uint32_t* completed, size_t max) override;
  size_t capacity() const override { return 2048; }

 private:
  // Lock-free be_idx allocator. Values stay in [0, 2^30) so they fit in
  // the tagged rid's low bits; failed enqueues leave harmless gaps.
  std::atomic<uint32_t> cmd_next_{0};
};

}  // namespace CCL
}  // namespace UKernel
