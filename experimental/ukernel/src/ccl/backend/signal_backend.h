#pragma once

#include "backend.h"
#include <cstdint>
#include <mutex>
#include <unordered_map>

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
  bool supports(ExecOpKind kind) const override;

  size_t do_enqueue(Cmd const* cmds, size_t n,
                    uint32_t* out_indices = nullptr) override;
  size_t do_drain(uint32_t* completed, size_t max) override;
  size_t capacity() const override { return 2048; }
  void release(uint32_t cmd_idx) override;

 private:
  // Signal (send) completions go through completion_ring_ → try_complete()
  std::unordered_map<unsigned, uint32_t> signal_send_rid_to_cmd_;

  // SignalWait completions go through signal_ring_ → try_complete_signals()
  std::unordered_map<unsigned, uint32_t> signal_wait_rid_to_cmd_;

  uint32_t cmd_next_ = 0;
  std::mutex mu_;
};

}  // namespace CCL
}  // namespace UKernel
