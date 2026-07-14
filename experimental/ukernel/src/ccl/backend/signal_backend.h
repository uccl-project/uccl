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

  // Reserve a be_idx and communicator rid without submitting.
  // Caller must write the slot table entry BEFORE calling
  // do_enqueue_reserved so the drain thread sees it when the
  // completion arrives (signal send is synchronous for IPC).
  uint32_t reserve_slot();

  size_t do_enqueue(Cmd const* cmds, size_t n,
                    uint32_t* out_indices = nullptr) override;
  bool do_enqueue_reserved(Cmd const& cmd, uint32_t be_idx);
  size_t do_drain(uint32_t* completed, size_t max) override;
  size_t capacity() const override { return 2048; }

 private:
  uint32_t cmd_next_ = 0;
  std::mutex mu_;
  std::unordered_map<uint32_t, unsigned> reserved_;
};

}  // namespace CCL
}  // namespace UKernel
