#pragma once

#include "backend.h"
#include <cstdint>
#include <unordered_map>

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
  bool supports(OpKind kind) const override;

  void init(BufSpec bufs[3]) override;
  size_t enqueue(Cmd const* cmds, size_t n,
                 uint32_t* out_indices = nullptr) override;
  size_t drain(uint32_t* completed, size_t max) override;
  size_t drain_signals(uint32_t* completed, size_t max);
  size_t capacity() const override { return 2048; }
  void release(uint32_t cmd_idx) override;

  std::unordered_map<unsigned, uint32_t> const& signal_rid_to_cmd() const {
    return signal_rid_to_cmd_;
  }
  uint8_t cmd_transport(uint32_t idx) const {
    auto it = cmd_transport_.find(idx);
    return it != cmd_transport_.end() ? it->second : 0;
  }

 private:
  UKernel::Transport::Communicator* comm_;
  std::unordered_map<unsigned, uint32_t> rid_to_cmd_;
  std::unordered_map<unsigned, uint32_t> signal_rid_to_cmd_;
  std::unordered_map<unsigned, uint32_t> signal_rid_to_caller_;
  std::unordered_map<uint32_t, uint8_t> cmd_transport_;
  uint32_t cmd_next_ = 0;
};

}  // namespace CCL
}  // namespace UKernel
