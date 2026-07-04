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
  bool supports(ExecOpKind kind) const override;

  size_t do_enqueue(Cmd const* cmds, size_t n,
                    uint32_t* out_indices = nullptr) override;
  size_t do_drain(uint32_t* completed, size_t max) override;
  size_t capacity() const override { return 2048; }
  void release(uint32_t cmd_idx) override;

 private:
  std::unordered_map<unsigned, uint32_t> rid_to_cmd_;
  uint32_t cmd_next_ = 0;
};

}  // namespace CCL
}  // namespace UKernel
