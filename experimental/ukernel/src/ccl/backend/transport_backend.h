#pragma once

#include "backend.h"
#include <cstdint>
#include <memory>
#include <vector>

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
  size_t enqueue(Cmd const* cmds, size_t n) override;
  size_t drain(uint32_t* completed, size_t max) override;
  size_t capacity() const override { return 1024; }
  void release(uint32_t cmd_idx) override;

 private:
  UKernel::Transport::Communicator* comm_;

  struct Pending {
    unsigned req_id;
    uint32_t cmd_idx;
  };
  std::vector<Pending> pending_;
  uint32_t cmd_next_ = 0;
};

}  // namespace CCL
}  // namespace UKernel
