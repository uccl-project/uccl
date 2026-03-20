#pragma once

#include "../../include/transport.h"
#include "backend.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace UKernel {
namespace CCL {

// Adapts transport::Communicator onto the CCL backend interface.
class CommunicatorTransportBackend final : public Backend {
 public:
  CommunicatorTransportBackend(UKernel::Transport::Communicator& comm,
                               int peer_rank, CollectiveBuffers buffers);
  ~CommunicatorTransportBackend() override;

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
  bool poll(BackendToken token) override;
  void release(BackendToken token) override;

 private:
  enum class RegisteredBuffer : uint64_t {
    LocalInput = 0x2001,
    RemoteInput = 0x2002,
    RemoteReduced = 0x2003,
    FinalOutput = 0x2004,
    RecvStaging = 0x2005,
  };

  struct RegisteredMr {
    uint32_t local_mr_id = 0;
    uint32_t remote_mr_id = 0;
  };

  struct PendingRequest {
    unsigned request_id = 0;
    bool completed = false;
  };

  void validate_topology(ExecutionOp const& op) const;
  void ensure_registered();
  void exchange_mrs();
  void register_buffer(RegisteredBuffer id, void* ptr);
  void deregister_registered();
  void* resolve_dst(BufferRole role, size_t offset) const;
  void const* resolve_src(BufferRole role, size_t offset) const;
  RegisteredMr resolve_mr(BufferRole role) const;
  static void* byte_offset(void* base, size_t offset);
  static void const* byte_offset(void const* base, size_t offset);

  UKernel::Transport::Communicator& comm_;
  int peer_rank_ = -1;
  CollectiveBuffers buffers_{};
  bool registered_ = false;
  std::unordered_map<uint64_t, RegisteredMr> registered_mrs_;
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, PendingRequest> pending_;
};

}  // namespace CCL
}  // namespace UKernel
