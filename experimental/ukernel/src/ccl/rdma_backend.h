#pragma once

#include "../transport/uccl_transport_adapter.h"
#include "../../include/transport.h"
#include "backend.h"
#include "plan.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace UKernel {
namespace CCL {

struct BufferBindings {
  void const* local_input = nullptr;
  void const* remote_input = nullptr;
  void const* remote_reduced = nullptr;
  void* final_output = nullptr;
  void* recv_staging = nullptr;
  size_t registration_bytes = 0;
};

class RdmaBackend final : public Backend {
 public:
  RdmaBackend(UKernel::Transport::UcclTransportAdapter& adapter,
              BufferBindings bindings);
  ~RdmaBackend() override;

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
  bool poll(BackendToken token) override;
  void release(BackendToken token) override;

 private:
  struct PendingRequest {
    uint64_t request_id = 0;
    bool completed = false;
  };

  enum class RegisteredBuffer : uint64_t {
    LocalInput = 0x1001,
    RemoteInput = 0x1002,
    RemoteReduced = 0x1003,
    FinalOutput = 0x1004,
    RecvStaging = 0x1005,
  };

  void ensure_registered();
  void register_buffer(RegisteredBuffer id, void* ptr);
  void deregister_registered();
  void* resolve_dst(BufferRole role, size_t offset) const;
  void const* resolve_src(BufferRole role, size_t offset) const;
  uint64_t resolve_mr_id_for_src(BufferRole role) const;
  uint64_t resolve_mr_id_for_dst(BufferRole role) const;

  static void* byte_offset(void* base, size_t offset);
  static void const* byte_offset(void const* base, size_t offset);

  UKernel::Transport::UcclTransportAdapter& adapter_;
  BufferBindings bindings_{};
  bool registered_ = false;
  uint64_t next_request_id_ = 1;
  std::unordered_map<uint64_t, PendingRequest> pending_;
};

class CommunicatorRdmaBackend final : public Backend {
 public:
  CommunicatorRdmaBackend(UKernel::Transport::Communicator& comm, int peer_rank,
                          BufferBindings bindings);
  ~CommunicatorRdmaBackend() override;

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
    uint16_t local_mr_id = 0;
    uint16_t remote_mr_id = 0;
  };

  struct PendingRequest {
    unsigned request_id = 0;
    bool completed = false;
  };

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
  BufferBindings bindings_{};
  bool registered_ = false;
  std::unordered_map<uint64_t, RegisteredMr> registered_mrs_;
  uint64_t next_token_ = 1;
  std::unordered_map<uint64_t, PendingRequest> pending_;
};

}  // namespace CCL
}  // namespace UKernel
