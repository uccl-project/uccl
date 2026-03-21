#pragma once

#include "../../include/transport.h"
#include "backend.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace UKernel {
namespace CCL {

// Adapts transport::Communicator onto the CCL backend interface.
class CommunicatorTransportBackend final : public Backend {
 public:
  CommunicatorTransportBackend(UKernel::Transport::Communicator& comm,
                               CollectiveMemory memory);
  ~CommunicatorTransportBackend() override = default;

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
  bool poll(BackendToken token) override;
  bool try_pop_completed(BackendToken& token) override;
  void release(BackendToken token) override;

  private:
  struct PendingRequest {
    unsigned request_id = 0;
    bool completed = false;
    bool released = false;
  };

  void* resolve_mutable(MemoryRef const& ref, size_t bytes) const;
  void const* resolve_const(MemoryRef const& ref, size_t bytes) const;
  uint32_t resolve_local_mr_id(MemoryRef const& ref, size_t bytes) const;
  uint32_t resolve_remote_mr_id(int peer_rank) const;
  void on_transport_completion(unsigned request_id);
  static void* byte_offset(void* base, size_t offset);
  static void const* byte_offset(void const* base, size_t offset);

  UKernel::Transport::Communicator& comm_;
  CollectiveMemory memory_{};
  uint64_t next_token_ = 1;
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, PendingRequest> pending_;
  std::unordered_map<unsigned, uint64_t> request_to_token_;
  std::deque<uint64_t> completed_tokens_;
  std::shared_ptr<void> completion_notifier_;
};

}  // namespace CCL
}  // namespace UKernel
