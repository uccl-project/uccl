#pragma once

#include "backend.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {
class Communicator;
struct CommunicatorConfig;
}  // namespace Transport
namespace CCL {

struct TransportBackendConfig {
  int gpu_id = 0;
  int rank = 0;
  int world_size = 1;
  std::shared_ptr<UKernel::Transport::CommunicatorConfig> communicator_config;
};

// Adapts transport::Communicator onto the CCL backend interface.
class CommunicatorTransportBackend final : public Backend {
 public:
  explicit CommunicatorTransportBackend(TransportBackendConfig const& config);
  ~CommunicatorTransportBackend() override;

  char const* name() const override;
  void validate(ExecutionPlan const& plan,
                CollectiveBinding& binding) const override;
  bool supports(ExecOpKind kind) const override;
  BackendToken submit(ExecOp const& op, CollectiveBinding& binding) override;
  bool poll(BackendToken token) override;
  bool try_pop_completed(BackendToken& token) override;
  void release(BackendToken token) override;

  UKernel::Transport::Communicator& communicator();
  UKernel::Transport::Communicator const& communicator() const;

 private:
  struct PendingRequest {
    unsigned request_id = 0;
    bool completed = false;
    bool released = false;
  };

  void ensure_memory_bindings_initialized(CollectiveBinding& binding) const;
  void initialize_memory_bindings(CollectiveBinding& binding) const;
  void* resolve_mutable(CollectiveBinding const& binding, BufferRef const& ref,
                        size_t bytes) const;
  void const* resolve_const(CollectiveBinding const& binding,
                            BufferRef const& ref, size_t bytes) const;
  uint32_t resolve_local_buffer_id(CollectiveBinding const& binding,
                               BufferRef const& ref, size_t bytes) const;
  int resolve_peer_rank(ExecOp const& op) const;
  uint32_t resolve_remote_buffer_id(CollectiveBinding const& binding,
                                BufferRef const& ref) const;
  void ensure_plan_paths(ExecutionPlan const& plan) const;
  void ensure_peer_paths(int peer_rank) const;
  void on_transport_completion(unsigned request_id);
  static void* byte_offset(void* base, size_t offset);
  static void const* byte_offset(void const* base, size_t offset);

  std::unique_ptr<UKernel::Transport::Communicator> communicator_;
  uint64_t next_token_ = 1;
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, PendingRequest> pending_;
  std::unordered_map<unsigned, uint64_t> request_to_token_;
  std::deque<uint64_t> completed_tokens_;
  std::shared_ptr<void> completion_notifier_;
  mutable std::mutex init_mu_;
  mutable std::mutex path_mu_;
  uint64_t backend_cache_key_ = 0;
  mutable std::vector<bool> peer_paths_ready_;
};

}  // namespace CCL
}  // namespace UKernel
