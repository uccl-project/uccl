#pragma once

#include "backend.h"
#include <cstddef>
#include <cstdint>
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

class CommunicatorTransportBackend final : public Backend {
 public:
  explicit CommunicatorTransportBackend(TransportBackendConfig const& config);
  ~CommunicatorTransportBackend() override;

  char const* name() const override;
  void validate(CollectivePlan const& plan,
                CollectiveBinding& binding) override;
  bool supports(OpKind kind) const override;
  BackendToken submit(Op const& op, CollectiveBinding& binding) override;
  size_t drain(BackendToken* out, size_t max_count) override;

  UKernel::Transport::Communicator& communicator();
  UKernel::Transport::Communicator const& communicator() const;

 private:
  void initialize_memory_bindings(CollectiveBinding& binding);
  void ensure_peer_path(int peer_rank, bool need_put, bool need_wait);

  uint32_t resolve_local_buffer_id(CollectiveBinding const& binding,
                                    BufferRef const& ref) const;
  int resolve_peer_rank(Op const& op) const;
  uint32_t resolve_remote_buffer_id(CollectiveBinding const& binding,
                                     BufferRef const& ref) const;
  bool is_transport_fresh(CollectiveBinding const& binding) const;

  std::unique_ptr<UKernel::Transport::Communicator> communicator_;
  uint64_t next_token_ = 1;
  uint64_t backend_cache_key_ = 0;
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, unsigned> pending_;     // token → request_id
  std::unordered_map<unsigned, uint64_t> req_to_token_; // request_id → token
  std::vector<bool> peer_put_ready_;
  std::vector<bool> peer_wait_ready_;
  mutable std::mutex init_mu_;
  int connect_retry_ms_ = 100;
  int connect_timeout_s_ = 30;
};

}  // namespace CCL
}  // namespace UKernel
