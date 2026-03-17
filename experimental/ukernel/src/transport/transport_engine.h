#pragma once

#include "../../include/config.h"
#include "oob.h"
#include "uccl_transport_adapter.h"
#include <memory>
#include <string>

namespace UKernel {
namespace Transport {

enum class PeerTransportKind { Uccl, Ipc };

char const* peer_transport_kind_name(PeerTransportKind kind);

PeerTransportKind resolve_peer_transport_kind(
    CommunicatorConfig const& config, CommunicatorMeta const& local_meta,
    CommunicatorMeta const& peer_meta);

class UcclTransportEngine {
 public:
  UcclTransportEngine(int local_gpu_idx, int world_size);

  bool connect_to_peer(int global_rank, int peer_rank,
                       std::shared_ptr<CommunicatorConfig> const& config,
                       std::shared_ptr<Exchanger> const& exchanger,
                       CommunicatorMeta const& local_meta,
                       CommunicatorMeta const& peer_meta);
  bool accept_from_peer(int global_rank, int peer_rank,
                        std::shared_ptr<CommunicatorConfig> const& config,
                        std::shared_ptr<Exchanger> const& exchanger,
                        CommunicatorMeta const& local_meta,
                        CommunicatorMeta const& peer_meta);

  bool is_initialized() const { return adapter_ != nullptr; }

  bool register_memory(uint64_t mr_id, void* ptr, size_t len);
  void deregister_memory(uint64_t mr_id);
  int send_async(int peer_rank, void* local_ptr, size_t len,
                 uint64_t local_mr_id, uint64_t remote_mr_id,
                 uint64_t request_id);
  int recv_async(int peer_rank, void* local_ptr, size_t len,
                 uint64_t local_mr_id, uint64_t request_id);
  bool poll_completion(uint64_t request_id);
  bool wait_completion(uint64_t request_id);

 private:
  bool ensure_adapter(std::shared_ptr<CommunicatorConfig> const& config,
                      CommunicatorMeta const& local_meta,
                      CommunicatorMeta const& peer_meta);

  int local_gpu_idx_;
  int world_size_;
  std::unique_ptr<UcclTransportAdapter> adapter_;
};

}  // namespace Transport
}  // namespace UKernel
