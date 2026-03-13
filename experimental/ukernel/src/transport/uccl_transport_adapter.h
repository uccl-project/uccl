#pragma once

#include "../../include/transport.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declare uccl classes
namespace uccl {
class RDMAEndpoint;
class Mhandle;
class UcclFlow;
struct ucclRequest;
class UcclRDMAEngine;
// Note: ConnID is defined in collective/rdma/transport.h as a struct
}  // namespace uccl

namespace UKernel {
namespace Transport {

struct UcclTransportConfig {
  // Must stay aligned with ucclParamNUM_ENGINES() inside collective/rdma.
  int num_engines = 0;
  int num_paths = 16;
  std::string local_ip;
  uint16_t listen_port = 0;
};

class UcclTransportAdapter {
 public:
  UcclTransportAdapter(int local_gpu_idx, int world_size,
                       UcclTransportConfig config);
  ~UcclTransportAdapter();

  bool connect_to_peer(int peer_rank, std::string remote_ip,
                       uint16_t remote_port, int local_dev_idx,
                       int local_gpu_idx, int remote_dev_idx,
                       int remote_gpu_idx);
  bool accept_from_peer(int peer_rank);

  // Get P2P listen port for the given device
  uint16_t get_p2p_listen_port(int dev_idx) const;

  // Get P2P listen IP for the given device
  std::string get_p2p_listen_ip(int dev_idx) const;

  // Get best RDMA device index for given GPU
  int get_best_dev_idx(int gpu_idx) const;

  bool has_peer(int peer_rank) const;
  bool has_send_peer(int peer_rank) const;
  bool has_recv_peer(int peer_rank) const;

  uint64_t register_memory(void* ptr, size_t len);
  bool register_memory(uint64_t mr_id, void* ptr, size_t len);
  void deregister_memory(uint64_t mr_id);

  int send_async(int peer_rank, void* local_ptr, size_t len,
                 uint64_t local_mr_id, uint64_t remote_mr_id,
                 uint64_t request_id);
  int recv_async(int peer_rank, void* local_ptr, size_t len,
                 uint64_t local_mr_id, uint64_t request_id);

  bool poll_completion(int* out_peer_rank, uint64_t* out_mr_id);
  bool wait_completion(uint64_t request_id);

 private:
  struct PeerContext {
    ::uccl::UcclFlow* send_flow = nullptr;
    ::uccl::UcclFlow* recv_flow = nullptr;
    uint64_t remote_mr_addr = 0;
    uint64_t remote_mr_rkey = 0;
    int peer_rank = -1;
  };

  int local_gpu_idx_;
  int world_size_;
  UcclTransportConfig config_;

  std::unique_ptr<::uccl::RDMAEndpoint> endpoint_;
  std::unordered_map<int, PeerContext> peer_contexts_;
  std::unordered_map<uint64_t, ::uccl::Mhandle*> mr_id_to_mhandle_;
  std::unordered_map<uint64_t, std::unique_ptr<::uccl::ucclRequest>>
      pending_requests_;
  mutable std::mutex mu_;
  std::atomic<uint64_t> next_mr_id_{1};
};

}  // namespace Transport
}  // namespace UKernel
