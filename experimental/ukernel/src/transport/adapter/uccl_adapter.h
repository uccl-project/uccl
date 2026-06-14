#pragma once

#include "transport_adapter.h"
#include "../util/jring.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace uccl {
class RDMAEndpoint;
class Mhandle;
class UcclFlow;
struct ucclRequest;
}

namespace UKernel {
namespace Transport {

struct UcclTransportConfig { int num_engines = 0; };

class UcclTransportAdapter final : public TransportAdapter {
 public:
  UcclTransportAdapter(int gpu_id, int world_size, UcclTransportConfig config);
  ~UcclTransportAdapter() override;

  uint16_t get_p2p_listen_port(int dev_idx) const;
  std::string get_p2p_listen_ip(int dev_idx) const;
  int get_best_dev_idx(int gpu_idx) const;

  bool is_memory_registered(uint32_t buffer_id) const;
  bool register_memory(uint32_t buffer_id, void* ptr, size_t len);
  void deregister_memory(uint32_t buffer_id);

  bool ensure_put_path(PeerConnectSpec const& spec) override;
  bool ensure_wait_path(PeerConnectSpec const& spec) override;
  bool has_put_path(int peer) const override;
  bool has_wait_path(int peer) const override;
  bool is_initialized() const { return endpoint_ != nullptr; }

  unsigned send_put_async(int peer, void* local_ptr, uint32_t local_buf,
                          void* remote_ptr, uint32_t remote_buf, size_t len,
                          unsigned comm_rid) override;
  unsigned send_signal_async(int peer, uint64_t tag, unsigned comm_rid) override;
  unsigned wait_signal_async(int peer, uint64_t tag, std::optional<WaitTarget>,
                             unsigned comm_rid) override;

 private:
  enum class Kind : uint8_t { DataPut, DataWait, Signal, SignalWait };

  struct RingElem {
    unsigned comm_rid;
    int peer;
    Kind kind;
    void* ptr;
    size_t len;
    uint32_t buffer_id;
    uint64_t tag;
  };

  struct PeerCtx {
    ::uccl::UcclFlow* send_flow = nullptr;
    ::uccl::UcclFlow* recv_flow = nullptr;
    ::uccl::Mhandle* control_mhandle = nullptr;
    uint64_t control_tag = 0;
  };

  void send_worker();
  void recv_worker();

  bool connect_to_peer(int peer_rank, std::string const& remote_ip,
                       uint16_t remote_port, int local_dev_idx,
                       int local_gpu_idx, int remote_dev_idx,
                       int remote_gpu_idx);
  bool accept_from_peer(int peer_rank, std::string const& expected_remote_ip,
                        int expected_remote_dev_idx,
                        int expected_remote_gpu_idx,
                        uint16_t expected_remote_port = 0);

  int gpu_id_;
  int local_dev_idx_ = -1;
  std::unique_ptr<::uccl::RDMAEndpoint> endpoint_;
  jring_t* send_ring_ = nullptr;
  jring_t* recv_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_th_;
  std::thread recv_th_;
  mutable std::mutex mu_;
  std::unordered_map<int, PeerCtx> peers_;
  std::unordered_map<uint32_t, ::uccl::Mhandle*> buffer_id_to_mhandle_;
};
}  // namespace Transport
}  // namespace UKernel
