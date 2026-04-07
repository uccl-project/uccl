#pragma once

#include "transport_adapter.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

class TcpTransportAdapter final : public TransportAdapter {
 public:
  struct AcceptedPeer {
    int remote_rank = -1;
    std::string remote_ip;
  };

  TcpTransportAdapter(std::string local_ip, int local_rank);
  ~TcpTransportAdapter() override;

  uint16_t get_listen_port() const;
  std::string const& get_listen_ip() const { return local_ip_; }

  bool connect_to_peer(int peer_rank, std::string remote_ip,
                       uint16_t remote_port);
  bool accept_from_peer(int peer_rank, std::string const& expected_remote_ip,
                        AcceptedPeer* accepted_peer = nullptr);
  bool connect(int peer_rank) override { return has_send_peer(peer_rank); }
  bool accept(int peer_rank) override { return has_recv_peer(peer_rank); }

  bool has_send_peer(int peer_rank) const;
  bool has_recv_peer(int peer_rank) const;
  bool has_send_path(int peer_rank) const override;
  bool has_recv_path(int peer_rank) const override;

  unsigned send_async(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id,
                      std::optional<RemoteSlice> remote_hint,
                      BounceBufferProvider bounce_provider = nullptr) override;
  unsigned recv_async(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id,
                      BounceBufferProvider bounce_provider = nullptr) override;

  bool poll_completion(unsigned id) override;
  bool wait_completion(unsigned id) override;
  bool request_failed(unsigned id) override;
  void release_request(unsigned id) override;

  int peer_count() const override {
    return static_cast<int>(peer_contexts_.size());
  }

  int send_async_tcp(int peer_rank, void const* host_ptr, size_t len,
                     unsigned request_id);
  int recv_async_tcp(int peer_rank, void* host_ptr, size_t len,
                     unsigned request_id);

 private:
  struct Handshake {
    uint32_t src_rank = 0;
  };

  struct HandshakeAck {
    uint32_t accepted = 1;
  };

  struct PeerContext {
    int send_fd = -1;
    int recv_fd = -1;
    std::string remote_ip;
    std::mutex send_mu;
    std::mutex recv_mu;
  };

  struct PendingRequest {
    std::atomic<bool> completed{false};
    std::atomic<bool> failed{false};
    std::thread worker;
  };

  static int create_listen_socket(uint16_t& out_port);
  static bool connect_socket(int& out_fd, std::string const& remote_ip,
                             uint16_t remote_port,
                             std::chrono::milliseconds timeout);
  static bool send_handshake(int fd, Handshake const& hs);
  static bool recv_handshake(int fd, Handshake& hs);
  static bool send_handshake_ack(int fd, HandshakeAck const& ack);
  static bool recv_handshake_ack(int fd, HandshakeAck& ack);
  static bool send_all(int fd, void const* buf, size_t len);
  static bool recv_all(int fd, void* buf, size_t len);

  void join_and_erase_request(unsigned request_id);

  std::string local_ip_;
  int local_rank_ = -1;
  int listen_fd_ = -1;
  uint16_t listen_port_ = 0;

  mutable std::mutex mu_;
  std::unordered_map<int, std::shared_ptr<PeerContext>> peer_contexts_;
  std::unordered_map<unsigned, std::shared_ptr<PendingRequest>> pending_;
  std::atomic<unsigned> next_request_id_{1};
};

}  // namespace Transport
}  // namespace UKernel
