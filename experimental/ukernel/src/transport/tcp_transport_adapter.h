#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace UKernel {
namespace Transport {

class TcpTransportAdapter {
 public:
  struct AcceptedPeer {
    int remote_rank = -1;
    std::string remote_ip;
  };

  TcpTransportAdapter(std::string local_ip, int local_rank, int world_size);
  ~TcpTransportAdapter();

  uint16_t get_listen_port() const;
  std::string const& get_listen_ip() const { return local_ip_; }

  bool connect_to_peer(int peer_rank, std::string remote_ip,
                       uint16_t remote_port);
  bool accept_from_peer(int peer_rank, std::string const& expected_remote_ip,
                        AcceptedPeer* accepted_peer = nullptr);

  bool has_send_peer(int peer_rank) const;
  bool has_recv_peer(int peer_rank) const;

  int send_async(int peer_rank, void const* host_ptr, size_t len,
                 uint64_t request_id);
  int recv_async(int peer_rank, void* host_ptr, size_t len, uint64_t request_id);

  bool poll_completion(uint64_t request_id);
  bool wait_completion(uint64_t request_id);
  bool request_failed(uint64_t request_id) const;
  void release_request(uint64_t request_id);

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

  void join_and_erase_request(uint64_t request_id);

  std::string local_ip_;
  int local_rank_ = -1;
  int world_size_ = 0;
  int listen_fd_ = -1;
  uint16_t listen_port_ = 0;

  mutable std::mutex mu_;
  std::unordered_map<int, std::shared_ptr<PeerContext>> peer_contexts_;
  std::unordered_map<uint64_t, std::shared_ptr<PendingRequest>> pending_;
};

}  // namespace Transport
}  // namespace UKernel
