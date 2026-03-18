#pragma once

#include "../../include/config.h"
#include "oob.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace UKernel {
namespace Transport {

class PeerSession {
 public:
  explicit PeerSession(int peer_rank);
  ~PeerSession() = default;

  int peer_rank() const { return peer_rank_; }

  bool has_meta() const;
  CommunicatorMeta const& meta() const;
  void set_meta(CommunicatorMeta const& meta);

  PeerTransportKind transport_kind() const;
  void set_transport_kind(PeerTransportKind kind);

  bool send_ready() const;
  bool recv_ready() const;
  void set_send_ready(bool ready);
  void set_recv_ready(bool ready);

  void reset();

 private:
  int const peer_rank_;
  bool has_meta_ = false;
  CommunicatorMeta meta_;
  PeerTransportKind kind_ = PeerTransportKind::Ipc;
  bool send_ready_ = false;
  bool recv_ready_ = false;
  mutable std::mutex mu_;
};

class PeerSessionManager {
 public:
  explicit PeerSessionManager(int world_size);

  PeerSession* get(int rank);
  PeerSession const* get(int rank) const;
  int world_size() const { return world_size_; }

  bool all_have_meta() const;
  bool all_ready() const;
  bool has_peer_send_path(int rank) const;
  bool has_peer_recv_path(int rank) const;

 private:
  int world_size_;
  std::vector<std::unique_ptr<PeerSession>> peers_;
  mutable std::mutex mu_;
};

}  // namespace Transport
}  // namespace UKernel
