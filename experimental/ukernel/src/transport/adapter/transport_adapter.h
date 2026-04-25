#pragma once

#include "../request.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace UKernel {
namespace Transport {

struct BounceBufferInfo {
  void* ptr = nullptr;
  uint32_t mr_id = 0;
  std::string shm_name;
  bool valid() const { return ptr != nullptr; }
};

enum class PeerConnectType : uint8_t { Connect = 0, Accept = 1 };

struct TcpPeerConnectSpec {
  std::string remote_ip;
  uint16_t remote_port = 0;
};

struct UcclPeerConnectSpec {
  std::string remote_ip;
  uint16_t remote_port = 0;
  int local_dev_idx = -1;
  int local_gpu_idx = -1;
  int remote_dev_idx = -1;
  int remote_gpu_idx = -1;
};

struct IpcPeerConnectSpec {};

struct PeerConnectSpec {
  int peer_rank = -1;
  PeerConnectType type = PeerConnectType::Connect;
  std::variant<std::monostate, TcpPeerConnectSpec, UcclPeerConnectSpec,
               IpcPeerConnectSpec>
      detail{};
};

class TransportAdapter {
 public:
  using BounceBufferProvider = std::function<BounceBufferInfo(size_t)>;

  virtual ~TransportAdapter() = default;

  virtual bool ensure_peer(PeerConnectSpec const& spec) = 0;
  // Duplex readiness: both directions are established for this peer.
  virtual bool has_peer(int peer_rank) const = 0;

  virtual unsigned send_async(
      int peer_rank, void* local_ptr, size_t len, uint64_t local_mr_id,
      std::optional<RemoteSlice> remote_hint,
      BounceBufferProvider bounce_provider = nullptr) = 0;
  virtual unsigned recv_async(
      int peer_rank, void* local_ptr, size_t len, uint64_t local_mr_id,
      BounceBufferProvider bounce_provider = nullptr) = 0;

  virtual bool poll_completion(unsigned id) = 0;
  virtual bool wait_completion(unsigned id) = 0;
  virtual bool request_failed(unsigned id) = 0;
  virtual void release_request(unsigned id) = 0;
};

}  // namespace Transport
}  // namespace UKernel
