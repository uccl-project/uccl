#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace UKernel {
namespace Transport {

struct BounceBufferInfo {
  void* ptr = nullptr;
  uint32_t buffer_id = 0;
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
  struct WaitTarget {
    void* local_ptr = nullptr;
    size_t len = 0;
    uint32_t local_buffer_id = 0;
  };

  virtual ~TransportAdapter() = default;

  // Directional path readiness.
  virtual bool ensure_put_path(PeerConnectSpec const& spec) = 0;
  virtual bool ensure_wait_path(PeerConnectSpec const& spec) = 0;
  virtual bool has_put_path(int peer_rank) const = 0;
  virtual bool has_wait_path(int peer_rank) const = 0;

  // Async data path.
  // local_ptr / local_buffer_id: source data (may be host bounce buffer).
  // remote_ptr / remote_buffer_id: resolved destination on peer, or nullptr/0
  //   (for send/recv model where peer posts its own recv buffer).
  virtual unsigned put_async(
      int peer_rank, void* local_ptr, uint32_t local_buffer_id,
      void* remote_ptr, uint32_t remote_buffer_id, size_t len) = 0;
  virtual unsigned signal_async(int peer_rank, uint64_t tag) = 0;
  virtual unsigned wait_async(
      int peer_rank, uint64_t expected_tag,
      std::optional<WaitTarget> target = std::nullopt) = 0;

  virtual bool poll_completion(unsigned id) = 0;
  virtual bool wait_completion(unsigned id) = 0;
  virtual bool request_failed(unsigned id) = 0;
  virtual void release_request(unsigned id) = 0;

  // Return opaque completion payload (e.g. SHM buffer_id for IPC relay).
  virtual uint64_t completion_payload(unsigned id) const { return 0; }
};

}  // namespace Transport
}  // namespace UKernel
