#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

extern "C" {
#include "../util/jring.h"
}

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

struct RdmaPeerConnectSpec {
  static constexpr int kMaxQPs = 4;

  uint32_t remote_data_qpns[kMaxQPs] = {};
  uint32_t remote_signal_qpn = 0;
  uint8_t num_qps = kMaxQPs;
  uint16_t remote_lid = 0;
  std::array<uint8_t, 16> remote_gid_raw = {};

  int local_dev_idx = -1;
  int local_gpu_idx = -1;
  int remote_dev_idx = -1;
  int remote_gpu_idx = -1;
};

struct PeerConnectSpec {
  int peer_rank = -1;
  PeerConnectType type = PeerConnectType::Connect;
  std::variant<std::monostate, TcpPeerConnectSpec, UcclPeerConnectSpec,
               IpcPeerConnectSpec, RdmaPeerConnectSpec>
      detail{};
};

// Pushed to completion ring by adapter worker threads on operation completion.
struct CompletionEvent {
  unsigned rid;       // Communicator-level request ID
  unsigned failed;    // 0 = success, 1 = failed
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

  // Async submission. comm_rid is pushed to completion_ring on completion.
  // Returns non-zero on success (the comm_rid itself is the identifier).
  virtual unsigned send_put_async(int peer_rank, void* local_ptr,
                                  uint32_t local_buffer_id, void* remote_ptr,
                                  uint32_t remote_buffer_id, size_t len,
                                  unsigned comm_rid) = 0;
  virtual unsigned send_signal_async(int peer_rank, uint64_t tag,
                                     unsigned comm_rid) = 0;
  virtual unsigned wait_signal_async(int peer_rank, uint64_t expected_tag,
                                     std::optional<WaitTarget> target,
                                     unsigned comm_rid) = 0;

  void set_completion_ring(jring_t* ring) { completion_ring_ = ring; }

 protected:
  jring_t* completion_ring_ = nullptr;

  void publish_completion(unsigned rid, bool failed) {
    if (!completion_ring_) return;
    CompletionEvent ev{rid, failed ? 1u : 0u};
    while (jring_mp_enqueue_bulk(completion_ring_, &ev, 1, nullptr) != 1)
      std::this_thread::yield();
  }
};

}  // namespace Transport
}  // namespace UKernel
