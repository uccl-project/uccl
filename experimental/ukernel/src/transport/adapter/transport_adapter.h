#pragma once

#include "../communicator.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace UKernel {
namespace Transport {

struct BounceBufferInfo {
  void* ptr = nullptr;
  uint32_t mr_id = 0;
  std::string shm_name;
  bool valid() const { return ptr != nullptr; }
};

class TransportAdapter {
 public:
  using BounceBufferProvider = std::function<BounceBufferInfo(size_t)>;

  virtual ~TransportAdapter() = default;

  virtual bool connect(int peer_rank) = 0;
  virtual bool accept(int peer_rank) = 0;
  virtual bool has_send_path(int peer_rank) const = 0;
  virtual bool has_recv_path(int peer_rank) const = 0;

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

  virtual int peer_count() const = 0;
};

}  // namespace Transport
}  // namespace UKernel
