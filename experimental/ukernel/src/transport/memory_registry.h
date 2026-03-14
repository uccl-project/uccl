#pragma once

#include "oob.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

class MemoryRegistry {
 public:
  struct ReleasedLocalMr {
    uint32_t local_mr_id = 0;
    bool has_local_mr_id = false;
  };

  MR track_local_buffer(void* local_buf, size_t len);
  ReleasedLocalMr release_local_buffer(void* local_buf);
  std::vector<void*> local_buffers() const;
  void clear_remote_ipc_cache();

  MR get_local_mr(void* local_buf) const;
  MR get_local_mr(uint32_t mr_id) const;

  void cache_remote_mrs(int remote_rank, std::vector<MR> const& mrs);
  bool take_pending_remote_mr(int remote_rank, MR& out);
  MR get_remote_mr(int remote_rank, uint32_t mr_id) const;

  bool register_remote_ipc_cache(int remote_rank, gpuIpcMemHandle_t handle,
                                 IpcCache const& cache);
  IpcCache get_remote_ipc_cache(int remote_rank,
                                gpuIpcMemHandle_t handle) const;

 private:
  using HandleKey = std::array<uint8_t, sizeof(gpuIpcMemHandle_t)>;

  static inline HandleKey make_handle_key(gpuIpcMemHandle_t const& h) {
    HandleKey k{};
    std::memcpy(k.data(), &h, k.size());
    return k;
  }

  struct HandleKeyHash {
    size_t operator()(HandleKey const& k) const noexcept {
      uint64_t hash = 1469598103934665603ull;
      for (uint8_t b : k) {
        hash ^= b;
        hash *= 1099511628211ull;
      }
      return static_cast<size_t>(hash);
    }
  };

  using HandleCacheMap = std::unordered_map<HandleKey, IpcCache, HandleKeyHash>;

  mutable std::mutex local_mu_;
  std::unordered_map<void*, uint32_t> ptr_to_local_mr_id_;
  std::unordered_map<uint32_t, MR> mr_id_to_local_mr_;
  uint32_t next_mr_id_ = 1;

  mutable std::mutex remote_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, MR>>
      rank_mr_id_to_remote_mr_;
  std::unordered_map<int, std::deque<MR>> rank_to_pending_remote_mrs_;

  mutable std::mutex ipc_cache_mu_;
  std::unordered_map<int, HandleCacheMap> rank_handle_to_ipc_cache_;
};

}  // namespace Transport
}  // namespace UKernel
