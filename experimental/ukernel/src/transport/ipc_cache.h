#pragma once

#include "oob.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

class IpcCacheManager {
 public:
  IpcCacheManager() = default;
  ~IpcCacheManager() = default;

  struct IpcCache {
    gpuIpcMemHandle_t handle;
    bool is_send;
    void* direct_ptr;
    uintptr_t offset;
    size_t size;
    int device_idx = -1;
  };

  bool register_cache(int remote_rank, gpuIpcMemHandle_t handle,
                     IpcCache const& cache);
  IpcCache get_cache(int remote_rank, gpuIpcMemHandle_t handle) const;
  void clear_all();

 private:
  using HandleKey = std::array<uint8_t, sizeof(gpuIpcMemHandle_t)>;

  static HandleKey make_handle_key(gpuIpcMemHandle_t const& h) {
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

  mutable std::mutex mu_;
  std::unordered_map<int, HandleCacheMap> rank_handle_to_cache_;
};

}  // namespace Transport
}  // namespace UKernel
