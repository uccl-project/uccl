#pragma once

#include "ipc_cache.h"
#include "oob.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

class MemoryRegistry {
 public:
  struct LocalRegistrationInfo {
    void* local_buf = nullptr;
    MR mr{};
  };

  struct ReleasedLocalMr {
    void* local_buf = nullptr;
    uint32_t local_mr_id = 0;
    size_t len = 0;
    bool has_local_mr_id = false;
  };

  struct TrackLocalBufferResult {
    MR mr{};
    ReleasedLocalMr replaced{};
    bool reused_existing = false;
  };

  MemoryRegistry();
  ~MemoryRegistry();

  TrackLocalBufferResult track_local_buffer(void* local_buf, size_t len);
  ReleasedLocalMr release_local_buffer(void* local_buf);
  std::vector<LocalRegistrationInfo> local_registrations() const;
  void clear_remote_ipc_cache();

  MR get_local_mr(void* local_buf) const;
  MR get_local_mr(uint32_t mr_id) const;

  void cache_remote_mrs(int remote_rank, std::vector<MR> const& mrs);
  MR get_remote_mr(int remote_rank, uint32_t mr_id) const;

  bool register_remote_ipc_cache(int remote_rank, gpuIpcMemHandle_t handle,
                                 IpcCacheManager::IpcCache const& cache);
  IpcCacheManager::IpcCache get_remote_ipc_cache(
      int remote_rank, gpuIpcMemHandle_t handle) const;

 private:
  struct LocalRegistration {
    uint32_t mr_id = 0;
    size_t len = 0;
  };

  mutable std::mutex local_mu_;
  std::unordered_map<void*, LocalRegistration> ptr_to_local_mr_;
  std::unordered_map<uint32_t, MR> mr_id_to_local_mr_;
  uint32_t next_mr_id_ = 1;

  mutable std::mutex remote_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, MR>>
      rank_mr_id_to_remote_mr_;

  IpcCacheManager ipc_cache_;
};

}  // namespace Transport
}  // namespace UKernel
