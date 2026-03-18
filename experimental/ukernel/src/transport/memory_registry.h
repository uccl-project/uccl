#pragma once

#include "ipc_cache.h"
#include "oob.h"
#include <cstddef>
#include <cstdint>
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

  MemoryRegistry();
  ~MemoryRegistry();

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
                                  IpcCacheManager::IpcCache const& cache);
  IpcCacheManager::IpcCache get_remote_ipc_cache(int remote_rank,
                                  gpuIpcMemHandle_t handle) const;

 private:
  mutable std::mutex local_mu_;
  std::unordered_map<void*, uint32_t> ptr_to_local_mr_id_;
  std::unordered_map<uint32_t, MR> mr_id_to_local_mr_;
  uint32_t next_mr_id_ = 1;

  mutable std::mutex remote_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, MR>>
      rank_mr_id_to_remote_mr_;
  std::unordered_map<int, std::deque<MR>> rank_to_pending_remote_mrs_;

  IpcCacheManager ipc_cache_;
};

}  // namespace Transport
}  // namespace UKernel
