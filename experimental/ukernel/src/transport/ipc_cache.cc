#include "ipc_cache.h"

namespace UKernel {
namespace Transport {

bool IpcCacheManager::register_cache(int remote_rank, gpuIpcMemHandle_t handle,
                                     IpcCache const& cache) {
  std::lock_guard<std::mutex> lk(mu_);
  rank_handle_to_cache_[remote_rank][make_handle_key(handle)] = cache;
  return true;
}

IpcCacheManager::IpcCache IpcCacheManager::get_cache(
    int remote_rank, gpuIpcMemHandle_t handle) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it_rank = rank_handle_to_cache_.find(remote_rank);
  if (it_rank == rank_handle_to_cache_.end()) return IpcCache{};

  auto it = it_rank->second.find(make_handle_key(handle));
  if (it == it_rank->second.end()) return IpcCache{};
  return it->second;
}

void IpcCacheManager::clear_all() {
  std::lock_guard<std::mutex> lk(mu_);
  int original_device = -1;
  bool have_original_device = (gpuGetDevice(&original_device) == gpuSuccess);
  for (auto& [rank, caches] : rank_handle_to_cache_) {
    (void)rank;
    for (auto& [handle, cache] : caches) {
      (void)handle;
      if (cache.direct_ptr == nullptr || cache.device_idx < 0) continue;
      GPU_RT_CHECK(gpuSetDevice(cache.device_idx));
      GPU_RT_CHECK(gpuIpcCloseMemHandle(cache.direct_ptr));
      cache.direct_ptr = nullptr;
    }
  }
  if (have_original_device) {
    GPU_RT_CHECK(gpuSetDevice(original_device));
  }
  rank_handle_to_cache_.clear();
}

}  // namespace Transport
}  // namespace UKernel
