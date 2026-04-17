#include "ipc_manager.h"
#include <unordered_set>

namespace UKernel {
namespace Transport {

bool IPCManager::has_handle(gpuIpcMemHandle_t const& h) {
  auto const* bytes = reinterpret_cast<unsigned char const*>(&h);
  for (size_t i = 0; i < sizeof(gpuIpcMemHandle_t); ++i) {
    if (bytes[i] != 0) return true;
  }
  return false;
}

void IPCManager::close_local_import_if_open(IPCItem& item) {
  if (item.is_local || item.direct_ptr == nullptr || item.device_idx < 0) {
    return;
  }
  int original_device = -1;
  bool have_original_device = (gpuGetDevice(&original_device) == gpuSuccess);
  GPU_RT_CHECK(gpuSetDevice(item.device_idx));
  GPU_RT_CHECK(gpuIpcCloseMemHandle(item.direct_ptr));
  item.direct_ptr = nullptr;
  if (have_original_device) {
    GPU_RT_CHECK(gpuSetDevice(original_device));
  }
}

bool IPCManager::register_remote_ipc(int rank, IPCItem const& ipc) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  IPCItem item = ipc;
  item.is_local = false;
  if (item.buffer_id != 0) {
    rank_buffer_id_cache_[rank][item.buffer_id] = item;
  }
  if (has_handle(item.handle)) {
    rank_handle_cache_[rank][make_ipc_handle_key(item.handle)] = item;
  }
  return true;
}

IPCItem IPCManager::create_local_ipc(void* ptr, size_t len, int device_idx) {
  if (ptr == nullptr || len == 0) return {};

  void* base = nullptr;
  size_t base_size = 0;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, ptr));
  uintptr_t key = reinterpret_cast<uintptr_t>(base);

  {
    std::lock_guard<std::mutex> lk(local_mu_);
    auto it = local_ipc_cache_.find(key);
    if (it != local_ipc_cache_.end()) {
      return it->second;
    }
  }

  IPCItem created{};
  GPU_RT_CHECK(gpuIpcGetMemHandle(&created.handle, base));
  created.base_addr = key;
  created.base_offset = reinterpret_cast<uintptr_t>(ptr) - key;
  created.bytes = len;
  created.device_idx = device_idx;
  created.is_local = true;
  created.valid = true;

  std::lock_guard<std::mutex> lk(local_mu_);
  auto [it, inserted] = local_ipc_cache_.emplace(key, created);
  if (!inserted) return it->second;
  return created;
}

IPCItem IPCManager::get_ipc(int rank, gpuIpcMemHandle_t handle) const {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = rank_handle_cache_.find(rank);
  if (rank_it == rank_handle_cache_.end()) return {};
  auto it = rank_it->second.find(make_ipc_handle_key(handle));
  if (it == rank_it->second.end()) return {};
  return it->second;
}

IPCItem IPCManager::get_ipc(int rank, uint32_t buffer_id) const {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = rank_buffer_id_cache_.find(rank);
  if (rank_it == rank_buffer_id_cache_.end()) return {};
  auto it = rank_it->second.find(buffer_id);
  if (it == rank_it->second.end()) return {};
  return it->second;
}

IPCItem IPCManager::get_ipc(void* local_ptr) const {
  if (local_ptr == nullptr) return {};
  void* base = nullptr;
  size_t base_size = 0;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, local_ptr));
  (void)base_size;
  uintptr_t key = reinterpret_cast<uintptr_t>(base);

  std::lock_guard<std::mutex> lk(local_mu_);
  auto it = local_ipc_cache_.find(key);
  if (it == local_ipc_cache_.end()) return {};
  return it->second;
}

bool IPCManager::delete_ipc(int rank, gpuIpcMemHandle_t handle) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = rank_handle_cache_.find(rank);
  if (rank_it == rank_handle_cache_.end()) return false;
  auto key = make_ipc_handle_key(handle);
  auto it = rank_it->second.find(key);
  if (it == rank_it->second.end()) return false;

  IPCItem item = it->second;
  close_local_import_if_open(item);
  rank_it->second.erase(it);
  if (item.buffer_id != 0) {
    auto id_rank_it = rank_buffer_id_cache_.find(rank);
    if (id_rank_it != rank_buffer_id_cache_.end()) {
      id_rank_it->second.erase(item.buffer_id);
    }
  }
  return true;
}

bool IPCManager::delete_ipc(int rank, uint32_t buffer_id) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = rank_buffer_id_cache_.find(rank);
  if (rank_it == rank_buffer_id_cache_.end()) return false;
  auto it = rank_it->second.find(buffer_id);
  if (it == rank_it->second.end()) return false;

  IPCItem item = it->second;
  close_local_import_if_open(item);
  rank_it->second.erase(it);
  if (has_handle(item.handle)) {
    auto handle_rank_it = rank_handle_cache_.find(rank);
    if (handle_rank_it != rank_handle_cache_.end()) {
      handle_rank_it->second.erase(make_ipc_handle_key(item.handle));
    }
  }
  return true;
}

bool IPCManager::delete_ipc(void* local_ptr) {
  if (local_ptr == nullptr) return false;
  uintptr_t key = reinterpret_cast<uintptr_t>(local_ptr);
  {
    std::lock_guard<std::mutex> lk(local_mu_);
    auto it = local_ipc_cache_.find(key);
    if (it != local_ipc_cache_.end()) {
      local_ipc_cache_.erase(it);
      return true;
    }
  }

  void* base = nullptr;
  size_t base_size = 0;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, local_ptr));
  (void)base_size;
  key = reinterpret_cast<uintptr_t>(base);
  std::lock_guard<std::mutex> lk(local_mu_);
  return local_ipc_cache_.erase(key) > 0;
}

void IPCManager::delete_ipc(int rank) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  std::unordered_set<void*> closed_ptrs;

  auto close_from_map = [&](auto& cache_map) {
    auto rank_it = cache_map.find(rank);
    if (rank_it == cache_map.end()) return;
    for (auto& kv : rank_it->second) {
      IPCItem& item = kv.second;
      if (item.direct_ptr == nullptr) continue;
      if (!closed_ptrs.insert(item.direct_ptr).second) continue;
      close_local_import_if_open(item);
    }
  };

  close_from_map(rank_handle_cache_);
  close_from_map(rank_buffer_id_cache_);
  rank_handle_cache_.erase(rank);
  rank_buffer_id_cache_.erase(rank);
}

}  // namespace Transport
}  // namespace UKernel
