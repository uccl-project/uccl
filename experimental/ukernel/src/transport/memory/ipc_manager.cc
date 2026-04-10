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

bool IPCManager::local_cache_contains(IPCItem const& item,
                                      uintptr_t ptr_addr) noexcept {
  if (!item.valid || item.base_addr == 0 || item.allocation_size == 0) {
    return false;
  }
  return ptr_addr >= item.base_addr &&
         (ptr_addr - item.base_addr) < item.allocation_size;
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
  if (item.ipc_id != 0) {
    rank_ipc_id_cache_[rank][item.ipc_id] = item;
  }
  if (has_handle(item.handle)) {
    rank_handle_cache_[rank][make_ipc_handle_key(item.handle)] = item;
  }
  return true;
}

IPCItem IPCManager::create_local_ipc(void* ptr, size_t len, int device_idx) {
  if (ptr == nullptr || len == 0) return {};
  uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);

  void* base = nullptr;
  size_t base_size = 0;
  {
    std::lock_guard<std::mutex> lk(local_mu_);
    for (auto const& kv : local_ipc_cache_) {
      if (!local_cache_contains(kv.second, ptr_addr)) continue;
      IPCItem hit = kv.second;
      hit.base_offset = ptr_addr - hit.base_addr;
      hit.bytes = len;
      return hit;
    }
  }

  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, ptr));
  uintptr_t key = reinterpret_cast<uintptr_t>(base);

  IPCItem created{};
  GPU_RT_CHECK(gpuIpcGetMemHandle(&created.handle, base));
  created.base_addr = key;
  created.base_offset = ptr_addr - key;
  created.bytes = len;
  created.allocation_size = base_size;
  created.device_idx = device_idx;
  created.is_local = true;
  created.valid = true;

  std::lock_guard<std::mutex> lk(local_mu_);
  auto [it, inserted] = local_ipc_cache_.emplace(key, created);
  IPCItem out = inserted ? created : it->second;
  out.base_offset = ptr_addr - out.base_addr;
  out.bytes = len;
  if (out.allocation_size == 0) out.allocation_size = base_size;
  return out;
}

IPCItem IPCManager::get_ipc(int rank, gpuIpcMemHandle_t handle) const {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = rank_handle_cache_.find(rank);
  if (rank_it == rank_handle_cache_.end()) return {};
  auto it = rank_it->second.find(make_ipc_handle_key(handle));
  if (it == rank_it->second.end()) return {};
  return it->second;
}

IPCItem IPCManager::get_ipc(int rank, uint32_t ipc_id) const {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = rank_ipc_id_cache_.find(rank);
  if (rank_it == rank_ipc_id_cache_.end()) return {};
  auto it = rank_it->second.find(ipc_id);
  if (it == rank_it->second.end()) return {};
  return it->second;
}

IPCItem IPCManager::get_ipc(void* local_ptr) const {
  if (local_ptr == nullptr) return {};
  uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(local_ptr);

  std::lock_guard<std::mutex> lk(local_mu_);
  for (auto const& kv : local_ipc_cache_) {
    if (!local_cache_contains(kv.second, ptr_addr)) continue;
    IPCItem hit = kv.second;
    hit.base_offset = ptr_addr - hit.base_addr;
    if (hit.allocation_size >= hit.base_offset) {
      hit.bytes = hit.allocation_size - hit.base_offset;
    }
    return hit;
  }
  return {};
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
  if (item.ipc_id != 0) {
    auto id_rank_it = rank_ipc_id_cache_.find(rank);
    if (id_rank_it != rank_ipc_id_cache_.end()) {
      id_rank_it->second.erase(item.ipc_id);
    }
  }
  return true;
}

bool IPCManager::delete_ipc(int rank, uint32_t ipc_id) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = rank_ipc_id_cache_.find(rank);
  if (rank_it == rank_ipc_id_cache_.end()) return false;
  auto it = rank_it->second.find(ipc_id);
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
  uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(local_ptr);
  {
    std::lock_guard<std::mutex> lk(local_mu_);
    for (auto it = local_ipc_cache_.begin(); it != local_ipc_cache_.end();
         ++it) {
      if (!local_cache_contains(it->second, ptr_addr)) continue;
      local_ipc_cache_.erase(it);
      return true;
    }
  }
  return false;
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
  close_from_map(rank_ipc_id_cache_);
  rank_handle_cache_.erase(rank);
  rank_ipc_id_cache_.erase(rank);
}

}  // namespace Transport
}  // namespace UKernel
