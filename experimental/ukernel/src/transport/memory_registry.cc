#include "memory_registry.h"
#include <stdexcept>

namespace UKernel {
namespace Transport {

MemoryRegistry::MemoryRegistry() = default;
MemoryRegistry::~MemoryRegistry() = default;

MR MemoryRegistry::track_local_buffer(void* local_buf, size_t len) {
  std::lock_guard<std::mutex> lk(local_mu_);

  auto existing = ptr_to_local_mr_id_.find(local_buf);
  if (existing != ptr_to_local_mr_id_.end()) {
    auto info = mr_id_to_local_mr_.find(existing->second);
    if (info != mr_id_to_local_mr_.end()) return info->second;
  }

  uint32_t id = next_mr_id_++;
  MR info{};
  info.id = id;
  info.address = reinterpret_cast<uint64_t>(local_buf);
  info.length = static_cast<uint32_t>(len);
  info.lkey = 0;
  info.key = 0;

  ptr_to_local_mr_id_[local_buf] = id;
  mr_id_to_local_mr_[id] = info;
  return info;
}

MemoryRegistry::ReleasedLocalMr MemoryRegistry::release_local_buffer(
    void* local_buf) {
  std::lock_guard<std::mutex> lk(local_mu_);

  ReleasedLocalMr released{};
  auto it = ptr_to_local_mr_id_.find(local_buf);
  if (it == ptr_to_local_mr_id_.end()) {
    return released;
  }

  released.local_mr_id = it->second;
  released.has_local_mr_id = true;
  ptr_to_local_mr_id_.erase(it);

  auto info_it = mr_id_to_local_mr_.find(released.local_mr_id);
  if (info_it != mr_id_to_local_mr_.end()) {
    mr_id_to_local_mr_.erase(info_it);
  }

  return released;
}

std::vector<void*> MemoryRegistry::local_buffers() const {
  std::lock_guard<std::mutex> lk(local_mu_);
  std::vector<void*> bufs;
  bufs.reserve(ptr_to_local_mr_id_.size());
  for (auto const& kv : ptr_to_local_mr_id_) {
    bufs.push_back(kv.first);
  }
  return bufs;
}

void MemoryRegistry::clear_remote_ipc_cache() {
  ipc_cache_.clear_all();
}

MR MemoryRegistry::get_local_mr(void* local_buf) const {
  uint64_t buf_addr = reinterpret_cast<uint64_t>(local_buf);

  std::lock_guard<std::mutex> lk(local_mu_);
  for (auto const& kv : mr_id_to_local_mr_) {
    MR const& mr = kv.second;
    if (buf_addr >= mr.address && buf_addr < mr.address + mr.length) {
      return mr;
    }
  }

  throw std::runtime_error("Local MR info not found");
}

MR MemoryRegistry::get_local_mr(uint32_t mr_id) const {
  std::lock_guard<std::mutex> lk(local_mu_);
  auto it = mr_id_to_local_mr_.find(mr_id);
  if (it == mr_id_to_local_mr_.end()) {
    throw std::runtime_error("Local MR not found for buffer");
  }
  return it->second;
}

void MemoryRegistry::cache_remote_mrs(int remote_rank,
                                      std::vector<MR> const& mrs) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto& cached = rank_mr_id_to_remote_mr_[remote_rank];
  auto& pending = rank_to_pending_remote_mrs_[remote_rank];
  for (auto const& mr : mrs) {
    if (cached.find(mr.id) != cached.end()) continue;
    cached[mr.id] = mr;
    pending.push_back(mr);
  }
}

bool MemoryRegistry::take_pending_remote_mr(int remote_rank, MR& out) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto& pending = rank_to_pending_remote_mrs_[remote_rank];
  if (pending.empty()) return false;
  out = pending.front();
  pending.pop_front();
  return true;
}

MR MemoryRegistry::get_remote_mr(int remote_rank, uint32_t mr_id) const {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto it_rank = rank_mr_id_to_remote_mr_.find(remote_rank);
  if (it_rank == rank_mr_id_to_remote_mr_.end()) {
    throw std::runtime_error("No MR cached for remote rank");
  }
  auto it_mr = it_rank->second.find(mr_id);
  if (it_mr == it_rank->second.end()) {
    throw std::runtime_error("Remote MR not found for id=" +
                             std::to_string(mr_id));
  }
  return it_mr->second;
}

bool MemoryRegistry::register_remote_ipc_cache(int remote_rank,
                                               gpuIpcMemHandle_t handle,
                                               IpcCacheManager::IpcCache const& cache) {
  return ipc_cache_.register_cache(remote_rank, handle, cache);
}

IpcCacheManager::IpcCache MemoryRegistry::get_remote_ipc_cache(
    int remote_rank, gpuIpcMemHandle_t handle) const {
  return ipc_cache_.get_cache(remote_rank, handle);
}

}  // namespace Transport
}  // namespace UKernel
