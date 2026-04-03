#include "memory_registry.h"
#include <stdexcept>

namespace UKernel {
namespace Transport {

MemoryRegistry::MemoryRegistry() = default;
MemoryRegistry::~MemoryRegistry() = default;

MemoryRegistry::TrackLocalBufferResult MemoryRegistry::track_local_buffer(
    void* local_buf, size_t len) {
  std::lock_guard<std::mutex> lk(local_mu_);

  TrackLocalBufferResult tracked{};

  auto existing = ptr_to_local_mr_.find(local_buf);
  if (existing != ptr_to_local_mr_.end()) {
    if (existing->second.len == len) {
      auto info = mr_id_to_local_mr_.find(existing->second.mr_id);
      if (info != mr_id_to_local_mr_.end()) {
        tracked.mr = info->second;
        tracked.reused_existing = true;
        return tracked;
      }
    } else {
      tracked.replaced.local_buf = local_buf;
      tracked.replaced.local_mr_id = existing->second.mr_id;
      tracked.replaced.len = existing->second.len;
      tracked.replaced.has_local_mr_id = true;
      mr_id_to_local_mr_.erase(existing->second.mr_id);
      ptr_to_local_mr_.erase(existing);
    }
  }

  uint32_t id = next_mr_id_++;
  MR info{};
  info.id = id;
  info.address = reinterpret_cast<uint64_t>(local_buf);
  info.length = static_cast<uint64_t>(len);
  info.lkey = 0;
  info.key = 0;

  ptr_to_local_mr_[local_buf] = LocalRegistration{id, len};
  mr_id_to_local_mr_[id] = info;
  tracked.mr = info;
  return tracked;
}

MemoryRegistry::ReleasedLocalMr MemoryRegistry::release_local_buffer(
    void* local_buf) {
  std::lock_guard<std::mutex> lk(local_mu_);

  ReleasedLocalMr released{};
  auto it = ptr_to_local_mr_.find(local_buf);
  if (it == ptr_to_local_mr_.end()) {
    return released;
  }

  released.local_buf = local_buf;
  released.local_mr_id = it->second.mr_id;
  released.len = it->second.len;
  released.has_local_mr_id = true;
  ptr_to_local_mr_.erase(it);

  auto info_it = mr_id_to_local_mr_.find(released.local_mr_id);
  if (info_it != mr_id_to_local_mr_.end()) {
    mr_id_to_local_mr_.erase(info_it);
  }

  return released;
}

std::vector<MemoryRegistry::LocalRegistrationInfo>
MemoryRegistry::local_registrations() const {
  std::lock_guard<std::mutex> lk(local_mu_);
  std::vector<LocalRegistrationInfo> regs;
  regs.reserve(ptr_to_local_mr_.size());
  for (auto const& kv : ptr_to_local_mr_) {
    auto info = mr_id_to_local_mr_.find(kv.second.mr_id);
    if (info == mr_id_to_local_mr_.end()) continue;
    regs.push_back(LocalRegistrationInfo{kv.first, info->second});
  }
  return regs;
}

void MemoryRegistry::clear_remote_ipc_cache() { ipc_cache_.clear_all(); }

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
  for (auto const& mr : mrs) {
    if (cached.find(mr.id) != cached.end()) continue;
    cached[mr.id] = mr;
  }
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

bool MemoryRegistry::register_remote_ipc_cache(
    int remote_rank, gpuIpcMemHandle_t handle,
    IpcCacheManager::IpcCache const& cache) {
  return ipc_cache_.register_cache(remote_rank, handle, cache);
}

IpcCacheManager::IpcCache MemoryRegistry::get_remote_ipc_cache(
    int remote_rank, gpuIpcMemHandle_t handle) const {
  return ipc_cache_.get_cache(remote_rank, handle);
}

}  // namespace Transport
}  // namespace UKernel
