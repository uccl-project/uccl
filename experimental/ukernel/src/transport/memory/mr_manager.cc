#include "mr_manager.h"

namespace UKernel {
namespace Transport {

MRItem MRManager::create_local_mr(uint32_t buffer_id, void* ptr, size_t len) {
  if (buffer_id == 0 || ptr == nullptr || len == 0) return {};
  std::lock_guard<std::mutex> lk(local_mu_);

  auto it = local_by_buffer_id_.find(buffer_id);
  if (it != local_by_buffer_id_.end()) {
    MRItem& existing = it->second;
    if (existing.mr.address == reinterpret_cast<uint64_t>(ptr) &&
        static_cast<size_t>(existing.mr.length) == len) {
      return existing;
    }
    local_buffer_id_by_ptr_.erase(static_cast<uintptr_t>(existing.mr.address));
  }

  MRItem created{};
  created.mr.id = buffer_id;
  created.mr.address = reinterpret_cast<uint64_t>(ptr);
  created.mr.length = static_cast<uint64_t>(len);
  created.mr.lkey = 0;
  created.mr.key = 0;
  created.is_local = true;
  created.rank = -1;
  created.valid = true;

  local_by_buffer_id_[buffer_id] = created;
  local_buffer_id_by_ptr_[reinterpret_cast<uintptr_t>(ptr)] = buffer_id;
  return created;
}

bool MRManager::register_remote_mr(int rank, MRItem const& item) {
  if (!item.valid || item.mr.id == 0) return false;
  std::lock_guard<std::mutex> lk(remote_mu_);
  MRItem v = item;
  v.is_local = false;
  v.rank = rank;
  remote_by_rank_[rank][v.mr.id] = v;
  return true;
}

void MRManager::register_remote_mrs(int rank,
                                    std::vector<MRItem> const& items) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto& dst = remote_by_rank_[rank];
  for (auto const& item : items) {
    if (!item.valid || item.mr.id == 0) continue;
    MRItem v = item;
    v.is_local = false;
    v.rank = rank;
    dst[v.mr.id] = v;
  }
}

MRItem MRManager::get_mr(void* local_ptr) const {
  if (local_ptr == nullptr) return {};
  std::lock_guard<std::mutex> lk(local_mu_);

  uintptr_t query = reinterpret_cast<uintptr_t>(local_ptr);
  auto exact_it = local_buffer_id_by_ptr_.find(query);
  if (exact_it != local_buffer_id_by_ptr_.end()) {
    auto item_it = local_by_buffer_id_.find(exact_it->second);
    if (item_it != local_by_buffer_id_.end()) return item_it->second;
  }

  for (auto const& [buffer_id, item] : local_by_buffer_id_) {
    (void)buffer_id;
    uintptr_t begin = static_cast<uintptr_t>(item.mr.address);
    uintptr_t end = begin + static_cast<uintptr_t>(item.mr.length);
    if (query >= begin && query < end) return item;
  }
  return {};
}

MRItem MRManager::get_mr(uint32_t local_buffer_id) const {
  if (local_buffer_id == 0) return {};
  std::lock_guard<std::mutex> lk(local_mu_);
  auto it = local_by_buffer_id_.find(local_buffer_id);
  if (it == local_by_buffer_id_.end()) return {};
  return it->second;
}

MRItem MRManager::get_mr(int remote_rank, uint32_t remote_buffer_id) const {
  if (remote_buffer_id == 0) return {};
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = remote_by_rank_.find(remote_rank);
  if (rank_it == remote_by_rank_.end()) return {};
  auto it = rank_it->second.find(remote_buffer_id);
  if (it == rank_it->second.end()) return {};
  return it->second;
}

bool MRManager::delete_mr(void* local_ptr) {
  if (local_ptr == nullptr) return false;
  std::lock_guard<std::mutex> lk(local_mu_);

  uintptr_t query = reinterpret_cast<uintptr_t>(local_ptr);
  auto exact_it = local_buffer_id_by_ptr_.find(query);
  if (exact_it != local_buffer_id_by_ptr_.end()) {
    uint32_t buffer_id = exact_it->second;
    local_buffer_id_by_ptr_.erase(exact_it);
    return local_by_buffer_id_.erase(buffer_id) > 0;
  }

  for (auto it = local_by_buffer_id_.begin(); it != local_by_buffer_id_.end();
       ++it) {
    uintptr_t begin = static_cast<uintptr_t>(it->second.mr.address);
    uintptr_t end = begin + static_cast<uintptr_t>(it->second.mr.length);
    if (query < begin || query >= end) continue;
    local_buffer_id_by_ptr_.erase(begin);
    local_by_buffer_id_.erase(it);
    return true;
  }
  return false;
}

bool MRManager::delete_mr(uint32_t local_buffer_id) {
  if (local_buffer_id == 0) return false;
  std::lock_guard<std::mutex> lk(local_mu_);
  auto it = local_by_buffer_id_.find(local_buffer_id);
  if (it == local_by_buffer_id_.end()) return false;
  local_buffer_id_by_ptr_.erase(static_cast<uintptr_t>(it->second.mr.address));
  local_by_buffer_id_.erase(it);
  return true;
}

bool MRManager::delete_mr(int remote_rank, uint32_t remote_buffer_id) {
  if (remote_buffer_id == 0) return false;
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = remote_by_rank_.find(remote_rank);
  if (rank_it == remote_by_rank_.end()) return false;
  return rank_it->second.erase(remote_buffer_id) > 0;
}

void MRManager::delete_mr(int remote_rank) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  remote_by_rank_.erase(remote_rank);
}

std::vector<std::pair<uint32_t, MRItem>> MRManager::list_local_mrs() const {
  std::lock_guard<std::mutex> lk(local_mu_);
  std::vector<std::pair<uint32_t, MRItem>> out;
  out.reserve(local_by_buffer_id_.size());
  for (auto const& kv : local_by_buffer_id_) {
    out.push_back(kv);
  }
  return out;
}

}  // namespace Transport
}  // namespace UKernel
