#include "mr_manager.h"

namespace UKernel {
namespace Transport {

MRItem MRManager::create_local_mr(uint32_t buffer_id, void* ptr, size_t len) {
  if (buffer_id == 0 || ptr == nullptr || len == 0) return {};
  std::lock_guard<std::mutex> lk(local_mu_);

  auto remove_local_ptr = [&](void* local_ptr, MRItem const& item) {
    local_ptr_by_buffer_id_.erase(static_cast<uint32_t>(item.mr.id));
    local_by_ptr_.erase(local_ptr);
  };

  auto id_it = local_ptr_by_buffer_id_.find(buffer_id);
  if (id_it != local_ptr_by_buffer_id_.end()) {
    auto item_it = local_by_ptr_.find(id_it->second);
    if (item_it == local_by_ptr_.end()) {
      local_ptr_by_buffer_id_.erase(id_it);
    } else {
      MRItem const& existing = item_it->second;
      if (item_it->first == ptr &&
          static_cast<size_t>(existing.mr.length) == len) {
        return item_it->second;
      }
      remove_local_ptr(item_it->first, existing);
    }
  }

  auto existing_it = local_by_ptr_.find(ptr);
  if (existing_it != local_by_ptr_.end()) {
    MRItem const& existing = existing_it->second;
    remove_local_ptr(existing_it->first, existing);
  }

  MRItem created{};
  created.mr.id = buffer_id;
  created.mr.address = reinterpret_cast<uint64_t>(ptr);
  created.mr.length = static_cast<uint64_t>(len);
  created.mr.lkey = 0;
  created.is_local = true;
  created.rank = -1;
  created.valid = true;

  local_by_ptr_[ptr] = created;
  local_ptr_by_buffer_id_[buffer_id] = ptr;
  return created;
}

bool MRManager::register_remote_mr(int rank, MRItem const& item) {
  if (!item.valid) return false;
  std::lock_guard<std::mutex> lk(remote_mu_);
  MRItem v = item;
  v.is_local = false;
  v.rank = rank;
  remote_by_rank_[rank][static_cast<uint32_t>(v.mr.id)] = v;
  return true;
}

void MRManager::register_remote_mrs(int rank,
                                    std::vector<MRItem> const& items) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto& dst = remote_by_rank_[rank];
  for (auto const& item : items) {
    if (!item.valid) continue;
    MRItem v = item;
    v.is_local = false;
    v.rank = rank;
    dst[static_cast<uint32_t>(v.mr.id)] = v;
  }
}

MRItem MRManager::get_mr(void* local_ptr) const {
  std::lock_guard<std::mutex> lk(local_mu_);
  auto exact = local_by_ptr_.find(local_ptr);
  if (exact != local_by_ptr_.end()) return exact->second;

  uintptr_t query = reinterpret_cast<uintptr_t>(local_ptr);
  for (auto const& [ptr, item] : local_by_ptr_) {
    (void)ptr;
    uintptr_t begin = static_cast<uintptr_t>(item.mr.address);
    uintptr_t end = begin + static_cast<uintptr_t>(item.mr.length);
    if (query >= begin && query < end) return item;
  }
  return {};
}

MRItem MRManager::get_mr(uint32_t local_buffer_id) const {
  std::lock_guard<std::mutex> lk(local_mu_);
  auto ptr_it = local_ptr_by_buffer_id_.find(local_buffer_id);
  if (ptr_it == local_ptr_by_buffer_id_.end()) return {};
  auto item_it = local_by_ptr_.find(ptr_it->second);
  if (item_it == local_by_ptr_.end()) return {};
  return item_it->second;
}

MRItem MRManager::get_mr(int remote_rank, uint32_t remote_buffer_id) const {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = remote_by_rank_.find(remote_rank);
  if (rank_it == remote_by_rank_.end()) return {};
  auto it = rank_it->second.find(remote_buffer_id);
  if (it == rank_it->second.end()) return {};
  return it->second;
}

bool MRManager::delete_mr(void* local_ptr) {
  std::lock_guard<std::mutex> lk(local_mu_);
  auto it = local_by_ptr_.find(local_ptr);
  if (it == local_by_ptr_.end()) return false;
  local_ptr_by_buffer_id_.erase(static_cast<uint32_t>(it->second.mr.id));
  local_by_ptr_.erase(it);
  return true;
}

bool MRManager::delete_mr(uint32_t local_buffer_id) {
  std::lock_guard<std::mutex> lk(local_mu_);
  auto ptr_it = local_ptr_by_buffer_id_.find(local_buffer_id);
  if (ptr_it == local_ptr_by_buffer_id_.end()) return false;
  auto item_it = local_by_ptr_.find(ptr_it->second);
  if (item_it != local_by_ptr_.end()) {
    local_by_ptr_.erase(item_it);
  }
  local_ptr_by_buffer_id_.erase(ptr_it);
  return true;
}

bool MRManager::delete_mr(int remote_rank, uint32_t remote_buffer_id) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto rank_it = remote_by_rank_.find(remote_rank);
  if (rank_it == remote_by_rank_.end()) return false;
  return rank_it->second.erase(remote_buffer_id) > 0;
}

void MRManager::delete_mr(int remote_rank) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  remote_by_rank_.erase(remote_rank);
}

std::vector<std::pair<void*, MRItem>> MRManager::list_local_mrs() const {
  std::lock_guard<std::mutex> lk(local_mu_);
  std::vector<std::pair<void*, MRItem>> out;
  out.reserve(local_by_ptr_.size());
  for (auto const& kv : local_by_ptr_) {
    out.push_back(kv);
  }
  return out;
}

}  // namespace Transport
}  // namespace UKernel
