#include "shm_manager.h"
#include "../oob/oob.h"
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

SHMManager::~SHMManager() {
  std::vector<Entry> entries_copy;
  {
    std::lock_guard<std::mutex> lk(mu_);
    entries_copy = entries_;
    entries_.clear();
    local_slot_by_id_.clear();
  }

  for (Entry& entry : entries_copy) {
    if (!entry.shm_name.empty()) {
      shm_unlink(entry.shm_name.c_str());
    }
    if (entry.ptr != nullptr) {
      if (entry.shm_name.empty()) {
        gpuFreeHost(entry.ptr);
      } else {
        munmap(entry.ptr, entry.bytes);
      }
    }
  }

  clear_remote_shm_cache();
}

SHMItem SHMManager::create_local_shm(size_t bytes, bool require_shared) {
  if (bytes == 0) return {};

  size_t empty_slot = static_cast<size_t>(-1);
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].ptr == nullptr) {
        empty_slot = i;
        break;
      }
    }
  }

  Entry entry{};
  entry.bytes = bytes;
  entry.id = next_shm_id_.fetch_add(1, std::memory_order_relaxed);

  if (require_shared) {
    std::string name = "/uccl_shm_" +
                       std::to_string(static_cast<long long>(::getpid())) +
                       "_" + std::to_string(entry.id);
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) throw std::runtime_error("failed to create shm");
    if (ftruncate(fd, entry.bytes) < 0) {
      close(fd);
      shm_unlink(name.c_str());
      throw std::runtime_error("failed to size shm");
    }
    entry.ptr =
        mmap(nullptr, entry.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (entry.ptr == MAP_FAILED) {
      shm_unlink(name.c_str());
      throw std::runtime_error("failed to mmap shm");
    }
    entry.shm_name = name;
  } else {
    GPU_RT_CHECK(gpuHostMalloc(&entry.ptr, bytes));
  }

  std::lock_guard<std::mutex> lk(mu_);
  size_t slot = empty_slot;
  if (slot != static_cast<size_t>(-1) && slot < entries_.size() &&
      entries_[slot].ptr == nullptr) {
    entries_[slot] = entry;
  } else {
    slot = entries_.size();
    entries_.push_back(entry);
  }
  local_slot_by_id_[entry.id] = slot;

  SHMItem out{};
  out.ptr = entry.ptr;
  out.bytes = entry.bytes;
  out.id = entry.id;
  out.shareable = !entry.shm_name.empty();
  out.is_local = true;
  out.valid = true;
  out.shm_name = entry.shm_name;
  return out;
}

SHMItem SHMManager::get_local_shm(uint32_t id) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto slot_it = local_slot_by_id_.find(id);
  if (slot_it == local_slot_by_id_.end()) return {};
  size_t slot = slot_it->second;
  if (slot >= entries_.size()) return {};
  Entry const& entry = entries_[slot];
  if (entry.ptr == nullptr) return {};

  SHMItem out{};
  out.ptr = entry.ptr;
  out.bytes = entry.bytes;
  out.id = entry.id;
  out.shareable = !entry.shm_name.empty();
  out.is_local = true;
  out.valid = true;
  out.shm_name = entry.shm_name;
  return out;
}

SHMItem SHMManager::open_remote_shm(std::string const& shm_name) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = shm_cache_.find(shm_name);
  if (it != shm_cache_.end()) {
    SHMItem out{};
    out.ptr = it->second.ptr;
    out.bytes = it->second.bytes;
    out.shm_name = shm_name;
    out.is_local = false;
    out.valid = true;
    return out;
  }

  int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (fd < 0) return {};
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    return {};
  }
  size_t size = static_cast<size_t>(st.st_size);
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) return {};

  shm_cache_[shm_name] = {ptr, size};
  SHMItem out{};
  out.ptr = ptr;
  out.bytes = size;
  out.shm_name = shm_name;
  out.is_local = false;
  out.valid = true;
  return out;
}

bool SHMManager::delete_local_shm(uint32_t id) {
  Entry evicted{};
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto slot_it = local_slot_by_id_.find(id);
    if (slot_it == local_slot_by_id_.end()) return false;
    size_t slot = slot_it->second;
    if (slot >= entries_.size()) return false;
    Entry& entry = entries_[slot];
    if (entry.ptr == nullptr) return false;
    evicted = entry;
    entry = Entry{};
    local_slot_by_id_.erase(slot_it);
  }

  if (!evicted.shm_name.empty()) {
    shm_unlink(evicted.shm_name.c_str());
  }
  if (evicted.ptr != nullptr) {
    if (evicted.shm_name.empty()) {
      gpuFreeHost(evicted.ptr);
    } else {
      munmap(evicted.ptr, evicted.bytes);
    }
  }
  return true;
}

bool SHMManager::close_remote_shm(std::string const& shm_name) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = shm_cache_.find(shm_name);
  if (it == shm_cache_.end()) return false;
  munmap(it->second.ptr, it->second.bytes);
  shm_cache_.erase(it);
  return true;
}

void SHMManager::clear_remote_shm_cache() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& kv : shm_cache_) {
    munmap(kv.second.ptr, kv.second.bytes);
  }
  shm_cache_.clear();
}

}  // namespace Transport
}  // namespace UKernel
