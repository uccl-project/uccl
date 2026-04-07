#include "memory_manager.h"
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr size_t kMinBucketBytes = 4 * 1024;
constexpr size_t kMaxIdlePerBucket = 2;

}  // namespace

MemoryManager::MemoryManager() = default;
MemoryManager::~MemoryManager() = default;

MemoryManager::TrackResult MemoryManager::register_local(void* ptr,
                                                         size_t len) {
  std::lock_guard<std::mutex> lk(local_mu_);

  TrackResult result;
  auto existing = ptr_to_local_mr_.find(ptr);
  if (existing != ptr_to_local_mr_.end()) {
    if (existing->second.len == len) {
      auto info = id_to_local_mr_.find(existing->second.mr_id);
      if (info != id_to_local_mr_.end()) {
        ++existing->second.ref_count;
        result.mr = info->second;
        return result;
      }
    } else {
      if (existing->second.ref_count > 1) {
        throw std::runtime_error(
            "cannot resize a tracked local buffer while requests still hold "
            "references");
      }
      result.replaced = true;
      result.replaced_mr_id = existing->second.mr_id;
      id_to_local_mr_.erase(existing->second.mr_id);
      ptr_to_local_mr_.erase(existing);
    }
  }

  uint32_t id = next_mr_id_++;
  LocalMR mr;
  mr.id = id;
  mr.address = reinterpret_cast<uint64_t>(ptr);
  mr.length = static_cast<uint64_t>(len);
  mr.lkey = 0;

  ptr_to_local_mr_[ptr] = LocalRegInfo{id, len, 1};
  id_to_local_mr_[id] = mr;
  result.mr = mr;
  return result;
}

MemoryManager::ReleaseResult MemoryManager::deregister_local(void* ptr) {
  std::lock_guard<std::mutex> lk(local_mu_);

  ReleaseResult result;
  auto it = ptr_to_local_mr_.find(ptr);
  if (it == ptr_to_local_mr_.end()) {
    return result;
  }

  result.mr_id = it->second.mr_id;
  if (it->second.ref_count > 1) {
    --it->second.ref_count;
    return result;
  }

  result.fully_released = true;
  ptr_to_local_mr_.erase(it);
  id_to_local_mr_.erase(result.mr_id);

  return result;
}

LocalMR MemoryManager::get_local_mr(uint32_t mr_id) const {
  std::lock_guard<std::mutex> lk(local_mu_);
  auto it = id_to_local_mr_.find(mr_id);
  if (it == id_to_local_mr_.end()) {
    throw std::runtime_error("Local MR not found for id");
  }
  return it->second;
}

LocalMR MemoryManager::find_local_by_ptr(void* ptr) const {
  std::lock_guard<std::mutex> lk(local_mu_);
  auto exact = ptr_to_local_mr_.find(ptr);
  if (exact != ptr_to_local_mr_.end()) {
    auto mr_it = id_to_local_mr_.find(exact->second.mr_id);
    if (mr_it == id_to_local_mr_.end()) {
      throw std::runtime_error("Local MR info not found");
    }
    return mr_it->second;
  }

  uintptr_t query = reinterpret_cast<uintptr_t>(ptr);
  for (auto const& [mr_id, mr] : id_to_local_mr_) {
    uintptr_t begin = static_cast<uintptr_t>(mr.address);
    uintptr_t end = begin + static_cast<uintptr_t>(mr.length);
    if (query >= begin && query < end) {
      return mr;
    }
  }
  throw std::runtime_error("Local MR not found for buffer");
}

std::vector<std::pair<void*, LocalMR>> MemoryManager::all_local_mrs() const {
  std::lock_guard<std::mutex> lk(local_mu_);
  std::vector<std::pair<void*, LocalMR>> result;
  result.reserve(ptr_to_local_mr_.size());
  for (auto const& kv : ptr_to_local_mr_) {
    auto mr_it = id_to_local_mr_.find(kv.second.mr_id);
    if (mr_it == id_to_local_mr_.end()) continue;
    result.push_back({kv.first, mr_it->second});
  }
  return result;
}

void MemoryManager::cache_remote_mrs(int rank,
                                     std::vector<RemoteMR> const& mrs) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto& cached = rank_mr_cache_[rank];
  for (auto const& mr : mrs) {
    if (cached.find(mr.id) != cached.end()) continue;
    cached[mr.id] = mr;
  }
}

RemoteMR MemoryManager::get_remote_mr(int rank, uint32_t mr_id) const {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto it_rank = rank_mr_cache_.find(rank);
  if (it_rank == rank_mr_cache_.end()) {
    throw std::runtime_error("No MR cached for remote rank");
  }
  auto it_mr = it_rank->second.find(mr_id);
  if (it_mr == it_rank->second.end()) {
    throw std::runtime_error("Remote MR not found for id");
  }
  return it_mr->second;
}

bool MemoryManager::has_remote_mr(int rank, uint32_t mr_id) const {
  std::lock_guard<std::mutex> lk(remote_mu_);
  auto it_rank = rank_mr_cache_.find(rank);
  if (it_rank == rank_mr_cache_.end()) return false;
  return it_rank->second.find(mr_id) != it_rank->second.end();
}

void MemoryManager::clear_remote_mrs(int rank) {
  std::lock_guard<std::mutex> lk(remote_mu_);
  rank_mr_cache_.erase(rank);
}

bool MemoryManager::register_remote_ipc(int rank, gpuIpcMemHandle_t handle,
                                        RemoteIpc const& ipc) {
  std::lock_guard<std::mutex> lk(ipc_mu_);
  rank_ipc_cache_[rank][make_ipc_handle_key(handle)] = ipc;
  return true;
}

RemoteIpc MemoryManager::get_remote_ipc(int rank,
                                        gpuIpcMemHandle_t handle) const {
  std::lock_guard<std::mutex> lk(ipc_mu_);
  auto it_rank = rank_ipc_cache_.find(rank);
  if (it_rank == rank_ipc_cache_.end()) return RemoteIpc{};

  auto key = make_ipc_handle_key(handle);
  auto it = it_rank->second.find(key);
  if (it == it_rank->second.end()) return RemoteIpc{};
  return it->second;
}

void MemoryManager::clear_remote_ipc_cache() {
  std::lock_guard<std::mutex> lk(ipc_mu_);
  int original_device = -1;
  bool have_original_device = (gpuGetDevice(&original_device) == gpuSuccess);
  for (auto& [rank, caches] : rank_ipc_cache_) {
    (void)rank;
    for (auto& [key, ipc] : caches) {
      (void)key;
      if (ipc.direct_ptr == nullptr || ipc.device_idx < 0) continue;
      GPU_RT_CHECK(gpuSetDevice(ipc.device_idx));
      GPU_RT_CHECK(gpuIpcCloseMemHandle(ipc.direct_ptr));
      ipc.direct_ptr = nullptr;
    }
  }
  if (have_original_device) {
    GPU_RT_CHECK(gpuSetDevice(original_device));
  }
  rank_ipc_cache_.clear();
}

bool MemoryManager::register_remote_ipc_buffer(int rank, uint32_t ipc_id,
                                               RemoteIpcBuffer const& buf) {
  std::lock_guard<std::mutex> lk(ipc_buf_mu_);
  rank_ipc_buffer_cache_[rank][ipc_id] = buf;
  return true;
}

MemoryManager::RemoteIpcBuffer MemoryManager::get_remote_ipc_buffer(
    int rank, uint32_t ipc_id) const {
  std::lock_guard<std::mutex> lk(ipc_buf_mu_);
  auto rank_it = rank_ipc_buffer_cache_.find(rank);
  if (rank_it == rank_ipc_buffer_cache_.end()) return {};
  auto it = rank_it->second.find(ipc_id);
  if (it == rank_it->second.end()) return {};
  return it->second;
}

void MemoryManager::clear_remote_ipc_buffers(int rank) {
  std::lock_guard<std::mutex> lk(ipc_buf_mu_);
  rank_ipc_buffer_cache_.erase(rank);
}

std::vector<std::pair<uint32_t, MemoryManager::RemoteIpcBuffer>>
MemoryManager::list_remote_ipc_buffers(int rank) const {
  std::lock_guard<std::mutex> lk(ipc_buf_mu_);
  auto rank_it = rank_ipc_buffer_cache_.find(rank);
  if (rank_it == rank_ipc_buffer_cache_.end()) return {};
  std::vector<std::pair<uint32_t, RemoteIpcBuffer>> result;
  for (auto const& kv : rank_it->second) {
    result.emplace_back(kv.first, kv.second);
  }
  return result;
}

BounceCpuBufferPool::BounceCpuBufferPool(RegisterFn register_fn,
                                         DeregisterFn deregister_fn)
    : register_fn_(std::move(register_fn)),
      deregister_fn_(std::move(deregister_fn)) {}

BounceCpuBufferPool::~BounceCpuBufferPool() {
  std::vector<Entry> entries_copy;
  {
    std::lock_guard<std::mutex> lk(mu_);
    entries_copy = entries_;
    entries_.clear();
  }

  for (Entry& entry : entries_copy) {
    if (entry.uccl_registered && deregister_fn_) {
      deregister_fn_(entry.mr_id);
    }
    if (entry.ptr != nullptr) {
      gpuFreeHost(entry.ptr);
      entry.ptr = nullptr;
    }
  }
}

bool BounceCpuBufferPool::ensure_uccl_registered(Entry& entry) {
  if (entry.uccl_registered) return true;
  if (!register_fn_) return false;
  if (!register_fn_(entry.mr_id, entry.ptr, entry.bytes)) return false;
  entry.uccl_registered = true;
  return true;
}

size_t BounceCpuBufferPool::bucket_capacity(size_t bytes) {
  size_t bucket = std::max(bytes, kMinBucketBytes);
  bucket--;
  bucket |= bucket >> 1;
  bucket |= bucket >> 2;
  bucket |= bucket >> 4;
  bucket |= bucket >> 8;
  bucket |= bucket >> 16;
  if constexpr (sizeof(size_t) >= 8) {
    bucket |= bucket >> 32;
  }
  return bucket + 1;
}

BounceCpuBuffer BounceCpuBufferPool::acquire(size_t bytes,
                                             bool require_uccl_registration,
                                             bool require_shared) {
  size_t requested_bucket = bucket_capacity(bytes);
  size_t candidate = BounceCpuBuffer::kInvalidSlot;
  size_t empty_slot = BounceCpuBuffer::kInvalidSlot;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < entries_.size(); ++i) {
      Entry& entry = entries_[i];
      if (entry.ptr == nullptr) {
        if (empty_slot == BounceCpuBuffer::kInvalidSlot) empty_slot = i;
        continue;
      }
      if (entry.in_use || entry.bucket_bytes < requested_bucket) continue;
      if (require_shared && entry.shm_name.empty()) continue;
      candidate = i;
      if (!require_uccl_registration || entry.uccl_registered) {
        entry.in_use = true;
        auto idle_it = idle_per_bucket_.find(entry.bucket_bytes);
        if (idle_it != idle_per_bucket_.end() && idle_it->second > 0) {
          --idle_it->second;
          if (idle_it->second == 0) {
            idle_per_bucket_.erase(idle_it);
          }
        }
        return BounceCpuBuffer{entry.ptr,
                               entry.bytes,
                               entry.mr_id,
                               entry.uccl_registered,
                               !entry.shm_name.empty(),
                               i};
      }
    }

    if (candidate != BounceCpuBuffer::kInvalidSlot) {
      auto& entry = entries_[candidate];
      auto idle_it = idle_per_bucket_.find(entry.bucket_bytes);
      if (idle_it != idle_per_bucket_.end() && idle_it->second > 0) {
        --idle_it->second;
        if (idle_it->second == 0) {
          idle_per_bucket_.erase(idle_it);
        }
      }
      entries_[candidate].in_use = true;
    }
  }

  if (candidate != BounceCpuBuffer::kInvalidSlot) {
    Entry snapshot;
    {
      std::lock_guard<std::mutex> lk(mu_);
      snapshot = entries_[candidate];
    }

    if (require_uccl_registration && !snapshot.uccl_registered) {
      Entry updated = snapshot;
      if (!ensure_uccl_registered(updated)) {
        std::lock_guard<std::mutex> lk(mu_);
        entries_[candidate].in_use = false;
        ++idle_per_bucket_[entries_[candidate].bucket_bytes];
        throw std::runtime_error(
            "failed to register pooled host bounce buffer with UCCL");
      }
      std::lock_guard<std::mutex> lk(mu_);
      entries_[candidate].uccl_registered = true;
      entries_[candidate].in_use = true;
      return BounceCpuBuffer{entries_[candidate].ptr,
                             entries_[candidate].bytes,
                             entries_[candidate].mr_id,
                             true,
                             false,
                             candidate};
    }
  }

  Entry entry;
  entry.bucket_bytes = requested_bucket;
  entry.bytes = requested_bucket;
  entry.mr_id = next_mr_id_.fetch_add(1, std::memory_order_relaxed);

  if (require_shared) {
    std::string name = "/uccl_bounce_" +
                       std::to_string(static_cast<long long>(::getpid())) +
                       "_" + std::to_string(entry.mr_id);
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
      throw std::runtime_error(
          "failed to create shared memory for bounce buffer");
    }
    if (ftruncate(fd, entry.bytes) < 0) {
      close(fd);
      shm_unlink(name.c_str());
      throw std::runtime_error(
          "failed to size shared memory for bounce buffer");
    }
    entry.ptr =
        mmap(nullptr, entry.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (entry.ptr == MAP_FAILED) {
      close(fd);
      shm_unlink(name.c_str());
      throw std::runtime_error(
          "failed to mmap shared memory for bounce buffer");
    }
    close(fd);
    entry.shm_name = name;
  } else {
    GPU_RT_CHECK(gpuHostMalloc(&entry.ptr, requested_bucket));
  }

  entry.in_use = true;
  if (require_uccl_registration && !ensure_uccl_registered(entry)) {
    if (require_shared) {
      munmap(entry.ptr, entry.bytes);
      shm_unlink(entry.shm_name.c_str());
    } else {
      gpuFreeHost(entry.ptr);
    }
    throw std::runtime_error(
        "failed to register new host bounce buffer with UCCL");
  }

  std::lock_guard<std::mutex> lk(mu_);
  size_t slot = empty_slot;
  if (slot != BounceCpuBuffer::kInvalidSlot && slot < entries_.size() &&
      entries_[slot].ptr == nullptr) {
    entries_[slot] = entry;
  } else {
    slot = entries_.size();
    entries_.push_back(entry);
  }
  return BounceCpuBuffer{entry.ptr,
                         entry.bytes,
                         entry.mr_id,
                         entry.uccl_registered,
                         !entry.shm_name.empty(),
                         slot};
}

void BounceCpuBufferPool::release(BounceCpuBuffer& lease) {
  if (!lease.valid()) return;

  Entry evicted;
  bool should_evict = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (lease.slot < entries_.size()) {
      Entry& released = entries_[lease.slot];
      released.in_use = false;
      size_t idle_in_bucket = ++idle_per_bucket_[released.bucket_bytes];
      if (idle_in_bucket > kMaxIdlePerBucket) {
        evicted = released;
        released = Entry{};
        should_evict = true;
        auto idle_it = idle_per_bucket_.find(evicted.bucket_bytes);
        if (idle_it != idle_per_bucket_.end()) {
          if (idle_it->second > 0) {
            --idle_it->second;
          }
          if (idle_it->second == 0) {
            idle_per_bucket_.erase(idle_it);
          }
        }
      }
    }
  }
  lease = BounceCpuBuffer{};

  if (should_evict) {
    if (evicted.uccl_registered && deregister_fn_) {
      deregister_fn_(evicted.mr_id);
    }
    if (!evicted.shm_name.empty()) {
      shm_unlink(evicted.shm_name.c_str());
    }
    if (evicted.ptr != nullptr) {
      gpuFreeHost(evicted.ptr);
    }
  }
}

bool BounceCpuBufferPool::share_buffer(BounceCpuBuffer& buf, void** out_shm_ptr,
                                       int* out_fd) {
  if (!buf.valid() || buf.slot >= entries_.size()) {
    return false;
  }

  std::lock_guard<std::mutex> lk(mu_);
  Entry& entry = entries_[buf.slot];

  if (entry.shm_name.empty()) {
    return false;
  }

  int fd = shm_open(entry.shm_name.c_str(), O_RDWR, 0666);
  if (fd < 0) return false;

  entry.shareable = true;
  buf.shareable = true;
  *out_shm_ptr = entry.ptr;
  *out_fd = fd;
  return true;
}

std::string BounceCpuBufferPool::get_shm_name(size_t slot) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (slot < entries_.size()) {
    return entries_[slot].shm_name;
  }
  return {};
}

void* BounceCpuBufferPool::get_or_open_shm(std::string const& shm_name,
                                           size_t size) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = shm_cache_.find(shm_name);
  if (it != shm_cache_.end()) {
    return it->second.ptr;
  }

  int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (fd < 0) {
    return nullptr;
  }
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    return nullptr;
  }
  shm_cache_[shm_name] = {ptr, size};
  return ptr;
}

void BounceCpuBufferPool::clear_shm_cache() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& kv : shm_cache_) {
    munmap(kv.second.ptr, kv.second.size);
  }
  shm_cache_.clear();
}

}  // namespace Transport
}  // namespace UKernel
