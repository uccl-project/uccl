#include "host_bounce_pool.h"
#include <stdexcept>
#include <utility>

namespace UKernel {
namespace Transport {

namespace {

constexpr size_t kMinBucketBytes = 4 * 1024;
constexpr size_t kMaxIdlePerBucket = 2;

}  // namespace

HostBouncePool::HostBouncePool(RegisterFn register_fn, DeregisterFn deregister_fn)
    : register_fn_(std::move(register_fn)),
      deregister_fn_(std::move(deregister_fn)) {}

HostBouncePool::~HostBouncePool() {
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

bool HostBouncePool::ensure_uccl_registered(Entry& entry) {
  if (entry.uccl_registered) return true;
  if (!register_fn_) return false;
  if (!register_fn_(entry.mr_id, entry.ptr, entry.bytes)) return false;
  entry.uccl_registered = true;
  return true;
}

size_t HostBouncePool::bucket_capacity(size_t bytes) {
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

HostBouncePool::Lease HostBouncePool::acquire(size_t bytes,
                                              bool require_uccl_registration) {
  size_t requested_bucket = bucket_capacity(bytes);
  size_t candidate = Lease::kInvalidSlot;
  size_t empty_slot = Lease::kInvalidSlot;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < entries_.size(); ++i) {
      Entry& entry = entries_[i];
      if (entry.ptr == nullptr) {
        if (empty_slot == Lease::kInvalidSlot) empty_slot = i;
        continue;
      }
      if (entry.in_use || entry.bucket_bytes < requested_bucket) continue;
      candidate = i;
      if (!require_uccl_registration || entry.uccl_registered) {
        entry.in_use = true;
        return Lease{entry.ptr, entry.bytes, entry.mr_id, entry.uccl_registered,
                     i};
      }
    }

    if (candidate != Lease::kInvalidSlot) {
      entries_[candidate].in_use = true;
    }
  }

  if (candidate != Lease::kInvalidSlot) {
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
        throw std::runtime_error("failed to register pooled host bounce buffer with UCCL");
      }
      std::lock_guard<std::mutex> lk(mu_);
      entries_[candidate].uccl_registered = true;
      entries_[candidate].in_use = true;
      return Lease{entries_[candidate].ptr, entries_[candidate].bytes,
                   entries_[candidate].mr_id, true, candidate};
    }
  }

  Entry entry;
  entry.bucket_bytes = requested_bucket;
  GPU_RT_CHECK(gpuHostMalloc(&entry.ptr, requested_bucket));
  entry.bytes = requested_bucket;
  entry.mr_id = next_mr_id_.fetch_add(1, std::memory_order_relaxed);
  entry.in_use = true;
  if (require_uccl_registration && !ensure_uccl_registered(entry)) {
    gpuFreeHost(entry.ptr);
    throw std::runtime_error("failed to register new host bounce buffer with UCCL");
  }

  std::lock_guard<std::mutex> lk(mu_);
  size_t slot = empty_slot;
  if (slot != Lease::kInvalidSlot && slot < entries_.size() &&
      entries_[slot].ptr == nullptr) {
    entries_[slot] = entry;
  } else {
    slot = entries_.size();
    entries_.push_back(entry);
  }
  return Lease{entry.ptr, entry.bytes, entry.mr_id, entry.uccl_registered,
               slot};
}

void HostBouncePool::release(Lease& lease) {
  if (!lease.valid()) return;

  Entry evicted;
  bool should_evict = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (lease.slot < entries_.size()) {
      Entry& released = entries_[lease.slot];
      released.in_use = false;

      size_t idle_in_bucket = 0;
      for (size_t i = 0; i < entries_.size(); ++i) {
        Entry const& entry = entries_[i];
        if (entry.ptr != nullptr && !entry.in_use &&
            entry.bucket_bytes == released.bucket_bytes) {
          ++idle_in_bucket;
        }
      }
      if (idle_in_bucket > kMaxIdlePerBucket) {
        evicted = released;
        released = Entry{};
        should_evict = true;
      }
    }
  }
  lease = Lease{};

  if (should_evict) {
    if (evicted.uccl_registered && deregister_fn_) {
      deregister_fn_(evicted.mr_id);
    }
    if (evicted.ptr != nullptr) {
      gpuFreeHost(evicted.ptr);
    }
  }
}

}  // namespace Transport
}  // namespace UKernel
