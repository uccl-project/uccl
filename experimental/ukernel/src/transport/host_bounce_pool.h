#pragma once

#include "../../include/gpu_rt.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace UKernel {
namespace Transport {

class HostBouncePool {
 public:
  // Reusable pinned host buffers shared by transport paths that need
  // Device <-> Host staging, such as TCP and host-staged UCCL/RDMA.
  struct Lease {
    void* ptr = nullptr;
    size_t bytes = 0;
    uint64_t mr_id = 0;
    bool uccl_registered = false;
    size_t slot = kInvalidSlot;

    bool valid() const { return ptr != nullptr; }
    static constexpr size_t kInvalidSlot = static_cast<size_t>(-1);
  };

  using RegisterFn = std::function<bool(uint64_t, void*, size_t)>;
  using DeregisterFn = std::function<void(uint64_t)>;

  HostBouncePool(RegisterFn register_fn, DeregisterFn deregister_fn);
  ~HostBouncePool();

  // Acquire a pinned host buffer with at least `bytes` capacity. When
  // `require_uccl_registration` is true, the returned lease is guaranteed to
  // carry a transport-visible MR id suitable for UCCL host-staging.
  Lease acquire(size_t bytes, bool require_uccl_registration);
  void release(Lease& lease);

 private:
  struct Entry {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t bucket_bytes = 0;
    uint64_t mr_id = 0;
    bool uccl_registered = false;
    bool in_use = false;
  };

  static size_t bucket_capacity(size_t bytes);
  bool ensure_uccl_registered(Entry& entry);

  RegisterFn register_fn_;
  DeregisterFn deregister_fn_;
  std::atomic<uint64_t> next_mr_id_{1ull << 62};
  std::mutex mu_;
  std::vector<Entry> entries_;
};

}  // namespace Transport
}  // namespace UKernel
