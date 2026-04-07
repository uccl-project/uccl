#pragma once

#include "../oob/oob.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

using LocalMR = MR;
using RemoteMR = MR;

struct RemoteIpc {
  gpuIpcMemHandle_t handle;
  void* direct_ptr = nullptr;
  uintptr_t offset = 0;
  size_t size = 0;
  int device_idx = -1;
};

struct BounceCpuBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;
  uint32_t mr_id = 0;
  bool uccl_registered = false;
  bool shareable = false;
  size_t slot = static_cast<size_t>(-1);

  bool valid() const { return ptr != nullptr; }
  static constexpr size_t kInvalidSlot = static_cast<size_t>(-1);
};

class BounceCpuBufferPool {
 public:
  using RegisterFn = std::function<bool(uint32_t, void*, size_t)>;
  using DeregisterFn = std::function<void(uint32_t)>;

  BounceCpuBufferPool(RegisterFn register_fn, DeregisterFn deregister_fn);
  ~BounceCpuBufferPool();

  BounceCpuBuffer acquire(size_t bytes, bool require_uccl_registration,
                          bool require_shared = false);
  void release(BounceCpuBuffer& buf);

  bool share_buffer(BounceCpuBuffer& buf, void** out_shm_ptr, int* out_fd);
  std::string get_shm_name(size_t slot) const;
  void* get_or_open_shm(std::string const& shm_name, size_t size);
  void clear_shm_cache();

 private:
  struct Entry {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t bucket_bytes = 0;
    uint32_t mr_id = 0;
    bool uccl_registered = false;
    bool shareable = false;
    bool in_use = false;
    std::string shm_name;
  };

  static size_t bucket_capacity(size_t bytes);
  bool ensure_uccl_registered(Entry& entry);

  RegisterFn register_fn_;
  DeregisterFn deregister_fn_;
  // Reserve the upper half of uint32 ids for bounce buffers to reduce
  // collision risk with regular MemoryManager local MR ids (which start at 1).
  std::atomic<uint32_t> next_mr_id_{0x80000000u};
  mutable std::mutex mu_;
  std::vector<Entry> entries_;
  std::unordered_map<size_t, size_t> idle_per_bucket_;
  struct ShmCacheEntry {
    void* ptr = nullptr;
    size_t size = 0;
  };
  std::unordered_map<std::string, ShmCacheEntry> shm_cache_;
};

class MemoryManager {
 public:
  struct TrackResult {
    LocalMR mr;
    bool replaced = false;
    uint32_t replaced_mr_id = 0;
  };

  struct ReleaseResult {
    bool fully_released = false;
    uint32_t mr_id = 0;
  };

  MemoryManager();
  ~MemoryManager();

  TrackResult register_local(void* ptr, size_t len);
  ReleaseResult deregister_local(void* ptr);

  LocalMR get_local_mr(uint32_t mr_id) const;
  LocalMR find_local_by_ptr(void* ptr) const;
  std::vector<std::pair<void*, LocalMR>> all_local_mrs() const;

  void cache_remote_mrs(int rank, std::vector<RemoteMR> const& mrs);
  RemoteMR get_remote_mr(int rank, uint32_t mr_id) const;
  bool has_remote_mr(int rank, uint32_t mr_id) const;
  void clear_remote_mrs(int rank);

  bool register_remote_ipc(int rank, gpuIpcMemHandle_t handle,
                           RemoteIpc const& ipc);
  RemoteIpc get_remote_ipc(int rank, gpuIpcMemHandle_t handle) const;
  void clear_remote_ipc_cache();

  struct RemoteIpcBuffer {
    gpuIpcMemHandle_t handle{};
    uint64_t binding_version = 0;
    uintptr_t base_offset = 0;
    size_t bytes = 0;
    int device_idx = -1;
    bool valid = false;
    void* direct_ptr = nullptr;
  };
  bool register_remote_ipc_buffer(int rank, uint32_t ipc_id,
                                  RemoteIpcBuffer const& buf);
  RemoteIpcBuffer get_remote_ipc_buffer(int rank, uint32_t ipc_id) const;
  std::vector<std::pair<uint32_t, RemoteIpcBuffer>> list_remote_ipc_buffers(
      int rank) const;
  void clear_remote_ipc_buffers(int rank);

 private:
  struct LocalRegInfo {
    uint32_t mr_id = 0;
    size_t len = 0;
    uint32_t ref_count = 0;
  };

  using IpcHandleKey = std::array<uint8_t, sizeof(gpuIpcMemHandle_t)>;

  struct IpcHandleHash {
    size_t operator()(IpcHandleKey const& k) const noexcept {
      uint64_t hash = 1469598103934665603ull;
      for (uint8_t b : k) {
        hash ^= b;
        hash *= 1099511628211ull;
      }
      return static_cast<size_t>(hash);
    }
  };

  static IpcHandleKey make_ipc_handle_key(gpuIpcMemHandle_t const& h) {
    IpcHandleKey k{};
    std::memcpy(k.data(), &h, k.size());
    return k;
  }

  mutable std::mutex local_mu_;
  std::unordered_map<void*, LocalRegInfo> ptr_to_local_mr_;
  std::unordered_map<uint32_t, LocalMR> id_to_local_mr_;
  uint32_t next_mr_id_ = 1;

  mutable std::mutex remote_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, RemoteMR>>
      rank_mr_cache_;

  mutable std::mutex ipc_mu_;
  std::unordered_map<int,
                     std::unordered_map<IpcHandleKey, RemoteIpc, IpcHandleHash>>
      rank_ipc_cache_;

  mutable std::mutex ipc_buf_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, RemoteIpcBuffer>>
      rank_ipc_buffer_cache_;
};

}  // namespace Transport
}  // namespace UKernel
