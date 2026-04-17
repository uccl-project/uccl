#pragma once

#include "../oob/oob.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace UKernel {
namespace Transport {

struct IPCItem {
  gpuIpcMemHandle_t handle{};
  void* direct_ptr = nullptr;
  uintptr_t base_addr = 0;
  uintptr_t base_offset = 0;
  size_t bytes = 0;
  int device_idx = -1;
  uint32_t buffer_id = 0;
  bool is_local = false;
  bool valid = false;
};

class IPCManager {
 public:
  // Register remote IPC metadata/cache item (supports both handle-keyed and
  // buffer_id-keyed records by filling `handle` and/or `buffer_id`).
  bool register_remote_ipc(int rank, IPCItem const& ipc);

  // Create or reuse local GPU IPC export item for a local pointer.
  IPCItem create_local_ipc(void* ptr, size_t len, int device_idx);

  // Unified lookup API.
  IPCItem get_ipc(int rank, gpuIpcMemHandle_t handle) const;
  IPCItem get_ipc(int rank, uint32_t buffer_id) const;
  IPCItem get_ipc(void* local_ptr) const;

  // Unified delete API.
  bool delete_ipc(int rank, gpuIpcMemHandle_t handle);
  bool delete_ipc(int rank, uint32_t buffer_id);
  bool delete_ipc(void* local_ptr);
  void delete_ipc(int rank);

 private:
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

  static bool has_handle(gpuIpcMemHandle_t const& h);
  // Close local imported mapping (gpuIpcOpenMemHandle result) if present.
  // This only affects this process's VA mapping, not remote process memory.
  static void close_local_import_if_open(IPCItem& item);

  mutable std::mutex remote_mu_;
  std::unordered_map<int,
                     std::unordered_map<IpcHandleKey, IPCItem, IpcHandleHash>>
      rank_handle_cache_;
  std::unordered_map<int, std::unordered_map<uint32_t, IPCItem>>
      rank_buffer_id_cache_;

  mutable std::mutex local_mu_;
  std::unordered_map<uintptr_t, IPCItem> local_ipc_cache_;
};

}  // namespace Transport
}  // namespace UKernel
