#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <pthread.h>

namespace UKernel {
namespace Transport {

struct ShmBufSlotInfo {
  void* ptr = nullptr;
  uint32_t buffer_id = 0;
};

// Shared SHM buffer pool for IPC relay.
// One pool per host.  Lazy-allocate data SHMs on first acquire.
class ShmBufPool {
 public:
  ShmBufPool(std::string host_id);
  ~ShmBufPool();

  ShmBufPool(ShmBufPool const&) = delete;
  ShmBufPool& operator=(ShmBufPool const&) = delete;

  ShmBufSlotInfo acquire(size_t min_bytes);
  void release(uint32_t buffer_id);
  void* get_ptr(uint32_t buffer_id);

 private:
  static constexpr size_t kMaxSlots = 64;
  static constexpr uint32_t kMagic = 0x53484250;  // "SHBP"

  // ---- Shared memory layout (must match mmap) ----
  struct Slot {
    std::atomic<uint32_t> state;  // 0=free, 1=in_use
    uint32_t buffer_id;
    size_t capacity;
    char shm_name[128];
  };

  struct Registry {
    uint32_t magic;
    uint32_t num_slots;
    uint32_t next_buffer_id;
    pthread_mutex_t mu;
    Slot slots[kMaxSlots];
  };

  // ---- Local process cache (not shared) ----
  struct LocalMapping {
    void* ptr = nullptr;
    int shm_fd = -1;
    size_t capacity = 0;
  };

  void* open_or_create_data_shm(char const* name, size_t cap, int* out_fd);
  Slot* find_free_slot_locked(size_t min_bytes);
  Slot* find_slot_by_id_locked(uint32_t buffer_id);

  bool is_creator_;
  std::string host_id_;
  std::string reg_shm_name_;
  int reg_shm_fd_ = -1;
  Registry* reg_ = nullptr;

  mutable std::mutex local_cache_mu_;
  std::unordered_map<uint32_t, LocalMapping> local_cache_;
};

}  // namespace Transport
}  // namespace UKernel
