#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

struct SHMItem {
  void* ptr = nullptr;
  size_t bytes = 0;
  uint32_t shm_id = 0;
  bool shareable = false;
  bool is_local = false;
  bool valid = false;
  std::string shm_name;
};

class SHMManager {
 public:
  SHMManager() = default;
  ~SHMManager();

  SHMItem create_local_shm(size_t bytes, bool require_shared = false);
  SHMItem get_local_shm(uint32_t shm_id) const;
  SHMItem open_remote_shm(std::string const& shm_name);

  bool delete_local_shm(uint32_t shm_id);
  bool close_remote_shm(std::string const& shm_name);
  void clear_remote_shm_cache();

 private:
  struct Entry {
    void* ptr = nullptr;
    size_t bytes = 0;
    uint32_t shm_id = 0;
    std::string shm_name;
  };

  struct ShmCacheEntry {
    void* ptr = nullptr;
    size_t size = 0;
  };

  std::atomic<uint32_t> next_shm_id_{1};
  mutable std::mutex mu_;
  std::vector<Entry> entries_;
  std::unordered_map<uint32_t, size_t> local_slot_by_id_;
  std::unordered_map<std::string, ShmCacheEntry> shm_cache_;
};

}  // namespace Transport
}  // namespace UKernel
