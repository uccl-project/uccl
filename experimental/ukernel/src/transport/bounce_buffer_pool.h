#pragma once

#include "adapter/transport_adapter.h"
#include "memory/mr_manager.h"
#include "memory/shm_manager.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace UKernel {
namespace Transport {

class BounceBufferPool {
 public:
  struct Slot {
    void* ptr = nullptr;
    size_t bytes = 0;
    uint32_t buffer_id = 0;
    std::string shm_name;
    uint32_t shm_id = 0;

    BounceBufferInfo to_bounce_info() const {
      BounceBufferInfo info;
      info.ptr = ptr;
      info.buffer_id = buffer_id;
      info.shm_name = shm_name;
      return info;
    }
  };

  BounceBufferPool(SHMManager& shm, MRManager& mr, bool needs_uccl_mr,
                   std::atomic<uint32_t>& ephemeral_id_gen);
  ~BounceBufferPool();

  BounceBufferPool(BounceBufferPool const&) = delete;
  BounceBufferPool& operator=(BounceBufferPool const&) = delete;

  // Creates a new buffer if no free buffer >= min_bytes; returns nullptr on
  // allocation failure.
  Slot* acquire(size_t min_bytes);

  // Return a buffer to the pool for reuse.
  void release(Slot* slot);

  // Release all SHM + MR resources explicitly (called before managers die).
  void shutdown();

 private:
  SHMManager& shm_;
  MRManager& mr_;
  bool needs_uccl_mr_;
  std::atomic<uint32_t>& ephemeral_id_gen_;

  std::mutex mu_;
  // Free buffers indexed by size for best-fit search.
  std::multimap<size_t, std::unique_ptr<Slot>> free_slots_;
  // All allocated slots (owned, for shutdown).
  std::vector<std::unique_ptr<Slot>> all_slots_;
};

}  // namespace Transport
}  // namespace UKernel
