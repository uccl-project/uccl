#pragma once

#include "../oob/oob.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UKernel {
namespace Transport {

struct MRItem {
  MR mr{};
  bool is_local = false;
  int rank = -1;
  bool backend_registered = false;
  bool valid = false;
};

class MRManager {
 public:
  using RegisterFn = std::function<bool(uint32_t, void*, size_t)>;
  using DeregisterFn = std::function<void(uint32_t)>;

  void bind_backend(RegisterFn register_fn, DeregisterFn deregister_fn);
  void sync_local_backend();

  MRItem create_local_mr(void* ptr, size_t len);
  bool register_remote_mr(int rank, MRItem const& item);
  void register_remote_mrs(int rank, std::vector<MRItem> const& items);

  MRItem get_mr(void* local_ptr) const;
  MRItem get_mr(uint32_t local_mr_id) const;
  MRItem get_mr(int remote_rank, uint32_t remote_mr_id) const;

  bool delete_mr(void* local_ptr);
  bool delete_mr(int remote_rank, uint32_t remote_mr_id);
  void delete_mr(int remote_rank);

  std::vector<std::pair<void*, MRItem>> list_local_mrs() const;

 private:
  mutable std::mutex local_mu_;
  std::unordered_map<void*, MRItem> local_by_ptr_;
  std::unordered_map<uint32_t, void*> local_ptr_by_id_;
  uint32_t next_mr_id_ = 1;
  RegisterFn register_fn_;
  DeregisterFn deregister_fn_;

  mutable std::mutex remote_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, MRItem>>
      remote_by_rank_;
};

}  // namespace Transport
}  // namespace UKernel
