#pragma once

#include "../oob/oob.h"
#include <cstddef>
#include <cstdint>
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
  bool valid = false;
};

class MRManager {
 public:
  MRItem create_local_mr(uint32_t buffer_id, void* ptr, size_t len);
  bool register_remote_mr(int rank, MRItem const& item);
  void register_remote_mrs(int rank, std::vector<MRItem> const& items);

  MRItem get_mr(void* local_ptr) const;
  MRItem get_mr(uint32_t local_buffer_id) const;
  MRItem get_mr(int remote_rank, uint32_t remote_buffer_id) const;

  bool delete_mr(void* local_ptr);
  bool delete_mr(uint32_t local_buffer_id);
  bool delete_mr(int remote_rank, uint32_t remote_buffer_id);
  void delete_mr(int remote_rank);

  std::vector<std::pair<void*, MRItem>> list_local_mrs() const;

 private:
  mutable std::mutex local_mu_;
  std::unordered_map<void*, MRItem> local_by_ptr_;
  std::unordered_map<uint32_t, void*> local_ptr_by_buffer_id_;

  mutable std::mutex remote_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, MRItem>> remote_by_rank_;
};

}  // namespace Transport
}  // namespace UKernel
