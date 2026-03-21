#pragma once

#include "plan.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace UKernel {
namespace CCL {

struct PeerTensorView {
  int rank = -1;
  void* ptr = nullptr;
  uint32_t mr_id = 0;
  bool same_node = false;
  bool peer_accessible = false;
};

struct SymmetricTensor {
  int local_rank = 0;
  void* local_ptr = nullptr;
  uint32_t local_mr_id = 0;
  size_t bytes = 0;
  std::vector<PeerTensorView> peers;
};

struct CollectiveMemory {
  SymmetricTensor tensor;
  void* recv_staging = nullptr;
  size_t recv_staging_bytes = 0;
};

struct BackendToken {
  uint64_t value = 0;
};

// Backend is the execution-side interface used by Executor after plan lowering
// has decided each op's concrete kind.
class Backend {
 public:
  virtual ~Backend() = default;

  virtual char const* name() const = 0;
  virtual bool supports(ExecutionOpKind kind) const = 0;
  virtual BackendToken submit(ExecutionOp const& op) = 0;
  virtual bool poll(BackendToken token) = 0;
  virtual bool try_pop_completed(BackendToken& token) = 0;
  virtual void release(BackendToken token) = 0;
};

struct ExecutorBackends {
  Backend* transport = nullptr;
  Backend* device = nullptr;
  Backend* fallback = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
