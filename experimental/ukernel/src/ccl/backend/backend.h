#pragma once

#include "../collective_memory.h"
#include "../plan.h"
#include <cstdint>

namespace UKernel {
namespace CCL {

struct BackendToken {
  uint64_t value = 0;
  bool failed = false;
};

// Minimal backend interface:
//   validate  — one-time init (memory bindings, peer paths, worker warmup)
//   submit    — enqueue one op, returns opaque token (value==0 = backpressure)
//   drain     — harvest all completed ops; backend cleans its own resources
//   stop      — end-of-flow cleanup (optional)
class Backend {
 public:
  virtual ~Backend() = default;

  virtual char const* name() const = 0;
  virtual void validate(CollectivePlan const& plan,
                        CollectiveBinding& binding) = 0;
  virtual bool supports(OpKind kind) const = 0;
  virtual BackendToken submit(Op const& op, CollectiveBinding& binding) = 0;
  virtual size_t drain(BackendToken* out, size_t max_count) = 0;
  virtual void stop(uint32_t flow_id) { (void)flow_id; }
};

struct ExecutorBackends {
  Backend* transport = nullptr;
  Backend* device = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
