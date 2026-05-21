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

class Backend {
 public:
  virtual ~Backend() = default;

  virtual char const* name() const = 0;
  virtual void validate(CollectivePlan const& plan,
                        CollectiveBinding& binding) const = 0;
  virtual bool supports(OpKind kind) const = 0;
  virtual BackendToken submit(Op const& op, CollectiveBinding& binding) = 0;
  virtual bool poll(BackendToken token) = 0;
  virtual bool try_pop_completed(BackendToken& token) = 0;
  virtual void release(BackendToken token) = 0;
  virtual void stop(uint32_t flow_id) { (void)flow_id; }
};

struct ExecutorBackends {
  Backend* transport = nullptr;
  Backend* device = nullptr;
  Backend* fallback = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
