#pragma once

#include "../memory.h"
#include "../selector.h"
#include <cstdint>

namespace UKernel {
namespace CCL {

struct BackendToken {
  uint64_t value = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual char const* name() const = 0;
  virtual void validate(ExecutionPlan const& plan) const = 0;
  virtual bool supports(ExecOpKind kind) const = 0;
  virtual BackendToken submit(ExecOp const& op) = 0;
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
