#pragma once

#include "plan.h"
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace CCL {

// Shared buffer bindings for collective execution backends. Backends may use
// only the fields relevant to their transport/execution path.
struct CollectiveBuffers {
  void const* local_input = nullptr;
  void const* remote_input = nullptr;
  void const* remote_reduced = nullptr;
  void* final_output = nullptr;
  void* recv_staging = nullptr;
  size_t registration_bytes = 0;
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
  virtual void release(BackendToken token) = 0;
};

struct ExecutorBackends {
  Backend* transport = nullptr;
  Backend* device = nullptr;
  Backend* fallback = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
