#pragma once

#include "backend.h"
#include "plan.h"
#include <cstddef>
#include <cstdint>

namespace UKernel {
namespace CCL {

enum class CollectiveOpStatus : uint32_t {
  Pending,
  Running,
  Completed,
  Failed,
};

struct CollectiveOpHandle {
  uint64_t value = 0;
};

class Executor {
 public:
  explicit Executor(ExecutorBackends backends);
  ~Executor();

  Executor(Executor const&) = delete;
  Executor& operator=(Executor const&) = delete;

  CollectiveOpHandle submit(CollectivePlan plan);
  bool poll(CollectiveOpHandle handle);
  void wait(CollectiveOpHandle handle);
  void release(CollectiveOpHandle handle);

  CollectiveOpStatus status(CollectiveOpHandle handle) const;
  size_t inflight_steps(CollectiveOpHandle handle) const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace CCL
}  // namespace UKernel
