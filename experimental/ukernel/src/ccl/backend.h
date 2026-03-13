#pragma once

#include "plan.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace UKernel {
namespace CCL {

struct BackendToken {
  uint64_t value = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual char const* name() const = 0;
  virtual bool supports(ExecutionOpKind kind) const = 0;
  virtual BackendToken submit(ExecutionOp const& op) = 0;
  virtual bool poll(BackendToken token) = 0;
  virtual void release(BackendToken token) = 0;
};

class MockBackend final : public Backend {
 public:
  explicit MockBackend(uint32_t polls_before_ready = 1);

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
  bool poll(BackendToken token) override;
  void release(BackendToken token) override;

  uint64_t submissions() const { return submissions_; }

 private:
  uint32_t polls_before_ready_ = 1;
  uint64_t next_token_ = 1;
  uint64_t submissions_ = 0;
  std::unordered_map<uint64_t, uint32_t> pending_polls_;
};

class PersistentKernelBackend final : public Backend {
 public:
  explicit PersistentKernelBackend(uint32_t polls_before_ready = 1);

  char const* name() const override;
  bool supports(ExecutionOpKind kind) const override;
  BackendToken submit(ExecutionOp const& op) override;
  bool poll(BackendToken token) override;
  void release(BackendToken token) override;

  uint64_t submissions() const { return submissions_; }

 private:
  uint32_t polls_before_ready_ = 1;
  uint64_t next_token_ = 1;
  uint64_t submissions_ = 0;
  std::unordered_map<uint64_t, uint32_t> pending_polls_;
};

struct ExecutorBackends {
  Backend* rdma = nullptr;
  Backend* ce = nullptr;
  Backend* persistent = nullptr;
  Backend* fallback = nullptr;
};

}  // namespace CCL
}  // namespace UKernel
