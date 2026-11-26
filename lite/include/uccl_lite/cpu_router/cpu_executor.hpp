#pragma once
#include "memory.hpp"
#include <cstdint>
#include <memory>
#include <vector>
namespace uccl_lite::cpu_router {
struct CpuTaskState;
class CpuCompletion {
 public:
  bool ready() const;
  void wait() const;

 private:
  std::shared_ptr<CpuTaskState> state_;
  uint64_t ticket_ = 0;
  friend class CpuExecutor;
};
// Fixed pinned lanes and bounded queues. Callbacks execute outside queue locks.
class CpuExecutor {
 public:
  explicit CpuExecutor(std::vector<int> cpus, size_t depth = 128,
                       uint32_t spinMicros = 100);
  ~CpuExecutor();
  using Callback = void (*)(void*, int);
  CpuCompletion enqueue(size_t lane, Callback run, void* context);
  // Publish callbacks before notifying sleeping lanes. Indices must be strictly
  // increasing. The caller tracks completion through the callback protocol.
  void enqueueBatch(size_t const* lanes, size_t count, Callback run,
                    void* context);
  CpuCompletion enqueueSum(size_t lane, MemorySpan const* sources, size_t count,
                           MemorySpan destination);
  CpuCompletion enqueueMerge(size_t lane, MemorySpan local, MemorySpan remote,
                             MemorySpan destination);
  void stop();
  static void sumBf16(MemorySpan const* sources, size_t count,
                      MemorySpan destination);
  static void mergeBf16(MemorySpan local, MemorySpan remote,
                        MemorySpan destination);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uccl_lite::cpu_router
