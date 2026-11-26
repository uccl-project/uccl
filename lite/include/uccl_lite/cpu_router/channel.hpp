#pragma once
#include "memory.hpp"
#include <memory>
#include <vector>
namespace uccl_lite::cpu_router {
enum class TransportBackend { GpuKernel, Dma };
struct ChannelOptions {
  TransportBackend backend = TransportBackend::GpuKernel;
  int device = 0;
  int blocks = 64;
  int threads = 256;
  size_t completionSlots = 128;
};
struct ChannelCompletionState;
class ChannelCompletion {
 public:
  bool ready() const;
  void wait() const;
  void waitOn(int device, cudaStream_t stream) const;

 private:
  std::shared_ptr<ChannelCompletionState> state_;
  friend class Channel;
};
// Resources stay registered through completion even if the returned ticket is
// dropped.
class Channel {
 public:
  explicit Channel(ChannelOptions options);
  ~Channel();
  ChannelCompletion put(MemorySpan destination, MemorySpan source,
                        cudaStream_t stream);
  ChannelCompletion get(MemorySpan destination, MemorySpan source,
                        cudaStream_t stream);
  // Pipeline-owned spans/streams provide lifetime and completion externally.
  void putAsync(MemorySpan destination, MemorySpan source,
                cudaStream_t stream) const;
  void getAsync(MemorySpan destination, MemorySpan source,
                cudaStream_t stream) const;

 private:
  ChannelOptions options_;
  std::vector<std::shared_ptr<ChannelCompletionState>> pending_;
};
}  // namespace uccl_lite::cpu_router
