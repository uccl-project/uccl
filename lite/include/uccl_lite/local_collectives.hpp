#pragma once

#include "uccl_lite/collective_dispatch.hpp"
#include <array>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

namespace uccl_lite {

// Single-process BF16 collectives for two four-GPU NUMA groups. GPU peers
// exchange data within each group; pinned host memory carries the cross-group
// partials. This is a caller-owned alternative to CPU-router, with no NCCL or
// MSCCL++ dependency. Calls and prepare are serialized by the caller.
class LocalCollectives {
 public:
  LocalCollectives(Topology topology, size_t maxTensorBytes);
  ~LocalCollectives();
  LocalCollectives(LocalCollectives const&) = delete;
  LocalCollectives& operator=(LocalCollectives const&) = delete;

  // Allocate scratch outside the timed or captured region. Enqueue requires
  // prepare and completes the submitted work before returning.
  void prepare();
  void enqueue(CollectiveCall const& call);
  bool initialized() const { return initialized_; }
  static char const* identity() { return "uccl-lite-local-two-numa-bf16-v3"; }

 private:
  void gather(CollectiveCall const& call, size_t bytes);
  void reduce(CollectiveCall const& call, size_t bytes);
  void reduceSmall(CollectiveCall const& call, size_t bytes);
  void reductionWorker(int worker);
  void syncAll(CollectiveCall const& call) const;
  void syncLeaders(CollectiveCall const& call) const;

  Topology topology_;
  size_t maxTensorBytes_;
  std::array<std::vector<int>, 2> groups_;
  std::array<int, 8> groupOfRank_{};
  std::array<int, 8> localRank_{};
  std::array<void*, 2> host_{};
  std::array<void*, 2> scratch_{};
  std::array<void*, 2> remote_{};
  size_t smallBytes_ = 0;
  std::array<void*, 8> smallInput_{};
  void* smallResult_ = nullptr;
  std::vector<std::thread> reductionWorkers_;
  std::mutex reductionMutex_;
  std::condition_variable reductionReady_;
  std::condition_variable reductionPhase_;
  std::condition_variable reductionDone_;
  size_t reductionGeneration_ = 0;
  size_t reductionCount_ = 0;
  CollectiveCall const* reductionCall_ = nullptr;
  int reductionDownloaded_ = 0;
  int reductionReduced_ = 0;
  int reductionCompleted_ = 0;
  bool reductionFailed_ = false;
  std::array<cudaError_t, 8> reductionErrors_{};
  bool reductionStopping_ = false;
  bool initialized_ = false;
};

}  // namespace uccl_lite
