#pragma once
#include "uccl_lite/collective_types.hpp"
#include "uccl_lite/collectives.h"
#include <vector>

namespace uccl_lite {
struct PeerGraphCall {
  NetworkOperation operation = NetworkOperation::AllReduce;
  void const* input = nullptr;
  void* output = nullptr;
  size_t count = 0;
};
// Same-host BF16/FP32 collectives. Direct mode uses a complete CUDA peer graph;
// tree mode uses only a validated connected set of bidirectional edges. Rank
// order is independent of device/NUMA order. No allocation until prepare.
// Calls are host-serialized; the executor chains work across caller streams.
// The strict ABI currently rejects capture before calling this executor.
class PeerStagedCollectives {
 public:
  static NetworkId createUniqueId();
  PeerStagedCollectives(NetworkId id, int rank, int device, int ranks,
                        size_t capacity);
  PeerStagedCollectives(NetworkId id, int rank, int device, int ranks,
                        size_t capacity, ScalarType dtype);
  // Source-ordered tree forwarding over only the supplied bidirectional edges.
  // Parents are in logical rank order, rooted at rank zero. Requires consistent
  // same-host CUDA visibility; every required edge is checked before
  // allocation.
  PeerStagedCollectives(NetworkId id, int rank, int device, int ranks,
                        size_t capacity, ScalarType dtype,
                        std::vector<int> parents);
  ~PeerStagedCollectives();
  PeerStagedCollectives(PeerStagedCollectives const&) = delete;
  PeerStagedCollectives& operator=(PeerStagedCollectives const&) = delete;
  void prepare();
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream);
  // Collective, idle preparation of fixed caller-owned bindings. Input/output
  // storage must remain valid until releaseGraph. The same operation/count
  // sequence is required on every rank; local pointers/aliases may differ.
  // This is an owned graph interface, not permission to capture enqueue.
  size_t prepareGraph(std::vector<PeerGraphCall> const& calls);
  void launchGraph(size_t graph, cudaStream_t stream);
  // Release only while idle. Handles are never recycled.
  void releaseGraph(size_t graph);
  void synchronize();
  bool initialized() const;
  bool idle() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  void validate(PeerGraphCall const& call) const;
  void submit(PeerGraphCall const& call, cudaStream_t stream);
  void chain(cudaStream_t stream);
  void record(cudaStream_t stream);
};
}  // namespace uccl_lite
