#pragma once

#include "backend_policy.hpp"
#include "collective_call.hpp"
#include "cpu_router/cpu_router.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace uccl_lite {

// Optional decorator around a caller-owned existing backend. No NCCL
// dependency. One process owns all device ports. Calls are serialized by the
// caller.
class CollectiveDispatcher {
 public:
  using Fallback = std::function<void(CollectiveCall const&)>;
  CollectiveDispatcher(Topology topology, BackendPolicy policy,
                       Fallback fallback,
                       cpu_router::CpuRouterOptions options = {});
  ~CollectiveDispatcher();
  BackendDecision decision(CollectiveCall const& call) const;
  // Initialization boundary: call with idle streams before enqueue, outside
  // capture. A fallback decision creates no CPU workers or pinned routing
  // buffers.
  BackendDecision prepare(CollectiveCall const& call);
  BackendDecision enqueue(CollectiveCall const& call);
  bool routerInitialized() const { return bool(router_); }
  Topology const& topology() const { return topology_; }

 private:
  void validatePorts(CollectiveCall const& call) const;
  size_t packetBytesFor(CollectiveCall const& call) const;
  int workersPerNodeFor(CollectiveCall const& call) const;
  int transportBlocksFor(CollectiveCall const& call) const;
  enum class SmallGpuProgress { None, Flat, PerSource };
  SmallGpuProgress smallGpuProgressFor(CollectiveCall const& call) const;
  Topology topology_;
  BackendPolicy policy_;
  Fallback fallback_;
  cpu_router::CpuRouterOptions options_;
  std::unique_ptr<cpu_router::CpuRouter> router_;
  size_t routerPacketBytes_ = 0;
  int routerWorkersPerNode_ = 0;
  int routerTransportBlocks_ = 0;
  SmallGpuProgress routerSmallGpuProgress_ = SmallGpuProgress::None;
};

}  // namespace uccl_lite
