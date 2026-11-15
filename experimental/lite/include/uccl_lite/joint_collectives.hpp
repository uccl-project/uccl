#pragma once

#include "collective_dispatch.hpp"
#include <memory>
#include <string>

namespace uccl_lite {

enum class JointMode { Peer, Router, Auto };

// Single process, 2, 4, or 8 GPUs, BF16 out-of-place collectives. Calls are
// host serialized. prepare() is an explicit initialization boundary with idle
// caller streams. enqueue() respects producers on each caller stream and queues
// output completion AND input-reader completion on every caller stream. The
// initial peer adapter is eager-only; graph capture is rejected before side
// effects.
//
// The peer implementation adapts peer_ag_rs.cu from upstream 5b9a7f86:
// direct pulls, large AG peer-copy fan-out, large RS peer copies into private
// destination scratch, and large AR reduce-scatter followed by peer allgather.
// All paths support BF16 on two/four ranks; ordered FP32 sums convert once
// to BF16 before final output distribution.
// These are adaptations, not the unchanged FP32 benchmark envelope.
// Staged peer calls require prepare(call) before the first use or any larger
// extent. Scratch allocation/growth is outside enqueue; failed allocation
// preserves earlier prepared storage and capacity. Large AR whose count cannot
// be partitioned equally across ranks retains the general peer pull path.
// AR staging starts at 4 MiB on two ranks and 64 MiB on four ranks.
// Two-rank staged reductions read their own local shard directly from input;
// public completion still covers that input reader.
// Peer calls may change caller stream sets; internal prior-completion waits
// protect scratch independently of caller stream ordering.
// Two-rank coordinators through 256 KiB execute on the first caller stream;
// their done event remains chained through later families and is awaited at
// destruction. Eight-rank calls use the capability-forced CPU router; direct
// peer mode is unavailable at eight ranks.
class JointCollectives {
public:
  JointCollectives(Topology topology, JointMode mode, BackendPolicy policy = {},
                   cpu_router::CpuRouterOptions options = {});
  ~JointCollectives();
  JointCollectives(JointCollectives const &) = delete;
  JointCollectives &operator=(JointCollectives const &) = delete;

  BackendDecision prepare(CollectiveCall const &call);
  BackendDecision enqueue(CollectiveCall const &call);
  BackendDecision decision(CollectiveCall const &call) const;
  Topology const &topology() const;
  bool routerInitialized() const;
  static std::string peerIdentity();
  static bool peerSupported(Topology const &topology);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace uccl_lite
