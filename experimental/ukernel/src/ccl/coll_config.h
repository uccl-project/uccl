#pragma once

#include "coll_types.h"
#include <cstdlib>
#include <cstddef>
#include <string>
#include <vector>

namespace UKernel {
namespace CCL {

struct CollectiveConfig {
  CollKind kind = CollKind::AllReduceRing;
  int nranks = 1;
  int rank = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  std::vector<size_t> input_split_bytes;
  std::vector<size_t> output_split_bytes;
  ScalarType dtype = ScalarType::Float32;
  ReductionKind reduction = ReductionKind::Sum;
  // Signal aggregation: one Signal/WaitSignal pair per this many tiles
  // (per chunk pair) instead of per tile. 1 = per-tile (finest pipeline),
  // larger values cut signal-op counts at the cost of coarser pipelining
  // at group boundaries. Both sides derive identical group tags.
  uint32_t signal_group_tiles = 1;
  // Explicit in-place declaration. The executor cannot infer it from
  // buffer pointers for every collective: NCCL AllGather/ReduceScatter
  // in-place layouts use DISTINCT pointers that physically overlap
  // (sendbuff = recvbuff + rank*sendcount), so `input == output` is
  // false. The shim sets this when it detects the overlapping layout;
  // for AllReduce/AllToAll the executor also accepts input==output as
  // the implicit form.
  bool inplace = false;
  // Out-of-place AllToAll: the caller (nccl.cc shim) performs the
  // self-slice Input->Output copy itself (cudaMemcpyAsync on the user
  // stream), so the plan must NOT emit the pairless local copy op —
  // which would otherwise run on the persistent worker (SM copy) and
  // make the collective slow at low BLK. Default false: the executor
  // level (spray) keeps the in-plan copy.
  bool external_self_slice = false;
  // Fused reduce-scatter (AllReduceRing): the receiver's reduce kernel
  // reads the peer's buffer directly over NVLink (LD/ST) instead of the
  // peer's CE put landing the data locally first; the sender's signal
  // fires when the data is ready (after its producing reduce, or at
  // start for the own shard). Cuts CE traffic in half (RS phase only;
  // the AG phase keeps the configured put path). Toggle via
  // UK_CCL_FUSE_RS_REDUCE (default 0).
  bool fuse_rs_reduce = false;
  // Fused reduce+copy (AllReduceRing RS phase): each RecvReduce task
  // also copies its reduced shard to the next rank's accumulation buffer
  // (device LD/ST write to peer, the alltoall-proven direction) and
  // writes the data-ready signal from the device when it completes.
  // This removes the reduce→put→signal host transitions from the ring's
  // per-hop critical path (the biggest measured latency component).
  // Toggle via UK_CCL_FUSE_REDUCE_COPY (default 0). Forces G=1 (the
  // signal fires per tile).
  bool fuse_reduce_copy = false;
  // Fused-task completion flags: the fused reduce+copy task writes its
  // data-ready signal to a per-slot device flag (plain store + fence,
  // no atomics); the matching WaitSignal polls the slot from the host.
  // Off: the signal is a separate host-written ring Signal. Defaults on
  // when fuse_reduce_copy is on (UK_CCL_DEVICE_FLAGS=0 to disable).
  bool device_flags = false;
  // Fused AG copy (AllReduceRing AG phase): each forward Put becomes a
  // device copy task (read my output, write next's output) with an
  // inline device-completion flag, removing the CE put + host signal
  // chain from the AG per-hop path. Toggle via UK_CCL_FUSE_AG_COPY
  // (default 0). Requires device_flags (the flag slot mechanism).
  bool fuse_ag_copy = false;
};

// Tile sizing rule, shared by the NCCL shim and the spray benchmarks:
// messages at or below the sweet spot move as ONE tile (tiling tiny
// messages only adds per-tile fixed overhead), and larger messages are
// tiled to at most kMaxTilesPerMessage tiles so per-tile overhead stays
// bounded at the large end. Measured on the A40 pair with the IPC put
// path fixed, 1MB is the best sweet spot: each per-tile hop costs a CPU
// round trip (scheduling, completion, signal), so fewer, bigger tiles
// win at every size — 256KB 98.9us -> 88us, 1MB 181us -> 113us, 4MB
// 519us -> 201us, 16MB 2267us -> 454us (native NCCL parity), 64MB
// 2041us -> 1581us. UK_CCL_TILE_MIN_BYTES overrides the sweet spot.
inline size_t adaptive_tile_bytes(size_t bytes) {
  constexpr size_t kMaxTilesPerMessage = 256;
  // Large messages target fewer, bigger tiles: per-tile fixed overhead
  // (IPC put launch/sync cadence, scheduling, signal matching) and the
  // multi-block per-task barrier both scale with tile count. Above the
  // 64MB floor the tile-count target drops to kLargeTiles (default 64),
  // so 256MB uses 4MB tiles (64 puts) instead of 1MB tiles (256 puts).
  // UK_CCL_LARGE_TILES overrides (256 = old behavior).
  constexpr size_t kLargeMessageFloor = 64u << 20;
  static size_t const kLargeTiles = [] {
    char const* v = std::getenv("UK_CCL_LARGE_TILES");
    return v ? static_cast<size_t>(std::stoull(v)) : 64;
  }();
  static size_t const kTileSweetSpotBytes = [] {
    char const* v = std::getenv("UK_CCL_TILE_MIN_BYTES");
    return v ? static_cast<size_t>(std::stoull(v)) : (size_t{1} << 20);
  }();
  size_t const target =
      (bytes > kLargeMessageFloor) ? kLargeTiles : kMaxTilesPerMessage;
  size_t tile = (bytes + target - 1) / target;
  tile = ((tile + 31) / 32) * 32;  // 32B-aligned tile boundaries
  return std::max(kTileSweetSpotBytes, tile);
}

}  // namespace CCL
}  // namespace UKernel
