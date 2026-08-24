// coll_algo.h — per-rank macro-op IR for collectives.
//
// A CollAlgo is ONE rank's view of a collective: a DAG of MacroOps
// (Put / Recv / RecvReduce) tiled and lowered into TiledOps by
// lower.cc. Semantic contract:
//
// - MacroOp deps are intra-rank RAW/ordering edges. Lowering
//   propagates them tile-by-tile with tile alignment (min of both
//   sides' tile counts); irregular tile-level dependencies are NOT
//   supported (extend deps granularity if that ever becomes needed).
// - Buffers are named and always authoritative: Input/Output are the
//   user tensors; Tmp(i) is an algorithm-declared intermediate region
//   (CollAlgo::tmp_bytes) the lowering maps into executor scratch.
//   A Recv's dst declares where the peer's data lands (mirroring the
//   peer Put's dst); RecvReduce accumulates src into dst.
// - pair_id is the cross-rank pairing identifier AND the signal-tag
//   namespace: sender and receiver derive the same id for the same
//   logical transfer (ring kinds: owner*2 for the reduce-scatter
//   phase, owner*2+1 for the allgather phase; alltoall: src*nranks+dst
//   for the directed rank pair). Signal tags are pair_id in the high
//   bits and the tile-group index in a plan-adaptive low field
//   (TiledResult::tag_group_bits); the all-ones group value is
//   reserved for the lowering's copies-done handshake.
// - Cross-rank anti-dependencies (WAR on mutable shared buffers) are
//   avoided by construction wherever possible: in-place allreduce
//   accumulates into a Tmp region so peer Puts never clobber unread
//   local input; the remaining case (in-place alltoall with variable
//   splits) is DECLARED per-op by the builder (stage_via_scratch) and
//   mechanized by the lowering — deliberately NOT encoded as deps in
//   the IR; this is an intentional architectural boundary.
//
// Correctness of the pairing conventions above is checked by
// verify_algo_pairing() (bottom of this header); tree / recursive
// halving-doubling / hierarchical algorithms are all expressible as
// per-step point-to-point exchanges with per-chunk deps and private
// pair_id namespaces per concurrent channel.
#pragma once

#include "backend/backend.h"
#include "coll_config.h"
#include "coll_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

// Ring topology with an explicit order (a rank permutation). The
// algorithm shape is code; the order is data — produced by a
// topology/NIC/incast planner (multi-ring / rail-aware ring order).
// The identity order reproduces the classic rank+1 ring exactly.
struct RingTopology {
  int nranks = 1;
  std::vector<int> order;       // the ring sequence (rank permutation)
  std::vector<int> next_rank_;  // rank -> successor in the ring
  std::vector<int> prev_rank_;  // rank -> predecessor in the ring
  std::vector<int> pos_;        // rank -> position in order

  RingTopology() : RingTopology(std::vector<int>{0}) {}
  explicit RingTopology(int n) : RingTopology(make_identity_ring(n)) {}
  explicit RingTopology(std::vector<int> ring_order)
      : nranks(static_cast<int>(ring_order.size())),
        order(std::move(ring_order)),
        next_rank_(static_cast<size_t>(nranks)),
        prev_rank_(static_cast<size_t>(nranks)),
        pos_(static_cast<size_t>(nranks)) {
    for (int i = 0; i < nranks; ++i) {
      pos_[static_cast<size_t>(order[static_cast<size_t>(i)])] = i;
      next_rank_[static_cast<size_t>(order[static_cast<size_t>(i)])] =
          order[static_cast<size_t>((i + 1) % nranks)];
      prev_rank_[static_cast<size_t>(order[static_cast<size_t>(i)])] =
          order[static_cast<size_t>((i - 1 + nranks) % nranks)];
    }
  }

  int next(int rank) const { return next_rank_[static_cast<size_t>(rank)]; }
  int prev(int rank) const { return prev_rank_[static_cast<size_t>(rank)]; }
  // Rank at position pos(rank)+delta in the order (mod n) — the chunk
  // owner a ring step rotates through.
  int rank_at_offset(int rank, int delta) const {
    int pos = pos_[static_cast<size_t>(rank)] + delta;
    pos %= nranks;
    if (pos < 0) pos += nranks;
    return order[static_cast<size_t>(pos)];
  }

 private:
  static std::vector<int> make_identity_ring(int n) {
    std::vector<int> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = i;
    return v;
  }
};

// Named buffer spaces a MacroOp can address. Input/Output are the user
// tensors; Tmp(i) is an algorithm-declared intermediate region (see
// CollAlgo::tmp_bytes) that the lowering maps into executor scratch.
enum class BufSpace : uint8_t { Input, Output, Tmp };

struct BufRef {
  BufSpace space;
  uint32_t index;  // only meaningful for Tmp
};

struct MacroOp {
  AlgoOpKind op = AlgoOpKind::Put;
  size_t bytes = 0;
  size_t src_off = 0;
  size_t dst_off = 0;
  int src_rank = -1;  // -1 = local buffer, >=0 = remote rank
  int dst_rank = -1;
  uint32_t pair_id = 0;  // matched Send/Recv pairs share the same id
  std::vector<uint32_t> deps;
  // Buffers the op reads from / writes to: ALWAYS authoritative — every
  // builder sets them explicitly for data-moving chunks (defaults keep
  // Input -> Output, the most common case). For a Recv, dst declares
  // where the peer's data lands (mirrors the peer Put's dst).
  BufRef src = {BufSpace::Input, 0};
  BufRef dst = {BufSpace::Output, 0};
  // Per-op put path override (None = auto). The AllToAll hybrid splits a
  // per-peer send into a CE half and a device-copy half so the CE engine
  // and the worker overlap on the same shard.
  PutPath put_path_hint = PutPath::None;
  // Fused reduce+copy (fuse_reduce_copy): after the reduce, copy dst to
  // the next rank's accumulation buffer (copy_dst) and device-write the
  // data-ready signal (pair_id, per tile). The copy and signal target
  // the peer given by copy_dst's owning rank.
  bool fuse_copy_to_peer = false;
  BufRef copy_dst = {BufSpace::Output, 0};
  int copy_peer = -1;  // fused-copy target rank (ring next)
  // Fused AG copy: this Put's data movement is a device copy task with
  // an inline device-completion flag (no CE, no host signal op).
  bool fuse_copy_flag = false;
  // The matching cross-rank sender emits a standalone Signal (not a
  // put), so this Recv's WaitSignal must be a plain one-arrival wait —
  // the lowering skips the put-fused group-count metadata.
  bool wait_standalone_signal = false;
  // Coordination requirement declared by the builder (the mechanism
  // lives in the lowering):
  // - stage_via_scratch (Put): the data must be staged through scratch
  //   before being sent out (in-place alltoall with variable splits,
  //   where peer writes can overlap not-yet-read local data).
  bool stage_via_scratch = false;
};

struct CollAlgo {
  CollKind kind = CollKind::AllReduceRing;
  int nranks = 1;
  int rank = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  ReductionKind reduction = ReductionKind::None;
  ScalarType dtype = ScalarType::Float32;
  // Declared intermediate buffers (Tmp regions), sizes in bytes. The
  // lowering lays them out back-to-back and sizes the executor scratch
  // to cover them (plus lowering-internal staging).
  std::vector<size_t> tmp_bytes;
  std::vector<MacroOp> chunks;
};

CollAlgo build_coll_algo(CollectiveConfig const& config, bool inplace);

// Cross-rank consistency check for a collective config: builds the
// CollAlgo of every rank (recomputing rank-dependent sizes for
// RS/AG) and verifies the invariants the runtime signaling protocol
// relies on — Put/Recv pairing (unique, same pair_id, bytes, dst
// buffer+offset mirror), local RecvReduce pairing, tag-space
// disjointness per directed rank pair, and acyclic backward-only local
// deps. Throws std::invalid_argument on the first violation. NOTE:
// full deadlock-freedom is out of scope (it depends on the runtime
// signal semantics); this checks the structural invariants only.
void verify_algo_pairing(CollectiveConfig const& config, bool inplace);

}  // namespace CCL
}  // namespace UKernel
