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

// Ring neighbor arithmetic used by the ring builders.
struct RingTopology {
  int nranks = 1;

  int wrap(int rank) const {
    if (nranks <= 0) return 0;
    int value = rank % nranks;
    return value < 0 ? value + nranks : value;
  }
  int next(int rank) const { return wrap(rank + 1); }
  int prev(int rank) const { return wrap(rank - 1); }
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
  // Per-op put path override (None = auto). The RS hybrid splits a
  // tile's send into a CE half and a device-copy half so the CE engine
  // and the peer's worker overlap on the same shard.
  PutPath put_path_hint = PutPath::None;
  // Fused RecvReduce (AllReduceRing, fuse_rs_reduce): src is the PEER's
  // send-source buffer (resolved through src_rank), src2 is this rank's
  // local Input contribution for the out-of-place 3-way reduce (dst =
  // src op src2). In-place fused reduces are 2-way RMW (dst = Input).
  BufRef src2 = {BufSpace::Input, 0};
  bool fuse_remote_src = false;
  uint8_t reduce_mode = 0;  // 0 = RMW dst=op(dst,src); 1 = dst=op(src,src2)
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
