#include "coll_algo.h"

#include "utils.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

size_t balanced_shard_size_bytes(size_t bytes, size_t elem_bytes, int nranks,
                                 int owner_rank);

size_t balanced_shard_offset_bytes(size_t bytes, size_t elem_bytes, int nranks,
                                   int owner_rank) {
  size_t off = 0;
  for (int r = 0; r < owner_rank; ++r)
    off += balanced_shard_size_bytes(bytes, elem_bytes, nranks, r);
  return off;
}

size_t balanced_shard_size_bytes(size_t bytes, size_t elem_bytes, int nranks,
                                 int owner_rank) {
  // 16B-aligned balanced shard layout: the CE copy engine and the
  // vectorized kernels require 16-byte alignment for addresses and
  // sizes. First n-1 shards are multiples of 16; the last shard absorbs
  // the residual (<16B, zero whenever the tensor is a 16B multiple —
  // the norm for collectives).
  size_t const align = 16;
  size_t base = (bytes / static_cast<size_t>(nranks)) & ~(align - 1);
  size_t rem = bytes - base * static_cast<size_t>(nranks);
  size_t extra = rem / align;
  size_t sz = base + (static_cast<size_t>(owner_rank) < extra ? align : 0);
  if (owner_rank == nranks - 1) sz += rem % align;
  (void)elem_bytes;
  return sz;
}

std::vector<size_t> equal_alltoall_splits(size_t total_bytes, int nranks) {
  std::vector<size_t> splits(static_cast<size_t>(nranks), 0);
  size_t shard = total_bytes / static_cast<size_t>(nranks);
  std::fill(splits.begin(), splits.end(), shard);
  return splits;
}

std::vector<size_t> prefix_bytes(std::vector<size_t> const& splits) {
  std::vector<size_t> prefix(splits.size(), 0);
  size_t running = 0;
  for (size_t i = 0; i < splits.size(); ++i) {
    prefix[i] = running;
    running += splits[i];
  }
  return prefix;
}

void validate_alltoall_splits(std::vector<size_t> const& splits, int nranks,
                              size_t elem_bytes, size_t total_bytes,
                              char const* which) {
  if (splits.size() != static_cast<size_t>(nranks)) {
    throw std::invalid_argument(std::string("alltoall ") + which +
                                " split count must equal nranks");
  }
  size_t sum = 0;
  for (size_t part : splits) {
    if (part % elem_bytes != 0) {
      throw std::invalid_argument(std::string("alltoall ") + which +
                                  " split bytes must align to dtype size");
    }
    sum += part;
  }
  if (sum != total_bytes) {
    throw std::invalid_argument(std::string("alltoall ") + which +
                                " split bytes must sum to tensor bytes");
  }
}

void require_collective_config(CollectiveConfig const& config) {
  if (config.nranks < 2) {
    throw std::invalid_argument("collective requires at least two ranks");
  }
  if (config.rank < 0 || config.rank >= config.nranks) {
    throw std::invalid_argument("collective rank out of range");
  }

  size_t elem_bytes = scalar_type_size(config.dtype);
  if (elem_bytes == 0) {
    throw std::invalid_argument("collective dtype has invalid element size");
  }

  if (config.kind == CollKind::AllReduceRing ||
      config.kind == CollKind::AllReduceTree) {
    if (config.input_bytes == 0) {
      throw std::invalid_argument("collective input_bytes must be positive");
    }
    if (config.input_bytes % elem_bytes != 0) {
      throw std::invalid_argument(
          "collective input_bytes must be aligned to dtype size");
    }
    return;
  }

  if (config.kind == CollKind::ReduceScatterRing) {
    if (config.input_bytes == 0 || config.input_bytes % elem_bytes != 0) {
      throw std::invalid_argument(
          "reduce-scatter input_bytes must be positive and aligned to dtype "
          "size");
    }
    size_t want = balanced_shard_size_bytes(config.input_bytes, elem_bytes,
                                            config.nranks, config.rank);
    if (config.output_bytes != want) {
      throw std::invalid_argument(
          "reduce-scatter output_bytes must equal this rank's balanced shard "
          "size of the input");
    }
    return;
  }

  if (config.kind == CollKind::AllGatherRing) {
    if (config.output_bytes == 0 || config.output_bytes % elem_bytes != 0) {
      throw std::invalid_argument(
          "all-gather output_bytes must be positive and aligned to dtype "
          "size");
    }
    size_t want = balanced_shard_size_bytes(config.output_bytes, elem_bytes,
                                            config.nranks, config.rank);
    if (config.input_bytes != want) {
      throw std::invalid_argument(
          "all-gather input_bytes must equal this rank's balanced shard size "
          "of the output");
    }
    return;
  }

  if (config.input_bytes == 0 || config.output_bytes == 0) {
    throw std::invalid_argument(
        "alltoall requires positive input/output tensor bytes");
  }
  if (config.input_bytes % elem_bytes != 0 ||
      config.output_bytes % elem_bytes != 0) {
    throw std::invalid_argument(
        "alltoall input/output tensor bytes must align to dtype size");
  }

  bool has_input_splits = !config.input_split_bytes.empty();
  bool has_output_splits = !config.output_split_bytes.empty();
  if (has_input_splits != has_output_splits) {
    throw std::invalid_argument(
        "alltoall split configuration must provide both input and output "
        "splits");
  }
  if (!has_input_splits) {
    size_t denom = static_cast<size_t>(config.nranks) * elem_bytes;
    if (config.input_bytes % denom != 0 || config.output_bytes % denom != 0) {
      throw std::invalid_argument(
          "equal-split alltoall requires input/output tensor bytes divisible "
          "by nranks * dtype size");
    }
    return;
  }

  validate_alltoall_splits(config.input_split_bytes, config.nranks, elem_bytes,
                           config.input_bytes, "input");
  validate_alltoall_splits(config.output_split_bytes, config.nranks, elem_bytes,
                           config.output_bytes, "output");
}

CollAlgo make_empty_algo(CollectiveConfig const& config) {
  CollAlgo algo;
  algo.nranks = config.nranks;
  algo.rank = config.rank;
  algo.input_bytes = config.input_bytes;
  algo.output_bytes = config.output_bytes;
  algo.reduction = config.reduction;
  algo.dtype = config.dtype;
  return algo;
}

struct AlgoBuilder {
  explicit AlgoBuilder(CollAlgo algo_in) : algo(std::move(algo_in)) {}

  uint32_t add_op(AlgoOpKind kind, size_t bytes, size_t src_off, size_t dst_off,
                  int src_rank, int dst_rank, std::vector<uint32_t> deps,
                  uint32_t pair_id = kNoPairId,
                  BufRef src = {BufSpace::Input, 0},
                  BufRef dst = {BufSpace::Output, 0}) {
    MacroOp chunk;
    chunk.op = kind;
    chunk.bytes = bytes;
    chunk.src_off = src_off;
    chunk.dst_off = dst_off;
    chunk.src_rank = src_rank;
    chunk.dst_rank = dst_rank;
    chunk.deps = std::move(deps);
    chunk.pair_id = pair_id;
    chunk.src = src;
    chunk.dst = dst;
    uint32_t idx = static_cast<uint32_t>(algo.chunks.size());
    algo.chunks.push_back(std::move(chunk));
    return idx;
  }

  CollAlgo algo;
};

// Reduce-scatter phase of ring allreduce: partial sums circulate
// rank -> rank+1; virtual shard v's full sum ends at rank wrap(v-1).
// First touch of a shard reads Input; afterwards the running sum.
// Out-of-place: everything lives in Output. In-place (Input aliases
// Output): partials accumulate in Tmp(0), laid out like the full
// input — a peer's Put then never clobbers unread local input, so no
// snapshot handshake is needed.
void emit_ring_reduce_scatter(RingTopology const& ring,
                              CollectiveConfig const& config,
                              AlgoBuilder& builder,
                              std::vector<uint32_t>& ready_ops,
                              bool inplace, size_t seg_bytes,
                              size_t channel_base,
                              uint32_t pair_base) {
  size_t elem_bytes = scalar_type_size(config.dtype);
  bool const fuse_copy = config.fuse_reduce_copy;
  // Shard offsets are relative to this channel's byte interval, then
  // shifted by the interval's base so different channels touch disjoint
  // regions of the same Input/Output buffers.
  auto off = [&](int owner) {
    return channel_base +
           balanced_shard_offset_bytes(seg_bytes, elem_bytes, config.nranks,
                                       owner);
  };
  BufRef const accum =
      inplace ? BufRef{BufSpace::Tmp, 0} : BufRef{BufSpace::Output, 0};
  for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
    int send_owner = ring.rank_at_offset(config.rank, -ring_step);
    int recv_owner = ring.rank_at_offset(config.rank, -ring_step - 1);
    int send_peer = ring.next(config.rank);
    int recv_peer = ring.prev(config.rank);
    size_t send_bytes = balanced_shard_size_bytes(
        seg_bytes, elem_bytes, config.nranks, send_owner);
    size_t recv_bytes = balanced_shard_size_bytes(
        seg_bytes, elem_bytes, config.nranks, recv_owner);

    if (send_bytes > 0 && !fuse_copy) {
      size_t offset = off(send_owner);
      uint32_t pair_id = pair_base + static_cast<uint32_t>(send_owner * 2);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(send_owner)]);
      builder.add_op(AlgoOpKind::Put, send_bytes, offset, offset, -1,
                     send_peer, std::move(deps), pair_id,
                     ring_step == 0 ? BufRef{BufSpace::Input, 0} : accum,
                     accum);
    } else if (send_bytes > 0) {
      // Fused reduce+copy: the send of shard wrap(rank-s) is done by the
      // PREVIOUS step's RecvReduce task (which produced that shard) as a
      // device copy + device signal. Only the step-0 send (the rank's
      // own shard, from Input) remains a standalone put.
      if (ring_step == 0) {
        size_t offset = off(send_owner);
        uint32_t pair_id = pair_base + static_cast<uint32_t>(send_owner * 2);
        builder.add_op(AlgoOpKind::Put, send_bytes, offset, offset, -1,
                       send_peer, {}, pair_id, BufRef{BufSpace::Input, 0},
                       accum);
      }
    }

    if (recv_bytes > 0) {
      size_t offset = off(recv_owner);
      uint32_t pair_id = pair_base + static_cast<uint32_t>(recv_owner * 2);
      uint32_t recv_op = 0;
      uint32_t reduce_op = 0;
      if (!fuse_copy) {
        recv_op = builder.add_op(AlgoOpKind::Recv, recv_bytes, offset, offset,
                                 recv_peer, -1, {}, pair_id,
                                 BufRef{BufSpace::Input, 0}, accum);
        reduce_op =
            builder.add_op(AlgoOpKind::RecvReduce, recv_bytes, offset, offset,
                           recv_peer, -1, {recv_op}, pair_id,
                           BufRef{BufSpace::Input, 0}, accum);
      } else {
        // Fused reduce+copy: the task reduces accum[off] += Input[off],
        // copies accum[off] to the NEXT rank's accumulation buffer, and
        // device-writes the data-ready signal. The copy replaces the
        // step-(ring_step+1) send put; the last RS reduce (the held
        // shard, ring_step == nranks-2) does not copy.
        recv_op = builder.add_op(AlgoOpKind::Recv, recv_bytes, offset, offset,
                                 recv_peer, -1, {}, pair_id,
                                 BufRef{BufSpace::Input, 0}, accum);
        if (ring_step > 0)
          builder.algo.chunks[recv_op].wait_standalone_signal = true;
        reduce_op =
            builder.add_op(AlgoOpKind::RecvReduce, recv_bytes, offset, offset,
                           recv_peer, -1, {recv_op}, pair_id,
                           BufRef{BufSpace::Input, 0}, accum);
        if (ring_step < config.nranks - 2) {
          builder.algo.chunks[reduce_op].fuse_copy_to_peer = true;
          builder.algo.chunks[reduce_op].copy_dst = accum;
          builder.algo.chunks[reduce_op].copy_peer = send_peer;
        }
      }
      ready_ops[static_cast<size_t>(recv_owner)] = reduce_op;
    }
  }
}

// All-gather phase of ring allreduce: fully-reduced shards circulate
// rank -> rank+1 from their RS holder until every rank holds every
// shard. Received shards land at their Output offset. In-place, the
// shard this rank holds from the RS phase lives in Tmp(0): it is both
// sent from Tmp at step 0 and published into the output layout by a
// local copy (the peer's AG-phase Put never arrives before this rank
// finished reading the region as input — the holder's send causally
// follows the RS reduce that consumed it).
void emit_ring_allgather(RingTopology const& ring,
                         CollectiveConfig const& config,
                         AlgoBuilder& builder,
                         std::vector<uint32_t>& ready_ops, bool inplace,
                         size_t seg_bytes, size_t channel_base,
                         uint32_t pair_base) {
  size_t elem_bytes = scalar_type_size(config.dtype);
  auto off = [&](int owner) {
    return channel_base +
           balanced_shard_offset_bytes(seg_bytes, elem_bytes, config.nranks,
                                       owner);
  };
  if (inplace) {
    // Publish the RS-held shard Tmp -> Output[own_offset].
    int own = ring.rank_at_offset(config.rank, 1);
    size_t own_bytes = balanced_shard_size_bytes(
        seg_bytes, elem_bytes, config.nranks, own);
    if (own_bytes > 0) {
      size_t offset = off(own);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(own)]);
      builder.add_op(AlgoOpKind::Put, own_bytes, offset, offset, -1, -1,
                     std::move(deps), kNoPairId, BufRef{BufSpace::Tmp, 0},
                     BufRef{BufSpace::Output, 0});
    }
  }
  for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
    int send_owner = ring.rank_at_offset(config.rank, 1 - ring_step);
    int recv_owner = ring.rank_at_offset(config.rank, -ring_step);
    int send_peer = ring.next(config.rank);
    int recv_peer = ring.prev(config.rank);
    size_t send_bytes = balanced_shard_size_bytes(
        seg_bytes, elem_bytes, config.nranks, send_owner);
    size_t recv_bytes = balanced_shard_size_bytes(
        seg_bytes, elem_bytes, config.nranks, recv_owner);

    if (send_bytes > 0) {
      size_t offset = off(send_owner);
      uint32_t pair_id = pair_base + static_cast<uint32_t>(send_owner * 2 + 1);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(send_owner)]);
      // In-place: the RS-held shard (sent at step 0) lives in Tmp(0);
      // everything else was received into Output.
      uint32_t put_op =
          builder.add_op(AlgoOpKind::Put, send_bytes, offset, offset, -1,
                         send_peer, std::move(deps), pair_id,
                         (inplace && ring_step == 0) ? BufRef{BufSpace::Tmp, 0}
                                                     : BufRef{BufSpace::Output, 0},
                         BufRef{BufSpace::Output, 0});
      if (config.fuse_ag_copy)
        builder.algo.chunks[put_op].fuse_copy_flag = true;
    }

    if (recv_bytes > 0) {
      size_t offset = off(recv_owner);
      uint32_t pair_id = pair_base + static_cast<uint32_t>(recv_owner * 2 + 1);
      uint32_t recv_op = builder.add_op(AlgoOpKind::Recv, recv_bytes, offset,
                                        offset, recv_peer, -1, {}, pair_id,
                                        BufRef{BufSpace::Input, 0},
                                        BufRef{BufSpace::Output, 0});
      if (config.fuse_ag_copy)
        builder.algo.chunks[recv_op].wait_standalone_signal = true;
      ready_ops[static_cast<size_t>(recv_owner)] = recv_op;
    }
  }
}

CollAlgo build_allreduce_ring_algo(CollectiveConfig const& config,
                                   bool inplace) {
  RingTopology ring = config.ring_order.empty()
                          ? RingTopology(config.nranks)
                          : RingTopology(config.ring_order);
  CollAlgo algo = make_empty_algo(config);
  algo.kind = CollKind::AllReduceRing;
  // In-place: RS partials accumulate in a full-input-layout Tmp region
  // (same footprint the snapshot staging used, so scratch does not
  // grow).
  if (inplace) algo.tmp_bytes = {config.input_bytes};

  AlgoBuilder builder(std::move(algo));
  uint32_t const channels = std::max<uint32_t>(1u, config.channels);
  size_t const elem_bytes = scalar_type_size(config.dtype);
  // Split the tensor into `channels` contiguous byte intervals. Each
  // interval is an independent ring (own pair_id namespace, no cross-
  // channel dependencies), so the executor sees `channels` parallel
  // ready ops per phase. Require an even split at 16B granularity;
  // non-divisible tensors fall back to a single ring (collectives are
  // normally multiples of 16B and of the rank count).
  size_t const align = 16;
  size_t const base_seg = (config.input_bytes / channels) & ~(align - 1);
  if (base_seg == 0 ||
      base_seg * channels != config.input_bytes ||
      (base_seg % static_cast<size_t>(config.nranks)) != 0) {
    std::vector<uint32_t> ready_ops(static_cast<size_t>(config.nranks),
                                    kNoOp);
    emit_ring_reduce_scatter(ring, config, builder, ready_ops, inplace,
                             config.input_bytes, 0, 0);
    emit_ring_allgather(ring, config, builder, ready_ops, inplace,
                        config.input_bytes, 0, 0);
    return std::move(builder.algo);
  }
  uint32_t const pair_stride = static_cast<uint32_t>(2 * config.nranks);
  for (uint32_t c = 0; c < channels; ++c) {
    // Each channel runs the full RS/AG circulation over its own byte
    // interval with an isolated ready set and pair namespace.
    std::vector<uint32_t> ready_ops(static_cast<size_t>(config.nranks),
                                    kNoOp);
    size_t const base = static_cast<size_t>(c) * base_seg;
    uint32_t const pair_base = c * pair_stride;
    emit_ring_reduce_scatter(ring, config, builder, ready_ops, inplace,
                             base_seg, base, pair_base);
    emit_ring_allgather(ring, config, builder, ready_ops, inplace,
                        base_seg, base, pair_base);
  }
  return std::move(builder.algo);
}

// Standalone reduce-scatter (ring): the same circulation as the
// allreduce RS phase, but rotated by one shard — the allreduce phase
// leaves virtual shard v's full sum at rank wrap(v-1), while NCCL
// reduce-scatter semantics require physical shard k at rank k, so the
// owner indices shift by one (shard k first moves from rank wrap(k+1)
// and arrives back at rank k in the last step). Partial sums live in a
// full-input-layout Scratch: the peer's Put lands there, each
// RecvReduce accumulates this rank's Input contribution into it, and
// the running sum is forwarded from it. The loop's last RecvReduce
// (recv_owner == rank) completes the rank's own shard in Scratch; a
// final local copy moves it to Output[0].
CollAlgo build_reduce_scatter_ring_algo(CollectiveConfig const& config) {
  RingTopology ring = config.ring_order.empty()
                          ? RingTopology(config.nranks)
                          : RingTopology(config.ring_order);
  CollAlgo algo = make_empty_algo(config);
  algo.kind = CollKind::ReduceScatterRing;
  // Partial sums live in a declared Tmp region laid out like the full
  // input; the lowering maps it into executor scratch.
  algo.tmp_bytes = {config.input_bytes};
  size_t elem_bytes = scalar_type_size(config.dtype);

  AlgoBuilder builder(std::move(algo));
  std::vector<uint32_t> ready_ops(static_cast<size_t>(config.nranks), kNoOp);

  for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
    int send_owner = ring.rank_at_offset(config.rank, -ring_step - 1);
    int recv_owner = ring.rank_at_offset(config.rank, -ring_step - 2);
    int send_peer = ring.next(config.rank);
    int recv_peer = ring.prev(config.rank);
    size_t send_bytes = balanced_shard_size_bytes(
        config.input_bytes, elem_bytes, config.nranks, send_owner);
    size_t recv_bytes = balanced_shard_size_bytes(
        config.input_bytes, elem_bytes, config.nranks, recv_owner);

    if (send_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.input_bytes, elem_bytes, config.nranks, send_owner);
      uint32_t pair_id = static_cast<uint32_t>(send_owner * 2);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(send_owner)]);
      // First touch of a shard reads Input; afterwards the running sum
      // in Tmp(0). The Put always lands in the peer's Tmp (same full
      // layout there). Explicit by ring_step: at step s this rank sends
      // shard wrap(rank-s-1), which it received at step s-1 — only
      // s==0 (shard wrap(rank-1), never received before) reads Input.
      // (Equivalent to the previous deps.empty() test: deps are empty
      // exactly at step 0.)
      builder.add_op(AlgoOpKind::Put, send_bytes, offset, offset, -1, send_peer,
                     std::move(deps), pair_id,
                     ring_step == 0 ? BufRef{BufSpace::Input, 0}
                                    : BufRef{BufSpace::Tmp, 0},
                     BufRef{BufSpace::Tmp, 0});
    }

    if (recv_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.input_bytes, elem_bytes, config.nranks, recv_owner);
      uint32_t pair_id = static_cast<uint32_t>(recv_owner * 2);
      uint32_t recv_op = builder.add_op(AlgoOpKind::Recv, recv_bytes, offset,
                                        offset, recv_peer, -1, {}, pair_id,
                                        BufRef{BufSpace::Input, 0},
                                        BufRef{BufSpace::Tmp, 0});
      uint32_t reduce_op =
          builder.add_op(AlgoOpKind::RecvReduce, recv_bytes, offset, offset,
                         recv_peer, -1, {recv_op}, pair_id,
                         BufRef{BufSpace::Input, 0},
                         BufRef{BufSpace::Tmp, 0});
      ready_ops[static_cast<size_t>(recv_owner)] = reduce_op;
    }
  }

  // Own shard: complete in Scratch after the loop's last RecvReduce;
  // copy it to Output[0] (NCCL RS layout: only this rank's shard).
  size_t own_bytes = balanced_shard_size_bytes(
      config.input_bytes, elem_bytes, config.nranks, config.rank);
  if (own_bytes > 0) {
    size_t offset = balanced_shard_offset_bytes(
        config.input_bytes, elem_bytes, config.nranks, config.rank);
    std::vector<uint32_t> deps;
    add_dep(deps, ready_ops[static_cast<size_t>(config.rank)]);
    builder.add_op(AlgoOpKind::Put, own_bytes, offset, 0, -1, -1,
                   std::move(deps), kNoPairId, BufRef{BufSpace::Tmp, 0},
                   BufRef{BufSpace::Output, 0});
  }

  return std::move(builder.algo);
}

// Standalone all-gather (ring): shard k starts at rank k (its whole
// input) and circulates rank -> rank+1 until every rank holds it — the
// allreduce AG phase with owner indices shifted by one (there shard v
// starts at its RS holder wrap(v-1)). The rank's own shard is sent
// straight from Input[0] on the first step (no staging dependency);
// the local copy Input[0] -> Output[offset(rank)] only publishes the
// own shard into the output layout (NCCL AG semantics) and completes
// with the run. Shards received from the ring are forwarded from their
// Output offset, exactly like the allreduce AG phase (Put with deps
// reads Output via the lowering's phase-2 rule).
//
// In-place (NCCL form: sendbuff == recvbuff + rank*sendcount): the
// rank's own shard already sits at Output[offset(rank)] — skip the
// local publish copy and source the step-0 send from there. Received
// shards land in Output exactly as out-of-place; the own shard is read
// once (step 0) before any peer write can touch it, and the peer's
// step-0 Put is gated on this rank's step-0 send having consumed it via
// the ring order (send causally follows recv on the sender side), so no
// snapshot handshake is needed.
CollAlgo build_allgather_ring_algo(CollectiveConfig const& config,
                                   bool inplace) {
  RingTopology ring = config.ring_order.empty()
                          ? RingTopology(config.nranks)
                          : RingTopology(config.ring_order);
  CollAlgo algo = make_empty_algo(config);
  algo.kind = CollKind::AllGatherRing;
  size_t elem_bytes = scalar_type_size(config.dtype);

  AlgoBuilder builder(std::move(algo));
  std::vector<uint32_t> ready_ops(static_cast<size_t>(config.nranks), kNoOp);

  // Local copy of the own shard into the output layout. Independent of
  // the send path; only run completion orders it before the user read.
  // In-place: the shard is already at Output[offset(rank)] — no copy.
  size_t own_bytes = balanced_shard_size_bytes(
      config.output_bytes, elem_bytes, config.nranks, config.rank);
  if (!inplace && own_bytes > 0) {
    size_t offset = balanced_shard_offset_bytes(
        config.output_bytes, elem_bytes, config.nranks, config.rank);
    builder.add_op(AlgoOpKind::Put, own_bytes, 0, offset, -1, -1, {},
                   kNoPairId, BufRef{BufSpace::Input, 0},
                   BufRef{BufSpace::Output, 0});
  }

  for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
    int send_owner = ring.rank_at_offset(config.rank, -ring_step);
    int recv_owner = ring.rank_at_offset(config.rank, -ring_step - 1);
    int send_peer = ring.next(config.rank);
    int recv_peer = ring.prev(config.rank);
    size_t send_bytes = balanced_shard_size_bytes(
        config.output_bytes, elem_bytes, config.nranks, send_owner);
    size_t recv_bytes = balanced_shard_size_bytes(
        config.output_bytes, elem_bytes, config.nranks, recv_owner);

    if (send_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.output_bytes, elem_bytes, config.nranks, send_owner);
      uint32_t pair_id = static_cast<uint32_t>(send_owner * 2 + 1);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(send_owner)]);
      if (send_owner == config.rank) {
        // First step: the own shard is sent straight from Input[0]
        // (out-of-place) or from its Output position (in-place).
        builder.add_op(AlgoOpKind::Put, send_bytes,
                       inplace ? offset : 0, offset, -1, send_peer,
                       std::move(deps), pair_id,
                       inplace ? BufRef{BufSpace::Output, 0}
                               : BufRef{BufSpace::Input, 0},
                       BufRef{BufSpace::Output, 0});
      } else {
        // Forwarded shards were received into the Output layout.
        builder.add_op(AlgoOpKind::Put, send_bytes, offset, offset, -1,
                       send_peer, std::move(deps), pair_id,
                       BufRef{BufSpace::Output, 0},
                       BufRef{BufSpace::Output, 0});
      }
    }

    if (recv_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.output_bytes, elem_bytes, config.nranks, recv_owner);
      uint32_t pair_id = static_cast<uint32_t>(recv_owner * 2 + 1);
      uint32_t recv_op = builder.add_op(AlgoOpKind::Recv, recv_bytes, offset,
                                        offset, recv_peer, -1, {}, pair_id,
                                        BufRef{BufSpace::Input, 0},
                                        BufRef{BufSpace::Output, 0});
      ready_ops[static_cast<size_t>(recv_owner)] = recv_op;
    }
  }

  return std::move(builder.algo);
}

// Binary-tree allreduce over the binary-heap indexing (parent =
// (r-1)/2, children 2r+1 / 2r+2 when < nranks; root = 0).
//
// Up phase (reduce): leaves send their Input to the parent. An inner
// rank reduces each child's partial into its own contribution: the
// only/last child lands in Output and the final RecvReduce leaves the
// subtree sum there (src = Input for a single child, Tmp(0) holding
// the first child's partial plus this rank's Input for two children);
// the FIRST of two children lands in Tmp(0) instead, so the final
// result always ends in Output. The rank then Puts to its parent
// (src = Input for leaves, Output for inner ranks).
//
// Down phase (broadcast): the root already holds the result in
// Output; every rank Recv's it from its parent into Output[0] and
// forwards Output to its children.
//
// WAR safety without local deps across the up/down boundary: a peer's
// down-phase Put into my Output (or a child's into mine) can only be
// sent after the sender received the broadcast from ITS parent, which
// is causally after the root's full reduction — and that reduction
// causally includes my up-phase Put having been consumed. So no local
// dep is needed between the up-put and the down-recv, the same
// argument as ring AG forwarding.
//
// Out-of-place only: in-place would need a final full-buffer Tmp->Output
// copy on top of the Tmp-landing reduce path (now available);
// rejected in build_coll_algo for now.
//
// Tmp(0) is declared only on two-children ranks; scratch lives under a
// fixed reserved buf id, so rank-asymmetric declaration is safe.
CollAlgo build_allreduce_tree_algo(CollectiveConfig const& config) {
  int const n = config.nranks;
  int const r = config.rank;
  CollAlgo algo = make_empty_algo(config);
  algo.kind = CollKind::AllReduceTree;
  size_t const bytes = config.input_bytes;

  AlgoBuilder builder(std::move(algo));
  std::vector<int> children;
  if (2 * r + 1 < n) children.push_back(2 * r + 1);
  if (2 * r + 2 < n) children.push_back(2 * r + 2);
  int const parent = (r - 1) / 2;  // used only when r > 0

  // Tmp(0) buffers the first child's partial on two-children ranks.
  // Declared unconditionally on EVERY rank: an up-phase put targeting a
  // two-children parent's Tmp(0) is addressed with the SENDER's scratch
  // id (make_cmd's role_to_buf), which only matches the parent's
  // registration if both ranks minted the same scratch id — that
  // requires rank-symmetric tmp declaration (same shape sequence => same
  // first-seen minting order). Rank-asymmetric declaration sent
  // dst_buf=0 and hit the wait_mr "key not found" timeout class.
  builder.algo.tmp_bytes = {bytes};

  // ---- Up phase (reduce) ----
  uint32_t last_reduce = kNoOp;
  for (size_t i = 0; i < children.size(); ++i) {
    int const c = children[static_cast<size_t>(i)];
    bool const last = (i + 1 == children.size());
    uint32_t const pair_id = static_cast<uint32_t>((c * n + r) * 2);
    BufRef const land =
        last ? BufRef{BufSpace::Output, 0} : BufRef{BufSpace::Tmp, 0};
    uint32_t recv_op =
        builder.add_op(AlgoOpKind::Recv, bytes, 0, 0, c, -1, {}, pair_id,
                       BufRef{BufSpace::Input, 0}, land);
    std::vector<uint32_t> rdeps{recv_op};
    add_dep(rdeps, last_reduce);
    BufRef const rdst =
        last ? BufRef{BufSpace::Output, 0} : BufRef{BufSpace::Tmp, 0};
    BufRef const rsrc = last && children.size() > 1
                            ? BufRef{BufSpace::Tmp, 0}
                            : BufRef{BufSpace::Input, 0};
    last_reduce = builder.add_op(AlgoOpKind::RecvReduce, bytes, 0, 0, c, -1,
                                 std::move(rdeps), pair_id, rsrc, rdst);
  }
  if (r > 0) {
    // Forward the subtree sum: leaves send Input, inner ranks the
    // reduced Output. The first of two children lands in the parent's
    // Tmp(0), every other child in the parent's Output (mirrors the
    // parent's landing rule above).
    uint32_t const pair_id = static_cast<uint32_t>((r * n + parent) * 2);
    BufRef const psrc = children.empty() ? BufRef{BufSpace::Input, 0}
                                         : BufRef{BufSpace::Output, 0};
    bool const parent_two = (2 * parent + 2 < n);
    BufRef const pdst = (parent_two && r == 2 * parent + 1)
                            ? BufRef{BufSpace::Tmp, 0}
                            : BufRef{BufSpace::Output, 0};
    std::vector<uint32_t> deps;
    add_dep(deps, last_reduce);
    builder.add_op(AlgoOpKind::Put, bytes, 0, 0, -1, parent, std::move(deps),
                   pair_id, psrc, pdst);
  }

  // ---- Down phase (broadcast) ----
  uint32_t down_recv = kNoOp;
  if (r > 0) {
    uint32_t const pair_id = static_cast<uint32_t>((parent * n + r) * 2 + 1);
    down_recv =
        builder.add_op(AlgoOpKind::Recv, bytes, 0, 0, parent, -1, {},
                       pair_id, BufRef{BufSpace::Input, 0},
                       BufRef{BufSpace::Output, 0});
  }
  for (int c : children) {
    uint32_t const pair_id = static_cast<uint32_t>((r * n + c) * 2 + 1);
    std::vector<uint32_t> deps;
    add_dep(deps, r > 0 ? down_recv : last_reduce);
    builder.add_op(AlgoOpKind::Put, bytes, 0, 0, -1, c, std::move(deps),
                   pair_id, BufRef{BufSpace::Output, 0},
                   BufRef{BufSpace::Output, 0});
  }

  return std::move(builder.algo);
}

CollAlgo build_alltoall_pairwise_algo(CollectiveConfig const& config,
                                      bool inplace) {
  CollAlgo algo = make_empty_algo(config);
  algo.kind = CollKind::AllToAllPairwise;

  std::vector<size_t> input_splits =
      config.input_split_bytes.empty()
          ? equal_alltoall_splits(config.input_bytes, config.nranks)
          : config.input_split_bytes;
  std::vector<size_t> output_splits =
      config.output_split_bytes.empty()
          ? equal_alltoall_splits(config.output_bytes, config.nranks)
          : config.output_split_bytes;
  std::vector<size_t> input_prefix = prefix_bytes(input_splits);
  std::vector<size_t> output_prefix = prefix_bytes(output_splits);

  AlgoBuilder builder(std::move(algo));

  size_t self_slice_bytes = input_splits[static_cast<size_t>(config.rank)];
  if (self_slice_bytes != output_splits[static_cast<size_t>(config.rank)])
    throw std::invalid_argument(
        "alltoall self split size must match between input and output");

  // Out-of-place (sendbuff != recvbuff, the nccl-tests/native shape):
  // no aliasing, so sends go straight out of Input via IPC — no staging
  // copy, no SM worker involvement. The self-slice is an explicit local
  // Input->Output copy (native sends partition r to itself).
  //
  // In-place (sendbuff == recvbuff): partition p is both the source of
  // my Put to peer p AND the target of peer p's Put into my buffer, so
  // every send must be staged through scratch — an unstaged send reads
  // data the peer's incoming copy may already have overwritten
  // (observed at 2/4 ranks: the receiver got its own data back once the
  // peer's source was clobbered mid-copy). The lowering emits a local
  // Input->Scratch copy ahead of each staged Put.
  bool const stage = inplace;

  if (!inplace && self_slice_bytes > 0 && !config.external_self_slice) {
    // pairless Put chunk -> plain local copy, no signal.
    builder.add_op(AlgoOpKind::Put, self_slice_bytes,
                   input_prefix[static_cast<size_t>(config.rank)],
                   output_prefix[static_cast<size_t>(config.rank)], -1, -1,
                   {}, kNoPairId, BufRef{BufSpace::Input, 0},
                   BufRef{BufSpace::Output, 0});
  }

  // Rotate the per-peer send order (rank r sends to r+1, r+2, ...).
  // The naive ascending order makes every rank's first copy target the
  // same peer at the synchronized collective start — an incast into one
  // destination that overloads its CE/ingress arbitration (measured
  // 2-3x per-copy penalty at 4/8 ranks). The rotation spreads the first
  // wave across distinct destinations (Latin square).
  for (int k = 1; k < config.nranks; ++k) {
    int peer = config.a2a_rotate_order
                   ? (config.rank + k) % config.nranks
                   : (k <= config.rank ? k - 1 : k);

    size_t send_offset = input_prefix[static_cast<size_t>(peer)];
    size_t send_bytes = input_splits[static_cast<size_t>(peer)];
    size_t recv_offset = output_prefix[static_cast<size_t>(peer)];
    size_t recv_bytes = output_splits[static_cast<size_t>(peer)];
    size_t dst_off = output_prefix[static_cast<size_t>(config.rank)];

    // Canonical directed pair ids: both ranks derive the same value for
    // the same transfer, so the sender's Signal tag matches the
    // receiver's WaitSignal tag. (The previous per-rank next_pair_id++
    // sequence only agreed for adjacent ranks — non-adjacent pairs
    // never matched.)
    uint32_t send_pair =
        static_cast<uint32_t>(config.rank * config.nranks + peer);
    uint32_t recv_pair =
        static_cast<uint32_t>(peer * config.nranks + config.rank);

    if (send_bytes > 0) {
      if (config.a2a_hybrid && !stage && config.a2a_hybrid_ce_pct > 0 &&
          config.a2a_hybrid_ce_pct < 100) {
        // CE+device hybrid: ce_pct of the per-peer send via CE, the rest
        // via this rank's worker (device LD/ST to the peer), overlapping
        // engines. The send side previously ignored ce_pct (hardcoded
        // 50/50), so UK_CCL_A2A_HYBRID_CE_PCT=100 still created device
        // ops and forced a worker launch.
        // Round the CE half down to the copy kernel's 16B vector width:
        // an unaligned split (e.g. 30% of a 32MB partition = 10066328 B,
        // 8 mod 16) makes the device half start at a misaligned offset —
        // the vectorized LD/ST worker stalls with the fifo head advanced
        // but tail frozen. pct=50 worked because both halves are 16B
        // aligned.
        size_t const elem = 16;
        size_t ce_bytes =
            send_bytes * config.a2a_hybrid_ce_pct / 100 / elem * elem;
        size_t const dev_bytes = send_bytes - ce_bytes;
        if (ce_bytes > 0) {
          uint32_t put_ce =
              builder.add_op(AlgoOpKind::Put, ce_bytes, send_offset, dst_off,
                             -1, peer, {}, send_pair * 2,
                             BufRef{BufSpace::Input, 0},
                             BufRef{BufSpace::Output, 0});
          builder.algo.chunks[put_ce].put_path_hint = PutPath::Ipc;
        }
        if (dev_bytes > 0) {
          uint32_t put_dev =
              builder.add_op(AlgoOpKind::Put, dev_bytes,
                             send_offset + ce_bytes, dst_off + ce_bytes, -1,
                             peer, {}, send_pair * 2 + 1,
                             BufRef{BufSpace::Input, 0},
                             BufRef{BufSpace::Output, 0});
          builder.algo.chunks[put_dev].put_path_hint = PutPath::Device;
        }
      } else {
        uint32_t put_op =
            builder.add_op(AlgoOpKind::Put, send_bytes, send_offset, dst_off,
                           -1, peer, {}, send_pair, BufRef{BufSpace::Input, 0},
                           BufRef{BufSpace::Output, 0});
        builder.algo.chunks[put_op].stage_via_scratch = stage;
      }
    }
    if (recv_bytes > 0) {
      if (config.a2a_hybrid && !stage && config.a2a_hybrid_ce_pct > 0 &&
          config.a2a_hybrid_ce_pct < 100) {
        size_t const elem = 16;
        size_t ce_bytes =
            recv_bytes * config.a2a_hybrid_ce_pct / 100 / elem * elem;
        size_t const dev_bytes = recv_bytes - ce_bytes;
        if (ce_bytes > 0) {
          builder.add_op(AlgoOpKind::Recv, ce_bytes, recv_offset, recv_offset,
                         peer, -1, {}, recv_pair * 2,
                         BufRef{BufSpace::Input, 0},
                         BufRef{BufSpace::Output, 0});
        }
        if (dev_bytes > 0) {
          builder.add_op(AlgoOpKind::Recv, dev_bytes,
                         recv_offset + ce_bytes, recv_offset + ce_bytes, peer,
                         -1, {}, recv_pair * 2 + 1,
                         BufRef{BufSpace::Input, 0},
                         BufRef{BufSpace::Output, 0});
        }
      } else {
        builder.add_op(AlgoOpKind::Recv, recv_bytes, recv_offset, recv_offset,
                       peer, -1, {}, recv_pair, BufRef{BufSpace::Input, 0},
                       BufRef{BufSpace::Output, 0});
      }
    }
  }

  return std::move(builder.algo);
}

}  // namespace

CollAlgo build_coll_algo(CollectiveConfig const& config, bool inplace) {
  require_collective_config(config);
  switch (config.kind) {
    case CollKind::AllReduceRing:
      return build_allreduce_ring_algo(config, inplace);
    case CollKind::AllReduceTree:
      if (inplace)
        throw std::invalid_argument(
            "binary-tree allreduce: in-place not supported yet (a final "
            "full-buffer Tmp->Output copy would enable it; left for when "
            "needed)");
      return build_allreduce_tree_algo(config);
    case CollKind::AllToAllPairwise:
      return build_alltoall_pairwise_algo(config, inplace);
    case CollKind::AllGatherRing:
      return build_allgather_ring_algo(config, inplace);
    case CollKind::ReduceScatterRing:
      // In-place RS needs no algorithm change: the NCCL layout
      // (recvbuff = sendbuff + rank*recvBytes) makes the final
      // Tmp->Output[0] copy land on Input[offset(rank)] via the
      // allocation-scoped registration, and partial sums accumulate in
      // Tmp so peer puts never touch Input.
      return build_reduce_scatter_ring_algo(config);
  }
  throw std::invalid_argument("unsupported collective kind");
}

void verify_algo_pairing(CollectiveConfig const& config, bool inplace) {
  size_t elem_bytes = scalar_type_size(config.dtype);
  int const n = config.nranks;
  uint32_t const G = config.signal_group_tiles ? config.signal_group_tiles : 1;

  auto fail = [](std::string const& msg) {
    throw std::invalid_argument("verify_algo_pairing: " + msg);
  };

  // Build every rank's CollAlgo. RS/AG have rank-dependent sizes;
  // recompute them per rank from the same formulas the validators use.
  std::vector<CollAlgo> algos;
  algos.reserve(static_cast<size_t>(n));
  for (int r = 0; r < n; ++r) {
    CollectiveConfig rcfg = config;
    rcfg.rank = r;
    if (config.kind == CollKind::ReduceScatterRing)
      rcfg.output_bytes = balanced_shard_size_bytes(
          config.input_bytes, elem_bytes, n, r);
    if (config.kind == CollKind::AllGatherRing)
      rcfg.input_bytes = balanced_shard_size_bytes(config.output_bytes,
                                                   elem_bytes, n, r);
    algos.push_back(build_coll_algo(rcfg, inplace));
  }

  for (int r = 0; r < n; ++r) {
    auto const& chunks = algos[static_cast<size_t>(r)].chunks;
    // Local deps reference strictly earlier ops (hence acyclic) and
    // must have equal byte counts (hard lowering invariant).
    for (size_t i = 0; i < chunks.size(); ++i)
      for (uint32_t d : chunks[i].deps) {
        if (d >= i)
          fail("rank " + std::to_string(r) + " op " + std::to_string(i) +
               " depends on a later op");
        if (chunks[d].bytes != chunks[i].bytes)
          fail("rank " + std::to_string(r) + " op " + std::to_string(i) +
               " dep endpoints with unequal byte counts");
      }
    // Every RecvReduce pairs with a local Recv of the same pair_id.
    for (auto const& c : chunks) {
      if (c.op != AlgoOpKind::RecvReduce) continue;
      bool found = false;
      for (auto const& o : chunks)
        if (o.op == AlgoOpKind::Recv && o.pair_id == c.pair_id) found = true;
      if (!found)
        fail("rank " + std::to_string(r) + " RecvReduce pair " +
             std::to_string(c.pair_id) + " has no local Recv");
    }
  }

  // Cross-rank Put/Recv pairing and tag-space disjointness. Each
  // directed (src,dst) pair may use a pair id at most once, and tile
  // groups must stay clear of the reserved all-ones handshake tag. The
  // layout mirrors the lowering's adaptive group_bits, derived from the
  // rank-independent max-tensor bound.
  size_t const max_bytes = std::max(config.input_bytes, config.output_bytes);
  size_t const max_tiles =
      (max_bytes + config.tile_bytes - 1) / config.tile_bytes;
  size_t const max_groups = (max_tiles + G - 1) / G;
  uint32_t group_bits = 1;
  while ((1u << group_bits) < max_groups + 1) ++group_bits;
  uint32_t const all_ones = (1u << group_bits) - 1;
  std::map<std::tuple<int, int, uint32_t>, size_t> tag_groups;
  for (int r = 0; r < n; ++r) {
    for (auto const& c : algos[static_cast<size_t>(r)].chunks) {
      bool const is_put = (c.op == AlgoOpKind::Put);
      bool const is_sig = (c.op == AlgoOpKind::Signal);
      bool const fused_copy = c.fuse_copy_to_peer;
      if ((!is_put && !is_sig && !fused_copy) || c.pair_id == kNoPairId)
        continue;
      int const p = fused_copy ? c.copy_peer : c.dst_rank;
      if (p < 0) continue;
      auto key = std::make_tuple(r, p, c.pair_id);
      if (tag_groups.count(key))
        fail("duplicate pair id " + std::to_string(c.pair_id) +
             " on rank " + std::to_string(r) + " -> " + std::to_string(p));
      size_t tiles = (c.bytes + config.tile_bytes - 1) / config.tile_bytes;
      size_t groups = (tiles + G - 1) / G;
      if (groups > all_ones)
        fail("pair " + std::to_string(c.pair_id) +
             " tag groups collide with the reserved handshake tag");
      tag_groups[key] = groups;
      int matches = 0;
      for (auto const& o : algos[static_cast<size_t>(p)].chunks) {
        if (o.op != AlgoOpKind::Recv || o.src_rank != r ||
            o.pair_id != c.pair_id)
          continue;
        ++matches;
        if (o.bytes != c.bytes)
          fail("pair " + std::to_string(c.pair_id) +
               " " + (is_put ? "Put" : is_sig ? "Signal" : "FusedReduce") +
               "/Recv bytes mismatch");
        // A Put declares where its data lands (mirroring the Recv's
        // dst); a standalone Signal carries no data, so only bytes and
        // pairing are checked.
        if (is_put && (o.dst.space != c.dst.space ||
                       o.dst.index != c.dst.index || o.dst_off != c.dst_off))
          fail("pair " + std::to_string(c.pair_id) +
               " Put/Recv dst buffer or offset mismatch");
      }
      if (matches != 1)
        fail(std::string(is_put ? "Put" : is_sig ? "Signal"
                                                  : "FusedReduce") +
             " rank " +
             std::to_string(r) + " -> " + std::to_string(p) + " pair " +
             std::to_string(c.pair_id) + " has " +
             std::to_string(matches) + " matching Recv (want exactly 1)");
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
