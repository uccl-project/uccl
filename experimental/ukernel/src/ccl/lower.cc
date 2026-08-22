#include "lower.h"
#include "utils.h"
#include "../transport/adapter/ipc_signal_ring.h"
#include "../../include/util/uk_debug.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace UKernel {
namespace CCL {

namespace {

// Cross-node fused reduce+copy via the software RDMA proxy.
static bool const kRdmaFusedProxy = []() {
  char const* v = std::getenv("UK_CCL_RDMA_FUSED_MODE");
  return v && std::string(v) == "proxy";
}();

struct TileResult {
  std::vector<Op> ops;
  std::vector<size_t> tiles_per_op;
};

TileResult tile_macro_ops(CollAlgo const& algo, size_t tile_bytes,
                       std::vector<size_t>& first_tile) {
  TileResult r;
  size_t n = algo.chunks.size();
  first_tile.resize(n);
  r.tiles_per_op.resize(n);

  size_t total_tiles = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t nt = ceil_div(algo.chunks[i].bytes, tile_bytes);
    r.tiles_per_op[i] = nt;
    total_tiles += nt;
  }
  r.ops.reserve(total_tiles);

  for (size_t i = 0; i < n; ++i) {
    auto const& c = algo.chunks[i];
    first_tile[i] = r.ops.size();
    size_t num_tiles = r.tiles_per_op[i];
    size_t off = 0;
    for (size_t t = 0; t < num_tiles; ++t) {
      Op tile;
      tile.kind = c.op;
      tile.bytes = std::min(tile_bytes, c.bytes - off);
      tile.src_off = c.src_off + off;
      tile.dst_off = c.dst_off + off;
      tile.src_peer =
          (c.src_rank < 0) ? ~0u : static_cast<uint32_t>(c.src_rank);
      tile.dst_peer =
          (c.dst_rank < 0) ? ~0u : static_cast<uint32_t>(c.dst_rank);
      tile.put_path_hint = c.put_path_hint;
      r.ops.push_back(tile);
      off += tile_bytes;
    }
  }
  return r;
}

void propagate_deps(std::vector<MacroOp> const& chunks,
                    std::vector<size_t> const& first_tile,
                    std::vector<size_t> const& tiles_per_op,
                    std::vector<Op>& ops) {
  size_t n = chunks.size();
  for (size_t i = 0; i < n; ++i) {
    auto const& c = chunks[i];
    size_t num_i = tiles_per_op[i];

    for (uint32_t dep_idx : c.deps) {
      assert(dep_idx < n);
      // Hard invariant: dep endpoints must have equal byte counts
      // (tile-by-tile propagation is undefined for unequal lengths).
      if (chunks[dep_idx].bytes != c.bytes)
        throw std::invalid_argument(
            "lower: dep endpoints with unequal byte counts are not "
            "supported");
      size_t num_d = tiles_per_op[dep_idx];
      size_t common = std::min(num_i, num_d);
      for (size_t t = 0; t < common; ++t)
        ops[first_tile[i] + t].deps.push_back(
            static_cast<uint32_t>(first_tile[dep_idx] + t));
    }
  }
}

// Signal tag layout: pair_id in the high bits, tile-group index in the
// low `group_bits` bits. group_bits is plan-adaptive (see lower_algo):
// sized from the largest possible chunk so every group fits while the
// all-ones group value stays reserved for the copies-done handshake.
// Derived from a rank-independent bound (max tensor bytes), so all
// ranks compute the SAME layout — per-rank maxima would diverge for
// non-uniform shards and break cross-rank tag matching.
uint64_t make_tag(uint32_t pair_id, uint32_t group, uint32_t group_bits) {
  return (static_cast<uint64_t>(pair_id) << group_bits) |
         static_cast<uint64_t>(group);
}

// Number of group values the layout must hold for this plan: groups of
// the largest possible chunk (bounded by the bigger tensor), plus one
// reserved all-ones slot.
uint32_t tag_group_bits(CollAlgo const& algo, size_t tile_bytes, uint32_t G) {
  size_t max_bytes = algo.input_bytes > algo.output_bytes
                         ? algo.input_bytes
                         : algo.output_bytes;
  size_t max_tiles = (max_bytes + tile_bytes - 1) / tile_bytes;
  size_t max_groups = (max_tiles + G - 1) / G;
  uint32_t bits = 1;
  while ((1u << bits) < max_groups + 1) ++bits;
  return bits;
}

TiledOp op_to_tiled(Op const& op) {
  TiledOp t;
  t.kind = ExecOpKind::Put;
  t.bytes = op.bytes;
  t.src_off = op.src_off;
  t.dst_off = op.dst_off;
  t.src_peer = op.src_peer;
  t.dst_peer = op.dst_peer;
  t.src_buf_role = CollectiveBufferRole::Input;
  t.dst_buf_role = CollectiveBufferRole::Output;
  t.deps = std::move(op.deps);
  t.put_path_hint = op.put_path_hint;
  return t;
}

// Fusion metadata sinks (filled for signal groups whose tag fits the
// 32-bit RDMA immediate).
struct FusionMetaOut {
  std::vector<std::pair<uint32_t, uint32_t>>* put_signal;
  std::vector<std::pair<uint32_t, uint32_t>>* sig_groups;
  std::vector<std::pair<uint32_t, uint32_t>>* wait_groups;
};

// Executor-flat view of a builder BufRef: Input/Output map 1:1 onto the
// executor's buffer roles; Tmp(i) maps onto the Scratch role with the
// tmp region's base offset folded in.
struct ResolvedBuf {
  CollectiveBufferRole role;
  size_t base_off;
};
static ResolvedBuf resolve_buf(BufRef const& b,
                               std::vector<size_t> const& tmp_base) {
  switch (b.space) {
    case BufSpace::Input:
      return {CollectiveBufferRole::Input, 0};
    case BufSpace::Output:
      return {CollectiveBufferRole::Output, 0};
    case BufSpace::Tmp:
      return {CollectiveBufferRole::Scratch, tmp_base.at(b.index)};
  }
  return {CollectiveBufferRole::Input, 0};
}

std::vector<TiledOp> lower_to_tiled(std::vector<Op>&& ops,
                                    std::vector<MacroOp> const& chunks,
                                    std::vector<size_t> const& first_tile,
                                    std::vector<size_t> const& tiles_per_op,
                                    std::vector<size_t> const& tmp_base,
                                    uint32_t group_bits,
                                    uint32_t signal_group_tiles,
                                    int rank, int nranks,
                                    size_t& staging_bytes, FusionMetaOut meta,
                                    bool device_flags, size_t max_tiles) {
  size_t n_old = ops.size();
  std::vector<TiledOp> out;
  out.reserve(n_old * 2);
  std::vector<uint32_t> old_to_new(n_old, kNoOp);
  // Signal aggregation factor: one Signal/WaitSignal pair per G tiles of
  // a chunk pair. G=1 reproduces the per-tile scheme exactly.
  uint32_t const G = signal_group_tiles ? signal_group_tiles : 1;

  // New deps created during lowering (e.g. Signal→Put, Reduce→staging).
  // Merged into the target op's deps after remapping so they don't collide
  // with old-to-new index mapping.
  struct NewDep {
    uint32_t target;
    uint32_t dep;
  };
  std::vector<NewDep> new_deps;

  size_t staging_bytes_out = 0;
  // Each staged chunk needs its own scratch region: the copy and the Put
  // of different chunks overlap in time, so reusing one region would let
  // a later chunk's staging clobber an earlier chunk's data before its
  // Put read it (4-rank alltoall sent wrong partitions). The base
  // advances by the previous chunk's staging size.
  size_t staging_base = 0;

  // Signal aggregation: one Signal per group of G tiles, fired once
  // every Put of the group has completed. Shared by the staged and
  // direct Put paths below.
  auto emit_group_signal = [&](std::vector<uint32_t>& group_puts,
                               MacroOp const& ch, uint32_t dst_peer, size_t t,
                               size_t num_tiles) {
    if (t % G != G - 1 && t + 1 != num_tiles) return;
    TiledOp sig;
    sig.kind = ExecOpKind::Signal;
    sig.dst_peer = dst_peer;
    sig.tag = make_tag(ch.pair_id, static_cast<uint32_t>(t / G), group_bits);
    uint32_t sig_idx = static_cast<uint32_t>(out.size());
    for (uint32_t pi : group_puts) new_deps.push_back({sig_idx, pi});
    out.push_back(sig);
    // Imm-sized tag: every Put of the group may carry the tag
    // (receiver counts arrivals); record the whole group.
    if (sig.tag <= 0xFFFFFFFFu) {
      for (uint32_t pi : group_puts) meta.put_signal->emplace_back(sig_idx, pi);
      meta.sig_groups->emplace_back(sig_idx,
                                    static_cast<uint32_t>(group_puts.size()));
    }
    group_puts.clear();
  };

  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    auto const& ch = chunks[ci];
    size_t num_tiles = tiles_per_op[ci];

    std::vector<uint32_t> cp_indices;
    std::vector<uint32_t> put_indices;
    std::vector<uint32_t>
        sig_group_puts;             // Put ops of the current signal group
    uint32_t cur_group_ws = kNoOp;  // WaitSignal of the current group
    size_t chunk_staging = 0;

    for (size_t t = 0; t < num_tiles; ++t) {
      size_t old_idx = first_tile[ci] + t;
      Op const& op = ops[old_idx];

      if (op.kind == AlgoOpKind::Put && ch.pair_id != kNoPairId) {
        if (ch.stage_via_scratch) {
          size_t staging_off = chunk_staging;
          chunk_staging += op.bytes;

          TiledOp cp;
          cp.kind = ExecOpKind::Put;
          cp.bytes = op.bytes;
          cp.src_off = op.src_off;
          cp.dst_off = staging_base + staging_off;
          cp.src_peer = ~0u;
          cp.dst_peer = ~0u;
          cp.src_buf_role = CollectiveBufferRole::Input;
          cp.dst_buf_role = CollectiveBufferRole::Scratch;
          cp.deps = std::move(op.deps);
          uint32_t cp_idx = static_cast<uint32_t>(out.size());
          cp_indices.push_back(cp_idx);
          out.push_back(cp);

          TiledOp put;
          put.kind = ExecOpKind::Put;
          put.bytes = op.bytes;
          put.src_off = staging_base + staging_off;
          put.dst_off = op.dst_off;
          put.src_peer = ~0u;
          put.dst_peer = op.dst_peer;
          put.src_buf_role = CollectiveBufferRole::Scratch;
          put.dst_buf_role = CollectiveBufferRole::Output;
          put.put_path_hint = op.put_path_hint;
          old_to_new[old_idx] = static_cast<uint32_t>(out.size());
          out.push_back(put);
          put_indices.push_back(static_cast<uint32_t>(out.size() - 1));
          // Put depends on its own Copy
          new_deps.push_back(
              {static_cast<uint32_t>(out.size() - 1), cp_indices.back()});

          sig_group_puts.push_back(put_indices.back());
          emit_group_signal(sig_group_puts, ch, op.dst_peer, t, num_tiles);
        } else {
          uint32_t put_idx = static_cast<uint32_t>(out.size());
          old_to_new[old_idx] = put_idx;
          TiledOp put = op_to_tiled(op);
          // Buffer refs are always authoritative from the builder;
          // Tmp(i) lands in scratch at the region's base offset.
          auto put_src = resolve_buf(ch.src, tmp_base);
          auto put_dst = resolve_buf(ch.dst, tmp_base);
          put.src_buf_role = put_src.role;
          put.dst_buf_role = put_dst.role;
          put.src_off += put_src.base_off;
          put.dst_off += put_dst.base_off;
          bool const fused_flag = ch.fuse_copy_flag && device_flags;
          if (ch.fuse_copy_flag && device_flags) {
            // Fused AG copy: the device task copies to the peer and
            // writes the completion flag itself — no host signal op.
            put.flag_slot = static_cast<uint32_t>(
                static_cast<uint64_t>(ch.pair_id) * max_tiles +
                static_cast<uint64_t>(t));
            put.tag = put.flag_slot;
          }
          out.push_back(put);
          put_indices.push_back(put_idx);
          if (!fused_flag) {
            sig_group_puts.push_back(put_idx);
            emit_group_signal(sig_group_puts, ch, op.dst_peer, t, num_tiles);
          }
        }

      } else if (op.kind == AlgoOpKind::Recv) {
        // One WaitSignal per group of G tiles; ops consuming any tile in
        // the group depend on it via old_to_new remapping.
        if (t % G == 0) {
          TiledOp ws;
          ws.kind = ExecOpKind::WaitSignal;
          ws.src_peer = op.src_peer;
          if (ch.wait_standalone_signal && device_flags) {
            // Fused-task signals (fuse_reduce_copy): each tile's task
            // writes its own slot with its own tag, so the group wait
            // polls flag_count consecutive slots and completes when ALL
            // match. Slots are collision-free within a run: pair*K+tile
            // with K = the plan's tile-count bound.
            ws.flag_slot = static_cast<uint32_t>(
                static_cast<uint64_t>(ch.pair_id) * max_tiles +
                static_cast<uint64_t>(t));
            ws.flag_count = static_cast<uint32_t>(
                std::min<size_t>(G, num_tiles - t));
            // Flag tag == slot index (salted by the executor): slots are
            // consecutive within a group, so the wait matches
            // slot[i] == tag+i without any tag-layout arithmetic.
            ws.tag = ws.flag_slot;
          } else {
            ws.tag = make_tag(ch.pair_id, static_cast<uint32_t>(t / G),
                              group_bits);
          }
          cur_group_ws = static_cast<uint32_t>(out.size());
          out.push_back(ws);
          // When the sender fuses this group, the wait must count one
          // arrival per tile instead of a single signal. A fused-RS Recv
          // pairs with a standalone Signal (no put), so its wait stays a
          // plain one-arrival wait.
          if (ws.tag <= 0xFFFFFFFFu && !ch.wait_standalone_signal) {
            uint32_t grp =
                static_cast<uint32_t>(std::min<size_t>(G, num_tiles - t));
            meta.wait_groups->emplace_back(cur_group_ws, grp);
          }
        }
        old_to_new[old_idx] = cur_group_ws;

      } else if (op.kind == AlgoOpKind::RecvReduce) {
        TiledOp red;
        red.kind = ExecOpKind::Reduce;
        red.bytes = op.bytes;
        red.src_off = op.src_off;
        red.dst_off = op.dst_off;
        red.dst_peer = ~0u;
        red.src_peer = ~0u;
        auto red_src = resolve_buf(ch.src, tmp_base);
        red.src_buf_role = red_src.role;
        red.src_off += red_src.base_off;
        auto red_dst = resolve_buf(ch.dst, tmp_base);
        red.dst_buf_role = red_dst.role;
        red.dst_off += red_dst.base_off;
        uint32_t red_idx = static_cast<uint32_t>(out.size());
        uint32_t put_idx = red_idx;
        if (ch.fuse_copy_to_peer) {
          auto red_copy = resolve_buf(ch.copy_dst, tmp_base);
          red.copy_dst_peer = static_cast<uint32_t>(ch.copy_peer);
          red.copy_dst_buf_role = red_copy.role;
          red.copy_dst_off = op.dst_off + red_copy.base_off;
          if (kRdmaFusedProxy) {
            // Cross-node RDMA proxy: reduce to local dst only; the
            // synthetic Put below is posted by the proxy after the device
            // notifies through the D2H ring.
            red.rdma_fused_proxy = true;
          } else {
            // Same-host fused reduce+copy: after the reduce, copy dst to
            // the peer's accumulation buffer. The data-ready signal is a
            // device-completion flag slot (when device_flags) written by
            // the task itself, or a separate host-written Signal op.
            red.fused_copy = true;
            if (device_flags) {
              // Per-tile slot + tag: tile t writes slot pair*K+t with the
              // slot index as the tag (salted). The receiver's group wait
              // polls consecutive slots and matches base_tag+i.
              red.flag_slot = static_cast<uint32_t>(
                  static_cast<uint64_t>(ch.pair_id) * max_tiles +
                  static_cast<uint64_t>(t));
              red.tag = red.flag_slot;
            }
          }
        }
        red.deps = op.deps;
        out.push_back(red);
        if (ch.fuse_copy_to_peer && kRdmaFusedProxy) {
          // Synthetic RDMA Put: depends on the Reduce; all downstream
          // ops wait for this Put instead of the Reduce.
          TiledOp put;
          put.kind = ExecOpKind::Put;
          put.bytes = red.bytes;
          put.src_off = red.dst_off;
          put.dst_off = red.copy_dst_off;
          put.dst_peer = red.copy_dst_peer;
          put.src_buf_role = red.dst_buf_role;
          put.dst_buf_role = red.copy_dst_buf_role;
          put.put_path_hint = PutPath::Rdma;
          put.rdma_fused_proxy = true;
          put.deps.push_back(red_idx);
          put_idx = static_cast<uint32_t>(out.size());
          out.push_back(put);
          red.fused_proxy_put_idx = static_cast<int32_t>(put_idx);
          old_to_new[old_idx] = put_idx;
        } else {
          old_to_new[old_idx] = red_idx;
        }
        if (ch.fuse_copy_to_peer && !device_flags) {
          TiledOp sig;
          sig.kind = ExecOpKind::Signal;
          sig.dst_peer = static_cast<uint32_t>(ch.copy_peer);
          sig.tag = make_tag(ch.pair_id, static_cast<uint32_t>(t / G),
                             group_bits);
          sig.deps.push_back(put_idx);  // remapped target (red or put)
          out.push_back(sig);
        }

      } else if (op.kind == AlgoOpKind::Signal) {
        // Fused-RS data-ready signal: one standalone Signal per G-tile
        // group, mirroring the receiver's WaitSignal group structure.
        // The group fires when its producing reduce tiles complete
        // (deps were propagated tile-by-tile from the producing chunk).
        if (t % G == 0) {
          TiledOp sig;
          sig.kind = ExecOpKind::Signal;
          sig.dst_peer = op.dst_peer;
          sig.tag = make_tag(ch.pair_id, static_cast<uint32_t>(t / G),
                             group_bits);
          size_t const gt = std::min<size_t>(G, num_tiles - t);
          for (size_t k = 0; k < gt; ++k) {
            Op const& gtile = ops[first_tile[ci] + t + k];
            for (uint32_t d : gtile.deps) sig.deps.push_back(d);
          }
          old_to_new[old_idx] = static_cast<uint32_t>(out.size());
          out.push_back(sig);
        }

      } else if (op.kind == AlgoOpKind::Put) {
        // Local copy (pairless Put chunk): plain Put tiles with no
        // signal — emitted by the AllGather/ReduceScatter builders for
        // their Input->Output / Scratch->Output copies.
        old_to_new[old_idx] = static_cast<uint32_t>(out.size());
        TiledOp put = op_to_tiled(op);
        auto cp_src = resolve_buf(ch.src, tmp_base);
        auto cp_dst = resolve_buf(ch.dst, tmp_base);
        put.src_buf_role = cp_src.role;
        put.dst_buf_role = cp_dst.role;
        put.src_off += cp_src.base_off;
        put.dst_off += cp_dst.base_off;
        out.push_back(put);
      }
    }

    // Cross-rank coordination: Signal "copies done" after all local
    // Copies complete. All Puts wait for peer's "copies done" so that
    // the peer's input data has been staged before we overwrite it.
    if (ch.stage_via_scratch && !put_indices.empty()) {
      // The all-ones group value is reserved for this handshake.
      uint32_t const kAllOnesGroup = (1u << group_bits) - 1;
      int peer = ch.dst_rank;

      // Signal "copies_done" → peer (depends on all copies). The tag
      // uses THIS chunk's pair: the peer's staged-put gating wait keys
      // on the pair of the data it overwrites in MY buffer, which is
      // exactly this send chunk's pair.
      uint64_t sig_tag = make_tag(ch.pair_id, kAllOnesGroup, group_bits);
      TiledOp sig_cd;
      sig_cd.kind = ExecOpKind::Signal;
      sig_cd.dst_peer = static_cast<uint32_t>(peer);
      sig_cd.tag = sig_tag;
      uint32_t sig_cd_idx = static_cast<uint32_t>(out.size());
      for (uint32_t ci : cp_indices) new_deps.push_back({sig_cd_idx, ci});
      out.push_back(sig_cd);

      // WaitSignal "copies_done" ← peer. My put writes into the peer's
      // partition `rank`; that partition is the peer's send source for
      // ITS chunk pair (peer*nranks+rank), so this wait must key on
      // that pair, not on ch.pair_id — otherwise the tags never match
      // and both sides wait on each other's never-sent signal
      // (2-rank alltoall deadlock seen once staging was enabled).
      uint32_t peer_send_pair =
          static_cast<uint32_t>(peer * nranks + rank);
      uint64_t wait_tag =
          make_tag(peer_send_pair, kAllOnesGroup, group_bits);
      TiledOp ws_cd;
      ws_cd.kind = ExecOpKind::WaitSignal;
      ws_cd.src_peer = static_cast<uint32_t>(peer);
      ws_cd.tag = wait_tag;
      uint32_t ws_cd_idx = static_cast<uint32_t>(out.size());
      out.push_back(ws_cd);

      // All Puts wait for peer's "copies_done"
      for (uint32_t pi : put_indices) new_deps.push_back({pi, ws_cd_idx});
    }

    staging_base += chunk_staging;
    if (staging_base > staging_bytes_out) staging_bytes_out = staging_base;
  }

  // Total staging footprint across all chunks (each chunk owns a
  // disjoint region; they are live concurrently).
  staging_bytes =
      staging_bytes_out > staging_bytes ? staging_bytes_out : staging_bytes;

  for (auto& o : out) {
    for (auto& dep : o.deps) {
      if (dep < n_old && old_to_new[dep] != kNoOp) dep = old_to_new[dep];
    }
  }

  for (auto& nd : new_deps) out[nd.target].deps.push_back(nd.dep);

  return out;
}

}  // namespace

TiledResult lower_algo(CollAlgo const& algo, size_t tile_bytes,
                       uint32_t signal_group_tiles, bool device_flags) {
  if (tile_bytes == 0)
    throw std::invalid_argument("tile_bytes must be positive");

  TiledResult result;
  result.input_bytes = algo.input_bytes;
  result.output_bytes = algo.output_bytes;
  result.rank = algo.rank;
  result.nranks = algo.nranks;
  result.reduction = algo.reduction;
  result.dtype = algo.dtype;
  if (algo.chunks.empty()) return result;

  std::vector<size_t> first_tile;
  auto tiled = tile_macro_ops(algo, tile_bytes, first_tile);
  propagate_deps(algo.chunks, first_tile, tiled.tiles_per_op, tiled.ops);

  // Device-flag slot bound: slot = pair*max_tiles + tile is
  // collision-free within a run when max_tiles >= every chunk's tile
  // count. The per-peer flag area is fixed-size, so plans that need more
  // slots fall back to host-written signals (both sides derive the same
  // bound, so the fallback is consistent).
  size_t const max_bytes =
      algo.input_bytes > algo.output_bytes ? algo.input_bytes
                                           : algo.output_bytes;
  size_t const max_tiles = (max_bytes + tile_bytes - 1) / tile_bytes;
  uint32_t max_pair = 0;
  for (auto const& c : algo.chunks)
    if (c.pair_id != kNoPairId && c.pair_id > max_pair) max_pair = c.pair_id;
  bool const flags_ok =
      (static_cast<uint64_t>(max_pair) + 1) * max_tiles <=
      Transport::kDeviceFlagSlots;
  if (device_flags && !flags_ok)
    UK_DBG(UK_DBG_LVL_EXEC,
           "[lower] device-flag slots exceed %zu (pairs=%u tiles=%zu) — "
           "falling back to host signals",
           Transport::kDeviceFlagSlots, max_pair, max_tiles);
  if (!flags_ok) device_flags = false;

  // Adaptive tag layout (see make_tag): the group field width is the
  // same on every rank because it derives from the rank-independent
  // max-tensor bound.
  uint32_t const G = signal_group_tiles ? signal_group_tiles : 1;
  uint32_t const group_bits = tag_group_bits(algo, tile_bytes, G);
  result.tag_group_bits = group_bits;
  {
    // Hard layout invariants: tile groups of the largest possible chunk
    // must fit below the reserved all-ones value, and the pair id must
    // fit in the remaining high bits of the 32-bit (fusion) tag.
    uint32_t const all_ones = (1u << group_bits) - 1;
    size_t max_groups = 0;
    for (auto const& c : algo.chunks) {
      if (c.pair_id == kNoPairId) continue;
      size_t tiles = (c.bytes + tile_bytes - 1) / tile_bytes;
      size_t groups = (tiles + G - 1) / G;
      if (groups > max_groups) max_groups = groups;
      if (c.pair_id > max_pair) max_pair = c.pair_id;
    }
    if (max_groups > all_ones)
      throw std::invalid_argument(
          "lower: chunk has too many tile groups for the tag layout");
    if (group_bits < 32 && max_pair >= (1u << (32 - group_bits)))
      throw std::invalid_argument(
          "lower: pair_id does not fit the adaptive tag layout");
  }

  // Declared Tmp regions are laid out back-to-back inside executor
  // scratch; their total sizes the scratch floor alongside
  // lowering-internal staging (alltoall).
  std::vector<size_t> tmp_base(algo.tmp_bytes.size(), 0);
  size_t tmp_total = 0;
  for (size_t i = 0; i < algo.tmp_bytes.size(); ++i) {
    tmp_base[i] = tmp_total;
    tmp_total += algo.tmp_bytes[i];
  }

  size_t staging_bytes = 0;
  FusionMetaOut meta{&result.fused_put_signal, &result.sig_group_size,
                     &result.wait_group_size};
  result.ops =
      lower_to_tiled(std::move(tiled.ops), algo.chunks, first_tile,
                     tiled.tiles_per_op, tmp_base, group_bits,
                     signal_group_tiles, algo.rank, algo.nranks,
                     staging_bytes, meta, device_flags, max_tiles);
  result.staging_bytes_required =
      staging_bytes > tmp_total ? staging_bytes : tmp_total;
  return result;
}

TiledResult build_tiled(CollectiveConfig const& config, bool inplace) {
  CollAlgo algo = build_coll_algo(config, inplace);
  return lower_algo(algo, config.tile_bytes, config.signal_group_tiles,
                    config.device_flags);
}

}  // namespace CCL
}  // namespace UKernel
