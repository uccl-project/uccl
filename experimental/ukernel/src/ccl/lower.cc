#include "lower.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace UKernel {
namespace CCL {

namespace {

struct TileResult {
  std::vector<Op> ops;
  std::vector<size_t> tiles_per_chunk;
};

TileResult tile_chunks(CollAlgo const& algo, size_t tile_bytes,
                       std::vector<size_t>& first_tile) {
  TileResult r;
  size_t n = algo.chunks.size();
  first_tile.resize(n);
  r.tiles_per_chunk.resize(n);

  size_t total_tiles = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t nt = ceil_div(algo.chunks[i].bytes, tile_bytes);
    r.tiles_per_chunk[i] = nt;
    total_tiles += nt;
  }
  r.ops.reserve(total_tiles);

  for (size_t i = 0; i < n; ++i) {
    auto const& c = algo.chunks[i];
    first_tile[i] = r.ops.size();
    size_t num_tiles = r.tiles_per_chunk[i];
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
      r.ops.push_back(tile);
      off += tile_bytes;
    }
  }
  return r;
}

void propagate_deps(std::vector<Chunk> const& chunks,
                    std::vector<size_t> const& first_tile,
                    std::vector<size_t> const& tiles_per_chunk,
                    std::vector<Op>& ops) {
  size_t n = chunks.size();
  for (size_t i = 0; i < n; ++i) {
    auto const& c = chunks[i];
    size_t num_i = tiles_per_chunk[i];

    for (uint32_t dep_idx : c.deps) {
      assert(dep_idx < n);
      size_t num_d = tiles_per_chunk[dep_idx];
      size_t common = std::min(num_i, num_d);
      for (size_t t = 0; t < common; ++t)
        ops[first_tile[i] + t].deps.push_back(
            static_cast<uint32_t>(first_tile[dep_idx] + t));
    }
  }
}

uint64_t make_tag(uint32_t pair_id, size_t tile_idx) {
  return (static_cast<uint64_t>(pair_id) << 16) |
         static_cast<uint64_t>(tile_idx);
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
  return t;
}

std::vector<TiledOp> lower_to_tiled(std::vector<Op>&& ops,
                                    std::vector<Chunk> const& chunks,
                                    std::vector<size_t> const& first_tile,
                                    std::vector<size_t> const& tiles_per_chunk,
                                    bool inplace, bool stage_puts,
                                    uint32_t signal_group_tiles,
                                    size_t& staging_bytes) {
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
  struct NewDep { uint32_t target; uint32_t dep; };
  std::vector<NewDep> new_deps;

  size_t staging_bytes_out = 0;

  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    auto const& ch = chunks[ci];
    size_t num_tiles = tiles_per_chunk[ci];

    std::vector<uint32_t> cp_indices;
    std::vector<uint32_t> put_indices;
    std::vector<uint32_t> sig_group_puts;  // Put ops of the current signal group
    uint32_t cur_group_ws = kNoOp;         // WaitSignal of the current group
    size_t chunk_staging = 0;

    for (size_t t = 0; t < num_tiles; ++t) {
      size_t old_idx = first_tile[ci] + t;
      Op const& op = ops[old_idx];

      if (op.kind == AlgoOpKind::Put && ch.pair_id != kNoPairId) {
        if (stage_puts) {
          size_t staging_off = chunk_staging;
          chunk_staging += op.bytes;

          TiledOp cp;
          cp.kind = ExecOpKind::Put;
          cp.bytes = op.bytes;
          cp.src_off = op.src_off;
          cp.dst_off = staging_off;
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
          put.src_off = staging_off;
          put.dst_off = op.dst_off;
          put.src_peer = ~0u;
          put.dst_peer = op.dst_peer;
          put.src_buf_role = CollectiveBufferRole::Scratch;
          put.dst_buf_role = CollectiveBufferRole::Output;
          old_to_new[old_idx] = static_cast<uint32_t>(out.size());
          out.push_back(put);
          put_indices.push_back(
              static_cast<uint32_t>(out.size() - 1));
          // Put depends on its own Copy
          new_deps.push_back(
              {static_cast<uint32_t>(out.size() - 1), cp_indices.back()});

          // Signal aggregation: one Signal per group of G tiles, fired
          // when every Put in the group completed.
          sig_group_puts.push_back(put_indices.back());
          if (t % G == G - 1 || t + 1 == num_tiles) {
            TiledOp sig;
            sig.kind = ExecOpKind::Signal;
            sig.dst_peer = op.dst_peer;
            sig.tag = make_tag(ch.pair_id, t / G);
            uint32_t sig_idx = static_cast<uint32_t>(out.size());
            for (uint32_t pi : sig_group_puts)
              new_deps.push_back({sig_idx, pi});
            out.push_back(sig);
            sig_group_puts.clear();
          }
        } else {
        uint32_t put_idx = static_cast<uint32_t>(out.size());
        old_to_new[old_idx] = put_idx;
        TiledOp put = op_to_tiled(op);
        // Phase-2 Put: data was already reduced, source is Output buffer
        if (!op.deps.empty() && op.src_peer == ~0u) {
          put.src_buf_role = CollectiveBufferRole::Output;
        }
        out.push_back(put);

        // Signal aggregation: one Signal per group of G tiles, fired
        // when every Put in the group completed.
        sig_group_puts.push_back(put_idx);
        if (t % G == G - 1 || t + 1 == num_tiles) {
          TiledOp sig;
          sig.kind = ExecOpKind::Signal;
          sig.dst_peer = op.dst_peer;
          sig.tag = make_tag(ch.pair_id, t / G);
          uint32_t sig_idx = static_cast<uint32_t>(out.size());
          for (uint32_t pi : sig_group_puts)
            new_deps.push_back({sig_idx, pi});
          out.push_back(sig);
          sig_group_puts.clear();
        }
        }

      } else if (op.kind == AlgoOpKind::Recv) {
        // One WaitSignal per group of G tiles; ops consuming any tile in
        // the group depend on it via old_to_new remapping.
        if (t % G == 0) {
          TiledOp ws;
          ws.kind = ExecOpKind::WaitSignal;
          ws.src_peer = op.src_peer;
          ws.tag = make_tag(ch.pair_id, t / G);
          cur_group_ws = static_cast<uint32_t>(out.size());
          out.push_back(ws);
        }
        old_to_new[old_idx] = cur_group_ws;

      } else if (op.kind == AlgoOpKind::RecvReduce) {
        if (inplace) {
          size_t staging_off = staging_bytes;
          staging_bytes += op.bytes;

          TiledOp cp;
          cp.kind = ExecOpKind::Put;
          cp.bytes = op.bytes;
          cp.src_off = op.dst_off;
          cp.dst_off = staging_off;
          cp.src_peer = ~0u;
          cp.dst_peer = ~0u;
          cp.src_buf_role = CollectiveBufferRole::Output;
          cp.dst_buf_role = CollectiveBufferRole::Scratch;
          cp.deps = op.deps;
          out.push_back(cp);
          uint32_t cp_idx = static_cast<uint32_t>(out.size() - 1);

          TiledOp red;
          red.kind = ExecOpKind::Reduce;
          red.bytes = op.bytes;
          red.src_off = staging_off;
          red.dst_off = op.dst_off;
          red.src_peer = ~0u;
          red.dst_peer = ~0u;
          red.src_buf_role = CollectiveBufferRole::Scratch;
          red.dst_buf_role = CollectiveBufferRole::Output;
          new_deps.push_back({static_cast<uint32_t>(out.size()), cp_idx});
          old_to_new[old_idx] = static_cast<uint32_t>(out.size());
          out.push_back(red);
        } else {
          old_to_new[old_idx] = static_cast<uint32_t>(out.size());
          TiledOp red;
          red.kind = ExecOpKind::Reduce;
          red.bytes = op.bytes;
          red.src_off = op.src_off;
          red.dst_off = op.dst_off;
          red.src_peer = ~0u;
          red.dst_peer = ~0u;
          red.src_buf_role = CollectiveBufferRole::Input;
          red.dst_buf_role = CollectiveBufferRole::Output;
          red.deps = op.deps;
          out.push_back(red);
        }
      }
    }

    // Cross-rank coordination: Signal "copies done" after all local
    // Copies complete. All Puts wait for peer's "copies done" so that
    // the peer's input data has been staged before we overwrite it.
    if (stage_puts && !put_indices.empty()) {
      uint64_t barrier_tag = make_tag(ch.pair_id, 0xFFFF);
      int peer = ch.dst_rank;

      // Signal "copies_done" → peer (depends on all copies)
      TiledOp sig_cd;
      sig_cd.kind = ExecOpKind::Signal;
      sig_cd.dst_peer = static_cast<uint32_t>(peer);
      sig_cd.tag = barrier_tag;
      uint32_t sig_cd_idx = static_cast<uint32_t>(out.size());
      for (uint32_t ci : cp_indices)
        new_deps.push_back({sig_cd_idx, ci});
      out.push_back(sig_cd);

      // WaitSignal "copies_done" ← peer
      TiledOp ws_cd;
      ws_cd.kind = ExecOpKind::WaitSignal;
      ws_cd.src_peer = static_cast<uint32_t>(peer);
      ws_cd.tag = barrier_tag;
      uint32_t ws_cd_idx = static_cast<uint32_t>(out.size());
      out.push_back(ws_cd);

      // All Puts wait for peer's "copies_done"
      for (uint32_t pi : put_indices)
        new_deps.push_back({pi, ws_cd_idx});
    }

    if (chunk_staging > staging_bytes_out)
      staging_bytes_out = chunk_staging;
  }

  // Use the larger of max-chunk-staging (AllToAll) or
  // cumulative (AllReduce inplace RecvReduce).
  staging_bytes = staging_bytes_out > staging_bytes ? staging_bytes_out : staging_bytes;

  for (auto& o : out) {
    for (auto& dep : o.deps) {
      if (dep < n_old && old_to_new[dep] != kNoOp) dep = old_to_new[dep];
    }
  }

  for (auto& nd : new_deps) out[nd.target].deps.push_back(nd.dep);

  return out;
}

}  // namespace

TiledResult lower_algo(CollAlgo const& algo, size_t tile_bytes, bool inplace,
                       bool stage_puts, uint32_t signal_group_tiles) {
  if (tile_bytes == 0)
    throw std::invalid_argument("tile_bytes must be positive");

  TiledResult result;
  result.input_bytes = algo.input_bytes;
  result.output_bytes = algo.output_bytes;
  result.rank = algo.rank;
  result.nranks = algo.nranks;
  result.reduction = algo.reduction;
  if (algo.chunks.empty()) return result;

  std::vector<size_t> first_tile;
  auto tiled = tile_chunks(algo, tile_bytes, first_tile);
  propagate_deps(algo.chunks, first_tile, tiled.tiles_per_chunk, tiled.ops);

  size_t staging_bytes = 0;
  result.ops = lower_to_tiled(std::move(tiled.ops), algo.chunks, first_tile,
                              tiled.tiles_per_chunk, inplace, stage_puts,
                              signal_group_tiles, staging_bytes);
  result.staging_bytes_required = staging_bytes;
  return result;
}

TiledResult build_tiled(CollectiveConfig const& config, bool inplace) {
  if (config.kind == CollKind::AllToAllPairwise && !inplace)
    throw std::invalid_argument(
        "AllToAll requires inplace (input == output)");
  CollAlgo algo = build_coll_algo(config, inplace);
  // Only stage for variable-split AllToAll; equal-split offsets never overlap.
  bool stage_puts = (config.kind == CollKind::AllToAllPairwise && inplace &&
                     !config.input_split_bytes.empty());
  return lower_algo(algo, config.tile_bytes, inplace, stage_puts,
                    config.signal_group_tiles);
}

}  // namespace CCL
}  // namespace UKernel
