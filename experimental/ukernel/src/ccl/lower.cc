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
  assert(op.kind == AlgoOpKind::Put);  // only Put reaches this path
  t.bytes = op.bytes;
  t.src_off = op.src_off;
  t.dst_off = op.dst_off;
  t.src_peer = op.src_peer;
  t.dst_peer = op.dst_peer;
  t.deps = std::move(op.deps);
  return t;
}

std::vector<TiledOp> lower_to_tiled(std::vector<Op>&& ops,
                                    std::vector<Chunk> const& chunks,
                                    std::vector<size_t> const& first_tile,
                                    std::vector<size_t> const& tiles_per_chunk,
                                    bool inplace, size_t& staging_bytes) {
  size_t n_old = ops.size();
  std::vector<TiledOp> out;
  out.reserve(n_old * 2);
  std::vector<uint32_t> old_to_new(n_old, kNoOp);

  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    auto const& ch = chunks[ci];
    size_t num_tiles = tiles_per_chunk[ci];

    for (size_t t = 0; t < num_tiles; ++t) {
      size_t old_idx = first_tile[ci] + t;
      Op const& op = ops[old_idx];

      if (op.kind == AlgoOpKind::Put && ch.pair_id != kNoPairId) {
        old_to_new[old_idx] = static_cast<uint32_t>(out.size());
        out.push_back(op_to_tiled(op));

        TiledOp sig;
        sig.kind = ExecOpKind::Signal;
        sig.dst_peer = op.dst_peer;
        sig.tag = make_tag(ch.pair_id, t);
        sig.deps = {old_to_new[old_idx]};
        out.push_back(sig);

      } else if (op.kind == AlgoOpKind::Recv) {
        TiledOp ws;
        ws.kind = ExecOpKind::WaitSignal;
        ws.src_peer = op.src_peer;
        ws.tag = make_tag(ch.pair_id, t);
        old_to_new[old_idx] = static_cast<uint32_t>(out.size());
        out.push_back(ws);

      } else if (op.kind == AlgoOpKind::RecvReduce) {
        if (inplace) {
          // Copy received data from output buf to staging before reducing
          size_t staging_off = staging_bytes;
          staging_bytes += op.bytes;

          TiledOp cp;
          cp.kind = ExecOpKind::Put;
          cp.bytes = op.bytes;
          cp.src_off = op.dst_off;
          cp.dst_off = staging_off;
          cp.src_peer = ~0u;
          cp.dst_peer = ~0u;
          cp.deps = op.deps;
          out.push_back(cp);
          uint32_t cp_idx = static_cast<uint32_t>(out.size() - 1);

          TiledOp red;
          red.kind = ExecOpKind::Reduce;
          red.bytes = op.bytes;
          red.src_off = staging_off;
          red.dst_off = op.dst_off;
          red.deps = {cp_idx};
          old_to_new[old_idx] = static_cast<uint32_t>(out.size());
          out.push_back(red);
        } else {
          old_to_new[old_idx] = static_cast<uint32_t>(out.size());
          TiledOp red;
          red.kind = ExecOpKind::Reduce;
          red.bytes = op.bytes;
          red.src_off = op.src_off;
          red.dst_off = op.dst_off;
          red.deps = op.deps;
          out.push_back(red);
        }

      } else {
        old_to_new[old_idx] = static_cast<uint32_t>(out.size());
        out.push_back(op_to_tiled(op));
      }
    }
  }

  for (auto& o : out) {
    for (auto& dep : o.deps) {
      if (dep < n_old && old_to_new[dep] != kNoOp)
        dep = old_to_new[dep];
    }
  }

  return out;
}

}  // namespace

TiledResult lower_algo(CollAlgo const& algo, size_t tile_bytes,
                       bool inplace) {
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
  propagate_deps(algo.chunks, first_tile, tiled.tiles_per_chunk,
                 tiled.ops);

  size_t staging_bytes = 0;
  result.ops = lower_to_tiled(std::move(tiled.ops), algo.chunks, first_tile,
                              tiled.tiles_per_chunk, inplace, staging_bytes);
  result.staging_bytes_required = staging_bytes;
  return result;
}

TiledResult build_tiled(CollectiveConfig const& config, bool inplace) {
  CollAlgo algo = build_coll_algo(config, inplace);
  return lower_algo(algo, config.tile_bytes, inplace);
}

}  // namespace CCL
}  // namespace UKernel
