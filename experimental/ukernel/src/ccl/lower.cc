#include "lower.h"
#include "utils.h"
#include <algorithm>
#include <queue>

namespace UKernel {
namespace CCL {

namespace {

struct TileResult {
  std::vector<Op> ops;
  std::vector<size_t> first_tile;
  std::vector<uint32_t> chunk_of;
};

TileResult tile_chunks(CollAlgo const& algo, size_t tile_bytes) {
  TileResult r;
  size_t n = algo.chunks.size();
  r.first_tile.resize(n);

  for (size_t i = 0; i < n; ++i) {
    auto const& c = algo.chunks[i];
    r.first_tile[i] = r.ops.size();
    size_t num_tiles = ceil_div(c.bytes, tile_bytes);
    for (size_t t = 0; t < num_tiles; ++t) {
      Op tile;
      tile.kind = c.op;
      tile.bytes = std::min(tile_bytes, c.bytes - t * tile_bytes);
      tile.src_off = c.src_off + t * tile_bytes;
      tile.dst_off = c.dst_off + t * tile_bytes;
      tile.src_peer = (c.src_rank < 0) ? ~0u : static_cast<uint32_t>(c.src_rank);
      tile.dst_peer = (c.dst_rank < 0) ? ~0u : static_cast<uint32_t>(c.dst_rank);
      tile.copy_from_staging =
          (c.op == OpKind::Copy && c.sequential_tiles);
      r.ops.push_back(tile);
      r.chunk_of.push_back(static_cast<uint32_t>(i));
    }
  }
  return r;
}

void propagate_deps(std::vector<Chunk> const& chunks,
                    std::vector<size_t> const& first_tile,
                    std::vector<Op>& ops) {
  size_t n = chunks.size();
  for (size_t i = 0; i < n; ++i) {
    auto const& c = chunks[i];
    size_t num_i =
        (i + 1 < n) ? first_tile[i + 1] - first_tile[i]
                    : ops.size() - first_tile[i];

    if (c.sequential_tiles)
      for (size_t t = 1; t < num_i; ++t)
        ops[first_tile[i] + t].deps.push_back(
            static_cast<uint32_t>(first_tile[i] + t - 1));

    for (uint32_t dep_idx : c.deps) {
      if (dep_idx >= n) continue;
      size_t num_d =
          (dep_idx + 1 < n)
              ? first_tile[dep_idx + 1] - first_tile[dep_idx]
              : ops.size() - first_tile[dep_idx];
      size_t common = std::min(num_i, num_d);
      for (size_t t = 0; t < common; ++t)
        ops[first_tile[i] + t].deps.push_back(
            static_cast<uint32_t>(first_tile[dep_idx] + t));
    }
  }
}

size_t compute_staging_bytes(CollAlgo const& algo, size_t tile_bytes) {
  for (auto const& c : algo.chunks)
    if (c.sequential_tiles)
      return static_cast<size_t>(algo.nranks - 1) * tile_bytes;
  return 0;
}

}  // namespace

std::vector<std::vector<uint32_t>> bfs_layers(std::vector<Op> const& ops) {
  size_t n = ops.size();
  std::vector<uint32_t> indegree(n, 0);
  std::vector<std::vector<uint32_t>> successors(n);

  for (uint32_t i = 0; i < n; ++i) {
    indegree[i] = static_cast<uint32_t>(ops[i].deps.size());
    for (uint32_t dep : ops[i].deps)
      if (dep < n) successors[dep].push_back(i);
  }

  std::queue<uint32_t> fifo;
  for (uint32_t i = 0; i < n; ++i)
    if (indegree[i] == 0) fifo.push(i);

  std::vector<std::vector<uint32_t>> layers;
  while (!fifo.empty()) {
    std::vector<uint32_t> layer;
    std::queue<uint32_t> next_fifo;
    layer.reserve(fifo.size());
    while (!fifo.empty()) {
      uint32_t idx = fifo.front();
      fifo.pop();
      layer.push_back(idx);
      for (uint32_t succ : successors[idx])
        if (--indegree[succ] == 0) next_fifo.push(succ);
    }
    layers.push_back(std::move(layer));
    fifo = std::move(next_fifo);
  }
  return layers;
}

TiledResult lower_algo(CollAlgo const& algo, size_t tile_bytes) {
  TiledResult result;
  result.input_bytes = algo.input_bytes;
  result.output_bytes = algo.output_bytes;
  result.rank = algo.rank;
  result.nranks = algo.nranks;
  result.reduction = algo.reduction;
  if (algo.chunks.empty()) return result;

  auto tiled = tile_chunks(algo, tile_bytes);
  propagate_deps(algo.chunks, tiled.first_tile, tiled.ops);

  result.ops = std::move(tiled.ops);
  result.chunk_of = std::move(tiled.chunk_of);
  result.staging_bytes_required = compute_staging_bytes(algo, tile_bytes);
  result.layers = bfs_layers(result.ops);
  return result;
}

TiledResult build_tiled(CollectiveConfig const& config, bool inplace) {
  CollAlgo algo = build_coll_algo(config, inplace);
  return lower_algo(algo, config.tile_bytes);
}

}  // namespace CCL
}  // namespace UKernel
