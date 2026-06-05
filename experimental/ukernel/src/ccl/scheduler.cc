#include "scheduler.h"
#include <algorithm>
#include <queue>

namespace UKernel {
namespace CCL {

namespace {

size_t ceil_div(size_t a, size_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

void expand_abstract_op(AlgoOp const& abs_op, size_t tile_bytes,
                        std::vector<Op>& out, size_t& first_idx) {
  first_idx = out.size();
  size_t num_tiles = ceil_div(abs_op.bytes, tile_bytes);
  for (size_t t = 0; t < num_tiles; ++t) {
    Op tile;
    tile.kind = abs_op.kind;
    tile.bytes = std::min(tile_bytes, abs_op.bytes - t * tile_bytes);
    tile.src_off = abs_op.src_off + t * tile_bytes;
    tile.dst_off = abs_op.dst_off + t * tile_bytes;
    tile.src_peer = abs_op.src_peer;
    tile.dst_peer = abs_op.dst_peer;
    tile.copy_from_staging = abs_op.copy_from_staging;
    out.push_back(tile);
  }
}

}  // namespace

TiledResult tile_and_schedule(CollAlgo const& algo, size_t tile_bytes) {
  TiledResult result;
  result.input_bytes = algo.input_bytes;
  result.output_bytes = algo.output_bytes;
  if (algo.ops.empty()) return result;

  size_t n = algo.ops.size();

  std::vector<size_t> first_tile(n);
  for (size_t i = 0; i < n; ++i)
    expand_abstract_op(algo.ops[i], tile_bytes, result.ops, first_tile[i]);

  for (size_t i = 0; i < n; ++i) {
    auto const& abs_op = algo.ops[i];
    size_t num_tiles_i = (i + 1 < n) ? first_tile[i + 1] - first_tile[i]
                                     : result.ops.size() - first_tile[i];

    if (abs_op.tile_order == TileOrder::Sequential) {
      for (size_t t = 1; t < num_tiles_i; ++t)
        result.ops[first_tile[i] + t].deps.push_back(
            static_cast<uint32_t>(first_tile[i] + t - 1));
    }

    for (uint32_t dep_idx : abs_op.deps) {
      if (dep_idx >= n) continue;
      size_t num_tiles_dep = (dep_idx + 1 < n)
                                 ? first_tile[dep_idx + 1] - first_tile[dep_idx]
                                 : result.ops.size() - first_tile[dep_idx];
      size_t common = std::min(num_tiles_i, num_tiles_dep);
      for (size_t t = 0; t < common; ++t) {
        uint32_t tile_dep = static_cast<uint32_t>(first_tile[dep_idx] + t);
        result.ops[first_tile[i] + t].deps.push_back(tile_dep);
      }
    }
  }

  // Compute staging_bytes_required.
  result.staging_bytes_required = 0;
  if (algo.collective == CollectiveKind::AllToAll && algo.ops.size() > 0) {
    bool has_sequential = false;
    for (auto const& op : algo.ops) {
      if (op.tile_order == TileOrder::Sequential) {
        has_sequential = true;
        break;
      }
    }
    if (has_sequential)
      result.staging_bytes_required =
          static_cast<size_t>(algo.nranks - 1) * tile_bytes;
  }

  result.schedule = schedule_ops(result.ops);
  return result;
}

Schedule schedule_ops(std::vector<Op> const& ops) {
  Schedule sched;
  if (ops.empty()) return sched;

  size_t n = ops.size();
  std::vector<uint32_t> indegree(n, 0);
  std::vector<std::vector<uint32_t>> successors(n);

  for (uint32_t i = 0; i < n; ++i) {
    indegree[i] = static_cast<uint32_t>(ops[i].deps.size());
    for (uint32_t dep : ops[i].deps) {
      if (dep < n) successors[dep].push_back(i);
    }
  }

  std::queue<uint32_t> fifo;
  for (uint32_t i = 0; i < n; ++i)
    if (indegree[i] == 0) fifo.push(i);

  uint32_t max_width = 0;
  while (!fifo.empty()) {
    max_width = std::max(max_width, static_cast<uint32_t>(fifo.size()));
    std::queue<uint32_t> next_fifo;
    while (!fifo.empty()) {
      uint32_t op_idx = fifo.front();
      fifo.pop();
      for (uint32_t succ : successors[op_idx]) {
        if (--indegree[succ] == 0) next_fifo.push(succ);
      }
    }
    fifo = std::move(next_fifo);
  }

  if (max_width == 0) max_width = 1;
  sched.num_streams = max_width;
  sched.stream_ops.resize(max_width);

  indegree.assign(n, 0);
  for (uint32_t i = 0; i < n; ++i) {
    indegree[i] = static_cast<uint32_t>(ops[i].deps.size());
  }

  fifo = std::queue<uint32_t>();
  for (uint32_t i = 0; i < n; ++i)
    if (indegree[i] == 0) fifo.push(i);

  while (!fifo.empty()) {
    std::queue<uint32_t> next_fifo;
    std::vector<uint32_t> layer;
    layer.reserve(fifo.size());
    while (!fifo.empty()) {
      uint32_t op_idx = fifo.front();
      fifo.pop();
      layer.push_back(op_idx);
    }

    for (size_t k = 0; k < layer.size(); ++k) {
      uint32_t op_idx = layer[k];
      uint32_t stream_id = static_cast<uint32_t>(k % max_width);
      sched.stream_ops[stream_id].push_back(op_idx);

      for (uint32_t succ : successors[op_idx]) {
        if (--indegree[succ] == 0) next_fifo.push(succ);
      }
    }
    fifo = std::move(next_fifo);
  }

  return sched;
}

TiledResult build_plan(CollectiveConfig const& config, bool inplace) {
  CollAlgo algo = build_coll_algo(config, inplace);
  return tile_and_schedule(algo, config.tile_bytes);
}

}  // namespace CCL
}  // namespace UKernel
