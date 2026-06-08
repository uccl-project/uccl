#include "chunk_graph.h"
#include "topology.h"
#include "utils.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {



size_t balanced_shard_offset_bytes(size_t bytes, size_t elem_bytes, int nranks,
                                   int owner_rank) {
  size_t total_elems = bytes / elem_bytes;
  size_t base_elems = total_elems / static_cast<size_t>(nranks);
  size_t extra_elems = total_elems % static_cast<size_t>(nranks);
  size_t offset_elems = static_cast<size_t>(owner_rank) * base_elems +
                        std::min(static_cast<size_t>(owner_rank), extra_elems);
  return offset_elems * elem_bytes;
}

size_t balanced_shard_size_bytes(size_t bytes, size_t elem_bytes, int nranks,
                                 int owner_rank) {
  size_t total_elems = bytes / elem_bytes;
  size_t base_elems = total_elems / static_cast<size_t>(nranks);
  size_t extra_elems = total_elems % static_cast<size_t>(nranks);
  size_t shard_elems =
      base_elems + (static_cast<size_t>(owner_rank) < extra_elems ? 1 : 0);
  return shard_elems * elem_bytes;
}

size_t alltoall_input_bytes(CollectiveConfig const& config) {
  return config.input_bytes;
}

size_t alltoall_output_bytes(CollectiveConfig const& config) {
  return config.output_bytes;
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

void require_collective_config(CollectiveConfig const& config, bool inplace) {
  (void)inplace;
  if (config.nranks < 2) {
    throw std::invalid_argument("collective requires at least two ranks");
  }
  if (config.rank < 0 || config.rank >= config.nranks) {
    throw std::invalid_argument("collective rank out of range");
  }
  if (config.tile_bytes == 0) {
    throw std::invalid_argument("collective tile_bytes must be positive");
  }

  size_t elem_bytes = scalar_type_size(config.dtype);
  if (elem_bytes == 0) {
    throw std::invalid_argument(
        "collective dtype has invalid element size");
  }
  if (config.tile_bytes % elem_bytes != 0) {
    throw std::invalid_argument(
        "collective tile_bytes must be aligned to dtype size");
  }

  if (config.kind == CollKind::AllReduceRing) {
    if (config.input_bytes == 0) {
      throw std::invalid_argument(
          "collective input_bytes must be positive");
    }
    if (config.input_bytes % elem_bytes != 0) {
      throw std::invalid_argument(
          "collective input_bytes must be aligned to dtype size");
    }
    return;
  }

  size_t input_bytes = alltoall_input_bytes(config);
  size_t output_bytes = alltoall_output_bytes(config);
  if (input_bytes == 0 || output_bytes == 0) {
    throw std::invalid_argument(
        "alltoall requires positive input/output tensor bytes");
  }
  if (input_bytes % elem_bytes != 0 || output_bytes % elem_bytes != 0) {
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
    if (input_bytes % denom != 0 || output_bytes % denom != 0) {
      throw std::invalid_argument(
          "equal-split alltoall requires input/output tensor bytes divisible "
          "by nranks * dtype size");
    }
    return;
  }

  validate_alltoall_splits(config.input_split_bytes, config.nranks, elem_bytes,
                           input_bytes, "input");
  validate_alltoall_splits(config.output_split_bytes, config.nranks, elem_bytes,
                           output_bytes, "output");
}

CollAlgo make_empty_algo(CollectiveConfig const& config) {
  CollAlgo algo;
  algo.kind = CollKind::AllReduceRing;
  algo.nranks = config.nranks;
  algo.rank = config.rank;
  algo.input_bytes = config.input_bytes;
  algo.output_bytes = config.output_bytes;
  algo.reduction = config.reduction;
  return algo;
}

struct ChunkBuilder {
  explicit ChunkBuilder(CollAlgo algo_in) : algo(std::move(algo_in)) {}

  uint32_t add_op(OpKind kind, size_t bytes, size_t src_off, size_t dst_off,
                  int src_rank, int dst_rank,
                  std::vector<uint32_t> deps,
                  bool sequential_tiles = false) {
    Chunk chunk;
    chunk.op = kind;
    chunk.bytes = bytes;
    chunk.src_off = src_off;
    chunk.dst_off = dst_off;
    chunk.src_rank = src_rank;
    chunk.dst_rank = dst_rank;
    chunk.deps = std::move(deps);
    chunk.sequential_tiles = sequential_tiles;
    uint32_t idx = static_cast<uint32_t>(algo.chunks.size());
    algo.chunks.push_back(std::move(chunk));
    return idx;
  }

  CollAlgo algo;
};

CollAlgo build_allreduce_ring_algo(CollectiveConfig const& config) {
  RingTopology ring{config.nranks};
  CollAlgo algo = make_empty_algo(config);
  algo.kind = CollKind::AllReduceRing;
  size_t elem_bytes = scalar_type_size(config.dtype);

  ChunkBuilder builder(std::move(algo));

  std::vector<uint32_t> ready_ops(static_cast<size_t>(config.nranks), kNoOp);

  for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
    int send_owner = ring.wrap(config.rank - ring_step);
    int recv_owner = ring.wrap(config.rank - ring_step - 1);
    int send_peer = ring.next(config.rank);
    int recv_peer = ring.prev(config.rank);
    size_t send_bytes = balanced_shard_size_bytes(
        config.input_bytes, elem_bytes, config.nranks, send_owner);
    size_t recv_bytes = balanced_shard_size_bytes(
        config.input_bytes, elem_bytes, config.nranks, recv_owner);

    if (send_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.input_bytes, elem_bytes, config.nranks, send_owner);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(send_owner)]);
      builder.add_op(OpKind::Send, send_bytes, offset, offset, -1,
                     send_peer, std::move(deps));
    }

    if (recv_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.input_bytes, elem_bytes, config.nranks, recv_owner);
      uint32_t reduce_op = builder.add_op(OpKind::RecvReduce, recv_bytes,
                                          offset, offset, recv_peer, -1, {});
      ready_ops[static_cast<size_t>(recv_owner)] = reduce_op;
    }
  }

  for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
    int send_owner = ring.wrap(config.rank + 1 - ring_step);
    int recv_owner = ring.wrap(config.rank - ring_step);
    int send_peer = ring.next(config.rank);
    int recv_peer = ring.prev(config.rank);
    size_t send_bytes = balanced_shard_size_bytes(
        config.input_bytes, elem_bytes, config.nranks, send_owner);
    size_t recv_bytes = balanced_shard_size_bytes(
        config.input_bytes, elem_bytes, config.nranks, recv_owner);

    if (send_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.input_bytes, elem_bytes, config.nranks, send_owner);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(send_owner)]);
      builder.add_op(OpKind::Send, send_bytes, offset, offset, -1,
                     send_peer, std::move(deps));
    }

    if (recv_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.input_bytes, elem_bytes, config.nranks, recv_owner);
      uint32_t recv_op = builder.add_op(OpKind::Recv, recv_bytes, offset,
                                        offset, recv_peer, -1, {});
      ready_ops[static_cast<size_t>(recv_owner)] = recv_op;
    }
  }

  return std::move(builder.algo);
}

CollAlgo build_alltoall_pairwise_algo_dma(
    CollectiveConfig const& config, bool inplace) {
  CollAlgo algo = make_empty_algo(config);
  algo.kind = CollKind::AllToAllPairwise;
  algo.input_bytes = alltoall_input_bytes(config);
  algo.output_bytes = alltoall_output_bytes(config);

  size_t input_bytes = alltoall_input_bytes(config);
  size_t output_bytes = alltoall_output_bytes(config);

  std::vector<size_t> input_splits =
      config.input_split_bytes.empty()
          ? equal_alltoall_splits(input_bytes, config.nranks)
          : config.input_split_bytes;
  std::vector<size_t> output_splits =
      config.output_split_bytes.empty()
          ? equal_alltoall_splits(output_bytes, config.nranks)
          : config.output_split_bytes;
  std::vector<size_t> input_prefix = prefix_bytes(input_splits);
  std::vector<size_t> output_prefix = prefix_bytes(output_splits);

  ChunkBuilder builder(std::move(algo));

  size_t self_input_offset = input_prefix[static_cast<size_t>(config.rank)];
  size_t self_output_offset = output_prefix[static_cast<size_t>(config.rank)];
  size_t self_slice_bytes = input_splits[static_cast<size_t>(config.rank)];
  if (self_slice_bytes != output_splits[static_cast<size_t>(config.rank)]) {
    throw std::invalid_argument(
        "alltoall self split size must match between input and output");
  }
  if (self_slice_bytes != 0 && !inplace) {
    builder.add_op(OpKind::Copy, self_slice_bytes, self_input_offset,
                   self_output_offset, -1, -1, {});
  }

  size_t peer_slot = 0;
  for (int peer = 0; peer < config.nranks; ++peer) {
    if (peer == config.rank) continue;

    size_t send_offset = input_prefix[static_cast<size_t>(peer)];
    size_t send_bytes = input_splits[static_cast<size_t>(peer)];
    size_t recv_offset = output_prefix[static_cast<size_t>(peer)];
    size_t recv_bytes = output_splits[static_cast<size_t>(peer)];
    size_t staging_offset = peer_slot * config.tile_bytes;

    uint32_t send_op = kNoOp;
    if (send_bytes > 0) {
      send_op = builder.add_op(
          OpKind::Send, send_bytes, send_offset, staging_offset, 0,
          peer, {}, /*sequential_tiles=*/true);
    }

    if (recv_bytes > 0) {
      uint32_t recv_op =
          builder.add_op(OpKind::Recv, recv_bytes, recv_offset,
                         staging_offset, peer, 0, {},
                         /*sequential_tiles=*/true);

      std::vector<uint32_t> copy_deps;
      add_dep(copy_deps, send_op);
      add_dep(copy_deps, recv_op);
      builder.add_op(OpKind::Copy, recv_bytes, staging_offset,
                     recv_offset, -1, -1, std::move(copy_deps),
                     /*sequential_tiles=*/true);
    }

    ++peer_slot;
  }

  return std::move(builder.algo);
}

CollAlgo build_alltoall_pairwise_algo_sm(
    CollectiveConfig const& config, bool inplace) {
  CollAlgo algo = make_empty_algo(config);
  algo.kind = CollKind::AllToAllPairwise;
  algo.input_bytes = alltoall_input_bytes(config);
  algo.output_bytes = alltoall_output_bytes(config);

  size_t input_bytes = alltoall_input_bytes(config);
  size_t output_bytes = alltoall_output_bytes(config);

  std::vector<size_t> input_splits =
      config.input_split_bytes.empty()
          ? equal_alltoall_splits(input_bytes, config.nranks)
          : config.input_split_bytes;
  std::vector<size_t> output_splits =
      config.output_split_bytes.empty()
          ? equal_alltoall_splits(output_bytes, config.nranks)
          : config.output_split_bytes;
  std::vector<size_t> input_prefix = prefix_bytes(input_splits);
  std::vector<size_t> output_prefix = prefix_bytes(output_splits);

  ChunkBuilder builder(std::move(algo));

  size_t self_input_offset = input_prefix[static_cast<size_t>(config.rank)];
  size_t self_output_offset = output_prefix[static_cast<size_t>(config.rank)];
  size_t self_slice_bytes = input_splits[static_cast<size_t>(config.rank)];
  if (self_slice_bytes != output_splits[static_cast<size_t>(config.rank)])
    throw std::invalid_argument(
        "alltoall self split size must match between input and output");

  if (self_slice_bytes != 0 && !inplace) {
    builder.add_op(OpKind::Copy, self_slice_bytes, self_input_offset,
                   self_output_offset, -1, -1, {});
  }

  for (int peer = 0; peer < config.nranks; ++peer) {
    if (peer == config.rank) continue;

    size_t send_offset = input_prefix[static_cast<size_t>(peer)];
    size_t send_bytes = input_splits[static_cast<size_t>(peer)];
    size_t recv_offset = output_prefix[static_cast<size_t>(peer)];
    size_t recv_bytes = output_splits[static_cast<size_t>(peer)];

    if (send_bytes > 0) {
      builder.add_op(OpKind::Send, send_bytes, send_offset, send_offset,
                     -1, peer, {});
    }
    if (recv_bytes > 0) {
      builder.add_op(OpKind::Recv, recv_bytes, recv_offset, recv_offset,
                     peer, -1, {});
    }
  }

  return std::move(builder.algo);
}

}  // namespace

CollAlgo build_coll_algo(CollectiveConfig const& config,
                                           bool inplace) {
  require_collective_config(config, inplace);
  switch (config.kind) {
    case CollKind::AllReduceRing:
      return build_allreduce_ring_algo(config);
    case CollKind::AllToAllPairwise:
      if (config.use_sm_ipc)
        return build_alltoall_pairwise_algo_sm(config, inplace);
      return build_alltoall_pairwise_algo_dma(config, inplace);
  }
  throw std::invalid_argument("unsupported collective kind");
}

}  // namespace CCL
}  // namespace UKernel
