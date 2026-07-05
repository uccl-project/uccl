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

  if (config.kind == CollKind::AllReduceRing) {
    if (config.input_bytes == 0) {
      throw std::invalid_argument("collective input_bytes must be positive");
    }
    if (config.input_bytes % elem_bytes != 0) {
      throw std::invalid_argument(
          "collective input_bytes must be aligned to dtype size");
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
  return algo;
}

struct ChunkBuilder {
  explicit ChunkBuilder(CollAlgo algo_in) : algo(std::move(algo_in)) {}

  uint32_t add_op(AlgoOpKind kind, size_t bytes, size_t src_off, size_t dst_off,
                  int src_rank, int dst_rank, std::vector<uint32_t> deps,
                  uint32_t pair_id = kNoPairId) {
    Chunk chunk;
    chunk.op = kind;
    chunk.bytes = bytes;
    chunk.src_off = src_off;
    chunk.dst_off = dst_off;
    chunk.src_rank = src_rank;
    chunk.dst_rank = dst_rank;
    chunk.deps = std::move(deps);
    chunk.pair_id = pair_id;
    uint32_t idx = static_cast<uint32_t>(algo.chunks.size());
    algo.chunks.push_back(std::move(chunk));
    return idx;
  }

  CollAlgo algo;
  uint32_t next_pair_id = 1;
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
      uint32_t pair_id = static_cast<uint32_t>(send_owner * 2);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(send_owner)]);
      builder.add_op(AlgoOpKind::Put, send_bytes, offset, offset, -1, send_peer,
                     std::move(deps), pair_id);
    }

    if (recv_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.input_bytes, elem_bytes, config.nranks, recv_owner);
      uint32_t pair_id = static_cast<uint32_t>(recv_owner * 2);
      uint32_t recv_op = builder.add_op(AlgoOpKind::Recv, recv_bytes, offset,
                                        offset, recv_peer, -1, {}, pair_id);
      uint32_t reduce_op = builder.add_op(AlgoOpKind::RecvReduce, recv_bytes,
                                          offset, offset, -1, -1, {recv_op});
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
      uint32_t pair_id = static_cast<uint32_t>(send_owner * 2 + 1);
      std::vector<uint32_t> deps;
      add_dep(deps, ready_ops[static_cast<size_t>(send_owner)]);
      builder.add_op(AlgoOpKind::Put, send_bytes, offset, offset, -1, send_peer,
                     std::move(deps), pair_id);
    }

    if (recv_bytes > 0) {
      size_t offset = balanced_shard_offset_bytes(
          config.input_bytes, elem_bytes, config.nranks, recv_owner);
      uint32_t pair_id = static_cast<uint32_t>(recv_owner * 2 + 1);
      uint32_t recv_op = builder.add_op(AlgoOpKind::Recv, recv_bytes, offset,
                                        offset, recv_peer, -1, {}, pair_id);
      ready_ops[static_cast<size_t>(recv_owner)] = recv_op;
    }
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

  ChunkBuilder builder(std::move(algo));

  size_t self_input_offset = input_prefix[static_cast<size_t>(config.rank)];
  size_t self_output_offset = output_prefix[static_cast<size_t>(config.rank)];
  size_t self_slice_bytes = input_splits[static_cast<size_t>(config.rank)];
  if (self_slice_bytes != output_splits[static_cast<size_t>(config.rank)])
    throw std::invalid_argument(
        "alltoall self split size must match between input and output");

  if (self_slice_bytes != 0 && !inplace) {
    builder.add_op(AlgoOpKind::Put, self_slice_bytes, self_input_offset,
                   self_output_offset, -1, -1, {});
  }

  for (int peer = 0; peer < config.nranks; ++peer) {
    if (peer == config.rank) continue;

    size_t send_offset = input_prefix[static_cast<size_t>(peer)];
    size_t send_bytes = input_splits[static_cast<size_t>(peer)];
    size_t recv_offset = output_prefix[static_cast<size_t>(peer)];
    size_t recv_bytes = output_splits[static_cast<size_t>(peer)];

    uint32_t pair_id = builder.next_pair_id++;

    if (send_bytes > 0) {
      builder.add_op(AlgoOpKind::Put, send_bytes, send_offset, send_offset, -1,
                     peer, {}, pair_id);
    }
    if (recv_bytes > 0) {
      builder.add_op(AlgoOpKind::Recv, recv_bytes, recv_offset, recv_offset,
                     peer, -1, {}, pair_id);
    }
  }

  return std::move(builder.algo);
}

}  // namespace

CollAlgo build_coll_algo(CollectiveConfig const& config, bool inplace) {
  require_collective_config(config);
  switch (config.kind) {
    case CollKind::AllReduceRing:
      return build_allreduce_ring_algo(config);
    case CollKind::AllToAllPairwise:
      return build_alltoall_pairwise_algo(config, inplace);
  }
  throw std::invalid_argument("unsupported collective kind");
}

}  // namespace CCL
}  // namespace UKernel
