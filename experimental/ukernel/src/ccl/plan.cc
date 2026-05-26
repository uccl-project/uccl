#include "plan.h"
#include "topology.h"
#include "utils.h"
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

constexpr uint32_t kNoOp = std::numeric_limits<uint32_t>::max();

size_t ceil_div(size_t a, size_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

size_t tile_offset(size_t tile_index, size_t tile_bytes) {
  return tile_index * tile_bytes;
}

size_t tile_size(size_t bytes, size_t tile_bytes, size_t tile_index) {
  size_t offset = tile_offset(tile_index, tile_bytes);
  if (offset >= bytes) return 0;
  return std::min(tile_bytes, bytes - offset);
}

size_t nominal_shard_bytes(size_t bytes, int nranks) {
  return ceil_div(bytes, static_cast<size_t>(nranks));
}

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

size_t max_balanced_shard_bytes(size_t bytes, size_t elem_bytes, int nranks) {
  if (bytes == 0) return 0;
  size_t total_elems = bytes / elem_bytes;
  return ceil_div(total_elems, static_cast<size_t>(nranks)) * elem_bytes;
}

uint32_t clamp_num_streams(uint32_t requested_streams, size_t tiles_per_unit) {
  size_t bounded = std::min<size_t>(std::max<size_t>(1, requested_streams),
                                    std::max<size_t>(1, tiles_per_unit));
  return static_cast<uint32_t>(bounded);
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
    throw std::invalid_argument("collective plan requires at least two ranks");
  }
  if (config.rank < 0 || config.rank >= config.nranks) {
    throw std::invalid_argument("collective rank out of range");
  }
  if (config.num_streams == 0) {
    throw std::invalid_argument("collective plan requires at least one stream");
  }
  if (config.tile_bytes == 0) {
    throw std::invalid_argument("collective plan tile_bytes must be positive");
  }

  size_t elem_bytes = scalar_type_size(config.dtype);
  if (elem_bytes == 0) {
    throw std::invalid_argument(
        "collective plan dtype has invalid element size");
  }
  if (config.tile_bytes % elem_bytes != 0) {
    throw std::invalid_argument(
        "collective plan tile_bytes must be aligned to dtype size");
  }

  if (config.collective == CollectiveKind::AllReduce) {
    if (config.input_bytes == 0) {
      throw std::invalid_argument(
          "collective plan input_bytes must be positive");
    }
    if (config.input_bytes % elem_bytes != 0) {
      throw std::invalid_argument(
          "collective plan input_bytes must be aligned to dtype size");
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

  validate_alltoall_splits(config.input_split_bytes, config.nranks,
                           elem_bytes, input_bytes, "input");
  validate_alltoall_splits(config.output_split_bytes, config.nranks,
                           elem_bytes, output_bytes, "output");
}

CollectivePlan make_empty_plan(CollectiveConfig const& config) {
  CollectivePlan plan;
  plan.collective = config.collective;
  plan.nranks = config.nranks;
  plan.rank = config.rank;
  plan.num_streams = config.num_streams;
  plan.input_bytes = config.input_bytes;
  plan.output_bytes = config.output_bytes;
  plan.tile_bytes = config.tile_bytes;
  plan.reduction = config.reduction;
  return plan;
}

CollectivePlan make_empty_alltoall_plan(CollectiveConfig const& config) {
  CollectivePlan plan;
  plan.collective = CollectiveKind::AllToAll;
  plan.nranks = config.nranks;
  plan.rank = config.rank;
  plan.num_streams = config.num_streams;
  plan.input_bytes = alltoall_input_bytes(config);
  plan.output_bytes = alltoall_output_bytes(config);
  plan.tile_bytes = config.tile_bytes;
  plan.reduction = config.reduction;
  return plan;
}

struct PlanBuilder {
  explicit PlanBuilder(CollectivePlan plan_in) : plan(std::move(plan_in)) {
    plan.stream_ops.resize(plan.num_streams);
  }

  uint32_t add_op(OpKind kind, uint32_t stream_index,
                  size_t bytes, size_t src_off, size_t dst_off,
                  uint32_t src_peer, uint32_t dst_peer,
                  std::vector<uint32_t> deps) {
    Op op;
    op.kind = kind;
    op.stream_index = stream_index;
    op.bytes = bytes;
    op.src_off = src_off;
    op.dst_off = dst_off;
    op.src_peer = src_peer;
    op.dst_peer = dst_peer;
    op.deps = std::move(deps);
    uint32_t idx = static_cast<uint32_t>(plan.ops.size());
    if (stream_index < plan.num_streams)
      plan.stream_ops[stream_index].push_back(idx);
    plan.ops.push_back(std::move(op));
    return idx;
  }

  CollectivePlan plan;
};


}  // namespace

uint32_t normalized_num_streams(CollectiveKind collective, int nranks,
                               size_t input_bytes, size_t tile_bytes,
                               ScalarType dtype, uint32_t requested_streams) {
  size_t unit_bytes = nominal_shard_bytes(input_bytes, nranks);
  switch (collective) {
    case CollectiveKind::AllReduce: {
      size_t elem_bytes = scalar_type_size(dtype);
      unit_bytes = max_balanced_shard_bytes(input_bytes, elem_bytes, nranks);
      [[fallthrough]];
    }
    case CollectiveKind::AllToAll: {
      size_t tiles_per_unit = ceil_div(unit_bytes, tile_bytes);
      return clamp_num_streams(requested_streams, tiles_per_unit);
    }
  }
  return std::max<uint32_t>(1, requested_streams);
}

namespace {

CollectivePlan build_allreduce_ring_plan(CollectiveConfig const& config) {
  RingTopology ring{config.nranks};
  CollectivePlan plan = make_empty_plan(config);
  plan.collective = CollectiveKind::AllReduce;
  size_t elem_bytes = scalar_type_size(config.dtype);
  size_t shard_bytes = max_balanced_shard_bytes(config.input_bytes,
                                                elem_bytes, config.nranks);
  size_t tiles_per_shard = ceil_div(shard_bytes, config.tile_bytes);
  uint32_t num_streams = normalized_num_streams(
      CollectiveKind::AllReduce, config.nranks, config.input_bytes,
      config.tile_bytes, config.dtype, config.num_streams);
  plan.num_streams = num_streams;
  plan.staging_bytes_required = 0;  // SM IPC: no staging

  PlanBuilder builder(std::move(plan));
  uint64_t tile_seq = 0;  // local counter — executor adds global base later

  std::vector<std::vector<uint32_t>> ready_ops(
      static_cast<size_t>(config.nranks),
      std::vector<uint32_t>(tiles_per_shard, kNoOp));

  for (uint32_t stream_slot = 0; stream_slot < num_streams; ++stream_slot) {
    for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
      int send_owner = ring.wrap(config.rank - ring_step);
      int recv_owner = ring.wrap(config.rank - ring_step - 1);
      int send_peer = ring.next(config.rank);
      int recv_peer = ring.prev(config.rank);
      size_t send_bytes = balanced_shard_size_bytes(
          config.input_bytes, elem_bytes, config.nranks, send_owner);
      size_t recv_bytes = balanced_shard_size_bytes(
          config.input_bytes, elem_bytes, config.nranks, recv_owner);

      for (size_t tile_index = stream_slot; tile_index < tiles_per_shard;
           tile_index += num_streams) {
        size_t send_tile_bytes =
            tile_size(send_bytes, config.tile_bytes, tile_index);
        size_t recv_tile_bytes =
            tile_size(recv_bytes, config.tile_bytes, tile_index);

        if (send_tile_bytes > 0 || recv_tile_bytes > 0) {
          // One seq per tile pair — send and reduce/recv share it.
          uint64_t ts = tile_seq++;
          if (send_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(config.input_bytes, elem_bytes,
                                          config.nranks, send_owner) +
              tile_offset(tile_index, config.tile_bytes);

          std::vector<uint32_t> deps;
          add_dep(deps, ready_ops[static_cast<size_t>(send_owner)][tile_index]);
          uint32_t send_op = builder.add_op(
              OpKind::DeviceSend, stream_slot, send_tile_bytes,
              offset, offset, ~0u, send_peer, std::move(deps));
          builder.plan.ops.back().seq = ts;
        }

        if (recv_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(config.input_bytes, elem_bytes,
                                          config.nranks, recv_owner) +
              tile_offset(tile_index, config.tile_bytes);

          uint32_t reduce_op =
              builder.add_op(OpKind::DeviceRecvReduce, stream_slot, recv_tile_bytes,
                             offset, offset, recv_peer, ~0u, {});
          builder.plan.ops.back().seq = ts;
          ready_ops[static_cast<size_t>(recv_owner)][tile_index] = reduce_op;
        }
        }
      }
    }
  }

  for (uint32_t stream_slot = 0; stream_slot < num_streams; ++stream_slot) {
    for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
      int send_owner = ring.wrap(config.rank + 1 - ring_step);
      int recv_owner = ring.wrap(config.rank - ring_step);
      int send_peer = ring.next(config.rank);
      int recv_peer = ring.prev(config.rank);
      size_t send_bytes = balanced_shard_size_bytes(
          config.input_bytes, elem_bytes, config.nranks, send_owner);
      size_t recv_bytes = balanced_shard_size_bytes(
          config.input_bytes, elem_bytes, config.nranks, recv_owner);

      for (size_t tile_index = stream_slot; tile_index < tiles_per_shard;
           tile_index += num_streams) {
        size_t send_tile_bytes =
            tile_size(send_bytes, config.tile_bytes, tile_index);
        size_t recv_tile_bytes =
            tile_size(recv_bytes, config.tile_bytes, tile_index);

        if (send_tile_bytes > 0 || recv_tile_bytes > 0) {
          uint64_t ts = tile_seq++;
          if (send_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(config.input_bytes, elem_bytes,
                                          config.nranks, send_owner) +
              tile_offset(tile_index, config.tile_bytes);

          std::vector<uint32_t> deps;
          add_dep(deps, ready_ops[static_cast<size_t>(send_owner)][tile_index]);
          uint32_t send_op = builder.add_op(
              OpKind::DeviceSend, stream_slot, send_tile_bytes,
              offset, offset, ~0u, send_peer, std::move(deps));
          builder.plan.ops.back().seq = ts;
        }

        if (recv_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(config.input_bytes, elem_bytes,
                                          config.nranks, recv_owner) +
              tile_offset(tile_index, config.tile_bytes);

          uint32_t recv_op = builder.add_op(
              OpKind::DeviceRecv, stream_slot, recv_tile_bytes,
              offset, offset, recv_peer, ~0u, {});
          builder.plan.ops.back().seq = ts;
          ready_ops[static_cast<size_t>(recv_owner)][tile_index] = recv_op;
        }
        }
      }
    }
  }

  return std::move(builder.plan);
}

CollectivePlan build_alltoall_pairwise_plan_dma(CollectiveConfig const& config,
                                            bool inplace) {
  CollectivePlan plan = make_empty_alltoall_plan(config);
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

  size_t max_slice_bytes = 0;
  for (size_t bytes : input_splits) {
    max_slice_bytes = std::max(max_slice_bytes, bytes);
  }
  for (size_t bytes : output_splits) {
    max_slice_bytes = std::max(max_slice_bytes, bytes);
  }
  size_t tiles_per_slice = ceil_div(max_slice_bytes, config.tile_bytes);
  uint32_t num_streams = clamp_num_streams(config.num_streams, tiles_per_slice);
  plan.num_streams = num_streams;
  plan.staging_bytes_required =
      static_cast<size_t>(config.nranks - 1) * config.tile_bytes;

  PlanBuilder builder(std::move(plan));

  std::vector<uint32_t> last_send_to_peer(static_cast<size_t>(config.nranks),
                                          kNoOp);
  std::vector<uint32_t> last_recv_from_peer(static_cast<size_t>(config.nranks),
                                            kNoOp);
  std::vector<uint32_t> last_staging_consumer(
      static_cast<size_t>(config.nranks > 0 ? config.nranks - 1 : 0), kNoOp);

  size_t self_input_offset = input_prefix[static_cast<size_t>(config.rank)];
  size_t self_output_offset = output_prefix[static_cast<size_t>(config.rank)];
  size_t self_slice_bytes = input_splits[static_cast<size_t>(config.rank)];
  if (self_slice_bytes != output_splits[static_cast<size_t>(config.rank)]) {
    throw std::invalid_argument(
        "alltoall self split size must match between input and output");
  }
  if (self_slice_bytes != 0 && !inplace) {
    size_t self_tiles = ceil_div(self_slice_bytes, config.tile_bytes);
    for (size_t tile_index = 0; tile_index < self_tiles; ++tile_index) {
      size_t bytes =
          tile_size(self_slice_bytes, config.tile_bytes, tile_index);
      if (bytes == 0) continue;
      size_t input_offset =
          self_input_offset + tile_offset(tile_index, config.tile_bytes);
      size_t output_offset =
          self_output_offset + tile_offset(tile_index, config.tile_bytes);
      builder.add_op(OpKind::DeviceCopy,
                     static_cast<uint32_t>(tile_index % num_streams),
                     bytes, input_offset, output_offset, ~0u, ~0u, {});
    }
  }

  size_t peer_slot = 0;
  for (int peer = 0; peer < config.nranks; ++peer) {
    if (peer == config.rank) continue;
    size_t remote_peer_slot = static_cast<size_t>(
        config.rank < peer ? config.rank : config.rank - 1);

    size_t send_offset = input_prefix[static_cast<size_t>(peer)];
    size_t send_bytes = input_splits[static_cast<size_t>(peer)];
    size_t recv_offset = output_prefix[static_cast<size_t>(peer)];
    size_t recv_bytes = output_splits[static_cast<size_t>(peer)];
    size_t staging_offset = peer_slot * config.tile_bytes;
    size_t send_tiles = ceil_div(send_bytes, config.tile_bytes);
    size_t recv_tiles = ceil_div(recv_bytes, config.tile_bytes);
    size_t tiles = std::max(send_tiles, recv_tiles);

    for (size_t tile_index = 0; tile_index < tiles; ++tile_index) {
      uint32_t send_op = kNoOp;
      if (tile_index < send_tiles) {
        size_t bytes = tile_size(send_bytes, config.tile_bytes, tile_index);
        size_t offset =
            send_offset + tile_offset(tile_index, config.tile_bytes);
        std::vector<uint32_t> send_deps;
        add_dep(send_deps, last_send_to_peer[static_cast<size_t>(peer)]);
        send_op = builder.add_op(
            OpKind::TransportSend,
            static_cast<uint32_t>(tile_index % num_streams),
            bytes, offset,
            remote_peer_slot * config.tile_bytes,
            0, peer, std::move(send_deps));
        last_send_to_peer[static_cast<size_t>(peer)] = send_op;
      }

      if (tile_index < recv_tiles) {
        size_t bytes = tile_size(recv_bytes, config.tile_bytes, tile_index);
        size_t offset =
            recv_offset + tile_offset(tile_index, config.tile_bytes);
        std::vector<uint32_t> recv_deps;
        add_dep(recv_deps, last_recv_from_peer[static_cast<size_t>(peer)]);
        add_dep(recv_deps, last_staging_consumer[peer_slot]);
        uint32_t recv_op = builder.add_op(
            OpKind::TransportRecv,
            static_cast<uint32_t>(tile_index % num_streams),
            bytes, offset, staging_offset, peer, 0,
            std::move(recv_deps));
        last_recv_from_peer[static_cast<size_t>(peer)] = recv_op;

        std::vector<uint32_t> copy_deps;
        add_dep(copy_deps, send_op);
        add_dep(copy_deps, recv_op);
        uint32_t copy_op = builder.add_op(
            OpKind::DeviceCopy,
            static_cast<uint32_t>(tile_index % num_streams),
            bytes, staging_offset, offset, ~0u, ~0u,
            std::move(copy_deps));
        builder.plan.ops.back().copy_from_staging = true;
        last_staging_consumer[peer_slot] = copy_op;
      }
    }

    ++peer_slot;
  }

  return std::move(builder.plan);
}

CollectivePlan build_alltoall_pairwise_plan_sm(CollectiveConfig const& config,
                                               bool inplace) {
  CollectivePlan plan = make_empty_alltoall_plan(config);
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

  size_t max_slice_bytes = 0;
  for (size_t bytes : input_splits) max_slice_bytes = std::max(max_slice_bytes, bytes);
  for (size_t bytes : output_splits) max_slice_bytes = std::max(max_slice_bytes, bytes);
  size_t tiles_per_slice = ceil_div(max_slice_bytes, config.tile_bytes);
  uint32_t num_streams = clamp_num_streams(config.num_streams, tiles_per_slice);
  plan.num_streams = num_streams;
  plan.staging_bytes_required = 0;  // SM IPC: no staging

  PlanBuilder builder(std::move(plan));
  uint64_t tile_seq = 0;

  size_t self_input_offset = input_prefix[static_cast<size_t>(config.rank)];
  size_t self_output_offset = output_prefix[static_cast<size_t>(config.rank)];
  size_t self_slice_bytes = input_splits[static_cast<size_t>(config.rank)];
  if (self_slice_bytes != output_splits[static_cast<size_t>(config.rank)])
    throw std::invalid_argument("alltoall self split size must match between input and output");

  // Self-copy: local DeviceCopy (same as DMA).
  if (self_slice_bytes != 0 && !inplace) {
    size_t self_tiles = ceil_div(self_slice_bytes, config.tile_bytes);
    for (size_t tile_index = 0; tile_index < self_tiles; ++tile_index) {
      size_t bytes = tile_size(self_slice_bytes, config.tile_bytes, tile_index);
      if (bytes == 0) continue;
      size_t in_off = self_input_offset + tile_offset(tile_index, config.tile_bytes);
      size_t out_off = self_output_offset + tile_offset(tile_index, config.tile_bytes);
      builder.add_op(OpKind::DeviceCopy,
                     static_cast<uint32_t>(tile_index % num_streams),
                     bytes, in_off, out_off, ~0u, ~0u, {});
    }
  }

  // Per-peer SM IPC: DeviceSend + DeviceRecv.
  for (int peer = 0; peer < config.nranks; ++peer) {
    if (peer == config.rank) continue;

    size_t send_offset = input_prefix[static_cast<size_t>(peer)];
    size_t send_bytes = input_splits[static_cast<size_t>(peer)];
    size_t recv_offset = output_prefix[static_cast<size_t>(peer)];
    size_t recv_bytes = output_splits[static_cast<size_t>(peer)];
    size_t send_tiles = ceil_div(send_bytes, config.tile_bytes);
    size_t recv_tiles = ceil_div(recv_bytes, config.tile_bytes);
    size_t tiles = std::max(send_tiles, recv_tiles);

    for (size_t tile_index = 0; tile_index < tiles; ++tile_index) {
      size_t stb = (tile_index < send_tiles)
          ? tile_size(send_bytes, config.tile_bytes, tile_index) : 0;
      size_t rtb = (tile_index < recv_tiles)
          ? tile_size(recv_bytes, config.tile_bytes, tile_index) : 0;
      if (stb > 0 || rtb > 0) {
        uint64_t ts = tile_seq++;
        if (stb > 0) {
          size_t off = send_offset + tile_offset(tile_index, config.tile_bytes);
          uint32_t send_op = builder.add_op(
              OpKind::DeviceSend,
              static_cast<uint32_t>(tile_index % num_streams),
              stb, off, off, ~0u, peer, {});
          builder.plan.ops.back().seq = ts;
        }
        if (rtb > 0) {
          size_t off = recv_offset + tile_offset(tile_index, config.tile_bytes);
          uint32_t recv_op = builder.add_op(
              OpKind::DeviceRecv,
              static_cast<uint32_t>(tile_index % num_streams),
              rtb, off, off, peer, ~0u, {});
          builder.plan.ops.back().seq = ts;
        }
      }
    }
  }

  return std::move(builder.plan);
}

char const* op_kind_name(OpKind kind) {
  switch (kind) {
    case OpKind::TransportSend:       return "send";
    case OpKind::TransportRecv:       return "recv";
    case OpKind::DeviceCopy:          return "copy";
    case OpKind::DeviceReduce:        return "reduce";
    case OpKind::DeviceSend:    return "dev_send";
    case OpKind::DeviceRecvReduce:  return "dev_recv_reduce";
    case OpKind::DeviceRecv:    return "dev_recv";
  }
  return "unknown";
}

char const* buffer_kind_name(BufferKind kind) {
  switch (kind) {
    case BufferKind::Local:
      return "local";
    case BufferKind::Remote:
      return "remote";
  }
  return "unknown";
}

}  // namespace

CollectivePlan build_plan(CollectiveConfig const& config, bool inplace) {
  require_collective_config(config, inplace);
  switch (config.collective) {
    case CollectiveKind::AllReduce:
      if (config.algorithm != AlgorithmKind::Ring) {
        throw std::invalid_argument(
            "allreduce currently supports only ring algorithm");
      }
      return build_allreduce_ring_plan(config);
    case CollectiveKind::AllToAll:
      if (config.algorithm != AlgorithmKind::Pairwise) {
        throw std::invalid_argument(
            "alltoall currently supports only pairwise algorithm");
      }
      if (config.use_sm_ipc)
        return build_alltoall_pairwise_plan_sm(config, inplace);
      return build_alltoall_pairwise_plan_dma(config, inplace);
  }
  throw std::invalid_argument("unsupported collective kind");
}

std::string to_string(CollectivePlan const& plan) {
  std::ostringstream oss;
  oss << "CollectivePlan(rank=" << plan.rank << "/" << plan.nranks
      << ", input_bytes=" << plan.input_bytes
      << ", output_bytes=" << plan.output_bytes
      << ", tile_bytes=" << plan.tile_bytes
      << ", staging_bytes_required=" << plan.staging_bytes_required
      << ", ops=" << plan.ops.size() << ")\n";
  for (size_t i = 0; i < plan.ops.size(); ++i) {
    auto const& op = plan.ops[i];
    oss << "  op " << i << " " << op_kind_name(op.kind)
        << " stream=" << op.stream_index;
    int peer_rank = op.src_peer != ~0u ? static_cast<int>(op.src_peer)
                    : op.dst_peer != ~0u ? static_cast<int>(op.dst_peer) : -1;
    if (peer_rank >= 0) {
      oss << " peer=" << peer_rank;
    }
    oss << " bytes=" << op.bytes
        << " src_off=" << op.src_off << " dst_off=" << op.dst_off
        << " src_peer=" << op.src_peer << " dst_peer=" << op.dst_peer;
    if (op.copy_from_staging) oss << " staging";
    if (!op.deps.empty()) {
      oss << " deps=[";
      for (size_t i = 0; i < op.deps.size(); ++i) {
        if (i != 0) oss << ",";
        oss << op.deps[i];
      }
      oss << "]";
    }
    oss << "\n";
  }
  return oss.str();
}

}  // namespace CCL
}  // namespace UKernel
