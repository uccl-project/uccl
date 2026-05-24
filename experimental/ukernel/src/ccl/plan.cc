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

uint32_t clamp_num_flows(uint32_t requested_flows, size_t tiles_per_unit) {
  size_t bounded = std::min<size_t>(std::max<size_t>(1, requested_flows),
                                    std::max<size_t>(1, tiles_per_unit));
  return static_cast<uint32_t>(bounded);
}

BufferRef input_ref(CollectiveConfig const& /*config*/, size_t offset_bytes) {
  return local_buffer_ref(PlanBuffer::Input, offset_bytes);
}

BufferRef output_ref(CollectiveConfig const& /*config*/, size_t offset_bytes) {
  return local_buffer_ref(PlanBuffer::Output, offset_bytes);
}

BufferRef staging_ref(CollectiveConfig const& /*config*/, size_t offset_bytes) {
  return local_buffer_ref(PlanBuffer::Scratch, offset_bytes);
}

BufferRef peer_input_ref(CollectiveConfig const& /*config*/, int rank,
                         size_t offset_bytes) {
  return remote_buffer_ref(PlanBuffer::Input, rank, offset_bytes);
}

BufferRef peer_staging_ref(CollectiveConfig const& /*config*/, int rank,
                           size_t offset_bytes) {
  return remote_buffer_ref(PlanBuffer::Scratch, rank, offset_bytes);
}

size_t alltoall_input_bytes(CollectiveConfig const& config) {
  return config.input_bytes != 0 ? config.input_bytes : config.tensor_bytes;
}

size_t alltoall_output_bytes(CollectiveConfig const& config) {
  return config.output_bytes != 0 ? config.output_bytes
                                  : config.tensor_bytes;
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
  if (config.num_flows == 0) {
    throw std::invalid_argument("collective plan requires at least one flow");
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
    if (config.tensor_bytes == 0) {
      throw std::invalid_argument(
          "collective plan tensor_bytes must be positive");
    }
    if (config.tensor_bytes % elem_bytes != 0) {
      throw std::invalid_argument(
          "collective plan tensor_bytes must be aligned to dtype size");
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
  plan.algorithm = config.algorithm;
  plan.nranks = config.nranks;
  plan.rank = config.rank;
  plan.num_flows = config.num_flows;
  plan.tensor_bytes = config.tensor_bytes;
  plan.input_bytes = config.tensor_bytes;
  plan.output_bytes = config.tensor_bytes;
  plan.tile_bytes = config.tile_bytes;
  plan.dtype = config.dtype;
  plan.reduction = config.reduction;
  return plan;
}

CollectivePlan make_empty_alltoall_plan(CollectiveConfig const& config) {
  CollectivePlan plan;
  plan.collective = CollectiveKind::AllToAll;
  plan.algorithm = config.algorithm;
  plan.nranks = config.nranks;
  plan.rank = config.rank;
  plan.num_flows = config.num_flows;
  plan.input_bytes = alltoall_input_bytes(config);
  plan.output_bytes = alltoall_output_bytes(config);
  plan.tensor_bytes = std::max(plan.input_bytes, plan.output_bytes);
  plan.tile_bytes = config.tile_bytes;
  plan.dtype = config.dtype;
  plan.reduction = config.reduction;
  plan.input_split_bytes = config.input_split_bytes;
  plan.output_split_bytes = config.output_split_bytes;
  return plan;
}

struct PlanBuilder {
  explicit PlanBuilder(CollectivePlan plan_in) : plan(std::move(plan_in)) {
    plan.flow_ops.resize(plan.num_flows);
  }

  uint32_t add_op(OpKind kind, TileRef tile, BufferRef src, BufferRef dst,
                  ScalarType dtype, ReductionKind reduction,
                  std::vector<uint32_t> deps) {
    Op op;
    op.op_id = next_op_id++;
    op.kind = kind;
    op.tile = tile;
    op.src = std::move(src);
    op.dst = std::move(dst);
    op.dtype = dtype;
    op.reduction = reduction;
    op.deps = std::move(deps);
    uint32_t fid = tile.flow_index;
    if (fid < plan.num_flows)
      plan.flow_ops[fid].push_back(op.op_id);
    plan.ops.push_back(std::move(op));
    return plan.ops.back().op_id;
  }

  CollectivePlan plan;
  uint32_t next_op_id = 0;
};

TileRef make_tile(CollectiveConfig const& config, int owner_rank,
                  uint32_t flow_index, size_t tile_index, size_t offset_bytes,
                  size_t size_bytes) {
  (void)config;
  TileRef tile;
  tile.owner_rank = static_cast<uint32_t>(owner_rank);
  tile.tile_index = static_cast<uint32_t>(tile_index);
  tile.flow_index = flow_index;
  tile.offset_bytes = offset_bytes;
  tile.size_bytes = size_bytes;
  return tile;
}

ScalarType op_dtype(CollectiveConfig const& config, OpKind kind) {
  switch (kind) {
    case OpKind::DeviceReduce:
      return config.dtype;
    case OpKind::DeviceCopy:
      return ScalarType::Int8;
    case OpKind::TransportSend:
    case OpKind::TransportRecv:
      return ScalarType::UInt8;
  }
  return ScalarType::UInt8;
}

ReductionKind op_reduction(CollectiveConfig const& config, OpKind kind) {
  return kind == OpKind::DeviceReduce ? config.reduction
                                      : ReductionKind::None;
}

}  // namespace

uint32_t normalized_num_flows(CollectiveKind collective, int nranks,
                              size_t tensor_bytes, size_t tile_bytes,
                              ScalarType dtype, uint32_t requested_flows) {
  size_t unit_bytes = nominal_shard_bytes(tensor_bytes, nranks);
  switch (collective) {
    case CollectiveKind::AllReduce: {
      size_t elem_bytes = scalar_type_size(dtype);
      unit_bytes = max_balanced_shard_bytes(tensor_bytes, elem_bytes, nranks);
      [[fallthrough]];
    }
    case CollectiveKind::AllToAll: {
      size_t tiles_per_unit = ceil_div(unit_bytes, tile_bytes);
      return clamp_num_flows(requested_flows, tiles_per_unit);
    }
  }
  return std::max<uint32_t>(1, requested_flows);
}

namespace {

CollectivePlan build_allreduce_ring_plan(CollectiveConfig const& config) {
  RingTopology ring{config.nranks};
  CollectivePlan plan = make_empty_plan(config);
  plan.collective = CollectiveKind::AllReduce;
  plan.algorithm = AlgorithmKind::Ring;
  size_t elem_bytes = scalar_type_size(config.dtype);
  size_t shard_bytes = max_balanced_shard_bytes(config.tensor_bytes,
                                                elem_bytes, config.nranks);
  size_t tiles_per_shard = ceil_div(shard_bytes, config.tile_bytes);
  uint32_t num_flows = normalized_num_flows(
      CollectiveKind::AllReduce, config.nranks, config.tensor_bytes,
      config.tile_bytes, config.dtype, config.num_flows);
  plan.num_flows = num_flows;
  plan.staging_bytes_required = 0;  // SM IPC: no staging

  PlanBuilder builder(std::move(plan));
  uint64_t tile_seq = 0;  // local counter — executor adds global base later

  std::vector<std::vector<uint32_t>> ready_ops(
      static_cast<size_t>(config.nranks),
      std::vector<uint32_t>(tiles_per_shard, kNoOp));

  for (uint32_t flow_slot = 0; flow_slot < num_flows; ++flow_slot) {
    for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
      int send_owner = ring.wrap(config.rank - ring_step);
      int recv_owner = ring.wrap(config.rank - ring_step - 1);
      int send_peer = ring.next(config.rank);
      int recv_peer = ring.prev(config.rank);
      size_t send_bytes = balanced_shard_size_bytes(
          config.tensor_bytes, elem_bytes, config.nranks, send_owner);
      size_t recv_bytes = balanced_shard_size_bytes(
          config.tensor_bytes, elem_bytes, config.nranks, recv_owner);

      for (size_t tile_index = flow_slot; tile_index < tiles_per_shard;
           tile_index += num_flows) {
        size_t send_tile_bytes =
            tile_size(send_bytes, config.tile_bytes, tile_index);
        size_t recv_tile_bytes =
            tile_size(recv_bytes, config.tile_bytes, tile_index);

        if (send_tile_bytes > 0 || recv_tile_bytes > 0) {
          // One seq per tile pair — send and reduce/recv share it.
          uint64_t ts = tile_seq++;
          if (send_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(config.tensor_bytes, elem_bytes,
                                          config.nranks, send_owner) +
              tile_offset(tile_index, config.tile_bytes);
          TileRef send_tile = make_tile(config, send_owner, flow_slot,
                                        tile_index, offset, send_tile_bytes);

          std::vector<uint32_t> deps;
          add_dep(deps, ready_ops[static_cast<size_t>(send_owner)][tile_index]);
          uint32_t send_op = builder.add_op(
              OpKind::DeviceSendRemote, send_tile,
              input_ref(config, send_tile.offset_bytes),
              peer_input_ref(config, send_peer, send_tile.offset_bytes),
              ScalarType::Int8, ReductionKind::None, std::move(deps));
          builder.plan.ops.back().signal_seq = ts;
        }

        if (recv_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(config.tensor_bytes, elem_bytes,
                                          config.nranks, recv_owner) +
              tile_offset(tile_index, config.tile_bytes);
          TileRef recv_tile = make_tile(config, recv_owner, flow_slot,
                                        tile_index, offset, recv_tile_bytes);

          uint32_t reduce_op =
              builder.add_op(OpKind::DeviceReduceRemote, recv_tile,
                             peer_input_ref(config, recv_peer, recv_tile.offset_bytes),
                             input_ref(config, recv_tile.offset_bytes),
                             config.dtype, config.reduction, {});
          builder.plan.ops.back().signal_seq = ts;
          ready_ops[static_cast<size_t>(recv_owner)][tile_index] = reduce_op;
        }
        }
      }
    }
  }

  for (uint32_t flow_slot = 0; flow_slot < num_flows; ++flow_slot) {
    for (int ring_step = 0; ring_step < config.nranks - 1; ++ring_step) {
      int send_owner = ring.wrap(config.rank + 1 - ring_step);
      int recv_owner = ring.wrap(config.rank - ring_step);
      int send_peer = ring.next(config.rank);
      int recv_peer = ring.prev(config.rank);
      size_t send_bytes = balanced_shard_size_bytes(
          config.tensor_bytes, elem_bytes, config.nranks, send_owner);
      size_t recv_bytes = balanced_shard_size_bytes(
          config.tensor_bytes, elem_bytes, config.nranks, recv_owner);

      for (size_t tile_index = flow_slot; tile_index < tiles_per_shard;
           tile_index += num_flows) {
        size_t send_tile_bytes =
            tile_size(send_bytes, config.tile_bytes, tile_index);
        size_t recv_tile_bytes =
            tile_size(recv_bytes, config.tile_bytes, tile_index);

        if (send_tile_bytes > 0 || recv_tile_bytes > 0) {
          uint64_t ts = tile_seq++;
          if (send_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(config.tensor_bytes, elem_bytes,
                                          config.nranks, send_owner) +
              tile_offset(tile_index, config.tile_bytes);
          TileRef send_tile = make_tile(config, send_owner, flow_slot,
                                        tile_index, offset, send_tile_bytes);

          std::vector<uint32_t> deps;
          add_dep(deps, ready_ops[static_cast<size_t>(send_owner)][tile_index]);
          uint32_t send_op = builder.add_op(
              OpKind::DeviceSendRemote, send_tile,
              input_ref(config, send_tile.offset_bytes),
              peer_input_ref(config, send_peer, send_tile.offset_bytes),
              ScalarType::Int8, ReductionKind::None, std::move(deps));
          builder.plan.ops.back().signal_seq = ts;
        }

        if (recv_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(config.tensor_bytes, elem_bytes,
                                          config.nranks, recv_owner) +
              tile_offset(tile_index, config.tile_bytes);
          TileRef recv_tile = make_tile(config, recv_owner, flow_slot,
                                        tile_index, offset, recv_tile_bytes);

          uint32_t recv_op = builder.add_op(
              OpKind::DeviceRecvRemote, recv_tile,
              peer_input_ref(config, recv_peer, recv_tile.offset_bytes),
              input_ref(config, recv_tile.offset_bytes),
              ScalarType::Int8, ReductionKind::None, {});
          builder.plan.ops.back().signal_seq = ts;
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
  plan.algorithm = AlgorithmKind::Pairwise;
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
  uint32_t num_flows = clamp_num_flows(config.num_flows, tiles_per_slice);
  plan.num_flows = num_flows;
  plan.input_split_bytes = input_splits;
  plan.output_split_bytes = output_splits;
  plan.staging_bytes_required =
      static_cast<size_t>(config.nranks - 1) * config.tile_bytes;
  if (config.staging_bytes < plan.staging_bytes_required) {
    throw std::invalid_argument(
        "alltoall pairwise requires staging_bytes >= (nranks - 1) * "
        "tile_bytes");
  }

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
      TileRef tile = make_tile(config, config.rank,
                               static_cast<uint32_t>(tile_index % num_flows),
                               tile_index, output_offset, bytes);
      builder.add_op(OpKind::DeviceCopy, tile,
                     input_ref(config, input_offset),
                     output_ref(config, output_offset),
                     op_dtype(config, OpKind::DeviceCopy),
                     op_reduction(config, OpKind::DeviceCopy), {});
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
        TileRef send_tile =
            make_tile(config, config.rank,
                      static_cast<uint32_t>(tile_index % num_flows), tile_index,
                      offset, bytes);
        std::vector<uint32_t> send_deps;
        add_dep(send_deps, last_send_to_peer[static_cast<size_t>(peer)]);
        send_op = builder.add_op(
            OpKind::TransportSend, send_tile, input_ref(config, offset),
            peer_staging_ref(config, peer,
                             remote_peer_slot * config.tile_bytes),
            op_dtype(config, OpKind::TransportSend),
            op_reduction(config, OpKind::TransportSend), std::move(send_deps));
        last_send_to_peer[static_cast<size_t>(peer)] = send_op;
      }

      if (tile_index < recv_tiles) {
        size_t bytes = tile_size(recv_bytes, config.tile_bytes, tile_index);
        size_t offset =
            recv_offset + tile_offset(tile_index, config.tile_bytes);
        TileRef recv_tile = make_tile(
            config, peer, static_cast<uint32_t>(tile_index % num_flows),
            tile_index, offset, bytes);
        std::vector<uint32_t> recv_deps;
        add_dep(recv_deps, last_recv_from_peer[static_cast<size_t>(peer)]);
        add_dep(recv_deps, last_staging_consumer[peer_slot]);
        uint32_t recv_op = builder.add_op(
            OpKind::TransportRecv, recv_tile,
            peer_input_ref(config, peer, offset),
            staging_ref(config, staging_offset),
            op_dtype(config, OpKind::TransportRecv),
            op_reduction(config, OpKind::TransportRecv), std::move(recv_deps));
        last_recv_from_peer[static_cast<size_t>(peer)] = recv_op;

        std::vector<uint32_t> copy_deps;
        add_dep(copy_deps, send_op);
        add_dep(copy_deps, recv_op);
        uint32_t copy_op = builder.add_op(
            OpKind::DeviceCopy, recv_tile,
            staging_ref(config, staging_offset), output_ref(config, offset),
            op_dtype(config, OpKind::DeviceCopy),
            op_reduction(config, OpKind::DeviceCopy), std::move(copy_deps));
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
  plan.algorithm = AlgorithmKind::Pairwise;
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
  uint32_t num_flows = clamp_num_flows(config.num_flows, tiles_per_slice);
  plan.num_flows = num_flows;
  plan.input_split_bytes = input_splits;
  plan.output_split_bytes = output_splits;
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
      TileRef tile = make_tile(config, config.rank,
                               static_cast<uint32_t>(tile_index % num_flows),
                               tile_index, out_off, bytes);
      builder.add_op(OpKind::DeviceCopy, tile,
                     input_ref(config, in_off), output_ref(config, out_off),
                     op_dtype(config, OpKind::DeviceCopy),
                     op_reduction(config, OpKind::DeviceCopy), {});
    }
  }

  // Per-peer SM IPC: DeviceSendRemote + DeviceRecvRemote.
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
          TileRef send_tile = make_tile(config, config.rank,
                                        static_cast<uint32_t>(tile_index % num_flows),
                                        tile_index, off, stb);
          uint32_t send_op = builder.add_op(
              OpKind::DeviceSendRemote, send_tile,
              input_ref(config, off),
              peer_input_ref(config, peer, off),
              ScalarType::Int8, ReductionKind::None, {});
          builder.plan.ops.back().signal_seq = ts;
        }
        if (rtb > 0) {
          size_t off = recv_offset + tile_offset(tile_index, config.tile_bytes);
          TileRef recv_tile = make_tile(config, peer,
                                        static_cast<uint32_t>(tile_index % num_flows),
                                        tile_index, off, rtb);
          uint32_t recv_op = builder.add_op(
              OpKind::DeviceRecvRemote, recv_tile,
              peer_input_ref(config, peer, off),
              output_ref(config, off),
              ScalarType::Int8, ReductionKind::None, {});
          builder.plan.ops.back().signal_seq = ts;
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
    case OpKind::DeviceSendRemote:    return "sm_send";
    case OpKind::DeviceReduceRemote:  return "sm_reduce";
    case OpKind::DeviceRecvRemote:    return "sm_recv";
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
      << ", tensor_bytes=" << plan.tensor_bytes
      << ", tile_bytes=" << plan.tile_bytes
      << ", staging_bytes_required=" << plan.staging_bytes_required
      << ", ops=" << plan.ops.size() << ")\n";
  for (auto const& op : plan.ops) {
    oss << "  op " << op.op_id << " " << op_kind_name(op.kind)
        << " lane=" << op.tile.flow_index;
    int peer_rank = -1;
    if (op.src.rank >= 0) {
      peer_rank = op.src.rank;
    } else if (op.dst.rank >= 0) {
      peer_rank = op.dst.rank;
    }
    if (peer_rank >= 0) {
      oss << " peer=" << peer_rank;
    }
    if (!op.deps.empty()) {
      oss << " deps=[";
      for (size_t i = 0; i < op.deps.size(); ++i) {
        if (i != 0) oss << ",";
        oss << op.deps[i];
      }
      oss << "]";
    }
    if (op.kind == OpKind::TransportSend || op.kind == OpKind::DeviceCopy ||
        op.kind == OpKind::DeviceReduce) {
      oss << " src=" << buffer_kind_name(op.src.kind) << "[" << op.src.buffer_id
          << "]@" << op.src.offset_bytes << "+" << op.tile.size_bytes;
    }
    if (op.kind == OpKind::TransportRecv || op.kind == OpKind::DeviceCopy ||
        op.kind == OpKind::DeviceReduce) {
      oss << " dst=" << buffer_kind_name(op.dst.kind) << "[" << op.dst.buffer_id
          << "]@" << op.dst.offset_bytes << "+" << op.tile.size_bytes;
    }
    oss << "\n";
  }
  return oss.str();
}

}  // namespace CCL
}  // namespace UKernel
