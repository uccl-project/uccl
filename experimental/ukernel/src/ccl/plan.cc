#include "plan.h"
#include "topology.h"
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

size_t shard_offset(size_t bytes, int nranks, int owner_rank) {
  return static_cast<size_t>(owner_rank) * nominal_shard_bytes(bytes, nranks);
}

size_t shard_size(size_t bytes, int nranks, int owner_rank) {
  size_t offset = shard_offset(bytes, nranks, owner_rank);
  if (offset >= bytes) return 0;
  return std::min(nominal_shard_bytes(bytes, nranks), bytes - offset);
}

uint32_t clamp_num_flows(uint32_t requested_flows, size_t tiles_per_unit) {
  size_t bounded = std::min<size_t>(std::max<size_t>(1, requested_flows),
                                    std::max<size_t>(1, tiles_per_unit));
  return static_cast<uint32_t>(bounded);
}

BufferRef input_ref(PlanRequest const& request, size_t offset_bytes) {
  return local_buffer_ref(request.roles.buffer_id(CollectiveBufferRole::Input),
                          offset_bytes);
}

BufferRef output_ref(PlanRequest const& request, size_t offset_bytes) {
  return local_buffer_ref(request.roles.buffer_id(CollectiveBufferRole::Output),
                          offset_bytes);
}

BufferRef staging_ref(PlanRequest const& request, size_t offset_bytes) {
  return local_buffer_ref(
      request.roles.buffer_id(CollectiveBufferRole::Scratch), offset_bytes);
}

BufferRef peer_input_ref(PlanRequest const& request, int rank,
                         size_t offset_bytes) {
  return remote_buffer_ref(request.roles.buffer_id(CollectiveBufferRole::Input),
                           rank, offset_bytes);
}

BufferRef peer_staging_ref(PlanRequest const& request, int rank,
                           size_t offset_bytes) {
  return remote_buffer_ref(
      request.roles.buffer_id(CollectiveBufferRole::Scratch), rank,
      offset_bytes);
}

void add_dep(std::vector<uint32_t>& deps, uint32_t dep) {
  if (dep == kNoOp) return;
  if (std::find(deps.begin(), deps.end(), dep) == deps.end()) {
    deps.push_back(dep);
  }
}

size_t alltoall_input_bytes(PlanRequest const& request) {
  return request.input_bytes != 0 ? request.input_bytes : request.tensor_bytes;
}

size_t alltoall_output_bytes(PlanRequest const& request) {
  return request.output_bytes != 0 ? request.output_bytes
                                   : request.tensor_bytes;
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

void require_plan_request(PlanRequest const& request) {
  if (request.nranks < 2) {
    throw std::invalid_argument("collective plan requires at least two ranks");
  }
  if (request.rank < 0 || request.rank >= request.nranks) {
    throw std::invalid_argument("collective rank out of range");
  }
  if (request.num_flows == 0) {
    throw std::invalid_argument("collective plan requires at least one flow");
  }
  if (request.tile_bytes == 0) {
    throw std::invalid_argument("collective plan tile_bytes must be positive");
  }
  request.roles.validate();

  size_t elem_bytes = scalar_type_size(request.dtype);
  if (elem_bytes == 0) {
    throw std::invalid_argument(
        "collective plan dtype has invalid element size");
  }
  if (request.tile_bytes % elem_bytes != 0) {
    throw std::invalid_argument(
        "collective plan tile_bytes must be aligned to dtype size");
  }
  if (request.collective == CollectiveKind::AllReduce) {
    if (request.tensor_bytes == 0) {
      throw std::invalid_argument(
          "collective plan tensor_bytes must be positive");
    }
    if (request.tensor_bytes % elem_bytes != 0) {
      throw std::invalid_argument(
          "collective plan tensor_bytes must be aligned to dtype size");
    }
    return;
  }

  size_t input_bytes = alltoall_input_bytes(request);
  size_t output_bytes = alltoall_output_bytes(request);
  if (input_bytes == 0 || output_bytes == 0) {
    throw std::invalid_argument(
        "alltoall requires positive input/output tensor bytes");
  }
  if (input_bytes % elem_bytes != 0 || output_bytes % elem_bytes != 0) {
    throw std::invalid_argument(
        "alltoall input/output tensor bytes must align to dtype size");
  }

  bool has_input_splits = !request.input_split_bytes.empty();
  bool has_output_splits = !request.output_split_bytes.empty();
  if (has_input_splits != has_output_splits) {
    throw std::invalid_argument(
        "alltoall split configuration must provide both input and output "
        "splits");
  }
  if (!has_input_splits) {
    size_t denom = static_cast<size_t>(request.nranks) * elem_bytes;
    if (input_bytes % denom != 0 || output_bytes % denom != 0) {
      throw std::invalid_argument(
          "equal-split alltoall requires input/output tensor bytes divisible "
          "by nranks * dtype size");
    }
    return;
  }

  validate_alltoall_splits(request.input_split_bytes, request.nranks,
                           elem_bytes, input_bytes, "input");
  validate_alltoall_splits(request.output_split_bytes, request.nranks,
                           elem_bytes, output_bytes, "output");
}

CollectivePlan make_empty_plan(PlanRequest const& request) {
  CollectivePlan plan;
  plan.collective = request.collective;
  plan.algorithm = request.algorithm;
  plan.nranks = request.nranks;
  plan.rank = request.rank;
  plan.num_flows = request.num_flows;
  if (request.collective == CollectiveKind::AllToAll) {
    plan.input_bytes = alltoall_input_bytes(request);
    plan.output_bytes = alltoall_output_bytes(request);
    plan.tensor_bytes = std::max(plan.input_bytes, plan.output_bytes);
    plan.input_split_bytes = request.input_split_bytes;
    plan.output_split_bytes = request.output_split_bytes;
  } else {
    plan.tensor_bytes = request.tensor_bytes;
    plan.input_bytes = request.tensor_bytes;
    plan.output_bytes = request.tensor_bytes;
  }
  plan.tile_bytes = request.tile_bytes;
  plan.dtype = request.dtype;
  plan.reduction = request.reduction;
  return plan;
}

struct PlanBuilder {
  explicit PlanBuilder(CollectivePlan plan_in) : plan(std::move(plan_in)) {}

  uint32_t add_op(PrimitiveOpKind kind, TileRef tile, BufferRef src,
                  BufferRef dst, ScalarType dtype, ReductionKind reduction,
                  std::vector<uint32_t> deps) {
    PrimitiveOp op;
    op.op_id = next_op_id++;
    op.kind = kind;
    op.tile = tile;
    op.src = std::move(src);
    op.dst = std::move(dst);
    op.dtype = dtype;
    op.reduction = reduction;
    op.deps = std::move(deps);
    plan.ops.push_back(std::move(op));
    return plan.ops.back().op_id;
  }

  CollectivePlan plan;
  uint32_t next_op_id = 0;
};

TileRef make_tile(PlanRequest const& request, int owner_rank,
                  uint32_t flow_index, size_t tile_index, size_t offset_bytes,
                  size_t size_bytes) {
  TileRef tile;
  tile.owner_rank = static_cast<uint32_t>(owner_rank);
  tile.tile_index = static_cast<uint32_t>(tile_index);
  tile.flow_index = flow_index;
  tile.offset_bytes = offset_bytes;
  tile.size_bytes = size_bytes;
  return tile;
}

ScalarType op_dtype(PlanRequest const& request, PrimitiveOpKind kind) {
  switch (kind) {
    case PrimitiveOpKind::Reduce:
      return request.dtype;
    case PrimitiveOpKind::Copy:
      return ScalarType::Int8;
    case PrimitiveOpKind::Send:
    case PrimitiveOpKind::Recv:
      return ScalarType::UInt8;
  }
  return ScalarType::UInt8;
}

ReductionKind op_reduction(PlanRequest const& request, PrimitiveOpKind kind) {
  return kind == PrimitiveOpKind::Reduce ? request.reduction
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

CollectivePlan build_allreduce_ring_plan(PlanRequest const& request) {
  RingTopology ring{request.nranks};
  CollectivePlan plan = make_empty_plan(request);
  plan.algorithm = AlgorithmKind::Ring;
  size_t elem_bytes = scalar_type_size(request.dtype);
  size_t shard_bytes = max_balanced_shard_bytes(request.tensor_bytes,
                                                elem_bytes, request.nranks);
  size_t tiles_per_shard = ceil_div(shard_bytes, request.tile_bytes);
  uint32_t num_flows = normalized_num_flows(
      request.collective, request.nranks, request.tensor_bytes,
      request.tile_bytes, request.dtype, request.num_flows);
  plan.num_flows = num_flows;
  plan.staging_bytes_required =
      static_cast<size_t>(num_flows) * request.tile_bytes;
  if (request.staging_bytes < plan.staging_bytes_required) {
    throw std::invalid_argument(
        "allreduce ring requires staging_bytes >= num_flows * tile_bytes");
  }

  PlanBuilder builder(std::move(plan));

  std::vector<std::vector<uint32_t>> ready_ops(
      static_cast<size_t>(request.nranks),
      std::vector<uint32_t>(tiles_per_shard, kNoOp));
  std::vector<uint32_t> last_send_to_peer(static_cast<size_t>(request.nranks),
                                          kNoOp);
  std::vector<uint32_t> last_recv_from_peer(static_cast<size_t>(request.nranks),
                                            kNoOp);
  std::vector<uint32_t> last_staging_consumer(num_flows, kNoOp);

  for (uint32_t flow_slot = 0; flow_slot < num_flows; ++flow_slot) {
    size_t staging_offset = static_cast<size_t>(flow_slot) * request.tile_bytes;

    for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
      int send_owner = ring.wrap(request.rank - ring_step);
      int recv_owner = ring.wrap(request.rank - ring_step - 1);
      int send_peer = ring.next(request.rank);
      int recv_peer = ring.prev(request.rank);
      size_t send_bytes = balanced_shard_size_bytes(
          request.tensor_bytes, elem_bytes, request.nranks, send_owner);
      size_t recv_bytes = balanced_shard_size_bytes(
          request.tensor_bytes, elem_bytes, request.nranks, recv_owner);

      for (size_t tile_index = flow_slot; tile_index < tiles_per_shard;
           tile_index += num_flows) {
        size_t send_tile_bytes =
            tile_size(send_bytes, request.tile_bytes, tile_index);
        size_t recv_tile_bytes =
            tile_size(recv_bytes, request.tile_bytes, tile_index);

        if (send_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(request.tensor_bytes, elem_bytes,
                                          request.nranks, send_owner) +
              tile_offset(tile_index, request.tile_bytes);
          TileRef send_tile = make_tile(request, send_owner, flow_slot,
                                        tile_index, offset, send_tile_bytes);

          std::vector<uint32_t> deps;
          add_dep(deps, ready_ops[static_cast<size_t>(send_owner)][tile_index]);
          add_dep(deps, last_send_to_peer[static_cast<size_t>(send_peer)]);
          uint32_t send_op = builder.add_op(
              PrimitiveOpKind::Send, send_tile,
              input_ref(request, send_tile.offset_bytes),
              peer_staging_ref(request, send_peer, staging_offset),
              op_dtype(request, PrimitiveOpKind::Send),
              op_reduction(request, PrimitiveOpKind::Send), std::move(deps));
          last_send_to_peer[static_cast<size_t>(send_peer)] = send_op;
        }

        if (recv_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(request.tensor_bytes, elem_bytes,
                                          request.nranks, recv_owner) +
              tile_offset(tile_index, request.tile_bytes);
          TileRef recv_tile = make_tile(request, recv_owner, flow_slot,
                                        tile_index, offset, recv_tile_bytes);

          std::vector<uint32_t> recv_deps;
          add_dep(recv_deps,
                  last_recv_from_peer[static_cast<size_t>(recv_peer)]);
          add_dep(recv_deps, last_staging_consumer[flow_slot]);
          uint32_t recv_op = builder.add_op(
              PrimitiveOpKind::Recv, recv_tile,
              peer_input_ref(request, recv_peer, recv_tile.offset_bytes),
              staging_ref(request, staging_offset),
              op_dtype(request, PrimitiveOpKind::Recv),
              op_reduction(request, PrimitiveOpKind::Recv),
              std::move(recv_deps));
          last_recv_from_peer[static_cast<size_t>(recv_peer)] = recv_op;

          std::vector<uint32_t> reduce_deps;
          add_dep(reduce_deps, recv_op);
          uint32_t reduce_op =
              builder.add_op(PrimitiveOpKind::Reduce, recv_tile,
                             staging_ref(request, staging_offset),
                             input_ref(request, recv_tile.offset_bytes),
                             op_dtype(request, PrimitiveOpKind::Reduce),
                             op_reduction(request, PrimitiveOpKind::Reduce),
                             std::move(reduce_deps));
          ready_ops[static_cast<size_t>(recv_owner)][tile_index] = reduce_op;
          last_staging_consumer[flow_slot] = reduce_op;
        }
      }
    }
  }

  for (uint32_t flow_slot = 0; flow_slot < num_flows; ++flow_slot) {
    for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
      // After reduce-scatter, rank r owns the fully reduced shard
      // (r + 1) mod nranks. Allgather then circulates those completed shards
      // around the ring in that order.
      int send_owner = ring.wrap(request.rank + 1 - ring_step);
      int recv_owner = ring.wrap(request.rank - ring_step);
      int send_peer = ring.next(request.rank);
      int recv_peer = ring.prev(request.rank);
      size_t send_bytes = balanced_shard_size_bytes(
          request.tensor_bytes, elem_bytes, request.nranks, send_owner);
      size_t recv_bytes = balanced_shard_size_bytes(
          request.tensor_bytes, elem_bytes, request.nranks, recv_owner);

      for (size_t tile_index = flow_slot; tile_index < tiles_per_shard;
           tile_index += num_flows) {
        size_t send_tile_bytes =
            tile_size(send_bytes, request.tile_bytes, tile_index);
        size_t recv_tile_bytes =
            tile_size(recv_bytes, request.tile_bytes, tile_index);

        if (send_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(request.tensor_bytes, elem_bytes,
                                          request.nranks, send_owner) +
              tile_offset(tile_index, request.tile_bytes);
          TileRef send_tile = make_tile(request, send_owner, flow_slot,
                                        tile_index, offset, send_tile_bytes);

          std::vector<uint32_t> deps;
          add_dep(deps, ready_ops[static_cast<size_t>(send_owner)][tile_index]);
          add_dep(deps, last_send_to_peer[static_cast<size_t>(send_peer)]);
          uint32_t send_op = builder.add_op(
              PrimitiveOpKind::Send, send_tile,
              input_ref(request, send_tile.offset_bytes),
              peer_input_ref(request, send_peer, send_tile.offset_bytes),
              op_dtype(request, PrimitiveOpKind::Send),
              op_reduction(request, PrimitiveOpKind::Send), std::move(deps));
          last_send_to_peer[static_cast<size_t>(send_peer)] = send_op;
        }

        if (recv_tile_bytes > 0) {
          size_t offset =
              balanced_shard_offset_bytes(request.tensor_bytes, elem_bytes,
                                          request.nranks, recv_owner) +
              tile_offset(tile_index, request.tile_bytes);
          TileRef recv_tile = make_tile(request, recv_owner, flow_slot,
                                        tile_index, offset, recv_tile_bytes);

          std::vector<uint32_t> recv_deps;
          add_dep(recv_deps,
                  last_recv_from_peer[static_cast<size_t>(recv_peer)]);
          uint32_t recv_op = builder.add_op(
              PrimitiveOpKind::Recv, recv_tile,
              peer_input_ref(request, recv_peer, recv_tile.offset_bytes),
              input_ref(request, recv_tile.offset_bytes),
              op_dtype(request, PrimitiveOpKind::Recv),
              op_reduction(request, PrimitiveOpKind::Recv),
              std::move(recv_deps));
          last_recv_from_peer[static_cast<size_t>(recv_peer)] = recv_op;
          ready_ops[static_cast<size_t>(recv_owner)][tile_index] = recv_op;
        }
      }
    }
  }

  return std::move(builder.plan);
}

CollectivePlan build_alltoall_pairwise_plan(PlanRequest const& request) {
  CollectivePlan plan = make_empty_plan(request);
  plan.algorithm = AlgorithmKind::Pairwise;
  size_t input_bytes = alltoall_input_bytes(request);
  size_t output_bytes = alltoall_output_bytes(request);

  std::vector<size_t> input_splits =
      request.input_split_bytes.empty()
          ? equal_alltoall_splits(input_bytes, request.nranks)
          : request.input_split_bytes;
  std::vector<size_t> output_splits =
      request.output_split_bytes.empty()
          ? equal_alltoall_splits(output_bytes, request.nranks)
          : request.output_split_bytes;
  std::vector<size_t> input_prefix = prefix_bytes(input_splits);
  std::vector<size_t> output_prefix = prefix_bytes(output_splits);

  size_t max_slice_bytes = 0;
  for (size_t bytes : input_splits) {
    max_slice_bytes = std::max(max_slice_bytes, bytes);
  }
  for (size_t bytes : output_splits) {
    max_slice_bytes = std::max(max_slice_bytes, bytes);
  }
  size_t tiles_per_slice = ceil_div(max_slice_bytes, request.tile_bytes);
  uint32_t num_flows = clamp_num_flows(request.num_flows, tiles_per_slice);
  plan.num_flows = num_flows;
  plan.input_split_bytes = input_splits;
  plan.output_split_bytes = output_splits;
  plan.staging_bytes_required =
      static_cast<size_t>(request.nranks - 1) * request.tile_bytes;
  if (request.staging_bytes < plan.staging_bytes_required) {
    throw std::invalid_argument(
        "alltoall pairwise requires staging_bytes >= (nranks - 1) * "
        "tile_bytes");
  }

  PlanBuilder builder(std::move(plan));

  std::vector<uint32_t> last_send_to_peer(static_cast<size_t>(request.nranks),
                                          kNoOp);
  std::vector<uint32_t> last_recv_from_peer(static_cast<size_t>(request.nranks),
                                            kNoOp);
  std::vector<uint32_t> last_staging_consumer(
      static_cast<size_t>(request.nranks > 0 ? request.nranks - 1 : 0), kNoOp);

  size_t self_input_offset = input_prefix[static_cast<size_t>(request.rank)];
  size_t self_output_offset = output_prefix[static_cast<size_t>(request.rank)];
  size_t self_slice_bytes = input_splits[static_cast<size_t>(request.rank)];
  if (self_slice_bytes != output_splits[static_cast<size_t>(request.rank)]) {
    throw std::invalid_argument(
        "alltoall self split size must match between input and output");
  }
  if (self_slice_bytes != 0 &&
      request.roles.input_buffer_id != request.roles.output_buffer_id) {
    size_t self_tiles = ceil_div(self_slice_bytes, request.tile_bytes);
    for (size_t tile_index = 0; tile_index < self_tiles; ++tile_index) {
      size_t bytes =
          tile_size(self_slice_bytes, request.tile_bytes, tile_index);
      if (bytes == 0) continue;
      size_t input_offset =
          self_input_offset + tile_offset(tile_index, request.tile_bytes);
      size_t output_offset =
          self_output_offset + tile_offset(tile_index, request.tile_bytes);
      TileRef tile = make_tile(request, request.rank,
                               static_cast<uint32_t>(tile_index % num_flows),
                               tile_index, output_offset, bytes);
      builder.add_op(PrimitiveOpKind::Copy, tile,
                     input_ref(request, input_offset),
                     output_ref(request, output_offset),
                     op_dtype(request, PrimitiveOpKind::Copy),
                     op_reduction(request, PrimitiveOpKind::Copy), {});
    }
  }

  size_t peer_slot = 0;
  for (int peer = 0; peer < request.nranks; ++peer) {
    if (peer == request.rank) continue;
    size_t remote_peer_slot = static_cast<size_t>(
        request.rank < peer ? request.rank : request.rank - 1);

    size_t send_offset = input_prefix[static_cast<size_t>(peer)];
    size_t send_bytes = input_splits[static_cast<size_t>(peer)];
    size_t recv_offset = output_prefix[static_cast<size_t>(peer)];
    size_t recv_bytes = output_splits[static_cast<size_t>(peer)];
    size_t staging_offset = peer_slot * request.tile_bytes;
    size_t send_tiles = ceil_div(send_bytes, request.tile_bytes);
    size_t recv_tiles = ceil_div(recv_bytes, request.tile_bytes);
    size_t tiles = std::max(send_tiles, recv_tiles);

    for (size_t tile_index = 0; tile_index < tiles; ++tile_index) {
      uint32_t send_op = kNoOp;
      if (tile_index < send_tiles) {
        size_t bytes = tile_size(send_bytes, request.tile_bytes, tile_index);
        size_t offset =
            send_offset + tile_offset(tile_index, request.tile_bytes);
        TileRef send_tile =
            make_tile(request, request.rank,
                      static_cast<uint32_t>(tile_index % num_flows), tile_index,
                      offset, bytes);
        std::vector<uint32_t> send_deps;
        add_dep(send_deps, last_send_to_peer[static_cast<size_t>(peer)]);
        send_op = builder.add_op(
            PrimitiveOpKind::Send, send_tile, input_ref(request, offset),
            peer_staging_ref(request, peer,
                             remote_peer_slot * request.tile_bytes),
            op_dtype(request, PrimitiveOpKind::Send),
            op_reduction(request, PrimitiveOpKind::Send), std::move(send_deps));
        last_send_to_peer[static_cast<size_t>(peer)] = send_op;
      }

      if (tile_index < recv_tiles) {
        size_t bytes = tile_size(recv_bytes, request.tile_bytes, tile_index);
        size_t offset =
            recv_offset + tile_offset(tile_index, request.tile_bytes);
        TileRef recv_tile = make_tile(
            request, peer, static_cast<uint32_t>(tile_index % num_flows),
            tile_index, offset, bytes);
        std::vector<uint32_t> recv_deps;
        add_dep(recv_deps, last_recv_from_peer[static_cast<size_t>(peer)]);
        add_dep(recv_deps, last_staging_consumer[peer_slot]);
        uint32_t recv_op = builder.add_op(
            PrimitiveOpKind::Recv, recv_tile,
            peer_input_ref(request, peer, offset),
            staging_ref(request, staging_offset),
            op_dtype(request, PrimitiveOpKind::Recv),
            op_reduction(request, PrimitiveOpKind::Recv), std::move(recv_deps));
        last_recv_from_peer[static_cast<size_t>(peer)] = recv_op;

        std::vector<uint32_t> copy_deps;
        add_dep(copy_deps, send_op);
        add_dep(copy_deps, recv_op);
        uint32_t copy_op = builder.add_op(
            PrimitiveOpKind::Copy, recv_tile,
            staging_ref(request, staging_offset), output_ref(request, offset),
            op_dtype(request, PrimitiveOpKind::Copy),
            op_reduction(request, PrimitiveOpKind::Copy), std::move(copy_deps));
        last_staging_consumer[peer_slot] = copy_op;
      }
    }

    ++peer_slot;
  }

  return std::move(builder.plan);
}

char const* primitive_name(PrimitiveOpKind kind) {
  switch (kind) {
    case PrimitiveOpKind::Send:
      return "send";
    case PrimitiveOpKind::Recv:
      return "recv";
    case PrimitiveOpKind::Copy:
      return "copy";
    case PrimitiveOpKind::Reduce:
      return "reduce";
  }
  return "unknown";
}

char const* buffer_name(BufferKind kind) {
  switch (kind) {
    case BufferKind::Local:
      return "local";
    case BufferKind::Remote:
      return "remote";
  }
  return "unknown";
}

}  // namespace

CollectivePlan build_plan(PlanRequest const& request) {
  require_plan_request(request);
  switch (request.collective) {
    case CollectiveKind::AllReduce:
      if (request.algorithm != AlgorithmKind::Ring) {
        throw std::invalid_argument(
            "allreduce currently supports only ring algorithm");
      }
      return build_allreduce_ring_plan(request);
    case CollectiveKind::AllToAll:
      if (request.algorithm != AlgorithmKind::Pairwise) {
        throw std::invalid_argument(
            "alltoall currently supports only pairwise algorithm");
      }
      return build_alltoall_pairwise_plan(request);
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
    oss << "  op " << op.op_id << " " << primitive_name(op.kind)
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
    if (op.kind == PrimitiveOpKind::Send || op.kind == PrimitiveOpKind::Copy ||
        op.kind == PrimitiveOpKind::Reduce) {
      oss << " src=" << buffer_name(op.src.kind) << "[" << op.src.buffer_id
          << "]@" << op.src.offset_bytes << "+" << op.tile.size_bytes;
    }
    if (op.kind == PrimitiveOpKind::Recv || op.kind == PrimitiveOpKind::Copy ||
        op.kind == PrimitiveOpKind::Reduce) {
      oss << " dst=" << buffer_name(op.dst.kind) << "[" << op.dst.buffer_id
          << "]@" << op.dst.offset_bytes << "+" << op.tile.size_bytes;
    }
    oss << "\n";
  }
  return oss.str();
}

}  // namespace CCL
}  // namespace UKernel
