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

size_t shard_offset(size_t bytes, int nranks, int owner_rank) {
  return static_cast<size_t>(owner_rank) * nominal_shard_bytes(bytes, nranks);
}

size_t shard_size(size_t bytes, int nranks, int owner_rank) {
  size_t offset = shard_offset(bytes, nranks, owner_rank);
  if (offset >= bytes) return 0;
  return std::min(nominal_shard_bytes(bytes, nranks), bytes - offset);
}

BufferRef make_buffer_ref(BufferKind kind, size_t offset_bytes) {
  BufferRef ref;
  ref.kind = kind;
  ref.offset_bytes = offset_bytes;
  return ref;
}

BufferRef tensor_ref(size_t offset_bytes) {
  return make_buffer_ref(BufferKind::Tensor, offset_bytes);
}

BufferRef staging_ref(size_t offset_bytes) {
  return make_buffer_ref(BufferKind::Staging, offset_bytes);
}

void add_dep(std::vector<uint32_t>& deps, uint32_t dep) {
  if (dep == kNoOp) return;
  if (std::find(deps.begin(), deps.end(), dep) == deps.end()) {
    deps.push_back(dep);
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
  if (request.tensor_bytes == 0) {
    throw std::invalid_argument(
        "collective plan tensor_bytes must be positive");
  }
  if (request.tile_bytes == 0) {
    throw std::invalid_argument("collective plan tile_bytes must be positive");
  }
}

CollectivePlan make_empty_plan(PlanRequest const& request) {
  CollectivePlan plan;
  plan.collective = request.collective;
  plan.algorithm = request.algorithm;
  plan.nranks = request.nranks;
  plan.rank = request.rank;
  plan.num_flows = request.num_flows;
  plan.tensor_bytes = request.tensor_bytes;
  plan.tile_bytes = request.tile_bytes;
  return plan;
}

struct PlanBuilder {
  explicit PlanBuilder(CollectivePlan plan_in) : plan(std::move(plan_in)) {}

  uint32_t add_op(PrimitiveOpKind kind, TileRef tile, BufferRef src,
                  BufferRef dst, int peer_rank, std::vector<uint32_t> deps) {
    PrimitiveOp op;
    op.op_id = next_op_id++;
    op.kind = kind;
    op.peer_rank = peer_rank;
    op.tile = tile;
    op.src = std::move(src);
    op.dst = std::move(dst);
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

CollectivePlan build_allreduce_ring_plan(PlanRequest const& request) {
  RingTopology ring{request.nranks};
  CollectivePlan plan = make_empty_plan(request);
  plan.algorithm = AlgorithmKind::Ring;
  plan.staging_bytes_required =
      static_cast<size_t>(request.num_flows) * request.tile_bytes;
  if (request.staging_bytes < plan.staging_bytes_required) {
    throw std::invalid_argument(
        "allreduce ring requires staging_bytes >= num_flows * tile_bytes");
  }

  PlanBuilder builder(std::move(plan));
  size_t shard_bytes =
      nominal_shard_bytes(request.tensor_bytes, request.nranks);
  size_t tiles_per_shard = ceil_div(shard_bytes, request.tile_bytes);

  std::vector<std::vector<uint32_t>> ready_ops(
      static_cast<size_t>(request.nranks),
      std::vector<uint32_t>(tiles_per_shard, kNoOp));
  std::vector<uint32_t> last_send_to_peer(static_cast<size_t>(request.nranks),
                                          kNoOp);
  std::vector<uint32_t> last_recv_from_peer(static_cast<size_t>(request.nranks),
                                            kNoOp);
  std::vector<uint32_t> last_staging_consumer(request.num_flows, kNoOp);

  for (uint32_t flow_slot = 0; flow_slot < request.num_flows; ++flow_slot) {
    size_t staging_offset = static_cast<size_t>(flow_slot) * request.tile_bytes;

    for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
      int send_owner = ring.wrap(request.rank - ring_step);
      int recv_owner = ring.wrap(request.rank - ring_step - 1);
      int send_peer = ring.next(request.rank);
      int recv_peer = ring.prev(request.rank);
      size_t send_bytes =
          shard_size(request.tensor_bytes, request.nranks, send_owner);
      size_t recv_bytes =
          shard_size(request.tensor_bytes, request.nranks, recv_owner);

      for (size_t tile_index = flow_slot; tile_index < tiles_per_shard;
           tile_index += request.num_flows) {
        size_t send_tile_bytes =
            tile_size(send_bytes, request.tile_bytes, tile_index);
        size_t recv_tile_bytes =
            tile_size(recv_bytes, request.tile_bytes, tile_index);

        if (send_tile_bytes > 0) {
          size_t offset =
              shard_offset(request.tensor_bytes, request.nranks, send_owner) +
              tile_offset(tile_index, request.tile_bytes);
          TileRef send_tile = make_tile(request, send_owner, flow_slot,
                                        tile_index, offset, send_tile_bytes);

          std::vector<uint32_t> deps;
          add_dep(deps, ready_ops[static_cast<size_t>(send_owner)][tile_index]);
          add_dep(deps, last_send_to_peer[static_cast<size_t>(send_peer)]);
          uint32_t send_op =
              builder.add_op(PrimitiveOpKind::Send, send_tile,
                             tensor_ref(send_tile.offset_bytes), BufferRef{},
                             send_peer, std::move(deps));
          last_send_to_peer[static_cast<size_t>(send_peer)] = send_op;
        }

        if (recv_tile_bytes > 0) {
          size_t offset =
              shard_offset(request.tensor_bytes, request.nranks, recv_owner) +
              tile_offset(tile_index, request.tile_bytes);
          TileRef recv_tile = make_tile(request, recv_owner, flow_slot,
                                        tile_index, offset, recv_tile_bytes);

          std::vector<uint32_t> recv_deps;
          add_dep(recv_deps,
                  last_recv_from_peer[static_cast<size_t>(recv_peer)]);
          add_dep(recv_deps, last_staging_consumer[flow_slot]);
          uint32_t recv_op = builder.add_op(
              PrimitiveOpKind::Recv, recv_tile, BufferRef{},
              staging_ref(staging_offset), recv_peer, std::move(recv_deps));
          last_recv_from_peer[static_cast<size_t>(recv_peer)] = recv_op;

          std::vector<uint32_t> reduce_deps;
          add_dep(reduce_deps, recv_op);
          uint32_t reduce_op = builder.add_op(
              PrimitiveOpKind::Reduce, recv_tile, staging_ref(staging_offset),
              tensor_ref(recv_tile.offset_bytes), -1, std::move(reduce_deps));
          ready_ops[static_cast<size_t>(recv_owner)][tile_index] = reduce_op;
          last_staging_consumer[flow_slot] = reduce_op;
        }
      }
    }
  }

  for (uint32_t flow_slot = 0; flow_slot < request.num_flows; ++flow_slot) {
    for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
      int send_owner = ring.wrap(request.rank - ring_step);
      int recv_owner = ring.wrap(request.rank - ring_step - 1);
      int send_peer = ring.next(request.rank);
      int recv_peer = ring.prev(request.rank);
      size_t send_bytes =
          shard_size(request.tensor_bytes, request.nranks, send_owner);
      size_t recv_bytes =
          shard_size(request.tensor_bytes, request.nranks, recv_owner);

      for (size_t tile_index = flow_slot; tile_index < tiles_per_shard;
           tile_index += request.num_flows) {
        size_t send_tile_bytes =
            tile_size(send_bytes, request.tile_bytes, tile_index);
        size_t recv_tile_bytes =
            tile_size(recv_bytes, request.tile_bytes, tile_index);

        if (send_tile_bytes > 0) {
          size_t offset =
              shard_offset(request.tensor_bytes, request.nranks, send_owner) +
              tile_offset(tile_index, request.tile_bytes);
          TileRef send_tile = make_tile(request, send_owner, flow_slot,
                                        tile_index, offset, send_tile_bytes);

          std::vector<uint32_t> deps;
          add_dep(deps, ready_ops[static_cast<size_t>(send_owner)][tile_index]);
          add_dep(deps, last_send_to_peer[static_cast<size_t>(send_peer)]);
          uint32_t send_op =
              builder.add_op(PrimitiveOpKind::Send, send_tile,
                             tensor_ref(send_tile.offset_bytes), BufferRef{},
                             send_peer, std::move(deps));
          last_send_to_peer[static_cast<size_t>(send_peer)] = send_op;
        }

        if (recv_tile_bytes > 0) {
          size_t offset =
              shard_offset(request.tensor_bytes, request.nranks, recv_owner) +
              tile_offset(tile_index, request.tile_bytes);
          TileRef recv_tile = make_tile(request, recv_owner, flow_slot,
                                        tile_index, offset, recv_tile_bytes);

          std::vector<uint32_t> recv_deps;
          add_dep(recv_deps,
                  last_recv_from_peer[static_cast<size_t>(recv_peer)]);
          uint32_t recv_op =
              builder.add_op(PrimitiveOpKind::Recv, recv_tile, BufferRef{},
                             tensor_ref(recv_tile.offset_bytes), recv_peer,
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
  plan.staging_bytes_required =
      static_cast<size_t>(request.nranks - 1) * request.tile_bytes;
  if (request.staging_bytes < plan.staging_bytes_required) {
    throw std::invalid_argument(
        "alltoall pairwise requires staging_bytes >= (nranks - 1) * "
        "tile_bytes");
  }

  PlanBuilder builder(std::move(plan));
  size_t slice_bytes =
      nominal_shard_bytes(request.tensor_bytes, request.nranks);
  size_t tiles_per_slice = ceil_div(slice_bytes, request.tile_bytes);

  std::vector<uint32_t> last_send_to_peer(static_cast<size_t>(request.nranks),
                                          kNoOp);
  std::vector<uint32_t> last_recv_from_peer(static_cast<size_t>(request.nranks),
                                            kNoOp);
  std::vector<uint32_t> last_staging_consumer(
      static_cast<size_t>(request.nranks > 0 ? request.nranks - 1 : 0), kNoOp);

  size_t peer_slot = 0;
  for (int peer = 0; peer < request.nranks; ++peer) {
    if (peer == request.rank) continue;

    size_t slice_offset =
        shard_offset(request.tensor_bytes, request.nranks, peer);
    size_t peer_slice_bytes =
        shard_size(request.tensor_bytes, request.nranks, peer);
    size_t staging_offset = peer_slot * request.tile_bytes;

    for (size_t tile_index = 0; tile_index < tiles_per_slice; ++tile_index) {
      size_t bytes =
          tile_size(peer_slice_bytes, request.tile_bytes, tile_index);
      if (bytes == 0) continue;

      TileRef tile = make_tile(
          request, peer, static_cast<uint32_t>(tile_index % request.num_flows),
          tile_index,
          slice_offset + tile_offset(tile_index, request.tile_bytes), bytes);

      std::vector<uint32_t> send_deps;
      add_dep(send_deps, last_send_to_peer[static_cast<size_t>(peer)]);
      uint32_t send_op = builder.add_op(
          PrimitiveOpKind::Send, tile, tensor_ref(tile.offset_bytes),
          BufferRef{}, peer, std::move(send_deps));
      last_send_to_peer[static_cast<size_t>(peer)] = send_op;

      std::vector<uint32_t> recv_deps;
      add_dep(recv_deps, last_recv_from_peer[static_cast<size_t>(peer)]);
      add_dep(recv_deps, last_staging_consumer[peer_slot]);
      uint32_t recv_op = builder.add_op(
          PrimitiveOpKind::Recv, tile, BufferRef{}, staging_ref(staging_offset),
          peer, std::move(recv_deps));
      last_recv_from_peer[static_cast<size_t>(peer)] = recv_op;

      std::vector<uint32_t> copy_deps;
      add_dep(copy_deps, send_op);
      add_dep(copy_deps, recv_op);
      uint32_t copy_op = builder.add_op(
          PrimitiveOpKind::Copy, tile, staging_ref(staging_offset),
          tensor_ref(tile.offset_bytes), -1, std::move(copy_deps));
      last_staging_consumer[peer_slot] = copy_op;
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
    case BufferKind::Tensor:
      return "tensor";
    case BufferKind::Staging:
      return "staging";
    case BufferKind::PeerTensor:
      return "peer-tensor";
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
    if (op.peer_rank >= 0) {
      oss << " peer=" << op.peer_rank;
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
      oss << " src=" << buffer_name(op.src.kind) << "@" << op.src.offset_bytes
          << "+" << op.tile.size_bytes;
    }
    if (op.kind == PrimitiveOpKind::Recv || op.kind == PrimitiveOpKind::Copy ||
        op.kind == PrimitiveOpKind::Reduce) {
      oss << " dst=" << buffer_name(op.dst.kind) << "@" << op.dst.offset_bytes
          << "+" << op.tile.size_bytes;
    }
    oss << "\n";
  }
  return oss.str();
}

}  // namespace CCL
}  // namespace UKernel
