#include "plan.h"
#include "topology.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace UKernel {
namespace CCL {

namespace {

size_t ceil_div(size_t a, size_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

size_t chunk_offset(size_t chunk_index, size_t chunk_bytes) {
  return chunk_index * chunk_bytes;
}

size_t chunk_size(size_t bytes, size_t chunk_bytes, size_t chunk_index) {
  size_t offset = chunk_offset(chunk_index, chunk_bytes);
  if (offset >= bytes) return 0;
  return std::min(chunk_bytes, bytes - offset);
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

ExecutionOp make_send_op(uint32_t op_id, int peer_rank, ChunkRange const& chunk,
                         MemoryRef src, std::vector<uint32_t> deps = {}) {
  ExecutionOp op;
  op.op_id = op_id;
  op.kind = ExecutionOpKind::Send;
  op.peer_rank = peer_rank;
  op.chunk = chunk;
  op.src = src;
  op.deps = std::move(deps);
  return op;
}

ExecutionOp make_recv_op(uint32_t op_id, int peer_rank, ChunkRange const& chunk,
                         MemoryRef dst, std::vector<uint32_t> deps = {}) {
  ExecutionOp op;
  op.op_id = op_id;
  op.kind = ExecutionOpKind::Recv;
  op.peer_rank = peer_rank;
  op.chunk = chunk;
  op.dst = dst;
  op.deps = std::move(deps);
  return op;
}

ExecutionOp make_copy_op(uint32_t op_id, ChunkRange const& chunk, MemoryRef src,
                         MemoryRef dst, std::vector<uint32_t> deps = {}) {
  ExecutionOp op;
  op.op_id = op_id;
  op.kind = ExecutionOpKind::Copy;
  op.chunk = chunk;
  op.src = src;
  op.dst = dst;
  op.deps = std::move(deps);
  return op;
}

ExecutionOp make_reduce_op(uint32_t op_id, ChunkRange const& chunk, MemoryRef src,
                           MemoryRef dst, std::vector<uint32_t> deps = {}) {
  ExecutionOp op;
  op.op_id = op_id;
  op.kind = ExecutionOpKind::Reduce;
  op.chunk = chunk;
  op.src = src;
  op.dst = dst;
  op.deps = std::move(deps);
  return op;
}

CollectivePlan build_allreduce_ring_plan(PlanRequest const& request) {
  RingTopology ring{request.nranks};
  CollectivePlan plan;
  plan.collective = request.collective;
  plan.algorithm = request.algorithm;
  plan.nranks = request.nranks;
  plan.rank = request.rank;
  plan.channels = request.channels;
  plan.bytes_per_rank = request.bytes_per_rank;
  plan.chunk_bytes = request.chunk_bytes;

  size_t shard_bytes = nominal_shard_bytes(request.bytes_per_rank, request.nranks);
  size_t chunks_per_shard = ceil_div(shard_bytes, request.chunk_bytes);
  uint32_t next_step_id = 0;
  uint32_t next_op_id = 0;
  std::vector<int32_t> last_step_for_channel(request.channels, -1);

  for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
    int send_owner = ring.wrap(request.rank - ring_step);
    int recv_owner = ring.wrap(request.rank - ring_step - 1);
    size_t send_bytes = shard_size(request.bytes_per_rank, request.nranks, send_owner);
    size_t recv_bytes = shard_size(request.bytes_per_rank, request.nranks, recv_owner);

    for (size_t chunk_index = 0; chunk_index < chunks_per_shard; ++chunk_index) {
      size_t send_chunk_bytes = chunk_size(send_bytes, request.chunk_bytes, chunk_index);
      size_t recv_chunk_bytes = chunk_size(recv_bytes, request.chunk_bytes, chunk_index);
      if (send_chunk_bytes == 0 && recv_chunk_bytes == 0) continue;

      ChunkRange chunk;
      chunk.owner_rank = static_cast<uint32_t>(recv_owner);
      chunk.chunk_index = static_cast<uint32_t>(chunk_index);
      chunk.channel_id = static_cast<uint32_t>(chunk_index % request.channels);
      chunk.offset_bytes =
          shard_offset(request.bytes_per_rank, request.nranks, recv_owner) +
          chunk_offset(chunk_index, request.chunk_bytes);
      chunk.size_bytes = recv_chunk_bytes;

      CollectiveStep step;
      step.step_id = next_step_id++;
      step.phase = StepPhase::ReduceScatter;
      step.chunk = chunk;

      int32_t pred = last_step_for_channel[chunk.channel_id];
      if (pred >= 0) step.predecessors.push_back(static_cast<uint32_t>(pred));

      if (send_chunk_bytes > 0) {
        ChunkRange send_chunk = chunk;
        send_chunk.owner_rank = static_cast<uint32_t>(send_owner);
        send_chunk.offset_bytes =
            shard_offset(request.bytes_per_rank, request.nranks, send_owner) +
            chunk_offset(chunk_index, request.chunk_bytes);
        send_chunk.size_bytes = send_chunk_bytes;
        step.ops.push_back(make_send_op(
            next_op_id++, ring.next(request.rank), send_chunk,
            MemoryRef{MemorySlot::SymmetricTensor, -1, send_chunk.offset_bytes}));
      }

      if (recv_chunk_bytes > 0) {
        uint32_t recv_op_id = next_op_id++;
        step.ops.push_back(make_recv_op(
            recv_op_id, ring.prev(request.rank), chunk,
            MemoryRef{MemorySlot::RecvStaging, -1,
                      chunk_offset(chunk_index, request.chunk_bytes)}));
        step.ops.push_back(make_reduce_op(
            next_op_id++, chunk,
            MemoryRef{MemorySlot::RecvStaging, -1,
                      chunk_offset(chunk_index, request.chunk_bytes)},
            MemoryRef{MemorySlot::SymmetricTensor, -1, chunk.offset_bytes},
            {recv_op_id}));
      }

      last_step_for_channel[chunk.channel_id] = static_cast<int32_t>(step.step_id);
      plan.steps.push_back(std::move(step));
    }
  }

  for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
    int send_owner = ring.wrap(request.rank - ring_step);
    int recv_owner = ring.wrap(request.rank - ring_step - 1);
    size_t send_bytes = shard_size(request.bytes_per_rank, request.nranks, send_owner);
    size_t recv_bytes = shard_size(request.bytes_per_rank, request.nranks, recv_owner);

    for (size_t chunk_index = 0; chunk_index < chunks_per_shard; ++chunk_index) {
      size_t send_chunk_bytes = chunk_size(send_bytes, request.chunk_bytes, chunk_index);
      size_t recv_chunk_bytes = chunk_size(recv_bytes, request.chunk_bytes, chunk_index);
      if (send_chunk_bytes == 0 && recv_chunk_bytes == 0) continue;

      ChunkRange chunk;
      chunk.owner_rank = static_cast<uint32_t>(recv_owner);
      chunk.chunk_index = static_cast<uint32_t>(chunk_index);
      chunk.channel_id = static_cast<uint32_t>(chunk_index % request.channels);
      chunk.offset_bytes =
          shard_offset(request.bytes_per_rank, request.nranks, recv_owner) +
          chunk_offset(chunk_index, request.chunk_bytes);
      chunk.size_bytes = recv_chunk_bytes;

      CollectiveStep step;
      step.step_id = next_step_id++;
      step.phase = StepPhase::AllGather;
      step.chunk = chunk;

      int32_t pred = last_step_for_channel[chunk.channel_id];
      if (pred >= 0) step.predecessors.push_back(static_cast<uint32_t>(pred));

      if (send_chunk_bytes > 0) {
        ChunkRange send_chunk = chunk;
        send_chunk.owner_rank = static_cast<uint32_t>(send_owner);
        send_chunk.offset_bytes =
            shard_offset(request.bytes_per_rank, request.nranks, send_owner) +
            chunk_offset(chunk_index, request.chunk_bytes);
        send_chunk.size_bytes = send_chunk_bytes;
        step.ops.push_back(make_send_op(
            next_op_id++, ring.next(request.rank), send_chunk,
            MemoryRef{MemorySlot::SymmetricTensor, -1, send_chunk.offset_bytes}));
      }

      if (recv_chunk_bytes > 0) {
        step.ops.push_back(make_recv_op(
            next_op_id++, ring.prev(request.rank), chunk,
            MemoryRef{MemorySlot::SymmetricTensor, -1, chunk.offset_bytes}));
      }

      last_step_for_channel[chunk.channel_id] = static_cast<int32_t>(step.step_id);
      plan.steps.push_back(std::move(step));
    }
  }

  return plan;
}

CollectivePlan build_alltoall_ring_plan(PlanRequest const& request) {
  RingTopology ring{request.nranks};
  CollectivePlan plan;
  plan.collective = request.collective;
  plan.algorithm = request.algorithm;
  plan.nranks = request.nranks;
  plan.rank = request.rank;
  plan.channels = request.channels;
  plan.bytes_per_rank = request.bytes_per_rank;
  plan.chunk_bytes = request.chunk_bytes;

  size_t slice_bytes = nominal_shard_bytes(request.bytes_per_rank, request.nranks);
  size_t chunks_per_slice = ceil_div(slice_bytes, request.chunk_bytes);
  uint32_t next_step_id = 0;
  uint32_t next_op_id = 0;
  std::vector<int32_t> last_step_for_channel(request.channels, -1);
  std::vector<std::vector<uint32_t>> send_op_ids(
      static_cast<size_t>(request.nranks),
      std::vector<uint32_t>(chunks_per_slice, 0));
  std::vector<std::vector<uint32_t>> recv_op_ids(
      static_cast<size_t>(request.nranks),
      std::vector<uint32_t>(chunks_per_slice, 0));

  for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
    int send_peer = ring.wrap(request.rank + ring_step + 1);
    int recv_peer = ring.wrap(request.rank - ring_step - 1);
    size_t send_bytes = shard_size(request.bytes_per_rank, request.nranks, send_peer);
    size_t recv_bytes = shard_size(request.bytes_per_rank, request.nranks, recv_peer);

    for (size_t chunk_index = 0; chunk_index < chunks_per_slice; ++chunk_index) {
      size_t send_chunk_bytes = chunk_size(send_bytes, request.chunk_bytes, chunk_index);
      size_t recv_chunk_bytes = chunk_size(recv_bytes, request.chunk_bytes, chunk_index);
      if (send_chunk_bytes == 0 && recv_chunk_bytes == 0) continue;

      ChunkRange chunk;
      chunk.owner_rank = static_cast<uint32_t>(recv_peer);
      chunk.chunk_index = static_cast<uint32_t>(chunk_index);
      chunk.channel_id = static_cast<uint32_t>(chunk_index % request.channels);
      chunk.offset_bytes =
          shard_offset(request.bytes_per_rank, request.nranks, recv_peer) +
          chunk_offset(chunk_index, request.chunk_bytes);
      chunk.size_bytes = recv_chunk_bytes;

      CollectiveStep step;
      step.step_id = next_step_id++;
      step.phase = StepPhase::AllToAll;
      step.chunk = chunk;

      int32_t pred = last_step_for_channel[chunk.channel_id];
      if (pred >= 0) step.predecessors.push_back(static_cast<uint32_t>(pred));

      if (send_chunk_bytes > 0) {
        ChunkRange send_chunk = chunk;
        send_chunk.owner_rank = static_cast<uint32_t>(request.rank);
        send_chunk.offset_bytes =
            shard_offset(request.bytes_per_rank, request.nranks, send_peer) +
            chunk_offset(chunk_index, request.chunk_bytes);
        send_chunk.size_bytes = send_chunk_bytes;
        uint32_t send_op_id = next_op_id++;
        step.ops.push_back(make_send_op(
            send_op_id, send_peer, send_chunk,
            MemoryRef{MemorySlot::SymmetricTensor, -1, send_chunk.offset_bytes}));
        send_op_ids[static_cast<size_t>(send_peer)][chunk_index] = send_op_id;
      }

      if (recv_chunk_bytes > 0) {
        uint32_t recv_op_id = next_op_id++;
        step.ops.push_back(make_recv_op(
            recv_op_id, recv_peer, chunk,
            MemoryRef{MemorySlot::RecvStaging, -1, chunk.offset_bytes}));
        recv_op_ids[static_cast<size_t>(recv_peer)][chunk_index] = recv_op_id;
      }

      last_step_for_channel[chunk.channel_id] = static_cast<int32_t>(step.step_id);
      plan.steps.push_back(std::move(step));
    }
  }

  for (int peer = 0; peer < request.nranks; ++peer) {
    if (peer == request.rank) continue;
    size_t peer_bytes = shard_size(request.bytes_per_rank, request.nranks, peer);
    for (size_t chunk_index = 0; chunk_index < chunks_per_slice; ++chunk_index) {
      size_t copy_bytes = chunk_size(peer_bytes, request.chunk_bytes, chunk_index);
      if (copy_bytes == 0) continue;

      ChunkRange chunk;
      chunk.owner_rank = static_cast<uint32_t>(peer);
      chunk.chunk_index = static_cast<uint32_t>(chunk_index);
      chunk.channel_id = static_cast<uint32_t>(chunk_index % request.channels);
      chunk.offset_bytes =
          shard_offset(request.bytes_per_rank, request.nranks, peer) +
          chunk_offset(chunk_index, request.chunk_bytes);
      chunk.size_bytes = copy_bytes;

      CollectiveStep step;
      step.step_id = next_step_id++;
      step.phase = StepPhase::AllToAll;
      step.chunk = chunk;
      std::vector<uint32_t> deps;

      uint32_t recv_dep =
          recv_op_ids[static_cast<size_t>(peer)][chunk_index];
      uint32_t send_dep =
          send_op_ids[static_cast<size_t>(peer)][chunk_index];
      if (recv_dep != 0) deps.push_back(recv_dep);
      if (send_dep != 0) deps.push_back(send_dep);

      step.ops.push_back(make_copy_op(
          next_op_id++, chunk,
          MemoryRef{MemorySlot::RecvStaging, -1, chunk.offset_bytes},
          MemoryRef{MemorySlot::SymmetricTensor, -1, chunk.offset_bytes},
          std::move(deps)));
      plan.steps.push_back(std::move(step));
    }
  }

  return plan;
}

char const* collective_name(CollectiveKind kind) {
  switch (kind) {
    case CollectiveKind::AllGather:
      return "AllGather";
    case CollectiveKind::AllReduce:
      return "AllReduce";
    case CollectiveKind::ReduceScatter:
      return "ReduceScatter";
    case CollectiveKind::Broadcast:
      return "Broadcast";
    case CollectiveKind::AllToAll:
      return "AllToAll";
  }
  return "Unknown";
}

char const* op_name(ExecutionOpKind kind) {
  switch (kind) {
    case ExecutionOpKind::Send:
      return "Send";
    case ExecutionOpKind::Recv:
      return "Recv";
    case ExecutionOpKind::Copy:
      return "Copy";
    case ExecutionOpKind::Reduce:
      return "Reduce";
    case ExecutionOpKind::EventWait:
      return "EventWait";
    case ExecutionOpKind::Barrier:
      return "Barrier";
  }
  return "Unknown";
}

char const* slot_name(MemorySlot slot) {
  switch (slot) {
    case MemorySlot::SymmetricTensor:
      return "tensor";
    case MemorySlot::RecvStaging:
      return "recv_staging";
  }
  return "unknown";
}

}  // namespace

CollectivePlan build_plan(PlanRequest const& request) {
  if (request.algorithm != AlgorithmKind::Ring) {
    throw std::invalid_argument("Only ring algorithm is supported");
  }
  if (request.nranks < 2) {
    throw std::invalid_argument("nranks must be >= 2");
  }
  if (request.rank < 0 || request.rank >= request.nranks) {
    throw std::invalid_argument("rank out of range");
  }
  if (request.channels == 0) {
    throw std::invalid_argument("channels must be >= 1");
  }
  if (request.chunk_bytes == 0) {
    throw std::invalid_argument("chunk_bytes must be >= 1");
  }

  switch (request.collective) {
    case CollectiveKind::AllReduce:
      return build_allreduce_ring_plan(request);
    case CollectiveKind::AllToAll:
      return build_alltoall_ring_plan(request);
    case CollectiveKind::AllGather:
    case CollectiveKind::ReduceScatter:
    case CollectiveKind::Broadcast:
      throw std::invalid_argument("Collective kind not implemented in CCL v1");
  }
  throw std::invalid_argument("Unsupported collective kind");
}

std::string to_string(CollectivePlan const& plan) {
  std::ostringstream oss;
  oss << collective_name(plan.collective) << " plan"
      << " rank=" << plan.rank << "/" << plan.nranks
      << " channels=" << plan.channels
      << " bytes_per_rank=" << plan.bytes_per_rank
      << " chunk_bytes=" << plan.chunk_bytes
      << " steps=" << plan.steps.size() << "\n";

  for (auto const& step : plan.steps) {
    oss << "step " << step.step_id
        << " owner=" << step.chunk.owner_rank
        << " chunk=" << step.chunk.chunk_index
        << " channel=" << step.chunk.channel_id
        << " off=" << step.chunk.offset_bytes
        << " size=" << step.chunk.size_bytes
        << " ops=" << step.ops.size() << " [";
    for (size_t i = 0; i < step.ops.size(); ++i) {
      if (i) oss << ", ";
      auto const& op = step.ops[i];
      oss << op_name(op.kind);
      if (op.peer_rank >= 0) oss << "@peer=" << op.peer_rank;
      if (op.kind == ExecutionOpKind::Copy || op.kind == ExecutionOpKind::Reduce ||
          op.kind == ExecutionOpKind::Send || op.kind == ExecutionOpKind::Recv) {
        oss << "(" << slot_name(op.src.slot) << ":" << op.src.rank
            << "->" << slot_name(op.dst.slot) << ":" << op.dst.rank << ")";
      }
    }
    oss << "]\n";
  }

  return oss.str();
}

}  // namespace CCL
}  // namespace UKernel
