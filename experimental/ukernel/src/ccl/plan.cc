#include "plan.h"
#include "topology.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

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

ExecutionOp make_copy_op(uint32_t op_id, int src_rank, int dst_rank,
                         ChunkRange const& chunk,
                         std::vector<uint32_t> deps = {},
                         uint32_t flags =
                             static_cast<uint32_t>(ExecutionOpFlags::None),
                         BufferRole src_role = BufferRole::RemoteInput,
                         BufferRole dst_role = BufferRole::FinalOutput) {
  ExecutionOp op;
  op.op_id = op_id;
  op.kind = ExecutionOpKind::PkCopy;
  op.src_rank = src_rank;
  op.dst_rank = dst_rank;
  op.chunk = chunk;
  op.deps = std::move(deps);
  op.flags = flags;
  op.src_role = src_role;
  op.dst_role = dst_role;
  return op;
}

ExecutionOp make_reduce_op(uint32_t op_id, int src_rank, int dst_rank,
                           ChunkRange const& chunk,
                           std::vector<uint32_t> deps = {},
                           BufferRole src_role = BufferRole::RecvStaging,
                           BufferRole dst_role = BufferRole::FinalOutput) {
  ExecutionOp op;
  op.op_id = op_id;
  op.kind = ExecutionOpKind::PkReduce;
  op.src_rank = src_rank;
  op.dst_rank = dst_rank;
  op.chunk = chunk;
  op.deps = std::move(deps);
  op.src_role = src_role;
  op.dst_role = dst_role;
  return op;
}

CollectivePlan build_allgather_ring_plan(PlanRequest const& request) {
  RingTopology ring{request.nranks};
  CollectivePlan plan;
  plan.collective = request.collective;
  plan.algorithm = request.algorithm;
  plan.nranks = request.nranks;
  plan.rank = request.rank;
  plan.channels = request.channels;
  plan.bytes_per_rank = request.bytes_per_rank;
  plan.chunk_bytes = request.chunk_bytes;

  uint32_t next_step_id = 0;
  uint32_t next_op_id = 0;
  std::vector<int32_t> last_step_for_channel(request.channels, -1);
  size_t chunks_per_rank = ceil_div(request.bytes_per_rank, request.chunk_bytes);

  for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
    for (size_t chunk_index = 0; chunk_index < chunks_per_rank; ++chunk_index) {
      ChunkRange chunk;
      chunk.owner_rank = static_cast<uint32_t>(
          ring.wrap(request.rank - ring_step - 1));
      chunk.chunk_index = static_cast<uint32_t>(chunk_index);
      chunk.channel_id = static_cast<uint32_t>(chunk_index % request.channels);
      chunk.offset_bytes =
          static_cast<size_t>(chunk.owner_rank) * request.bytes_per_rank +
          chunk_offset(chunk_index, request.chunk_bytes);
      chunk.size_bytes =
          chunk_size(request.bytes_per_rank, request.chunk_bytes, chunk_index);

      CollectiveStep step;
      step.step_id = next_step_id++;
      step.phase = StepPhase::AllGather;
      step.src_rank = ring.prev(request.rank);
      step.dst_rank = request.rank;
      step.chunk = chunk;
      step.has_forward_chunk = true;
      step.forward_src_rank = request.rank;
      step.forward_dst_rank = ring.next(request.rank);
      step.forward_chunk.owner_rank =
          static_cast<uint32_t>(ring.wrap(request.rank - ring_step));
      step.forward_chunk.chunk_index = static_cast<uint32_t>(chunk_index);
      step.forward_chunk.channel_id =
          static_cast<uint32_t>(chunk_index % request.channels);
      step.forward_chunk.offset_bytes =
          static_cast<size_t>(step.forward_chunk.owner_rank) *
              request.bytes_per_rank +
          chunk_offset(chunk_index, request.chunk_bytes);
      step.forward_chunk.size_bytes =
          chunk_size(request.bytes_per_rank, request.chunk_bytes, chunk_index);
      step.forward_src_role = BufferRole::FinalOutput;

      int32_t pred = last_step_for_channel[chunk.channel_id];
      if (pred >= 0) step.predecessors.push_back(static_cast<uint32_t>(pred));

      step.ops.push_back(
          make_copy_op(next_op_id++, ring.prev(request.rank), request.rank, chunk,
                       {}, static_cast<uint32_t>(ExecutionOpFlags::None),
                       BufferRole::RemoteInput, BufferRole::FinalOutput));

      last_step_for_channel[chunk.channel_id] = static_cast<int32_t>(step.step_id);
      plan.steps.push_back(std::move(step));
    }
  }

  return plan;
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

  size_t shard_bytes = ceil_div(request.bytes_per_rank, static_cast<size_t>(request.nranks));
  size_t chunks_per_shard = ceil_div(shard_bytes, request.chunk_bytes);
  uint32_t next_step_id = 0;
  uint32_t next_op_id = 0;
  std::vector<int32_t> last_step_for_channel(request.channels, -1);

  for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
    int reduced_owner = ring.wrap(request.rank - ring_step - 1);
    for (size_t chunk_index = 0; chunk_index < chunks_per_shard; ++chunk_index) {
      ChunkRange chunk;
      chunk.owner_rank = static_cast<uint32_t>(reduced_owner);
      chunk.chunk_index = static_cast<uint32_t>(chunk_index);
      chunk.channel_id = static_cast<uint32_t>(chunk_index % request.channels);
      chunk.offset_bytes =
          static_cast<size_t>(reduced_owner) * shard_bytes +
          chunk_offset(chunk_index, request.chunk_bytes);
      chunk.size_bytes =
          chunk_size(shard_bytes, request.chunk_bytes, chunk_index);

      CollectiveStep step;
      step.step_id = next_step_id++;
      step.phase = StepPhase::ReduceScatter;
      step.src_rank = ring.prev(request.rank);
      step.dst_rank = request.rank;
      step.chunk = chunk;

      int32_t pred = last_step_for_channel[chunk.channel_id];
      if (pred >= 0) step.predecessors.push_back(static_cast<uint32_t>(pred));

      uint32_t copy_op_id = next_op_id++;
      step.ops.push_back(
          make_copy_op(copy_op_id, ring.prev(request.rank), request.rank, chunk,
                       {},
                       static_cast<uint32_t>(ExecutionOpFlags::StageForReduce)));
      step.ops.back().src_role = BufferRole::RemoteInput;
      step.ops.back().dst_role = BufferRole::RecvStaging;
      step.ops.push_back(make_reduce_op(next_op_id++, request.rank, request.rank,
                                        chunk, {copy_op_id},
                                        BufferRole::RecvStaging,
                                        BufferRole::FinalOutput));

      last_step_for_channel[chunk.channel_id] = static_cast<int32_t>(step.step_id);
      plan.steps.push_back(std::move(step));
    }
  }

  for (int ring_step = 0; ring_step < request.nranks - 1; ++ring_step) {
    int gathered_owner = ring.wrap(request.rank - ring_step);
    for (size_t chunk_index = 0; chunk_index < chunks_per_shard; ++chunk_index) {
      ChunkRange chunk;
      chunk.owner_rank = static_cast<uint32_t>(gathered_owner);
      chunk.chunk_index = static_cast<uint32_t>(chunk_index);
      chunk.channel_id = static_cast<uint32_t>(chunk_index % request.channels);
      chunk.offset_bytes =
          static_cast<size_t>(gathered_owner) * shard_bytes +
          chunk_offset(chunk_index, request.chunk_bytes);
      chunk.size_bytes =
          chunk_size(shard_bytes, request.chunk_bytes, chunk_index);

      CollectiveStep step;
      step.step_id = next_step_id++;
      step.phase = StepPhase::AllGather;
      step.src_rank = ring.prev(request.rank);
      step.dst_rank = request.rank;
      step.chunk = chunk;

      int32_t pred = last_step_for_channel[chunk.channel_id];
      if (pred >= 0) step.predecessors.push_back(static_cast<uint32_t>(pred));

      step.ops.push_back(
          make_copy_op(next_op_id++, ring.prev(request.rank), request.rank, chunk,
                       {}, static_cast<uint32_t>(ExecutionOpFlags::None),
                       BufferRole::RemoteReduced, BufferRole::FinalOutput));

      last_step_for_channel[chunk.channel_id] = static_cast<int32_t>(step.step_id);
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
  }
  return "Unknown";
}

char const* op_name(ExecutionOpKind kind) {
  switch (kind) {
    case ExecutionOpKind::RdmaSend:
      return "RdmaSend";
    case ExecutionOpKind::RdmaRecv:
      return "RdmaRecv";
    case ExecutionOpKind::CeCopy:
      return "CeCopy";
    case ExecutionOpKind::PkCopy:
      return "PkCopy";
    case ExecutionOpKind::PkReduce:
      return "PkReduce";
    case ExecutionOpKind::EventWait:
      return "EventWait";
    case ExecutionOpKind::Barrier:
      return "Barrier";
  }
  return "Unknown";
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
    case CollectiveKind::AllGather:
      return build_allgather_ring_plan(request);
    case CollectiveKind::AllReduce:
      return build_allreduce_ring_plan(request);
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
    oss << "step " << step.step_id << " src=" << step.src_rank
        << " dst=" << step.dst_rank
        << " owner=" << step.chunk.owner_rank
        << " chunk=" << step.chunk.chunk_index
        << " channel=" << step.chunk.channel_id
        << " off=" << step.chunk.offset_bytes
        << " size=" << step.chunk.size_bytes
        << " ops=" << step.ops.size() << " [";
    for (size_t i = 0; i < step.ops.size(); ++i) {
      if (i) oss << ", ";
      oss << op_name(step.ops[i].kind);
    }
    oss << "]\n";
  }

  return oss.str();
}

}  // namespace CCL
}  // namespace UKernel
