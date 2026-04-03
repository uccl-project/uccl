#include "selector.h"
#include "backend_test_utils.h"
#include "test_utils.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <string>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

using TestUtil::throws;
using TestUtil::wait_until_terminal;

struct SimRankState {
  CollectivePlan plan;
  std::vector<float> tensor;
  std::vector<float> staging;
  std::vector<uint32_t> remaining_deps;
  std::vector<std::vector<uint32_t>> successors;
  std::vector<bool> completed;
  std::vector<bool> recv_posted;
  std::unordered_map<uint32_t, uint64_t> send_seq;
  std::unordered_map<uint32_t, uint64_t> recv_seq;
};

uint64_t make_match_key(int src_rank, int dst_rank, uint64_t seq) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(src_rank)) << 48) ^
         (static_cast<uint64_t>(static_cast<uint32_t>(dst_rank)) << 32) ^ seq;
}

float* local_buffer_ptr(SimRankState& state, BufferRef const& ref) {
  if (ref.kind != BufferKind::Local) {
    throw std::invalid_argument(
        "remote ref is not addressable as a local buffer");
  }
  switch (ref.buffer_id) {
    case kDefaultInputBufferId:
      return state.tensor.data() + ref.offset_bytes / sizeof(float);
    case kDefaultScratchBufferId:
      return state.staging.data() + ref.offset_bytes / sizeof(float);
    default:
      break;
  }
  throw std::invalid_argument("buffer id is not addressable in simulator");
}

void complete_sim_op(SimRankState& state, uint32_t op_id) {
  if (state.completed[op_id]) return;
  state.completed[op_id] = true;
  for (uint32_t succ : state.successors[op_id]) {
    assert(state.remaining_deps[succ] > 0);
    --state.remaining_deps[succ];
  }
}

void build_sim_state_dependencies(SimRankState& state) {
  size_t nops = state.plan.ops.size();
  state.remaining_deps.assign(nops, 0);
  state.successors.assign(nops, {});
  state.completed.assign(nops, false);
  state.recv_posted.assign(nops, false);
  for (size_t i = 0; i < nops; ++i) {
    state.remaining_deps[i] =
        static_cast<uint32_t>(state.plan.ops[i].deps.size());
    for (uint32_t dep : state.plan.ops[i].deps) {
      state.successors[dep].push_back(static_cast<uint32_t>(i));
    }
  }

  std::vector<uint64_t> next_send_seq(static_cast<size_t>(state.plan.nranks), 1);
  std::vector<uint64_t> next_recv_seq(static_cast<size_t>(state.plan.nranks), 1);
  for (auto const& op : state.plan.ops) {
    if (op.kind == PrimitiveOpKind::Send) {
      int peer = op.dst.rank;
      assert(peer >= 0 && peer < state.plan.nranks);
      state.send_seq.emplace(op.op_id, next_send_seq[static_cast<size_t>(peer)]++);
    } else if (op.kind == PrimitiveOpKind::Recv) {
      int peer = op.src.rank;
      assert(peer >= 0 && peer < state.plan.nranks);
      state.recv_seq.emplace(op.op_id, next_recv_seq[static_cast<size_t>(peer)]++);
    }
  }
}

CollectivePlan build_test_plan(CollectiveKind kind,
                               CollectiveConfig const& config) {
  CollectiveBinding binding =
      Testing::make_test_memory(config.rank, config.nranks,
                                std::max(config.tensor_bytes,
                                         config.staging_bytes));
  return build_plan(make_plan_request(kind, config, binding.roles));
}

void assert_ref_uses_only_roles(BufferRef const& ref,
                                CollectiveBufferRoles const& roles) {
  assert(ref.buffer_id == roles.input_buffer_id ||
         ref.buffer_id == roles.output_buffer_id ||
         ref.buffer_id == roles.scratch_buffer_id);
}

std::string simulate_three_rank_allreduce_and_find_error(size_t tensor_bytes,
                                                        uint32_t num_flows) {
  constexpr int kNranks = 3;
  constexpr size_t kTileBytes = 64 << 10;

  std::vector<SimRankState> ranks;
  ranks.reserve(kNranks);
  for (int rank = 0; rank < kNranks; ++rank) {
    CollectiveConfig cfg =
        Testing::make_test_config(kNranks, rank, tensor_bytes, kTileBytes,
                                  num_flows);
    cfg.dtype = ScalarType::Float32;
    cfg.reduction = ReductionKind::Sum;

    SimRankState state;
    state.plan = build_test_plan(CollectiveKind::AllReduce, cfg);
    size_t nelems = tensor_bytes / sizeof(float);
    state.tensor.resize(nelems, 0.0f);
    state.staging.resize(kTileBytes / sizeof(float), 0.0f);
    for (size_t i = 0; i < nelems; ++i) {
      state.tensor[i] = static_cast<float>(rank * 1000) + static_cast<float>(i);
    }
    build_sim_state_dependencies(state);
    ranks.push_back(std::move(state));
  }

  struct PostedRecv {
    int dst_rank = -1;
    uint32_t recv_op_id = 0;
  };
  std::unordered_map<uint64_t, PostedRecv> posted_recvs;

  bool progress = true;
  while (progress) {
    progress = false;
    for (int rank = 0; rank < kNranks; ++rank) {
      SimRankState& state = ranks[rank];
      for (size_t op_id = 0; op_id < state.plan.ops.size(); ++op_id) {
        if (state.completed[op_id] || state.remaining_deps[op_id] != 0) {
          continue;
        }
        PrimitiveOp const& op = state.plan.ops[op_id];
        switch (op.kind) {
          case PrimitiveOpKind::Recv: {
            if (state.recv_posted[op_id]) continue;
            auto seq_it = state.recv_seq.find(static_cast<uint32_t>(op_id));
            assert(seq_it != state.recv_seq.end());
            uint64_t key =
                make_match_key(op.src.rank, rank, seq_it->second);
            auto [_, inserted] = posted_recvs.emplace(
                key, PostedRecv{rank, static_cast<uint32_t>(op_id)});
            assert(inserted && "duplicate posted recv in simulator");
            state.recv_posted[op_id] = true;
            progress = true;
            break;
          }
          case PrimitiveOpKind::Send: {
            auto seq_it = state.send_seq.find(static_cast<uint32_t>(op_id));
            assert(seq_it != state.send_seq.end());
            uint64_t key =
                make_match_key(rank, op.dst.rank, seq_it->second);
            auto posted_it = posted_recvs.find(key);
            if (posted_it == posted_recvs.end()) continue;

            PostedRecv posted = posted_it->second;
            SimRankState& dst_state = ranks[posted.dst_rank];
            PrimitiveOp const& recv_op = dst_state.plan.ops[posted.recv_op_id];
            assert(recv_op.kind == PrimitiveOpKind::Recv);
            assert(recv_op.src.rank == rank);
            assert(recv_op.tile.size_bytes == op.tile.size_bytes);

            size_t count = op.tile.size_bytes / sizeof(float);
            float const* src_ptr = local_buffer_ptr(state, op.src);
            float* dst_ptr = local_buffer_ptr(dst_state, recv_op.dst);
            std::copy(src_ptr, src_ptr + count, dst_ptr);

            posted_recvs.erase(posted_it);
            complete_sim_op(state, static_cast<uint32_t>(op_id));
            complete_sim_op(dst_state, posted.recv_op_id);
            progress = true;
            break;
          }
          case PrimitiveOpKind::Copy: {
            size_t count = op.tile.size_bytes / sizeof(float);
            float const* src_ptr = local_buffer_ptr(state, op.src);
            float* dst_ptr = local_buffer_ptr(state, op.dst);
            std::copy(src_ptr, src_ptr + count, dst_ptr);
            complete_sim_op(state, static_cast<uint32_t>(op_id));
            progress = true;
            break;
          }
          case PrimitiveOpKind::Reduce: {
            size_t count = op.tile.size_bytes / sizeof(float);
            float const* src_ptr = local_buffer_ptr(state, op.src);
            float* dst_ptr = local_buffer_ptr(state, op.dst);
            for (size_t i = 0; i < count; ++i) {
              dst_ptr[i] += src_ptr[i];
            }
            complete_sim_op(state, static_cast<uint32_t>(op_id));
            progress = true;
            break;
          }
        }
      }
    }
  }

  for (int rank = 0; rank < kNranks; ++rank) {
    SimRankState const& state = ranks[rank];
    for (size_t op_id = 0; op_id < state.plan.ops.size(); ++op_id) {
      if (!state.completed[op_id]) {
        return "simulator stalled before completing all ops";
      }
    }
    for (size_t i = 0; i < state.tensor.size(); ++i) {
      float expected = 0.0f;
      for (int src_rank = 0; src_rank < kNranks; ++src_rank) {
        expected += static_cast<float>(src_rank * 1000) + static_cast<float>(i);
      }
      if (std::fabs(state.tensor[i] - expected) >= 1e-3f) {
        return "rank " + std::to_string(rank) + " mismatch at index " +
               std::to_string(i) + ", got=" +
               std::to_string(state.tensor[i]) + ", expected=" +
               std::to_string(expected);
      }
    }
  }

  return {};
}

void test_lowering_routes_ops_to_execution_kinds() {
  printf("[test] lowering routes ops...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectivePlan plan = build_test_plan(CollectiveKind::AllReduce, cfg);
  ExecutionPlan exec = lower_plan(plan);

  Testing::validate_basic_exec_plan(exec);
  assert(Testing::count_exec_ops(exec, ExecOpKind::TransportSend) > 0);
  assert(Testing::count_exec_ops(exec, ExecOpKind::TransportRecv) > 0);
  assert(Testing::count_exec_ops(exec, ExecOpKind::DeviceReduce) > 0);
}

void test_planner_emits_valid_collective_dags() {
  printf("[test] planner emits valid collective DAGs...\n");

  CollectiveConfig allreduce_cfg =
      Testing::make_test_config(4, 2, 4096, 512, 2);
  CollectivePlan allreduce_plan =
      build_test_plan(CollectiveKind::AllReduce, allreduce_cfg);
  Testing::validate_basic_plan(allreduce_plan);
  assert(allreduce_plan.collective == CollectiveKind::AllReduce);
  assert(allreduce_plan.algorithm == AlgorithmKind::Ring);
  assert(Testing::count_ops(allreduce_plan, PrimitiveOpKind::Send) > 0);
  assert(Testing::count_ops(allreduce_plan, PrimitiveOpKind::Recv) > 0);
  assert(Testing::count_ops(allreduce_plan, PrimitiveOpKind::Reduce) > 0);
  assert(allreduce_plan.staging_bytes_required ==
         allreduce_cfg.num_flows * allreduce_cfg.tile_bytes);

  CollectiveConfig alltoall_cfg = Testing::make_test_config(4, 1, 1024, 256, 2);
  alltoall_cfg.algorithm = AlgorithmKind::Pairwise;
  CollectivePlan alltoall_plan =
      build_test_plan(CollectiveKind::AllToAll, alltoall_cfg);
  Testing::validate_basic_plan(alltoall_plan);
  assert(alltoall_plan.collective == CollectiveKind::AllToAll);
  assert(alltoall_plan.algorithm == AlgorithmKind::Pairwise);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Send) > 0);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Recv) > 0);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Copy) > 0);
  assert(alltoall_plan.staging_bytes_required ==
         static_cast<size_t>(alltoall_cfg.nranks - 1) * alltoall_cfg.tile_bytes);
}

void test_planner_clamps_flow_count_to_available_tiles() {
  printf("[test] planner clamps flow count to available tiles...\n");

  CollectiveConfig allreduce_cfg =
      Testing::make_test_config(3, 0, 196608, 65536, 2);
  allreduce_cfg.dtype = ScalarType::Float32;
  allreduce_cfg.reduction = ReductionKind::Sum;
  CollectivePlan allreduce_plan =
      build_test_plan(CollectiveKind::AllReduce, allreduce_cfg);
  Testing::validate_basic_plan(allreduce_plan);
  assert(allreduce_plan.num_flows == 1);
  assert(allreduce_plan.staging_bytes_required == allreduce_cfg.tile_bytes);
  for (auto const& op : allreduce_plan.ops) {
    assert(op.tile.flow_index < allreduce_plan.num_flows);
  }

  CollectiveConfig alltoall_cfg = Testing::make_test_config(4, 0, 1024, 256, 2);
  alltoall_cfg.algorithm = AlgorithmKind::Pairwise;
  CollectivePlan alltoall_plan =
      build_test_plan(CollectiveKind::AllToAll, alltoall_cfg);
  Testing::validate_basic_plan(alltoall_plan);
  assert(alltoall_plan.num_flows == 1);
  assert(alltoall_plan.staging_bytes_required ==
         static_cast<size_t>(alltoall_cfg.nranks - 1) * alltoall_cfg.tile_bytes);
  for (auto const& op : alltoall_plan.ops) {
    assert(op.tile.flow_index < alltoall_plan.num_flows);
  }
}

void test_planner_builds_variable_split_alltoall_out_of_place() {
  printf("[test] planner builds variable-split alltoall out-of-place...\n");

  CollectiveConfig cfg = Testing::make_test_config(3, 1, 96, 16, 4);
  cfg.algorithm = AlgorithmKind::Pairwise;
  cfg.input_bytes = 96;
  cfg.output_bytes = 96;
  cfg.input_split_bytes = {16, 32, 48};
  cfg.output_split_bytes = {24, 32, 40};

  CollectiveBufferRoles roles{};
  roles.input_buffer_id = 7;
  roles.output_buffer_id = 9;
  roles.scratch_buffer_id = 11;
  roles.validate();

  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllToAll, cfg, roles));
  Testing::validate_basic_plan(plan);
  assert(plan.input_split_bytes == cfg.input_split_bytes);
  assert(plan.output_split_bytes == cfg.output_split_bytes);

  std::vector<size_t> sent_bytes(3, 0);
  std::vector<size_t> recv_bytes(3, 0);
  size_t self_copy_bytes = 0;
  for (auto const& op : plan.ops) {
    if (op.kind == PrimitiveOpKind::Send) {
      assert(op.dst.rank >= 0 && op.dst.rank < 3);
      sent_bytes[static_cast<size_t>(op.dst.rank)] += op.tile.size_bytes;
    } else if (op.kind == PrimitiveOpKind::Recv) {
      assert(op.src.rank >= 0 && op.src.rank < 3);
      recv_bytes[static_cast<size_t>(op.src.rank)] += op.tile.size_bytes;
    } else if (op.kind == PrimitiveOpKind::Copy &&
               op.src.kind == BufferKind::Local &&
               op.dst.kind == BufferKind::Local &&
               op.src.buffer_id == roles.input_buffer_id &&
               op.dst.buffer_id == roles.output_buffer_id) {
      self_copy_bytes += op.tile.size_bytes;
    }
  }

  // rank=1 sends to peers 0/2, receives from peers 0/2, and self-copies split[1].
  assert(sent_bytes[0] == 16);
  assert(sent_bytes[1] == 0);
  assert(sent_bytes[2] == 48);
  assert(recv_bytes[0] == 24);
  assert(recv_bytes[1] == 0);
  assert(recv_bytes[2] == 40);
  assert(self_copy_bytes == 32);
}

void test_planner_honors_custom_role_buffer_ids() {
  printf("[test] planner honors custom role buffer ids...\n");

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectiveBufferRoles roles{};
  roles.input_buffer_id = 7;
  roles.scratch_buffer_id = 11;
  roles.validate();

  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, cfg, roles));
  Testing::validate_basic_plan(plan);
  bool saw_input = false;
  bool saw_scratch = false;
  for (auto const& op : plan.ops) {
    for (BufferRef const* ref : {&op.src, &op.dst}) {
      assert_ref_uses_only_roles(*ref, roles);
      saw_input = saw_input || ref->buffer_id == roles.input_buffer_id;
      saw_scratch = saw_scratch || ref->buffer_id == roles.scratch_buffer_id;
    }
  }
  assert(saw_input);
  assert(saw_scratch);

  ExecutionPlan exec = lower_plan(plan);
  Testing::validate_basic_exec_plan(exec);
  for (auto const& op : exec.ops) {
    assert_ref_uses_only_roles(op.src, roles);
    assert_ref_uses_only_roles(op.dst, roles);
  }
}

void test_two_rank_ring_allreduce_rotates_reduced_shard_in_allgather() {
  printf("[test] two-rank ring allreduce rotates reduced shard...\n");

  CollectiveConfig cfg = Testing::make_test_config(2, 0, 512, 512, 1);
  CollectivePlan plan = build_test_plan(CollectiveKind::AllReduce, cfg);
  Testing::validate_basic_plan(plan);

  assert(plan.ops.size() == 5);

  PrimitiveOp const& rs_send = plan.ops[0];
  PrimitiveOp const& rs_recv = plan.ops[1];
  PrimitiveOp const& rs_reduce = plan.ops[2];
  PrimitiveOp const& ag_send = plan.ops[3];
  PrimitiveOp const& ag_recv = plan.ops[4];

  assert(rs_send.kind == PrimitiveOpKind::Send);
  assert(rs_send.tile.owner_rank == 0);
  assert(rs_recv.kind == PrimitiveOpKind::Recv);
  assert(rs_recv.tile.owner_rank == 1);
  assert(rs_reduce.kind == PrimitiveOpKind::Reduce);
  assert(rs_reduce.tile.owner_rank == 1);

  // The first allgather send must forward the fully reduced shard produced by
  // the reduce-scatter phase, which for rank 0 in a two-rank ring is shard 1.
  assert(ag_send.kind == PrimitiveOpKind::Send);
  assert(ag_send.tile.owner_rank == 1);
  assert(ag_recv.kind == PrimitiveOpKind::Recv);
  assert(ag_recv.tile.owner_rank == 0);
}

void test_three_rank_ring_allreduce_simulator_handles_multi_tile_case() {
  printf("[test] three-rank ring allreduce simulator handles multi-tile case...\n");
  std::string error =
      simulate_three_rank_allreduce_and_find_error(1048572, 1);
  assert(error.empty() && "three-rank multi-tile allreduce simulation failed");
}

void test_three_rank_ring_allreduce_simulator_handles_single_tile_case() {
  printf("[test] three-rank ring allreduce simulator handles single-tile case...\n");
  std::string error =
      simulate_three_rank_allreduce_and_find_error(196608, 2);
  assert(error.empty() && "three-rank single-tile allreduce simulation failed");
}

void test_three_rank_ring_allreduce_plans_tail_elements() {
  printf("[test] three-rank ring allreduce plans tail elements...\n");
  CollectiveConfig cfg = Testing::make_test_config(3, 1, 4100, 512, 2);
  cfg.dtype = ScalarType::Float32;
  cfg.reduction = ReductionKind::Sum;
  CollectivePlan plan = build_test_plan(CollectiveKind::AllReduce, cfg);
  Testing::validate_basic_plan(plan);
  bool saw_short_tail = false;
  for (auto const& op : plan.ops) {
    if (op.tile.size_bytes % sizeof(float) != 0) {
      assert(false && "allreduce tail tile must remain dtype aligned");
    }
    if (op.tile.size_bytes != cfg.tile_bytes) {
      saw_short_tail = true;
    }
  }
  assert(saw_short_tail &&
         "non-divisible allreduce should produce a shorter tail tile");
}

void test_three_rank_ring_allreduce_simulator_handles_tail_elements_case() {
  printf("[test] three-rank ring allreduce simulator handles tail elements case...\n");
  std::string error =
      simulate_three_rank_allreduce_and_find_error(4100, 2);
  assert(error.empty() && "three-rank tail-element allreduce simulation failed");
}

void test_lowering_preserves_dependency_dag() {
  printf("[test] lowering preserves dependency DAG...\n");

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectivePlan plan = build_test_plan(CollectiveKind::AllReduce, cfg);
  ExecutionPlan exec = lower_plan(plan);

  assert(exec.ops.size() == plan.ops.size());
  Testing::validate_basic_exec_plan(exec);

  for (size_t i = 0; i < exec.ops.size(); ++i) {
    auto const& op = exec.ops[i];
    assert(op.op_id == i);
    assert(op.tile.owner_rank == plan.ops[i].tile.owner_rank);
    assert(op.tile.tile_index == plan.ops[i].tile.tile_index);
    assert(op.src.kind == plan.ops[i].src.kind);
    assert(op.src.rank == plan.ops[i].src.rank);
    assert(op.dst.kind == plan.ops[i].dst.kind);
    assert(op.dst.rank == plan.ops[i].dst.rank);
    assert(op.deps == plan.ops[i].deps);
    for (uint32_t dep : op.deps) {
      assert(dep < op.op_id);
    }
  }
}

void test_executor_completes_collectives_with_background_progress() {
  printf("[test] executor background progress...\n");

  Testing::MockDeviceBackend device_backend;
  Testing::MockBackend transport_backend;

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  auto memory = std::make_shared<CollectiveBinding>(
      Testing::make_test_memory(1, 4, 4096));

  CollectiveConfig allreduce_cfg =
      Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectiveOpHandle ar_handle = executor.submit_allreduce(allreduce_cfg, memory);
  assert(wait_until_terminal(executor, ar_handle, std::chrono::seconds(2)));
  assert(executor.status(ar_handle) == CollectiveOpStatus::Completed);
  executor.release(ar_handle);

  CollectiveConfig alltoall_cfg = Testing::make_test_config(4, 1, 1024, 256, 2);
  alltoall_cfg.algorithm = AlgorithmKind::Pairwise;
  CollectiveOpHandle at_handle = executor.submit_alltoall(alltoall_cfg, memory);
  assert(wait_until_terminal(executor, at_handle, std::chrono::seconds(2)));
  assert(executor.status(at_handle) == CollectiveOpStatus::Completed);
  executor.release(at_handle);
}

void test_executor_queues_collectives_serially() {
  printf("[test] executor queues collectives serially...\n");

  Testing::MockDeviceBackend device_backend(1000);
  Testing::MockBackend transport_backend(1000);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  auto memory = std::make_shared<CollectiveBinding>(
      Testing::make_test_memory(1, 4, 2048));

  CollectiveConfig allreduce_cfg =
      Testing::make_test_config(4, 1, 2048, 256, 2);
  CollectiveConfig alltoall_cfg =
      Testing::make_test_config(4, 1, 2048, 256, 2);
  alltoall_cfg.algorithm = AlgorithmKind::Pairwise;

  CollectiveOpHandle first = executor.submit_allreduce(allreduce_cfg, memory);
  CollectiveOpHandle second = executor.submit_alltoall(alltoall_cfg, memory);

  assert(executor.status(second) == CollectiveOpStatus::Queued);

  assert(wait_until_terminal(executor, first, std::chrono::seconds(2)));
  assert(executor.status(first) == CollectiveOpStatus::Completed);

  assert(wait_until_terminal(executor, second, std::chrono::seconds(2)));
  assert(executor.status(second) == CollectiveOpStatus::Completed);

  executor.release(first);
  executor.release(second);
}

void test_executor_rejects_releasing_queued_or_running_collectives() {
  printf("[test] executor rejects releasing queued or running collectives...\n");

  Testing::MockDeviceBackend device_backend(1000);
  Testing::MockBackend transport_backend(1000);
  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  auto memory = std::make_shared<CollectiveBinding>(
      Testing::make_test_memory(1, 4, 2048));

  CollectiveConfig first_cfg = Testing::make_test_config(4, 1, 2048, 256, 2);
  CollectiveConfig second_cfg = Testing::make_test_config(4, 1, 2048, 256, 2);
  second_cfg.algorithm = AlgorithmKind::Pairwise;

  CollectiveOpHandle first = executor.submit_allreduce(first_cfg, memory);
  CollectiveOpHandle second = executor.submit_alltoall(second_cfg, memory);

  assert(throws([&] { executor.release(first); }));
  assert(throws([&] { executor.release(second); }));

  assert(wait_until_terminal(executor, first, std::chrono::seconds(2)));
  assert(wait_until_terminal(executor, second, std::chrono::seconds(2)));

  executor.release(first);
  executor.release(second);
}

void test_executor_uses_fallback_backend_when_specialized_backends_are_missing() {
  printf("[test] executor uses fallback backend when specialized backends are missing...\n");

  Testing::MockBackend fallback_backend;
  ExecutorBackends backends{};
  backends.fallback = &fallback_backend;
  Executor executor(backends);
  auto memory = std::make_shared<CollectiveBinding>(
      Testing::make_test_memory(1, 4, 4096));

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectiveOpHandle handle = executor.submit_allreduce(cfg, memory);
  assert(wait_until_terminal(executor, handle, std::chrono::seconds(2)));
  assert(executor.status(handle) == CollectiveOpStatus::Completed);
  assert(fallback_backend.submissions() > 0);
  executor.release(handle);
}

void test_executor_reports_submit_failure() {
  printf("[test] executor reports backend submit failures...\n");

  Testing::ThrowingBackend throwing_backend("mock backend submit failure");
  ExecutorBackends backends{};
  backends.fallback = &throwing_backend;
  Executor executor(backends);
  auto memory = std::make_shared<CollectiveBinding>(
      Testing::make_test_memory(1, 4, 2048));

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 2048, 256, 1);
  CollectiveOpHandle handle = executor.submit_allreduce(cfg, memory);

  assert(wait_until_terminal(executor, handle, std::chrono::seconds(2)));
  assert(executor.status(handle) == CollectiveOpStatus::Failed);
  assert(executor.error_message(handle).find("mock backend submit failure") !=
         std::string::npos);
  executor.release(handle);
}

}  // namespace

}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;

  printf("=== CCL Component Tests ===\n\n");
  test_lowering_routes_ops_to_execution_kinds();
  test_planner_emits_valid_collective_dags();
  test_planner_clamps_flow_count_to_available_tiles();
  test_planner_builds_variable_split_alltoall_out_of_place();
  test_planner_honors_custom_role_buffer_ids();
  test_two_rank_ring_allreduce_rotates_reduced_shard_in_allgather();
  test_three_rank_ring_allreduce_simulator_handles_multi_tile_case();
  test_three_rank_ring_allreduce_simulator_handles_single_tile_case();
  test_three_rank_ring_allreduce_plans_tail_elements();
  test_three_rank_ring_allreduce_simulator_handles_tail_elements_case();
  test_lowering_preserves_dependency_dag();
  test_executor_completes_collectives_with_background_progress();
  test_executor_queues_collectives_serially();
  test_executor_rejects_releasing_queued_or_running_collectives();
  test_executor_uses_fallback_backend_when_specialized_backends_are_missing();
  test_executor_reports_submit_failure();
  printf("\n=== Component tests PASSED ===\n");
  return 0;
}
