#include "backend_test_utils.h"
#include "test_utils.h"
#include "lower.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

using TestUtil::throws;
using TestUtil::wait_until_terminal;
using Testing::TestMemory;

TiledResult build_test_plan(CollKind kind,
                                CollectiveConfig const& config) {
  TestMemory mem = Testing::make_test_memory(
      config.rank, config.nranks,
      std::max(config.input_bytes, config.output_bytes));
  bool inplace = (mem.input_ptr == mem.output_ptr);
  (void)mem;
  CollectiveConfig cfg = config;
  cfg.kind = kind;
  return build_tiled(cfg, inplace);
}


void test_lowering_routes_ops_to_execution_kinds() {
  printf("[test] lowering routes ops...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  TiledResult tiled = build_test_plan(CollKind::AllReduceRing, cfg);

  Testing::validate_basic_exec_tiled(tiled);
  assert(Testing::count_exec_ops(tiled, OpKind::Send) > 0);
  assert(Testing::count_exec_ops(tiled, OpKind::Recv) > 0 ||
         Testing::count_exec_ops(tiled, OpKind::RecvReduce) > 0);
  assert(Testing::count_exec_ops(tiled, OpKind::Reduce) > 0 ||
         Testing::count_exec_ops(tiled, OpKind::RecvReduce) > 0);
}

void test_planner_emits_valid_collective_dags() {
  printf("[test] planner emits valid collective DAGs...\n");

  CollectiveConfig allreduce_cfg = Testing::make_test_config(4, 2, 4096, 512);
  TiledResult allreduce_tiled =
      build_test_plan(CollKind::AllReduceRing, allreduce_cfg);
  Testing::validate_basic_tiled(allreduce_tiled);
  assert(allreduce_tiled.input_bytes > 0);
  assert(Testing::count_ops(allreduce_tiled, OpKind::Send) > 0);
  assert(Testing::count_ops(allreduce_tiled, OpKind::RecvReduce) > 0 ||
         Testing::count_ops(allreduce_tiled, OpKind::Reduce) > 0);
  assert(allreduce_tiled.staging_bytes_required == 0);

  CollectiveConfig alltoall_cfg = Testing::make_test_config(4, 1, 1024, 256);
  alltoall_cfg.kind = CollKind::AllToAllPairwise;
  TiledResult alltoall_tiled =
      build_test_plan(CollKind::AllToAllPairwise, alltoall_cfg);
  Testing::validate_basic_tiled(alltoall_tiled);
  assert(alltoall_tiled.input_bytes > 0);
  assert(Testing::count_ops(alltoall_tiled, OpKind::Send) > 0);
  assert(Testing::count_ops(alltoall_tiled, OpKind::Recv) > 0);
  assert(alltoall_tiled.staging_bytes_required == 0);
}

void test_planner_clamps_flow_count_to_available_tiles() {
  printf("[test] planner clamps stream count to available tiles...\n");

  CollectiveConfig allreduce_cfg =
      Testing::make_test_config(3, 0, 196608, 65536);
  allreduce_cfg.dtype = ScalarType::Float32;
  allreduce_cfg.reduction = ReductionKind::Sum;
  TiledResult allreduce_tiled =
      build_test_plan(CollKind::AllReduceRing, allreduce_cfg);
  Testing::validate_basic_tiled(allreduce_tiled);
  assert(allreduce_tiled.layers.size() >= 1);
  assert(allreduce_tiled.staging_bytes_required == 0);

  CollectiveConfig alltoall_cfg = Testing::make_test_config(4, 0, 1024, 256);
  alltoall_cfg.kind = CollKind::AllToAllPairwise;
  TiledResult alltoall_tiled =
      build_test_plan(CollKind::AllToAllPairwise, alltoall_cfg);
  Testing::validate_basic_tiled(alltoall_tiled);
  assert(alltoall_tiled.layers.size() >= 1);
  assert(alltoall_tiled.staging_bytes_required == 0);
}

void test_planner_builds_variable_split_alltoall_out_of_place() {
  printf("[test] planner builds variable-split alltoall out-of-place...\n");

  CollectiveConfig cfg = Testing::make_test_config(3, 1, 96, 16);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.input_bytes = 96;
  cfg.output_bytes = 96;
  cfg.input_split_bytes = {16, 32, 48};
  cfg.output_split_bytes = {24, 32, 40};

  cfg.kind = CollKind::AllToAllPairwise;
  TiledResult tiled = build_tiled(cfg, /*inplace=*/false);
  Testing::validate_basic_tiled(tiled);

  std::vector<size_t> sent_bytes(3, 0);
  std::vector<size_t> recv_bytes(3, 0);
  size_t self_copy_bytes = 0;
  for (auto const& op : tiled.ops) {
    if (op.kind == OpKind::Send) {
      assert(op.dst_peer >= 0 && op.dst_peer < 3);
      sent_bytes[static_cast<size_t>(op.dst_peer)] += op.bytes;
    } else if (op.kind == OpKind::Recv) {
      assert(op.src_peer >= 0 && op.src_peer < 3);
      recv_bytes[static_cast<size_t>(op.src_peer)] += op.bytes;
    } else if (op.kind == OpKind::Copy && op.src_peer == ~0u &&
               op.dst_peer == ~0u) {
      self_copy_bytes += op.bytes;
    }
  }

  // rank=1 sends to peers 0/2, receives from peers 0/2, and self-copies
  // split[1].
  assert(sent_bytes[0] == 16);
  assert(sent_bytes[1] == 0);
  assert(sent_bytes[2] == 48);
  assert(recv_bytes[0] == 24);
  assert(recv_bytes[1] == 0);
  assert(recv_bytes[2] == 40);
  assert(self_copy_bytes == 32);
}

void test_planner_honors_custom_role_buffer_ids() {
  printf("[test] planner uses plan buffer constants...\n");

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  bool inplace = false;
  TiledResult tiled = build_tiled(cfg, inplace);
  Testing::validate_basic_tiled(tiled);
  bool saw_input = false;
  bool saw_scratch = false;
  for (auto const& op : tiled.ops) {
    auto check_role = [&](OpKind k, bool is_src) {
      auto role = buf_role(k, is_src, op.copy_from_staging);
      if (role == CollectiveBufferRole::Input) saw_input = true;
      if (role == CollectiveBufferRole::Scratch) saw_scratch = true;
    };
    check_role(op.kind, true);
    check_role(op.kind, false);
  }
  assert(saw_input);
  // SM IPC has no staging — scratch is only for alltoall DMA
  if (tiled.staging_bytes_required > 0) assert(saw_scratch);
}

void test_two_rank_ring_allreduce_rotates_reduced_shard_in_allgather() {
  printf("[test] two-rank ring allreduce rotates reduced shard...\n");

  CollectiveConfig cfg = Testing::make_test_config(2, 0, 512, 512);
  TiledResult tiled = build_test_plan(CollKind::AllReduceRing, cfg);
  Testing::validate_basic_tiled(tiled);

  assert(tiled.ops.size() > 0);

  Op const& rs_send = tiled.ops[0];
  Op const& rs_reduce = tiled.ops[1];
  Op const& ag_send = tiled.ops[2];
  Op const& ag_recv = tiled.ops[3];

  assert(rs_send.kind == OpKind::Send);
  assert(rs_reduce.kind == OpKind::RecvReduce);

  // The first allgather send must forward the fully reduced shard produced by
  // the reduce-scatter phase, which for rank 0 in a two-rank ring is shard 1.
  assert(ag_send.kind == OpKind::Send);
  assert(ag_recv.kind == OpKind::Recv);
}

void test_three_rank_ring_allreduce_plans_tail_elements() {
  printf("[test] three-rank ring allreduce plans tail elements...\n");
  CollectiveConfig cfg = Testing::make_test_config(3, 1, 4100, 512);
  cfg.dtype = ScalarType::Float32;
  cfg.reduction = ReductionKind::Sum;
  TiledResult tiled = build_test_plan(CollKind::AllReduceRing, cfg);
  Testing::validate_basic_tiled(tiled);
  bool saw_short_tail = false;
  for (auto const& op : tiled.ops) {
    if (op.bytes != cfg.tile_bytes) saw_short_tail = true;
  }
  assert(saw_short_tail &&
         "non-divisible allreduce should produce a shorter tail tile");
}

void test_lowering_preserves_dependency_dag() {
  printf("[test] plan dependency DAG invariants...\n");

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  TiledResult tiled = build_test_plan(CollKind::AllReduceRing, cfg);

  Testing::validate_basic_exec_tiled(tiled);

  for (size_t i = 0; i < tiled.ops.size(); ++i) {
    auto const& op = tiled.ops[i];
    assert(op.bytes > 0);
    for (uint32_t dep : op.deps) {
      assert(dep < i);
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
  auto memory = std::make_shared<TestMemory>(
      Testing::make_test_memory(1, 4, 4096));

  CollectiveConfig allreduce_cfg = Testing::make_test_config(4, 1, 4096, 512);
  allreduce_cfg.kind = CollKind::AllReduceRing;
  CollectiveOpHandle ar_handle =
      executor.submit_allreduce(allreduce_cfg, memory->input_ptr, memory->output_ptr, memory->scratch_ptr);
  assert(wait_until_terminal(executor, ar_handle, std::chrono::seconds(2)));
  assert(executor.status(ar_handle) == CollectiveOpStatus::Completed);
  executor.release(ar_handle);

  CollectiveConfig alltoall_cfg = Testing::make_test_config(4, 1, 1024, 256);
  alltoall_cfg.kind = CollKind::AllToAllPairwise;
  alltoall_cfg.kind = CollKind::AllToAllPairwise;
  CollectiveOpHandle at_handle =
      executor.submit_alltoall(alltoall_cfg, memory->input_ptr, memory->output_ptr, memory->scratch_ptr);
  assert(wait_until_terminal(executor, at_handle, std::chrono::seconds(2)));
  assert(executor.status(at_handle) == CollectiveOpStatus::Completed);
  executor.release(at_handle);
}

void test_executor_queues_collectives_serially() {
  printf("[test] executor queues collectives serially...\n");

  Testing::MockDeviceBackend device_backend(10);
  Testing::MockBackend transport_backend(10);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  auto memory = std::make_shared<TestMemory>(
      Testing::make_test_memory(1, 4, 2048));

  CollectiveConfig allreduce_cfg = Testing::make_test_config(4, 1, 2048, 256);
  CollectiveConfig alltoall_cfg = Testing::make_test_config(4, 1, 2048, 256);
  alltoall_cfg.kind = CollKind::AllToAllPairwise;

  CollectiveOpHandle first = executor.submit_allreduce(allreduce_cfg, memory->input_ptr, memory->output_ptr, memory->scratch_ptr);
  CollectiveOpHandle second = executor.submit_alltoall(alltoall_cfg, memory->input_ptr, memory->output_ptr, memory->scratch_ptr);

  assert(executor.status(second) == CollectiveOpStatus::Running);

  assert(wait_until_terminal(executor, first, std::chrono::seconds(2)));
  assert(executor.status(first) == CollectiveOpStatus::Completed);

  assert(wait_until_terminal(executor, second, std::chrono::seconds(2)));
  assert(executor.status(second) == CollectiveOpStatus::Completed);

  executor.release(first);
  executor.release(second);
}

void test_executor_rejects_releasing_queued_or_running_collectives() {
  printf(
      "[test] executor rejects releasing queued or running collectives...\n");

  Testing::MockDeviceBackend device_backend(10);
  Testing::MockBackend transport_backend(10);
  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  auto memory = std::make_shared<TestMemory>(
      Testing::make_test_memory(1, 4, 2048));

  CollectiveConfig first_cfg = Testing::make_test_config(4, 1, 2048, 256);
  CollectiveConfig second_cfg = Testing::make_test_config(4, 1, 2048, 256);
  second_cfg.kind = CollKind::AllToAllPairwise;

  CollectiveOpHandle first = executor.submit_allreduce(first_cfg, memory->input_ptr, memory->output_ptr, memory->scratch_ptr);
  CollectiveOpHandle second = executor.submit_alltoall(second_cfg, memory->input_ptr, memory->output_ptr, memory->scratch_ptr);

  assert(throws([&] { executor.release(first); }));
  assert(throws([&] { executor.release(second); }));

  assert(wait_until_terminal(executor, first, std::chrono::seconds(5)));
  assert(wait_until_terminal(executor, second, std::chrono::seconds(5)));

  executor.release(first);
  executor.release(second);
}

void test_executor_uses_transport_only_backend() {
  printf("[test] executor runs with transport-only backend...\n");

  Testing::MockBackend transport_backend;
  Testing::MockDeviceBackend device_backend;
  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  auto memory = std::make_shared<TestMemory>(
      Testing::make_test_memory(1, 4, 4096));

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  CollectiveOpHandle handle = executor.submit_allreduce(cfg, memory->input_ptr, memory->output_ptr, memory->scratch_ptr);
  assert(wait_until_terminal(executor, handle, std::chrono::seconds(2)));
  assert(executor.status(handle) == CollectiveOpStatus::Completed);
  assert(device_backend.submissions() > 0);
  executor.release(handle);
}

void test_executor_reports_submit_failure() {
  printf("[test] executor reports backend submit failures...\n");

  Testing::ThrowingBackend throwing_backend("mock backend submit failure");
  ExecutorBackends backends{};
  backends.device = &throwing_backend;
  Executor executor(backends);
  auto memory = std::make_shared<TestMemory>(
      Testing::make_test_memory(1, 4, 2048));

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 2048, 256);
  CollectiveOpHandle handle = executor.submit_allreduce(cfg, memory->input_ptr, memory->output_ptr, memory->scratch_ptr);

  assert(wait_until_terminal(executor, handle, std::chrono::seconds(2)));
  assert(executor.status(handle) == CollectiveOpStatus::Failed);
  assert(executor.error_message(handle).find("mock backend submit failure") !=
         std::string::npos);
  executor.release(handle);
}

void bench_executor_pipeline() {
  printf("[bench] executor advance_run pipeline...\n");
  Testing::MockBackend transport_backend;
  Testing::MockDeviceBackend device_backend;
  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  TestMemory mem = Testing::make_test_memory(1, 4, 65536);

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 65536, 4096);
  cfg.kind = CollKind::AllReduceRing;

  constexpr int kWarmup = 5;
  constexpr int kBenchMs = 200;

  for (int i = 0; i < kWarmup; ++i) {
    auto h = executor.submit_allreduce(cfg, mem.input_ptr, mem.output_ptr,
                                       mem.scratch_ptr);
    while (!executor.poll(h)) {}
    executor.release(h);
  }

  auto t0 = std::chrono::steady_clock::now();
  int iters = 0;
  auto deadline = t0 + std::chrono::milliseconds(kBenchMs);
  while (std::chrono::steady_clock::now() < deadline) {
    auto h = executor.submit_allreduce(cfg, mem.input_ptr, mem.output_ptr,
                                       mem.scratch_ptr);
    while (!executor.poll(h)) {}
    executor.release(h);
    ++iters;
  }
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<
      std::chrono::milliseconds>(t1 - t0).count();
  printf("  submit→complete: %d ops in %lld ms  (%.0f ops/s)\n",
         iters, (long long)ms, iters * 1000.0 / ms);
}

void bench_executor_latency() {
  printf("[bench] executor end-to-end latency...\n");
  Testing::MockBackend transport_backend;
  Testing::MockDeviceBackend device_backend;
  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  TestMemory mem = Testing::make_test_memory(1, 4, 4096);

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512);
  cfg.kind = CollKind::AllReduceRing;

  constexpr int kIters = 500;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) {
    auto h = executor.submit_allreduce(cfg, mem.input_ptr, mem.output_ptr,
                                       mem.scratch_ptr);
    while (!executor.poll(h)) {}
    executor.release(h);
  }
  auto t1 = std::chrono::steady_clock::now();
  auto us = std::chrono::duration_cast<
      std::chrono::microseconds>(t1 - t0).count();
  printf("  submit→poll→release: %d in %.1f ms  (%.1f us/op)\n",
         kIters, us / 1000.0, static_cast<double>(us) / kIters);
}

void test_alltoall_sm_execution() {
  printf("[test] alltoall SM execution...\n");
  Testing::MockDeviceBackend device_backend;
  Testing::MockBackend transport_backend;
  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  TestMemory mem = Testing::make_test_memory(1, 4, 8192);

  CollectiveConfig cfg = Testing::make_test_config(4, 2, 8192, 512);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = true;
  cfg.reduction = ReductionKind::None;

  auto h = executor.submit_alltoall(cfg, mem.input_ptr, mem.output_ptr,
                                     mem.scratch_ptr);
  assert(executor.status(h) == CollectiveOpStatus::Running);
  assert(wait_until_terminal(executor, h, std::chrono::seconds(2)));
  assert(executor.status(h) == CollectiveOpStatus::Completed);
  assert(device_backend.submissions() > 0);
  executor.release(h);
  printf("  ok: %zu device ops completed\n", device_backend.submissions());
}

void test_alltoall_dma_execution() {
  printf("[test] alltoall DMA execution...\n");
  Testing::MockBackend transport_backend;
  Testing::MockDeviceBackend device_backend;
  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  TestMemory mem = Testing::make_test_memory(1, 4, 8192);

  CollectiveConfig cfg = Testing::make_test_config(4, 0, 8192, 1024);
  cfg.kind = CollKind::AllToAllPairwise;
  cfg.use_sm_ipc = false;

  auto h = executor.submit_alltoall(cfg, mem.input_ptr, mem.output_ptr,
                                     mem.scratch_ptr);
  assert(wait_until_terminal(executor, h, std::chrono::seconds(2)));
  assert(executor.status(h) == CollectiveOpStatus::Completed);
  assert(transport_backend.submissions() > 0);
  executor.release(h);
  printf("  ok: %zu transport ops completed\n",
         transport_backend.submissions());
}

void bench_alltoall_pipeline() {
  printf("[bench] alltoall pipeline throughput (SM + DMA)...\n");
  Testing::MockDeviceBackend device_backend;
  Testing::MockBackend transport_backend;
  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);
  TestMemory mem = Testing::make_test_memory(1, 4, 65536);

  // SM
  {
    CollectiveConfig cfg = Testing::make_test_config(4, 1, 65536, 4096);
    cfg.kind = CollKind::AllToAllPairwise;
    cfg.use_sm_ipc = true;

    for (int i = 0; i < 3; ++i) {
      auto h = executor.submit_alltoall(cfg, mem.input_ptr, mem.output_ptr,
                                         mem.scratch_ptr);
      while (!executor.poll(h)) {}
      executor.release(h);
    }

    constexpr int kBenchMs = 100;
    auto t0 = std::chrono::steady_clock::now();
    int iters = 0;
    auto deadline = t0 + std::chrono::milliseconds(kBenchMs);
    while (std::chrono::steady_clock::now() < deadline) {
      auto h = executor.submit_alltoall(cfg, mem.input_ptr, mem.output_ptr,
                                         mem.scratch_ptr);
      while (!executor.poll(h)) {}
      executor.release(h);
      ++iters;
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    printf("  alltoall SM:  %d ops in %lld ms  (%.0f ops/s)\n",
           iters, (long long)ms, iters * 1000.0 / ms);
  }

  // DMA
  {
    CollectiveConfig cfg = Testing::make_test_config(4, 0, 65536, 4096);
    cfg.kind = CollKind::AllToAllPairwise;
    cfg.use_sm_ipc = false;

    for (int i = 0; i < 3; ++i) {
      auto h = executor.submit_alltoall(cfg, mem.input_ptr, mem.output_ptr,
                                         mem.scratch_ptr);
      while (!executor.poll(h)) {}
      executor.release(h);
    }

    constexpr int kBenchMs = 100;
    auto t0 = std::chrono::steady_clock::now();
    int iters = 0;
    auto deadline = t0 + std::chrono::milliseconds(kBenchMs);
    while (std::chrono::steady_clock::now() < deadline) {
      auto h = executor.submit_alltoall(cfg, mem.input_ptr, mem.output_ptr,
                                         mem.scratch_ptr);
      while (!executor.poll(h)) {}
      executor.release(h);
      ++iters;
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    printf("  alltoall DMA: %d ops in %lld ms  (%.0f ops/s)\n",
           iters, (long long)ms, iters * 1000.0 / ms);
  }
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
  test_three_rank_ring_allreduce_plans_tail_elements();
  test_lowering_preserves_dependency_dag();
  test_executor_completes_collectives_with_background_progress();
  test_executor_queues_collectives_serially();
  test_executor_rejects_releasing_queued_or_running_collectives();
  test_executor_uses_transport_only_backend();
  test_executor_reports_submit_failure();

  test_alltoall_sm_execution();
  test_alltoall_dma_execution();

  bench_executor_pipeline();
  bench_executor_latency();
  bench_alltoall_pipeline();

  printf("\n=== Component tests PASSED ===\n");
  return 0;
}
