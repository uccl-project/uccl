#include "../selector.h"
#include "backend_test_utils.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace CCL {

namespace {

bool wait_until_terminal(Executor& executor, CollectiveOpHandle handle,
                         std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (executor.poll(handle)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return executor.poll(handle);
}

void test_lowering_routes_ops_to_execution_kinds() {
  printf("[test] lowering routes ops...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, cfg));
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
      build_plan(make_plan_request(CollectiveKind::AllReduce, allreduce_cfg));
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
      build_plan(make_plan_request(CollectiveKind::AllToAll, alltoall_cfg));
  Testing::validate_basic_plan(alltoall_plan);
  assert(alltoall_plan.collective == CollectiveKind::AllToAll);
  assert(alltoall_plan.algorithm == AlgorithmKind::Pairwise);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Send) > 0);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Recv) > 0);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Copy) > 0);
  assert(alltoall_plan.staging_bytes_required ==
         static_cast<size_t>(alltoall_cfg.nranks - 1) * alltoall_cfg.tile_bytes);
}

void test_lowering_preserves_dependency_dag() {
  printf("[test] lowering preserves dependency DAG...\n");

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, cfg));
  ExecutionPlan exec = lower_plan(plan);

  assert(exec.ops.size() == plan.ops.size());
  Testing::validate_basic_exec_plan(exec);

  for (size_t i = 0; i < exec.ops.size(); ++i) {
    auto const& op = exec.ops[i];
    assert(op.op_id == i);
    assert(op.tile.owner_rank == plan.ops[i].tile.owner_rank);
    assert(op.tile.tile_index == plan.ops[i].tile.tile_index);
    assert(op.src.kind == plan.ops[i].src.kind);
    assert(op.dst.kind == plan.ops[i].dst.kind);
    assert(op.peer_rank == plan.ops[i].peer_rank);
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

  CollectiveConfig allreduce_cfg =
      Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectiveOpHandle ar_handle = executor.submit_allreduce(allreduce_cfg);
  assert(wait_until_terminal(executor, ar_handle, std::chrono::seconds(2)));
  assert(executor.status(ar_handle) == CollectiveOpStatus::Completed);
  executor.release(ar_handle);

  CollectiveConfig alltoall_cfg = Testing::make_test_config(4, 1, 1024, 256, 2);
  alltoall_cfg.algorithm = AlgorithmKind::Pairwise;
  CollectiveOpHandle at_handle = executor.submit_alltoall(alltoall_cfg);
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

  CollectiveConfig allreduce_cfg =
      Testing::make_test_config(4, 1, 2048, 256, 2);
  CollectiveConfig alltoall_cfg =
      Testing::make_test_config(4, 1, 2048, 256, 2);
  alltoall_cfg.algorithm = AlgorithmKind::Pairwise;

  CollectiveOpHandle first = executor.submit_allreduce(allreduce_cfg);
  CollectiveOpHandle second = executor.submit_alltoall(alltoall_cfg);

  assert(executor.status(second) == CollectiveOpStatus::Queued);

  assert(wait_until_terminal(executor, first, std::chrono::seconds(2)));
  assert(executor.status(first) == CollectiveOpStatus::Completed);

  assert(wait_until_terminal(executor, second, std::chrono::seconds(2)));
  assert(executor.status(second) == CollectiveOpStatus::Completed);

  executor.release(first);
  executor.release(second);
}

}  // namespace

}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;

  printf("=== CCL Component Tests ===\n\n");
  test_lowering_routes_ops_to_execution_kinds();
  test_planner_emits_valid_collective_dags();
  test_lowering_preserves_dependency_dag();
  test_executor_completes_collectives_with_background_progress();
  test_executor_queues_collectives_serially();
  printf("\n=== Component tests PASSED ===\n");
  return 0;
}
