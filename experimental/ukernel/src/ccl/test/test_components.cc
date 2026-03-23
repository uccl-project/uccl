#include "backend_test_utils.h"
#include "../selector.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>

namespace UKernel {
namespace CCL {

namespace {

void test_selector_routing() {
  printf("[test] selector routing...\n");
  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, cfg));
  ExecutionPlan exec = lower_plan(plan);

  size_t transport_ops = 0;
  size_t device_ops = 0;
  for (auto const& op : exec.ops) {
    if (op.kind == ExecOpKind::TransportSend ||
        op.kind == ExecOpKind::TransportRecv) {
      ++transport_ops;
    }
    if (op.kind == ExecOpKind::DeviceCopy ||
        op.kind == ExecOpKind::DeviceReduce) {
      ++device_ops;
    }
  }
  assert(transport_ops > 0);
  assert(device_ops > 0);
}

void test_plan_builders() {
  printf("[test] plan builders...\n");

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

  CollectiveConfig alltoall_cfg =
      Testing::make_test_config(4, 1, 1024, 256, 2);
  alltoall_cfg.algorithm = AlgorithmKind::Pairwise;
  CollectivePlan alltoall_plan =
      build_plan(make_plan_request(CollectiveKind::AllToAll, alltoall_cfg));
  Testing::validate_basic_plan(alltoall_plan);
  assert(alltoall_plan.collective == CollectiveKind::AllToAll);
  assert(alltoall_plan.algorithm == AlgorithmKind::Pairwise);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Send) > 0);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Recv) > 0);
  assert(Testing::count_ops(alltoall_plan, PrimitiveOpKind::Copy) > 0);
}

void test_lowering() {
  printf("[test] lowering...\n");

  CollectiveConfig cfg = Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectivePlan plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, cfg));
  ExecutionPlan exec = lower_plan(plan);

  assert(exec.ops.size() == plan.ops.size());

  size_t transport_ops = 0;
  size_t device_ops = 0;
  for (size_t i = 0; i < exec.ops.size(); ++i) {
    auto const& op = exec.ops[i];
    assert(op.op_id == i);
    if (op.kind == ExecOpKind::TransportSend ||
        op.kind == ExecOpKind::TransportRecv) {
      ++transport_ops;
    }
    if (op.kind == ExecOpKind::DeviceCopy ||
        op.kind == ExecOpKind::DeviceReduce) {
      ++device_ops;
    }
    for (uint32_t dep : op.deps) {
      assert(dep < op.op_id);
    }
  }
  assert(transport_ops > 0);
  assert(device_ops > 0);
}

void test_executor_with_mock_backends() {
  printf("[test] executor with mock backends...\n");

  Testing::MockDeviceBackend device_backend;
  Testing::MockBackend transport_backend;

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);

  CollectiveConfig allreduce_cfg =
      Testing::make_test_config(4, 1, 4096, 512, 2);
  CollectiveOpHandle ar_handle = executor.submit_allreduce(allreduce_cfg);
  executor.wait(ar_handle);
  assert(executor.status(ar_handle) == CollectiveOpStatus::Completed);
  executor.release(ar_handle);

  CollectiveConfig alltoall_cfg =
      Testing::make_test_config(4, 1, 1024, 256, 2);
  alltoall_cfg.algorithm = AlgorithmKind::Pairwise;
  CollectiveOpHandle at_handle = executor.submit_alltoall(alltoall_cfg);
  executor.wait(at_handle);
  assert(executor.status(at_handle) == CollectiveOpStatus::Completed);
  executor.release(at_handle);
}

void test_single_active_collective_constraint() {
  printf("[test] single active collective constraint...\n");

  Testing::MockDeviceBackend device_backend(2);
  Testing::MockBackend transport_backend(2);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);

  CollectiveConfig config = Testing::make_test_config(4, 1, 2048, 256, 2);
  CollectiveOpHandle handle = executor.submit_allreduce(config);

  bool threw = false;
  try {
    static_cast<void>(executor.submit_alltoall(config));
  } catch (std::runtime_error const&) {
    threw = true;
  }
  assert(threw);

  executor.wait(handle);
  executor.release(handle);
}

}  // namespace

}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;

  printf("=== CCL Component Tests ===\n\n");
  test_selector_routing();
  test_plan_builders();
  test_lowering();
  test_executor_with_mock_backends();
  test_single_active_collective_constraint();
  printf("\n=== Component tests PASSED ===\n");
  return 0;
}
