#include "backend_test_utils.h"
#include "../backend/mock_transport_backend.h"
#include "../selector.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>

namespace UKernel {
namespace CCL {

namespace {

void test_selector_routing() {
  printf("[test] selector routing...\n");

  RuntimeCapabilities caps;
  caps.peers.resize(4);
  caps.peers[1].same_node = true;
  caps.peers[1].peer_accessible = true;
  caps.peers[1].has_nvlink = true;
  caps.peers[1].has_copy_engine_path = true;
  caps.peers[2].supports_rdma = true;

  BackendSelectorConfig cfg{};

  ExecutionOp send{};
  send.kind = ExecutionOpKind::Send;
  send.peer_rank = 1;
  send.chunk.size_bytes = 1024;
  assert(resolve_backend_kind(BackendKind::Auto, send, caps, cfg) ==
         BackendKind::Transport);

  ExecutionOp reduce{};
  reduce.kind = ExecutionOpKind::Reduce;
  reduce.chunk.size_bytes = 1024;
  assert(resolve_backend_kind(BackendKind::Auto, reduce, caps, cfg) ==
         BackendKind::Device);

  ExecutionOp local_copy{};
  local_copy.kind = ExecutionOpKind::Copy;
  local_copy.chunk.size_bytes = 1024;
  local_copy.src = MemoryRef{MemorySlot::SymmetricTensor, -1, 0};
  local_copy.dst = MemoryRef{MemorySlot::SymmetricTensor, -1, 0};
  assert(resolve_backend_kind(BackendKind::Auto, local_copy, caps, cfg) ==
         BackendKind::Device);

  ExecutionOp small_remote_copy{};
  small_remote_copy.kind = ExecutionOpKind::Copy;
  small_remote_copy.chunk.size_bytes = 4 * 1024;
  small_remote_copy.src = MemoryRef{MemorySlot::SymmetricTensor, 1, 0};
  small_remote_copy.dst = MemoryRef{MemorySlot::SymmetricTensor, -1, 0};
  assert(resolve_backend_kind(BackendKind::Auto, small_remote_copy, caps, cfg) ==
         BackendKind::Device);

  ExecutionOp large_remote_copy = small_remote_copy;
  large_remote_copy.chunk.size_bytes =
      cfg.transport_copy_threshold_bytes + 1;
  assert(resolve_backend_kind(BackendKind::Auto, large_remote_copy, caps, cfg) ==
         BackendKind::Transport);
}

void test_plan_builders() {
  printf("[test] plan builders...\n");

  CollectiveConfig allreduce_cfg =
      Testing::make_ring_config(4, 2, 4096, 512, 2);
  CollectivePlan allreduce_plan =
      build_plan(make_plan_request(CollectiveKind::AllReduce, allreduce_cfg));
  Testing::validate_basic_plan(allreduce_plan);
  assert(allreduce_plan.collective == CollectiveKind::AllReduce);
  assert(Testing::count_ops(allreduce_plan, ExecutionOpKind::Send) > 0);
  assert(Testing::count_ops(allreduce_plan, ExecutionOpKind::Recv) > 0);
  assert(Testing::count_ops(allreduce_plan, ExecutionOpKind::Reduce) > 0);

  CollectiveConfig alltoall_cfg =
      Testing::make_ring_config(4, 1, 1024, 256, 2);
  CollectivePlan alltoall_plan =
      build_plan(make_plan_request(CollectiveKind::AllToAll, alltoall_cfg));
  Testing::validate_basic_plan(alltoall_plan);
  assert(alltoall_plan.collective == CollectiveKind::AllToAll);
  assert(Testing::count_ops(alltoall_plan, ExecutionOpKind::Send) > 0);
  assert(Testing::count_ops(alltoall_plan, ExecutionOpKind::Recv) > 0);
  assert(Testing::count_ops(alltoall_plan, ExecutionOpKind::Copy) > 0);

  bool saw_copy_with_deps = false;
  for (auto const& step : alltoall_plan.steps) {
    for (auto const& op : step.ops) {
      if (op.kind == ExecutionOpKind::Copy && !op.deps.empty()) {
        saw_copy_with_deps = true;
      }
    }
  }
  assert(saw_copy_with_deps);
}

void test_executor_with_mock_backends() {
  printf("[test] executor with mock backends...\n");

  Testing::MockDeviceBackend device_backend;
  MockCommunicator comm;
  CollectiveMemory memory = Testing::make_test_memory(1, 4, 4096);
  MockTransportBackend transport_backend(comm, memory);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);

  CollectiveConfig allreduce_cfg =
      Testing::make_ring_config(4, 1, 4096, 512, 2);
  CollectiveOpHandle ar_handle = executor.submit_allreduce(allreduce_cfg);
  executor.wait(ar_handle);
  assert(executor.status(ar_handle) == CollectiveOpStatus::Completed);
  executor.release(ar_handle);

  CollectiveConfig alltoall_cfg =
      Testing::make_ring_config(4, 1, 1024, 256, 2);
  CollectiveOpHandle at_handle = executor.submit_alltoall(alltoall_cfg);
  executor.wait(at_handle);
  assert(executor.status(at_handle) == CollectiveOpStatus::Completed);
  executor.release(at_handle);
}

void test_single_active_collective_constraint() {
  printf("[test] single active collective constraint...\n");

  Testing::MockDeviceBackend device_backend(2);
  MockCommunicator comm;
  CollectiveMemory memory = Testing::make_test_memory(1, 4, 2048);
  MockTransportBackend transport_backend(comm, memory);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);

  CollectiveConfig config = Testing::make_ring_config(4, 1, 2048, 256, 2);
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
  test_executor_with_mock_backends();
  test_single_active_collective_constraint();
  printf("\n=== Component tests PASSED ===\n");
  return 0;
}
