#include "backend_test_utils.h"
#include "../executor.h"
#include "../backend/device_backend.h"
#include "../backend/mock_transport_backend.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

namespace UKernel {
namespace CCL {

void test_plan_properties();
void test_executor_basic();

}  // namespace CCL
}  // namespace UKernel

namespace UKernel {
namespace CCL {

void test_plan_allgather() {
  printf("[test] testing AllGather plan generation...\n");
  
  CollectiveConfig config{};
  config.nranks = 4;
  config.rank = 1;
  config.channels = 2;
  config.bytes_per_rank = 1024;
  config.chunk_bytes = 256;
  config.algorithm = AlgorithmKind::Ring;

  PlanRequest request = make_plan_request(CollectiveKind::AllGather, config);
  CollectivePlan plan = build_plan(request);

  assert(plan.collective == CollectiveKind::AllGather);
  assert(plan.nranks == 4);
  assert(plan.rank == 1);
  assert(plan.channels == 2);

  size_t chunks_per_rank = config.bytes_per_rank / config.chunk_bytes;
  size_t expected_steps = (config.nranks - 1) * chunks_per_rank;
  assert(plan.steps.size() == expected_steps);

  printf("[test] AllGather plan: %zu steps\n", plan.steps.size());
  printf("[test] AllGather plan PASSED\n");
}

void test_plan_allreduce() {
  printf("[test] testing AllReduce plan generation...\n");
  
  CollectiveConfig config{};
  config.nranks = 4;
  config.rank = 2;
  config.channels = 2;
  config.bytes_per_rank = 4096;
  config.chunk_bytes = 512;
  config.algorithm = AlgorithmKind::Ring;

  PlanRequest request = make_plan_request(CollectiveKind::AllReduce, config);
  CollectivePlan plan = build_plan(request);

  assert(plan.collective == CollectiveKind::AllReduce);
  assert(plan.nranks == 4);
  assert(plan.rank == 2);
  assert(plan.channels == 2);

  assert(plan.steps.size() > 0);

  for (auto const& step : plan.steps) {
    assert(step.ops.size() >= 1);
  }

  printf("[test] AllReduce plan: %zu steps\n", plan.steps.size());
  printf("[test] AllReduce plan PASSED\n");
}

void test_plan_alltoall() {
  printf("[test] testing AllToAll plan generation...\n");
  
  CollectiveConfig config{};
  config.nranks = 4;
  config.rank = 1;
  config.channels = 2;
  config.bytes_per_rank = 1024;
  config.chunk_bytes = 256;
  config.algorithm = AlgorithmKind::Ring;

  PlanRequest request = make_plan_request(CollectiveKind::AllToAll, config);
  CollectivePlan plan = build_plan(request);

  assert(plan.collective == CollectiveKind::AllToAll);
  assert(plan.nranks == 4);
  assert(plan.rank == 1);

  assert(plan.steps.size() > 0);

  printf("[test] AllToAll plan: %zu steps\n", plan.steps.size());
  printf("[test] AllToAll plan PASSED\n");
}

void test_executor_allgather() {
  printf("[test] testing AllGather executor with mock backend...\n");
  
  // Use void* as placeholder for worker pool since our mock doesn't use real workerpool
  void* mock_worker_pool = nullptr;
  UKernel::CCL::CollectiveBuffers buffers{};
  buffers.registration_bytes = 1024 * 4; // 4 ranks * 1024 bytes per rank
  
  UKernel::CCL::DeviceBackend device_backend(
      mock_worker_pool, buffers, 0, 0, 1);
      
  UKernel::CCL::MockCommunicator comm;
  UKernel::CCL::MockTransportBackend transport_backend(
      comm, 1, buffers);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);

  CollectiveConfig config{};
  config.algorithm = AlgorithmKind::Ring;
  config.nranks = 4;
  config.rank = 1;
  config.channels = 2;
  config.bytes_per_rank = 1024;
  config.chunk_bytes = 256;

  CollectiveOpHandle handle = executor.submit_allgather(config);
  assert(executor.status(handle) == CollectiveOpStatus::Running);
  
  executor.wait(handle);
  assert(executor.status(handle) == CollectiveOpStatus::Completed);
  
  executor.release(handle);

  printf("[test] AllGather executor PASSED\n");
}

void test_executor_allreduce() {
  printf("[test] testing AllReduce executor with mock backend...\n");
  
  // Use void* as placeholder for worker pool since our mock doesn't use real workerpool
  void* mock_worker_pool = nullptr;
  UKernel::CCL::CollectiveBuffers buffers{};
  buffers.registration_bytes = 4096 * 4; // 4 ranks * 4096 bytes per rank
  
  UKernel::CCL::DeviceBackend device_backend(
      mock_worker_pool, buffers, 0, 0, 1);
      
  UKernel::CCL::MockCommunicator comm;
  UKernel::CCL::MockTransportBackend transport_backend(
      comm, 1, buffers);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);

  CollectiveConfig config{};
  config.algorithm = AlgorithmKind::Ring;
  config.nranks = 4;
  config.rank = 1;
  config.channels = 2;
  config.bytes_per_rank = 4096;
  config.chunk_bytes = 512;

  CollectiveOpHandle handle = executor.submit_allreduce(config);
  executor.wait(handle);
  assert(executor.status(handle) == CollectiveOpStatus::Completed);
  executor.release(handle);

  printf("[test] AllReduce executor PASSED\n");
}

void test_executor_alltoall() {
  printf("[test] testing AllToAll executor with mock backend...\n");
  
  // Use void* as placeholder for worker pool since our mock doesn't use real workerpool
  void* mock_worker_pool = nullptr;
  UKernel::CCL::CollectiveBuffers buffers{};
  buffers.registration_bytes = 1024 * 4; // 4 ranks * 1024 bytes per rank
  
  UKernel::CCL::DeviceBackend device_backend(
      mock_worker_pool, buffers, 0, 0, 1);
      
  UKernel::CCL::MockCommunicator comm;
  UKernel::CCL::MockTransportBackend transport_backend(
      comm, 1, buffers);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);

  CollectiveConfig config{};
  config.algorithm = AlgorithmKind::Ring;
  config.nranks = 4;
  config.rank = 1;
  config.channels = 2;
  config.bytes_per_rank = 1024;
  config.chunk_bytes = 256;

  CollectiveOpHandle handle = executor.submit_alltoall(config);
  executor.wait(handle);
  assert(executor.status(handle) == CollectiveOpStatus::Completed);
  executor.release(handle);

  printf("[test] AllToAll executor PASSED\n");
}

}  // namespace CCL
}  // namespace UKernel

int main() {
  using namespace UKernel::CCL;
  
  printf("=== CCL Core Tests ===\n\n");
  
  printf("--- Plan Tests ---\n");
  test_plan_allgather();
  test_plan_allreduce();
  test_plan_alltoall();
  
  printf("\n--- Executor Tests ---\n");
  test_executor_allgather();
  test_executor_allreduce();
  test_executor_alltoall();
  
  printf("\n=== Plan/Executor Verification ===\n");
  test_plan_properties();
  test_executor_basic();

  printf("\n=== All tests PASSED ===\n");
  return 0;
}

void UKernel::CCL::test_plan_properties() {
  printf("[test] testing plan properties...\n");
  
  // Test different collective algorithms
  for (int nranks : {2, 3, 4}) {
    for (int rank = 0; rank < nranks; ++rank) {
      UKernel::CCL::CollectiveConfig config{};
      config.nranks = nranks;
      config.rank = rank;
      config.channels = 1;
      config.bytes_per_rank = 1024;
      config.chunk_bytes = 256;
      config.algorithm = UKernel::CCL::AlgorithmKind::Ring;

      // Test AllGather
      UKernel::CCL::PlanRequest ag_request = make_plan_request(UKernel::CCL::CollectiveKind::AllGather, config);
      UKernel::CCL::CollectivePlan ag_plan = build_plan(ag_request);

      printf("  AllGather %d/%d: %zu steps\n", rank, nranks, ag_plan.steps.size());
      
      // Check that each step has valid properties
      for (const auto& step : ag_plan.steps) {
        // In ring algorithm, each rank should receive from prev rank and send to next rank
        assert(step.src_rank >= 0 && step.src_rank < nranks);
        assert(step.dst_rank >= 0 && step.dst_rank < nranks);
        assert(step.chunk.size_bytes <= config.chunk_bytes);
      }

      // Test AllReduce  
      UKernel::CCL::PlanRequest ar_request = make_plan_request(UKernel::CCL::CollectiveKind::AllReduce, config);
      UKernel::CCL::CollectivePlan ar_plan = build_plan(ar_request);

      printf("  AllReduce %d/%d: %zu steps\n", rank, nranks, ar_plan.steps.size());
      
      for (const auto& step : ar_plan.steps) {
        assert(step.src_rank >= 0 && step.src_rank < nranks);
        assert(step.dst_rank >= 0 && step.dst_rank < nranks);
        assert(step.chunk.size_bytes <= config.chunk_bytes);
      }

      // Test AllToAll
      UKernel::CCL::PlanRequest at_request = make_plan_request(UKernel::CCL::CollectiveKind::AllToAll, config);
      UKernel::CCL::CollectivePlan at_plan = build_plan(at_request);

      printf("  AllToAll %d/%d: %zu steps\n", rank, nranks, at_plan.steps.size());
      
      for (const auto& step : at_plan.steps) {
        assert(step.src_rank >= 0 && step.src_rank < nranks);
        assert(step.dst_rank >= 0 && step.dst_rank < nranks);
        assert(step.chunk.size_bytes <= config.chunk_bytes);
      }
    }
  }

  printf("[test] Plan properties PASSED\n");
}

void UKernel::CCL::test_executor_basic() {
  printf("[test] testing executor basic functionality...\n");
  
  // Create mock backends
  void* mock_worker_pool = nullptr;
  UKernel::CCL::CollectiveBuffers buffers{};
  buffers.registration_bytes = 1024 * 4; // 4 ranks * 1024 bytes per rank

  UKernel::CCL::DeviceBackend device_backend(mock_worker_pool, buffers, 0, 0, 1);
  UKernel::CCL::MockCommunicator comm;
  UKernel::CCL::MockTransportBackend transport_backend(comm, 2, buffers); // Mock peer rank 2

  UKernel::CCL::ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  UKernel::CCL::Executor executor(backends);

  UKernel::CCL::CollectiveConfig config{};
  config.algorithm = UKernel::CCL::AlgorithmKind::Ring;
  config.nranks = 4;
  config.rank = 1;
  config.channels = 1;
  config.bytes_per_rank = 1024;
  config.chunk_bytes = 256;

  // Test AllGather execution
  UKernel::CCL::CollectiveOpHandle ag_handle = executor.submit_allgather(config);
  executor.wait(ag_handle);
  assert(executor.status(ag_handle) == UKernel::CCL::CollectiveOpStatus::Completed);
  executor.release(ag_handle);

  // Test AllReduce execution
  UKernel::CCL::CollectiveOpHandle ar_handle = executor.submit_allreduce(config);
  executor.wait(ar_handle);
  assert(executor.status(ar_handle) == UKernel::CCL::CollectiveOpStatus::Completed);
  executor.release(ar_handle);

  // Test AllToAll execution
  UKernel::CCL::CollectiveOpHandle at_handle = executor.submit_alltoall(config);
  executor.wait(at_handle);
  assert(executor.status(at_handle) == UKernel::CCL::CollectiveOpStatus::Completed);
  executor.release(at_handle);

  printf("[test] Executor basic functionality PASSED\n");
}
