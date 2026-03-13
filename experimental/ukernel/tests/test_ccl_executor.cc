#include "backend.h"
#include "executor.h"
#include "plan.h"
#include "test.h"
#include <cassert>
#include <iostream>

namespace {

UKernel::CCL::CollectivePlan make_backend_routing_plan() {
  using namespace UKernel::CCL;

  CollectivePlan plan;
  plan.collective = CollectiveKind::AllGather;
  plan.algorithm = AlgorithmKind::Ring;
  plan.nranks = 2;
  plan.rank = 0;
  plan.channels = 1;
  plan.bytes_per_rank = 256;
  plan.chunk_bytes = 128;

  CollectiveStep ce_step;
  ce_step.step_id = 0;
  ce_step.phase = StepPhase::DirectCopy;
  ce_step.src_rank = 0;
  ce_step.dst_rank = 1;
  ce_step.chunk = ChunkRange{0, 0, 0, 0, 128};
  ce_step.ops.push_back(ExecutionOp{0, ExecutionOpKind::CeCopy, 0, 1,
                                    ce_step.chunk, {}});
  plan.steps.push_back(ce_step);

  CollectiveStep rdma_step;
  rdma_step.step_id = 1;
  rdma_step.phase = StepPhase::DirectCopy;
  rdma_step.src_rank = 1;
  rdma_step.dst_rank = 0;
  rdma_step.chunk = ChunkRange{1, 0, 0, 128, 128};
  rdma_step.predecessors = {0};
  rdma_step.ops.push_back(ExecutionOp{1, ExecutionOpKind::RdmaSend, 1, 0,
                                      rdma_step.chunk, {}});
  plan.steps.push_back(rdma_step);

  CollectiveStep pk_step;
  pk_step.step_id = 2;
  pk_step.phase = StepPhase::ReduceScatter;
  pk_step.src_rank = 0;
  pk_step.dst_rank = 0;
  pk_step.chunk = ChunkRange{0, 1, 0, 128, 128};
  pk_step.predecessors = {1};
  pk_step.ops.push_back(
      ExecutionOp{2, ExecutionOpKind::PkReduce, 0, 0, pk_step.chunk, {}});
  plan.steps.push_back(pk_step);

  return plan;
}

}  // namespace

void test_ccl_executor() {
  using namespace UKernel::CCL;

  MockBackend fallback_backend(1);
  PersistentKernelBackend persistent_backend(2);

  ExecutorBackends pk_backends{};
  pk_backends.persistent = &persistent_backend;
  pk_backends.fallback = &fallback_backend;

  Executor pk_executor(pk_backends);

  CollectiveConfig gather{};
  gather.algorithm = AlgorithmKind::Ring;
  gather.nranks = 4;
  gather.rank = 1;
  gather.channels = 2;
  gather.bytes_per_rank = 1024;
  gather.chunk_bytes = 256;

  CollectiveOpHandle gather_handle = pk_executor.submit_allgather(gather);
  assert(pk_executor.status(gather_handle) == CollectiveOpStatus::Running);
  pk_executor.wait(gather_handle);
  assert(pk_executor.status(gather_handle) == CollectiveOpStatus::Completed);
  assert(persistent_backend.submissions() == 12);
  pk_executor.release(gather_handle);

  CollectiveConfig reduce{};
  reduce.algorithm = AlgorithmKind::Ring;
  reduce.nranks = 4;
  reduce.rank = 1;
  reduce.channels = 2;
  reduce.bytes_per_rank = 4096;
  reduce.chunk_bytes = 512;

  CollectiveOpHandle reduce_handle = pk_executor.submit_allreduce(reduce);
  pk_executor.wait(reduce_handle);
  assert(pk_executor.status(reduce_handle) == CollectiveOpStatus::Completed);
  assert(persistent_backend.submissions() == 30);
  pk_executor.release(reduce_handle);

  MockBackend rdma_backend(1);
  MockBackend ce_backend(1);
  MockBackend fallback_backend2(1);
  PersistentKernelBackend persistent_backend2(1);

  ExecutorBackends routed_backends{};
  routed_backends.rdma = &rdma_backend;
  routed_backends.ce = &ce_backend;
  routed_backends.persistent = &persistent_backend2;
  routed_backends.fallback = &fallback_backend2;

  Executor routed_executor(routed_backends);
  CollectiveOpHandle routed_handle =
      routed_executor.submit(make_backend_routing_plan());
  routed_executor.wait(routed_handle);
  assert(routed_executor.status(routed_handle) == CollectiveOpStatus::Completed);
  assert(ce_backend.submissions() == 1);
  assert(rdma_backend.submissions() == 1);
  assert(persistent_backend2.submissions() == 1);
  routed_executor.release(routed_handle);

  std::cout << "[test_ccl_executor] PK-only executor PASSED\n";
  std::cout << "[test_ccl_executor] collective submit API PASSED\n";
  std::cout << "[test_ccl_executor] mixed backend routing PASSED\n";
}
