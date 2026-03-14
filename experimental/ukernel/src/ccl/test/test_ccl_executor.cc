#include "backend_test_utils.h"
#include "executor.h"
#include "plan.h"
#include "test.h"
#include <cassert>
#include <iostream>

namespace {

struct DagProbeState {
  bool copy_completed = false;
  bool reduce_submitted_before_copy_complete = false;
};

class DagCeBackend final : public UKernel::CCL::Backend {
 public:
  explicit DagCeBackend(DagProbeState& state) : state_(state) {}

  char const* name() const override { return "dag-ce"; }
  bool supports(UKernel::CCL::ExecutionOpKind kind) const override {
    return kind == UKernel::CCL::ExecutionOpKind::CeCopy;
  }
  UKernel::CCL::BackendToken submit(UKernel::CCL::ExecutionOp const&) override {
    pending_ = true;
    return UKernel::CCL::BackendToken{1};
  }
  bool poll(UKernel::CCL::BackendToken) override {
    if (!pending_) return true;
    pending_ = false;
    state_.copy_completed = true;
    return true;
  }
  void release(UKernel::CCL::BackendToken) override {}

 private:
  DagProbeState& state_;
  bool pending_ = false;
};

class DagPkBackend final : public UKernel::CCL::Backend {
 public:
  explicit DagPkBackend(DagProbeState& state) : state_(state) {}

  char const* name() const override { return "dag-pk"; }
  bool supports(UKernel::CCL::ExecutionOpKind kind) const override {
    return kind == UKernel::CCL::ExecutionOpKind::PkReduce;
  }
  UKernel::CCL::BackendToken submit(UKernel::CCL::ExecutionOp const&) override {
    if (!state_.copy_completed) {
      state_.reduce_submitted_before_copy_complete = true;
    }
    return UKernel::CCL::BackendToken{1};
  }
  bool poll(UKernel::CCL::BackendToken) override { return true; }
  void release(UKernel::CCL::BackendToken) override {}

 private:
  DagProbeState& state_;
};

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

UKernel::CCL::CollectivePlan make_op_dag_plan() {
  using namespace UKernel::CCL;

  CollectivePlan plan;
  plan.collective = CollectiveKind::AllReduce;
  plan.algorithm = AlgorithmKind::Ring;
  plan.nranks = 2;
  plan.rank = 0;
  plan.channels = 1;
  plan.bytes_per_rank = 128;
  plan.chunk_bytes = 128;

  CollectiveStep step;
  step.step_id = 0;
  step.phase = StepPhase::ReduceScatter;
  step.src_rank = 1;
  step.dst_rank = 0;
  step.chunk = ChunkRange{0, 0, 0, 0, 128};
  step.ops.push_back(ExecutionOp{7, ExecutionOpKind::CeCopy, 1, 0, step.chunk, {}});
  step.ops.push_back(
      ExecutionOp{8, ExecutionOpKind::PkReduce, 0, 0, step.chunk, {7}});
  plan.steps.push_back(step);

  return plan;
}

}  // namespace

void test_ccl_executor() {
  using namespace UKernel::CCL;
  using namespace UKernel::CCL::Testing;

  MockBackend fallback_backend(1);
  MockPersistentKernelBackend persistent_backend(2);

  ExecutorBackends pk_backends{};
  pk_backends.persistent_kernel = &persistent_backend;
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
  assert(persistent_backend.submissions() == 38);
  pk_executor.release(reduce_handle);

  MockBackend rdma_backend(1);
  MockBackend ce_backend(1);
  MockBackend fallback_backend2(1);
  MockPersistentKernelBackend persistent_backend2(1);

  ExecutorBackends routed_backends{};
  routed_backends.transport = &rdma_backend;
  routed_backends.copy_engine = &ce_backend;
  routed_backends.persistent_kernel = &persistent_backend2;
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

  MockBackend ce_selector_backend(1);
  MockPersistentKernelBackend pk_selector_backend(1);
  ExecutorBackends ce_selected_backends{};
  ce_selected_backends.copy_engine = &ce_selector_backend;
  ce_selected_backends.persistent_kernel = &pk_selector_backend;
  Executor ce_selected_executor(ce_selected_backends);

  CollectiveConfig ce_gather{};
  ce_gather.algorithm = AlgorithmKind::Ring;
  ce_gather.nranks = 4;
  ce_gather.rank = 1;
  ce_gather.channels = 2;
  ce_gather.bytes_per_rank = 1024;
  ce_gather.chunk_bytes = 256;
  ce_gather.requested_backend = BackendKind::Auto;
  ce_gather.runtime_caps.has_copy_engine_path = true;
  ce_gather.backend_selector.copy_engine_threshold_bytes = 1;

  CollectiveOpHandle ce_gather_handle =
      ce_selected_executor.submit_allgather(ce_gather);
  ce_selected_executor.wait(ce_gather_handle);
  assert(ce_selected_executor.status(ce_gather_handle) ==
         CollectiveOpStatus::Completed);
  assert(ce_selector_backend.submissions() == 12);
  assert(pk_selector_backend.submissions() == 0);
  ce_selected_executor.release(ce_gather_handle);

  MockBackend rdma_selector_backend(1);
  ExecutorBackends rdma_selected_backends{};
  rdma_selected_backends.transport = &rdma_selector_backend;
  Executor rdma_selected_executor(rdma_selected_backends);

  CollectiveConfig rdma_gather{};
  rdma_gather.algorithm = AlgorithmKind::Ring;
  rdma_gather.nranks = 4;
  rdma_gather.rank = 1;
  rdma_gather.channels = 2;
  rdma_gather.bytes_per_rank = 1024;
  rdma_gather.chunk_bytes = 256;
  rdma_gather.requested_backend = BackendKind::Auto;
  rdma_gather.runtime_caps.is_same_node = false;
  rdma_gather.runtime_caps.supports_rdma = true;

  CollectiveOpHandle rdma_gather_handle =
      rdma_selected_executor.submit_allgather(rdma_gather);
  rdma_selected_executor.wait(rdma_gather_handle);
  assert(rdma_selected_executor.status(rdma_gather_handle) ==
         CollectiveOpStatus::Completed);
  assert(rdma_selector_backend.submissions() == 24);
  rdma_selected_executor.release(rdma_gather_handle);

  DagProbeState dag_state{};
  DagCeBackend dag_ce_backend(dag_state);
  DagPkBackend dag_pk_backend(dag_state);
  ExecutorBackends dag_backends{};
  dag_backends.copy_engine = &dag_ce_backend;
  dag_backends.persistent_kernel = &dag_pk_backend;
  Executor dag_executor(dag_backends);
  auto dag_handle = dag_executor.submit(make_op_dag_plan());
  dag_executor.wait(dag_handle);
  assert(dag_executor.status(dag_handle) == CollectiveOpStatus::Completed);
  assert(!dag_state.reduce_submitted_before_copy_complete);
  dag_executor.release(dag_handle);

  std::cout << "[test_ccl_executor] PK-only executor PASSED\n";
  std::cout << "[test_ccl_executor] collective submit API PASSED\n";
  std::cout << "[test_ccl_executor] mixed backend routing PASSED\n";
  std::cout << "[test_ccl_executor] CE selector routing PASSED\n";
  std::cout << "[test_ccl_executor] RDMA rewrite routing PASSED\n";
  std::cout << "[test_ccl_executor] op DAG scheduling PASSED\n";
}
