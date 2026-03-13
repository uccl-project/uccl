#include "plan.h"
#include "test.h"
#include <cassert>
#include <iostream>

void test_ccl_plan() {
  using namespace UKernel::CCL;

  PlanRequest gather{};
  gather.collective = CollectiveKind::AllGather;
  gather.algorithm = AlgorithmKind::Ring;
  gather.nranks = 4;
  gather.rank = 1;
  gather.channels = 2;
  gather.bytes_per_rank = 1024;
  gather.chunk_bytes = 256;

  CollectivePlan gather_plan = build_plan(gather);
  assert(gather_plan.steps.size() == 12);
  assert(gather_plan.steps.front().ops.size() == 1);
  assert(gather_plan.steps.front().ops.front().kind == ExecutionOpKind::PkCopy);

  PlanRequest reduce{};
  reduce.collective = CollectiveKind::AllReduce;
  reduce.algorithm = AlgorithmKind::Ring;
  reduce.nranks = 4;
  reduce.rank = 1;
  reduce.channels = 2;
  reduce.bytes_per_rank = 4096;
  reduce.chunk_bytes = 512;

  CollectivePlan reduce_plan = build_plan(reduce);
  assert(!reduce_plan.steps.empty());
  assert(reduce_plan.steps.front().ops.size() == 2);
  assert(reduce_plan.steps.front().ops[0].kind == ExecutionOpKind::PkCopy);
  assert(reduce_plan.steps.front().ops[1].kind == ExecutionOpKind::PkReduce);

  std::cout << "[test_ccl_plan] AllGather steps=" << gather_plan.steps.size()
            << "\n";
  std::cout << "[test_ccl_plan] AllReduce steps=" << reduce_plan.steps.size()
            << "\n";
  std::cout << to_string(gather_plan);
}
