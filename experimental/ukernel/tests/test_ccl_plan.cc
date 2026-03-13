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
  assert(gather_plan.steps.front().ops.front().src_role == BufferRole::RemoteInput);
  assert(gather_plan.steps.front().ops.front().dst_role == BufferRole::FinalOutput);
  assert(gather_plan.steps.front().chunk.owner_rank == 0);
  assert(gather_plan.steps[4].chunk.owner_rank == 3);
  assert(gather_plan.steps[8].chunk.owner_rank == 2);
  assert(gather_plan.steps.front().chunk.offset_bytes == 0);
  assert(gather_plan.steps[4].chunk.offset_bytes == 3 * gather.bytes_per_rank);

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
  assert(reduce_plan.steps.front().ops[0].flags ==
         static_cast<uint32_t>(ExecutionOpFlags::StageForReduce));
  assert(reduce_plan.steps.front().ops[0].src_role == BufferRole::RemoteInput);
  assert(reduce_plan.steps.front().ops[0].dst_role == BufferRole::RecvStaging);
  assert(reduce_plan.steps.front().ops[1].src_role == BufferRole::RecvStaging);
  assert(reduce_plan.steps.front().ops[1].dst_role == BufferRole::FinalOutput);
  assert(reduce_plan.steps.front().chunk.owner_rank == 0);
  assert(reduce_plan.steps.back().chunk.owner_rank == 3);
  assert(reduce_plan.steps.back().ops.front().src_role == BufferRole::RemoteReduced);
  assert(reduce_plan.steps.back().ops.front().dst_role == BufferRole::FinalOutput);

  PlanRequest gather2{};
  gather2.collective = CollectiveKind::AllGather;
  gather2.algorithm = AlgorithmKind::Ring;
  gather2.nranks = 2;
  gather2.rank = 0;
  gather2.channels = 2;
  gather2.bytes_per_rank = 512;
  gather2.chunk_bytes = 256;
  CollectivePlan gather2_plan = build_plan(gather2);
  assert(gather2_plan.steps.size() == 2);
  assert(gather2_plan.steps.front().chunk.owner_rank == 1);
  assert(gather2_plan.steps.front().chunk.offset_bytes == 512);
  assert(gather2_plan.steps.back().chunk.offset_bytes == 768);
  assert(gather2_plan.steps.front().ops.front().src_role == BufferRole::RemoteInput);

  PlanRequest reduce2{};
  reduce2.collective = CollectiveKind::AllReduce;
  reduce2.algorithm = AlgorithmKind::Ring;
  reduce2.nranks = 2;
  reduce2.rank = 0;
  reduce2.channels = 1;
  reduce2.bytes_per_rank = 1024;
  reduce2.chunk_bytes = 512;
  CollectivePlan reduce2_plan = build_plan(reduce2);
  assert(reduce2_plan.steps.size() == 2);
  assert(reduce2_plan.steps.front().phase == StepPhase::ReduceScatter);
  assert(reduce2_plan.steps.front().chunk.owner_rank == 1);
  assert(reduce2_plan.steps.front().chunk.offset_bytes == 512);
  assert(reduce2_plan.steps.front().ops[0].flags ==
         static_cast<uint32_t>(ExecutionOpFlags::StageForReduce));
  assert(reduce2_plan.steps.back().phase == StepPhase::AllGather);
  assert(reduce2_plan.steps.back().chunk.owner_rank == 0);
  assert(reduce2_plan.steps.back().chunk.offset_bytes == 0);
  assert(reduce2_plan.steps.back().ops.front().src_role == BufferRole::RemoteReduced);

  std::cout << "[test_ccl_plan] AllGather steps=" << gather_plan.steps.size()
            << "\n";
  std::cout << "[test_ccl_plan] AllReduce steps=" << reduce_plan.steps.size()
            << "\n";
  std::cout << to_string(gather_plan);
}
