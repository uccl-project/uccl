#include "selector.h"

namespace UKernel {
namespace CCL {

namespace {

ExecOpKind lower_kind(PrimitiveOpKind kind) {
  switch (kind) {
    case PrimitiveOpKind::Send:
      return ExecOpKind::TransportSend;
    case PrimitiveOpKind::Recv:
      return ExecOpKind::TransportRecv;
    case PrimitiveOpKind::Copy:
      return ExecOpKind::DeviceCopy;
    case PrimitiveOpKind::Reduce:
      return ExecOpKind::DeviceReduce;
  }
  return ExecOpKind::DeviceCopy;
}

}  // namespace

ExecutionPlan lower_plan(CollectivePlan const& plan) {
  ExecutionPlan exec;
  exec.collective = plan.collective;
  exec.algorithm = plan.algorithm;
  exec.nranks = plan.nranks;
  exec.rank = plan.rank;
  exec.num_flows = plan.num_flows;
  exec.tensor_bytes = plan.tensor_bytes;
  exec.tile_bytes = plan.tile_bytes;
  exec.staging_bytes_required = plan.staging_bytes_required;
  exec.dtype = plan.dtype;
  exec.reduction = plan.reduction;
  exec.ops.reserve(plan.ops.size());

  for (auto const& op : plan.ops) {
    ExecOp exec_op;
    exec_op.op_id = op.op_id;
    exec_op.kind = lower_kind(op.kind);
    exec_op.peer_rank = op.peer_rank;
    exec_op.tile = op.tile;
    exec_op.src = op.src;
    exec_op.dst = op.dst;
    exec_op.dtype = op.dtype;
    exec_op.reduction = op.reduction;
    exec_op.deps = op.deps;
    exec.ops.push_back(std::move(exec_op));
  }

  return exec;
}

}  // namespace CCL
}  // namespace UKernel
