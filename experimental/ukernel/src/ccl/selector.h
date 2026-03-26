#pragma once

#include "collective_types.h"
#include "collective_memory.h"
#include "plan.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

enum class ExecOpKind : uint32_t {
  TransportSend,
  TransportRecv,
  DeviceCopy,
  DeviceReduce,
};

struct ExecOp {
  uint32_t op_id = 0;
  ExecOpKind kind = ExecOpKind::DeviceCopy;
  TileRef tile;
  BufferRef src;
  BufferRef dst;
  ScalarType dtype = ScalarType::UInt8;
  ReductionKind reduction = ReductionKind::None;
  std::vector<uint32_t> deps;
  // Resolved runtime bindings for execution backends. Planner/lowering keep
  // these unset; executor fills them before submitting the op.
  void const* resolved_src = nullptr;
  void* resolved_dst = nullptr;
  int src_device = -1;
  int dst_device = -1;
};

struct ExecutionPlan {
  CollectiveKind collective = CollectiveKind::AllReduce;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  int nranks = 1;
  int rank = 0;
  uint32_t num_flows = 1;
  size_t tensor_bytes = 0;
  size_t tile_bytes = 0;
  size_t staging_bytes_required = 0;
  ScalarType dtype = ScalarType::UInt8;
  ReductionKind reduction = ReductionKind::None;
  std::vector<ExecOp> ops;
};

ExecutionPlan lower_plan(CollectivePlan const& plan);

}  // namespace CCL
}  // namespace UKernel
