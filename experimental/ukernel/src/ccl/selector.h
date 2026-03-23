#pragma once

#include "memory.h"
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
  int peer_rank = -1;
  TileRef tile;
  BufferRef src;
  BufferRef dst;
  std::vector<uint32_t> deps;
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
  std::vector<ExecOp> ops;
};

ExecutionPlan lower_plan(CollectivePlan const& plan);

}  // namespace CCL
}  // namespace UKernel
