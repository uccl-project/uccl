#pragma once

#include "collective_memory.h"
#include "collective_types.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace UKernel {
namespace CCL {

enum class CollectiveKind : uint32_t {
  AllReduce,
  AllToAll,
};

enum class AlgorithmKind : uint32_t {
  Ring,
  Pairwise,
};

// OpKind unifies planner-level primitive kinds and backend execution kinds
// into a single type — no lowering pass needed.
enum class OpKind : uint32_t {
  TransportSend,
  TransportRecv,
  DeviceCopy,
  DeviceReduce,
};

// TileRef is the planner's logical scheduling unit: one tensor tile that moves
// through the collective pipeline. In the current implementation a tile is a
// contiguous byte range, but the name leaves room for richer tensor tiling.
struct TileRef {
  uint32_t owner_rank = 0;
  uint32_t tile_index = 0;
  uint32_t flow_index = 0;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;
};

// Op is a single operation in the collective DAG.  Planner fills in the
// static scheduling fields; the Executor fills in the resolved pointer
// fields before submitting to backends.
struct Op {
  uint32_t op_id = 0;
  OpKind kind = OpKind::DeviceCopy;
  TileRef tile;
  BufferRef src;
  BufferRef dst;
  ScalarType dtype = ScalarType::UInt8;
  ReductionKind reduction = ReductionKind::None;
  std::vector<uint32_t> deps;

  // Runtime bindings — filled by the Executor, unset by the planner.
  void const* resolved_src = nullptr;
  void* resolved_dst = nullptr;
  int src_device = -1;
  int dst_device = -1;
};

// CollectiveConfig bundles all parameters needed to plan and execute a
// single collective invocation.
struct CollectiveConfig {
  CollectiveKind collective = CollectiveKind::AllReduce;
  int nranks = 1;
  int rank = 0;
  uint32_t num_flows = 1;
  size_t tensor_bytes = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  size_t staging_bytes = 0;
  std::vector<size_t> input_split_bytes;
  std::vector<size_t> output_split_bytes;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  ScalarType dtype = ScalarType::Float32;
  ReductionKind reduction = ReductionKind::Sum;
};

// Immutable plan — can be cached and reused across invocations with the
// same config shape.
struct CollectivePlan {
  CollectiveKind collective = CollectiveKind::AllReduce;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  int nranks = 1;
  int rank = 0;
  uint32_t num_flows = 1;
  size_t tensor_bytes = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  size_t staging_bytes_required = 0;
  ScalarType dtype = ScalarType::UInt8;
  ReductionKind reduction = ReductionKind::None;
  std::vector<size_t> input_split_bytes;
  std::vector<size_t> output_split_bytes;
  std::vector<Op> ops;
  std::vector<std::vector<uint32_t>> flow_ops;
};

uint32_t normalized_num_flows(CollectiveKind collective, int nranks,
                              size_t tensor_bytes, size_t tile_bytes,
                              ScalarType dtype, uint32_t requested_flows);

CollectivePlan build_plan(CollectiveConfig const& config, bool inplace);
std::string to_string(CollectivePlan const& plan);

}  // namespace CCL
}  // namespace UKernel
