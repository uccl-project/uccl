#pragma once

#include "collective_types.h"
#include "collective_memory.h"
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

enum class PrimitiveOpKind : uint32_t {
  Send,
  Recv,
  Copy,
  Reduce,
};

// TileRef is the planner's logical scheduling unit: one tensor tile that moves
// through the collective pipeline. In the current implementation a tile is a
// contiguous byte range, but the name leaves room for richer tensor tiling.
struct TileRef {
  uint32_t owner_rank = 0;
  uint32_t tile_index = 0;
  // flow_index selects the logical pipeline lane this tile belongs to.
  uint32_t flow_index = 0;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;
};

struct PrimitiveOp {
  uint32_t op_id = 0;
  PrimitiveOpKind kind = PrimitiveOpKind::Copy;
  TileRef tile;
  BufferRef src;
  BufferRef dst;
  ScalarType dtype = ScalarType::UInt8;
  ReductionKind reduction = ReductionKind::None;
  std::vector<uint32_t> deps;
};

struct CollectivePlan {
  CollectiveKind collective = CollectiveKind::AllReduce;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  int nranks = 1;
  int rank = 0;
  // num_flows is the effective number of rank-local pipeline lanes the planner
  // builds after clamping the request to the available tile parallelism.
  uint32_t num_flows = 1;
  // For allreduce this is the input tensor size. For alltoall this is the
  // max(input_bytes, output_bytes) envelope used by plan metadata validation.
  size_t tensor_bytes = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  // staging_bytes_required is the minimum temporary buffer storage required
  // by the planned algorithm on this rank.
  size_t staging_bytes_required = 0;
  ScalarType dtype = ScalarType::UInt8;
  ReductionKind reduction = ReductionKind::None;
  std::vector<size_t> input_split_bytes;
  std::vector<size_t> output_split_bytes;
  std::vector<PrimitiveOp> ops;
};

struct PlanRequest {
  CollectiveKind collective = CollectiveKind::AllReduce;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
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
  CollectiveBufferRoles roles{};
  ScalarType dtype = ScalarType::Float32;
  ReductionKind reduction = ReductionKind::Sum;
};

uint32_t normalized_num_flows(CollectiveKind collective, int nranks,
                              size_t tensor_bytes, size_t tile_bytes,
                              ScalarType dtype, uint32_t requested_flows);

CollectivePlan build_plan(PlanRequest const& request);
std::string to_string(CollectivePlan const& plan);

}  // namespace CCL
}  // namespace UKernel
