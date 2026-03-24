#pragma once

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
  int peer_rank = -1;
  TileRef tile;
  BufferRef src;
  BufferRef dst;
  std::vector<uint32_t> deps;
};

struct CollectivePlan {
  CollectiveKind collective = CollectiveKind::AllReduce;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  int nranks = 1;
  int rank = 0;
  // num_flows controls how many rank-local pipeline lanes the planner builds.
  uint32_t num_flows = 1;
  size_t tensor_bytes = 0;
  size_t tile_bytes = 0;
  // staging_bytes_required is the minimum temporary tensor storage required
  // by the planned algorithm on this rank.
  size_t staging_bytes_required = 0;
  std::vector<PrimitiveOp> ops;
};

struct PlanRequest {
  CollectiveKind collective = CollectiveKind::AllReduce;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  int nranks = 1;
  int rank = 0;
  uint32_t num_flows = 1;
  size_t tensor_bytes = 0;
  size_t tile_bytes = 0;
  size_t staging_bytes = 0;
};

CollectivePlan build_plan(PlanRequest const& request);
std::string to_string(CollectivePlan const& plan);

}  // namespace CCL
}  // namespace UKernel
