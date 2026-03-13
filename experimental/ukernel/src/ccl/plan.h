#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace UKernel {
namespace CCL {

enum class CollectiveKind : uint32_t { AllGather, AllReduce };
enum class AlgorithmKind : uint32_t { Ring };
enum class StepPhase : uint32_t { DirectCopy, ReduceScatter, AllGather };
enum class ExecutionOpKind : uint32_t {
  RdmaSend,
  RdmaRecv,
  CeCopy,
  PkCopy,
  PkReduce,
  EventWait,
  Barrier,
};

struct ChunkRange {
  uint32_t owner_rank = 0;
  uint32_t chunk_index = 0;
  uint32_t channel_id = 0;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;
};

struct ExecutionOp {
  uint32_t op_id = 0;
  ExecutionOpKind kind = ExecutionOpKind::PkCopy;
  int src_rank = -1;
  int dst_rank = -1;
  ChunkRange chunk;
  std::vector<uint32_t> deps;
};

struct CollectiveStep {
  uint32_t step_id = 0;
  StepPhase phase = StepPhase::DirectCopy;
  int src_rank = -1;
  int dst_rank = -1;
  ChunkRange chunk;
  std::vector<uint32_t> predecessors;
  std::vector<ExecutionOp> ops;
};

struct CollectivePlan {
  CollectiveKind collective = CollectiveKind::AllGather;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  int nranks = 1;
  int rank = 0;
  uint32_t channels = 1;
  size_t bytes_per_rank = 0;
  size_t chunk_bytes = 0;
  std::vector<CollectiveStep> steps;
};

struct PlanRequest {
  CollectiveKind collective = CollectiveKind::AllGather;
  AlgorithmKind algorithm = AlgorithmKind::Ring;
  int nranks = 1;
  int rank = 0;
  uint32_t channels = 1;
  size_t bytes_per_rank = 0;
  size_t chunk_bytes = 0;
};

CollectivePlan build_plan(PlanRequest const& request);
std::string to_string(CollectivePlan const& plan);

}  // namespace CCL
}  // namespace UKernel
