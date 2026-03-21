#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace UKernel {
namespace CCL {

enum class CollectiveKind : uint32_t { AllGather, AllReduce, ReduceScatter, Broadcast, AllToAll };
enum class AlgorithmKind : uint32_t { Ring };
enum class StepPhase : uint32_t { Init, Exchange, ReduceScatter, AllGather, AllToAll };
enum class ExecutionOpKind : uint32_t {
  Send,
  Recv,
  Copy,
  Reduce,
  EventWait,
  Barrier,
};

enum class MemorySlot : uint32_t {
  SymmetricTensor,
  RecvStaging,
};

enum class ExecutionOpFlags : uint32_t {
  None = 0,
  InPlace = 1u << 0,
};

struct ChunkRange {
  uint32_t owner_rank = 0;
  uint32_t chunk_index = 0;
  uint32_t channel_id = 0;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;
};

struct MemoryRef {
  MemorySlot slot = MemorySlot::SymmetricTensor;
  // -1 means the local endpoint's symmetric tensor.
  int rank = -1;
  size_t offset_bytes = 0;
};

struct ExecutionOp {
  uint32_t op_id = 0;
  ExecutionOpKind kind = ExecutionOpKind::Copy;
  int peer_rank = -1;
  ChunkRange chunk;
  std::vector<uint32_t> deps;
  uint32_t flags = static_cast<uint32_t>(ExecutionOpFlags::None);
  MemoryRef src;
  MemoryRef dst;
};

struct CollectiveStep {
  uint32_t step_id = 0;
  StepPhase phase = StepPhase::Exchange;
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
