#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

enum class CollKind : uint32_t {
  AllReduceRing,
  AllToAllPairwise,
  AllGatherRing,
  ReduceScatterRing,
  AllReduceTree,
};

// Planner-level op kinds (used in MacroOp DAG)

enum class AlgoOpKind : uint32_t {
  Put,
  Recv,
  RecvReduce,
};

// Executor-level op kinds (used in TiledOp, lower output)

enum class ExecOpKind : uint32_t {
  Put,
  Reduce,
  Signal,
  WaitSignal,
};

enum class ScalarType : uint32_t {
  UInt8,
  Int8,
  Int16,
  Int32,
  Int64,
  Float16,
  Float32,
  Float64,
  BFloat16,
  Bool,
};

inline constexpr size_t scalar_type_size(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::UInt8:
    case ScalarType::Int8:
    case ScalarType::Bool:
      return 1;
    case ScalarType::Int16:
    case ScalarType::Float16:
    case ScalarType::BFloat16:
      return 2;
    case ScalarType::Int32:
    case ScalarType::Float32:
      return 4;
    case ScalarType::Int64:
    case ScalarType::Float64:
      return 8;
  }
  return 0;
}

enum class ReductionKind : uint32_t {
  None,
  Sum,
  Prod,
  Max,
  Min,
  BitwiseAnd,
};

enum class CollectiveBufferRole : uint32_t {
  Input,
  Output,
  Scratch,
};

// Internal op (planner DAG → lower pipeline)

struct Op {
  AlgoOpKind kind = AlgoOpKind::Put;
  size_t bytes = 0;
  size_t src_off = 0;
  size_t dst_off = 0;
  uint32_t src_peer = 0;
  uint32_t dst_peer = 0;
  std::vector<uint32_t> deps;
};

// Tiled op (lower output → executor input)

struct TiledOp {
  ExecOpKind kind = ExecOpKind::Put;
  size_t bytes = 0;
  size_t src_off = 0;
  size_t dst_off = 0;
  uint32_t src_peer = 0;
  uint32_t dst_peer = 0;
  uint64_t tag = 0;
  std::vector<uint32_t> deps;
  CollectiveBufferRole src_buf_role = CollectiveBufferRole::Input;
  CollectiveBufferRole dst_buf_role = CollectiveBufferRole::Output;
};

struct TiledResult {
  std::vector<TiledOp> ops;
  size_t staging_bytes_required = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  int rank = 0;
  int nranks = 1;
  ReductionKind reduction = ReductionKind::None;
  ScalarType dtype = ScalarType::Float32;
  // Signal-tag layout: low tag_group_bits bits hold the tile-group
  // index (all-ones reserved for the copies-done handshake), pair_id
  // sits above. Plan-adaptive, identical on all ranks.
  uint32_t tag_group_bits = 0;
  // PutSignal fusion metadata for signal groups whose tag fits the
  // 32-bit RDMA immediate:
  // - fused_put_signal: (signal_op_idx, put_op_idx) for EVERY Put of an
  //   eligible group (G entries per group; each Put may carry the group
  //   tag as an imm — the receiver then counts G arrivals).
  // - sig_group_size: (signal_op_idx, group_put_count).
  // - wait_group_size: (wait_op_idx, group_tile_count) — the expected
  //   tag arrivals when the sender fuses the group (else 1).
  std::vector<std::pair<uint32_t, uint32_t>> fused_put_signal;
  std::vector<std::pair<uint32_t, uint32_t>> sig_group_size;
  std::vector<std::pair<uint32_t, uint32_t>> wait_group_size;
};

}  // namespace CCL
}  // namespace UKernel
