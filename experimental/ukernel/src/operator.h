#pragma once
#include <torch/extension.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Runtime {

enum class OpType {
  // Point-to-point communication
  P2P,

  // Collective communication
  Collective,

  // Generic Compute
  Compute,

  // Moe
  Moe,
};

enum class P2PKind {
  Send,
  Recv,
};

enum class CollectiveKind {
  // Reductions
  Reduce,
  AllReduce,
  ReduceScatter,

  // Gather family
  Gather,
  AllGather,

  // Scatter family
  Scatter,

  // Exchange
  AllToAll,

  // Sync / control
  Barrier,
};

enum class ReduceKind {
  Sum,
  Sub,
  Avg,
  Max,
  Min,
};

enum class ComputeKind {
  Gemm,
};

enum class MoeKind {
  Routing,
  ExpertGemm,
  Combine,
};

struct ParallelRule {
  int num_tasks = 1;       // logical partitions
  int tiles_per_task = 1;  // tile-group size
};

struct Operator {
  uint64_t id = 0;
  std::vector<uint64_t> deps;

  OpType type;

  std::optional<P2PKind> p2p_kind;
  std::optional<CollectiveKind> collective_kind;
  std::optional<ReduceKind> reduce_kind;
  std::optional<MoeKind> moe_kind;

  ParallelRule parallel_rule;

  torch::Tensor src;
  torch::Tensor dst;

  // New-style IO/attrs (placeholders; keep src/dst for compatibility)
  std::vector<torch::Tensor> inputs;
  std::vector<torch::Tensor> outputs;
  std::unordered_map<std::string, std::string> attrs;
  std::optional<std::pair<std::string, std::string>>
      layout;  // in_layout, out_layout

  std::vector<int64_t> shape;
  int64_t numel = 0;
};

struct OperatorFactory {
  static Operator base(uint64_t id, OpType type, torch::Tensor src,
                       torch::Tensor dst, ParallelRule rule,
                       std::vector<uint64_t> deps) {
    Operator op;
    op.id = id;
    op.type = type;
    op.src = src;
    op.dst = dst;
    op.parallel_rule = rule;
    op.deps = std::move(deps);

    op.inputs = {src};
    op.outputs = {dst};

    op.shape = src.sizes().vec();
    op.numel = src.numel();
    return op;
  }

  // P2P
  static Operator P2PSend(uint64_t id, torch::Tensor src, torch::Tensor dst,
                          ParallelRule rule, std::vector<uint64_t> deps = {}) {
    Operator op = base(id, OpType::P2P, src, dst, rule, std::move(deps));
    op.p2p_kind = P2PKind::Send;
    return op;
  }

  static Operator P2PRecv(uint64_t id, torch::Tensor src, torch::Tensor dst,
                          ParallelRule rule, std::vector<uint64_t> deps = {}) {
    Operator op = base(id, OpType::P2P, src, dst, rule, std::move(deps));
    op.p2p_kind = P2PKind::Recv;
    return op;
  }

  // Collective
  static Operator AllReduce(uint64_t id, torch::Tensor src, torch::Tensor dst,
                            ReduceKind kind, ParallelRule rule,
                            std::vector<uint64_t> deps = {}) {
    Operator op = base(id, OpType::Collective, src, dst, rule, std::move(deps));
    op.collective_kind = CollectiveKind::AllReduce;
    op.reduce_kind = kind;
    return op;
  }

  static Operator AllToAll(uint64_t id, torch::Tensor src, torch::Tensor dst,
                           ParallelRule rule, std::vector<uint64_t> deps = {}) {
    Operator op = base(id, OpType::Collective, src, dst, rule, std::move(deps));
    op.collective_kind = CollectiveKind::AllToAll;
    return op;
  }

  // Generic Compute
  static Operator Gemm(uint64_t id, torch::Tensor src, torch::Tensor dst,
                       ParallelRule rule, std::vector<uint64_t> deps = {}) {
    return base(id, OpType::Compute, src, dst, rule, std::move(deps));
  }

  // MoE
  static Operator MoeRouting(uint64_t id, torch::Tensor src, torch::Tensor dst,
                             ParallelRule rule,
                             std::vector<uint64_t> deps = {}) {
    Operator op = base(id, OpType::Moe, src, dst, rule, std::move(deps));
    op.moe_kind = MoeKind::Routing;
    return op;
  }

  static Operator MoeExpertGemm(uint64_t id, torch::Tensor src,
                                torch::Tensor dst, ParallelRule rule,
                                std::vector<uint64_t> deps = {}) {
    Operator op = base(id, OpType::Moe, src, dst, rule, std::move(deps));
    op.moe_kind = MoeKind::ExpertGemm;
    return op;
  }

  static Operator MoeCombine(uint64_t id, torch::Tensor src, torch::Tensor dst,
                             ParallelRule rule,
                             std::vector<uint64_t> deps = {}) {
    Operator op = base(id, OpType::Moe, src, dst, rule, std::move(deps));
    op.moe_kind = MoeKind::Combine;
    return op;
  }
};

}  // namespace Runtime
}  // namespace UKernel