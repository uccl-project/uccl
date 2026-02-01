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
  Dispatch,
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

  std::vector<torch::Tensor> inputs;
  std::vector<torch::Tensor> outputs;

  std::unordered_map<std::string, std::string> attrs;
  std::optional<std::pair<std::string, std::string>>
      layout;  // in_layout, out_layout

  std::vector<int64_t> shape;
  int64_t numel = 0;
};

struct OperatorFactory {
 private:
  static inline std::atomic<uint64_t> g_next_id{1};

  static uint64_t next_id() {
    return g_next_id.fetch_add(1, std::memory_order_relaxed);
  }

  static void infer_shape_meta_(Operator& op) {
    torch::Tensor ref;
    if (!op.inputs.empty() && op.inputs[0].defined()) {
      ref = op.inputs[0];
    } else if (!op.outputs.empty() && op.outputs[0].defined()) {
      ref = op.outputs[0];
    }

    if (ref.defined()) {
      op.shape = ref.sizes().vec();
      op.numel = ref.numel();
    } else {
      op.shape.clear();
      op.numel = 0;
    }
  }

  static Operator base(OpType type, std::vector<torch::Tensor> inputs,
                       std::vector<torch::Tensor> outputs, ParallelRule rule,
                       std::vector<uint64_t> deps) {
    Operator op;
    op.id = next_id();
    op.type = type;
    op.parallel_rule = rule;
    op.deps = std::move(deps);
    op.inputs = std::move(inputs);
    op.outputs = std::move(outputs);

    infer_shape_meta_(op);
    return op;
  }

 public:
  static void ResetId(uint64_t start_from = 1) {
    g_next_id.store(start_from, std::memory_order_relaxed);
  }

  // P2P
  // Send: inputs[0] is the local buffer to send
  static Operator P2PSend(torch::Tensor src, ParallelRule rule, int peer_rank,
                          size_t offset = 0,
                          std::optional<size_t> len = std::nullopt,
                          bool on_gpu = true, std::vector<uint64_t> deps = {}) {
    if (!src.defined()) {
      throw std::runtime_error("P2PSend: src tensor is not defined");
    }

    // Ensure the operator has the correct inputs and outputs
    Operator op = base(OpType::P2P, {src}, {}, rule, std::move(deps));
    op.p2p_kind = P2PKind::Send;

    op.attrs["peer"] = std::to_string(peer_rank);
    op.attrs["offset"] = std::to_string(offset);
    if (len) op.attrs["len"] = std::to_string(*len);
    op.attrs["on_gpu"] = on_gpu ? "1" : "0";
    return op;
  }

  // Recv: outputs[0] is the local buffer to receive into
  static Operator P2PRecv(torch::Tensor dst, ParallelRule rule, int peer_rank,
                          size_t offset = 0,
                          std::optional<size_t> len = std::nullopt,
                          bool on_gpu = true, std::vector<uint64_t> deps = {}) {
    if (!dst.defined()) {
      throw std::runtime_error("P2PRecv: dst tensor is not defined");
    }

    // Ensure the operator has the correct inputs and outputs
    Operator op = base(OpType::P2P, {}, {dst}, rule, std::move(deps));
    op.p2p_kind = P2PKind::Recv;

    op.attrs["peer"] = std::to_string(peer_rank);
    op.attrs["offset"] = std::to_string(offset);
    if (len) op.attrs["len"] = std::to_string(*len);
    op.attrs["on_gpu"] = on_gpu ? "1" : "0";
    return op;
  }

  // Collective
  static Operator AllReduce(torch::Tensor src, torch::Tensor dst,
                            ReduceKind kind, ParallelRule rule,
                            std::vector<uint64_t> deps = {}) {
    if (!src.defined() || !dst.defined()) {
      throw std::runtime_error("AllReduce: src/dst tensor not defined");
    }

    Operator op = base(OpType::Collective, {src}, {dst}, rule, std::move(deps));
    op.collective_kind = CollectiveKind::AllReduce;
    op.reduce_kind = kind;
    return op;
  }

  static Operator AllToAll(torch::Tensor src, torch::Tensor dst,
                           ParallelRule rule, std::vector<uint64_t> deps = {}) {
    if (!src.defined() || !dst.defined()) {
      throw std::runtime_error("AllToAll: src/dst tensor not defined");
    }

    Operator op = base(OpType::Collective, {src}, {dst}, rule, std::move(deps));
    op.collective_kind = CollectiveKind::AllToAll;
    return op;
  }

  // Generic Compute
  static Operator Gemm(torch::Tensor src, torch::Tensor dst, ParallelRule rule,
                       std::vector<uint64_t> deps = {}) {
    if (!src.defined() || !dst.defined()) {
      throw std::runtime_error("Gemm: src/dst tensor not defined");
    }
    return base(OpType::Compute, {src}, {dst}, rule, std::move(deps));
  }

  // MoE
  static Operator MoeRouting(torch::Tensor src, torch::Tensor dst,
                             ParallelRule rule,
                             std::vector<uint64_t> deps = {}) {
    if (!src.defined() || !dst.defined()) {
      throw std::runtime_error("MoeRouting: src/dst tensor not defined");
    }

    Operator op = base(OpType::Moe, {src}, {dst}, rule, std::move(deps));
    op.moe_kind = MoeKind::Routing;
    return op;
  }

  static Operator MoeDispatch(torch::Tensor src, torch::Tensor dst,
                              ParallelRule rule,
                              std::vector<uint64_t> deps = {}) {
    if (!src.defined() || !dst.defined()) {
      throw std::runtime_error("MoeDispatch: src/dst tensor not defined");
    }

    Operator op = base(OpType::Moe, {src}, {dst}, rule, std::move(deps));
    op.moe_kind = MoeKind::Dispatch;
    return op;
  }

  static Operator MoeExpertGemm(torch::Tensor src, torch::Tensor dst,
                                ParallelRule rule,
                                std::vector<uint64_t> deps = {}) {
    if (!src.defined() || !dst.defined()) {
      throw std::runtime_error("MoeExpertGemm: src/dst tensor not defined");
    }

    Operator op = base(OpType::Moe, {src}, {dst}, rule, std::move(deps));
    op.moe_kind = MoeKind::ExpertGemm;
    return op;
  }

  static Operator MoeCombine(torch::Tensor src, torch::Tensor dst,
                             ParallelRule rule,
                             std::vector<uint64_t> deps = {}) {
    if (!src.defined() || !dst.defined()) {
      throw std::runtime_error("MoeCombine: src/dst tensor not defined");
    }

    Operator op = base(OpType::Moe, {src}, {dst}, rule, std::move(deps));
    op.moe_kind = MoeKind::Combine;
    return op;
  }
};

}  // namespace Runtime
}  // namespace UKernel