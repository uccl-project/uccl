import torch
import ukernel


def build_moe_dag():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Typical MoE params:
    mini_batch = 2
    seq_len = 8
    hidden_size = 128
    ffn_hidden = 256
    num_experts = 8
    topk = 2
    world_size = 4

    tokens_per_batch = mini_batch * seq_len
    tokens_local = max(1, tokens_per_batch // world_size)

    # --- Tensors (placeholders but shape-realistic) ---
    tokens = torch.randn(tokens_per_batch, hidden_size, device=device)
    gate_weights = torch.randn(hidden_size, num_experts, device=device)
    weight_up = torch.randn(hidden_size, ffn_hidden, device=device)
    weight_gate = torch.randn(hidden_size, ffn_hidden, device=device)
    weight_down = torch.randn(ffn_hidden, hidden_size, device=device)

    # Routing metadata (typical shapes)
    expert_ids = torch.empty(tokens_per_batch, topk, dtype=torch.int32, device=device)
    expert_scores = torch.empty(tokens_per_batch, topk, dtype=tokens.dtype, device=device)

    # Dispatch outputs (shape-realistic)
    tokens_expert = torch.empty(tokens_local, hidden_size, device=device)
    token_index_local = torch.empty(tokens_local, dtype=torch.int32, device=device)
    expert_scores_local = torch.empty(tokens_local, topk, dtype=tokens.dtype, device=device)
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

    # FFN / Combine buffers
    expert_output = torch.empty(tokens_local, hidden_size, device=device)
    gathered_output = torch.empty(tokens_local, hidden_size, device=device)
    output = torch.empty(tokens_per_batch, hidden_size, device=device)

    # Parallel rule (task tiling)
    rule = ukernel.ParallelRule(num_tasks=2, tiles_per_task=4)

    # --- Operator DAG (Router -> Dispatch -> FFN -> Combine) ---
    # Router: light GEMM + softmax/topk -> metadata
    router_op = ukernel.moe_routing(tokens, tokens, rule)
    router_op.set_inputs([tokens, gate_weights])
    router_op.set_outputs([expert_ids, expert_scores])
    router_op.set_attr("topk", str(topk))
    router_op.set_attr("capacity_factor", "1.0")

    # Dispatch: permutation + AllToAll (placeholder; AllToAllV later)
    dispatch_op = ukernel.all_to_all(tokens, tokens_expert, rule, deps=[router_op.id])
    dispatch_op.set_inputs([tokens, expert_ids, expert_scores])
    dispatch_op.set_outputs([tokens_expert, token_index_local, expert_scores_local, expert_offsets])
    dispatch_op.set_layout("SP", "EP")

    # FFN: heavy GEMM + activation (SwiGLU etc.) -> use expert_gemm as placeholder
    ffn_op = ukernel.moe_expert_gemm(tokens_expert, expert_output, rule, deps=[dispatch_op.id])
    ffn_op.set_inputs([tokens_expert, expert_offsets, weight_up, weight_gate, weight_down])
    ffn_op.set_outputs([expert_output])
    ffn_op.set_attr("activation", "swiglu")

    # Combine: gather + reduction/scatter -> use AllToAll + combine as placeholder
    gather_op = ukernel.all_to_all(expert_output, gathered_output, rule, deps=[ffn_op.id])
    gather_op.set_inputs([expert_output])
    gather_op.set_outputs([gathered_output])
    gather_op.set_layout("EP", "SP")

    combine_op = ukernel.moe_combine(gathered_output, output, rule, deps=[gather_op.id])
    combine_op.set_inputs([gathered_output, token_index_local])
    combine_op.set_outputs([output])

    ops = [router_op, dispatch_op, ffn_op, gather_op, combine_op]

    meta = {
        "gate_weights": gate_weights,
        "expert_ids": expert_ids,
        "expert_scores": expert_scores,
        "token_idx_local": token_index_local,
        "expert_scores_local": expert_scores_local,
        "expert_offsets": expert_offsets,
    }
    return ops, meta


def main():
    cfg = ukernel.SchedulerConfig()
    cfg.dummy = 0
    ukernel.init(cfg)

    ops, _ = build_moe_dag()
    for op in ops:
        ukernel.add(op)
    
    ukernel.run()
    ukernel.sync_all()

if __name__ == "__main__":
    main()
