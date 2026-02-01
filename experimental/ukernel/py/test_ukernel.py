import torch
import ukernel
import os


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

    # Operator DAG:
    # Router: routing via GEMM + softmax/topk -> metadata
    router_op = ukernel.moe_routing(tokens, tokens_expert, rule)

    # Dispatch: expert dispatching (using a placeholder for AllToAll)
    dispatch_op = ukernel.moe_dispatch(
        tokens_expert,
        expert_output,
        rule,
        deps=[router_op.id]
    )

    # FFN: Expert GEMM operation (heavy computation)
    ffn_op = ukernel.moe_expert_gemm(
        expert_output,
        gathered_output,
        rule,
        deps=[dispatch_op.id]
    )

    # Combine: gather results from all experts
    combine_op = ukernel.moe_combine(
        gathered_output,
        output,
        rule,
        deps=[ffn_op.id]
    )

    # List of all operations in the DAG
    ops = [router_op, dispatch_op, ffn_op, combine_op]

    return ops


def main():
    # Set environment variables for communicator configuration
    os.environ["UHM_EXCHANGER_SERVER_IP"] = "127.0.0.1"
    os.environ["UHM_EXCHANGER_SERVER_PORT"] = "6980"

    # Scheduler config
    cfg = ukernel.SchedulerConfig()
    cfg.gpu_id = 0
    cfg.rank = 0
    cfg.world_size = 1

    # Initialize UKernel
    ukernel.init(cfg)

    # Build DAG
    ops = build_moe_dag()

    # Add operators to the scheduler
    for op in ops:
        ukernel.add(op)

    # Run the scheduler and sync
    ukernel.run()
    ukernel.sync_all()


if __name__ == "__main__":
    main()
