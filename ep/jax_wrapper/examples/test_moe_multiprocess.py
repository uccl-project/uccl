"""Multi-process ``moe_dispatch`` / ``moe_combine`` example.

Launch this script through a JAX multi-process bootstrapper such as
``torchrun`` or ``mpirun``. Each process owns one GPU. The user is
responsible for calling ``jax.distributed.initialize()`` in their
launcher (we don't do it here to avoid double-init issues).

Intranode example (one host, 8 GPUs, 8 processes):

    torchrun --standalone --nproc_per_node=8 \\
        examples/test_moe_multiprocess.py \\
        --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 32

Internode example (2 hosts, 8 GPUs each, 16 processes; ``num_ranks=16``
> ``local_device_count=8``, so the internode primitive is selected):

    # on host 0
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
        --master_addr=<host0_ip> --master_port=12355 \\
        examples/test_moe_multiprocess.py \\
        --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 64

    # on host 1
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \\
        --master_addr=<host0_ip> --master_port=12355 \\
        examples/test_moe_multiprocess.py \\
        --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 64
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import uccl_ep_jax as ucx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=32)
    args = parser.parse_args()

    # The user must have called jax.distributed.initialize() before
    # running this example. If they haven't, bail out early.
    if jax.process_count() <= 1:
        raise SystemExit(
            "This example requires jax.process_count() > 1. Initialize "
            "JAX distributed (e.g. via torchrun/mpirun) before running."
        )
    assert ucx.detect_execution_mode() is ucx.JaxExecutionMode.MULTI_PROCESS

    world_size = jax.process_count() * jax.local_device_count()
    assert (
        args.num_experts % world_size == 0
    ), f"num_experts ({args.num_experts}) must be divisible by world_size ({world_size})"

    # High-throughput moe_dispatch / moe_combine need:
    #   * num_nvl_bytes for the NVLink/xGMI intranode scratch,
    #   * num_rdma_bytes for the internode RDMA scratch (uses the
    #     high-throughput sizing formula, not the low-latency one).
    hidden_bytes = args.hidden * 2  # bf16 input
    num_nvl_bytes = ucx.get_nvl_buffer_size_hint(hidden_bytes, world_size)
    num_rdma_bytes = ucx.get_rdma_buffer_size_hint(hidden_bytes, world_size)
    buf = ucx.initialize(
        num_rdma_bytes=num_rdma_bytes,
        num_nvl_bytes=num_nvl_bytes,
        low_latency_mode=False,
        num_experts=args.num_experts,
        num_qps_per_rank=max(args.num_experts // world_size, 1),
        explicitly_destroy=True,
    )

    if jax.process_index() == 0:
        print(
            f"Running moe_dispatch/combine with world_size={world_size} "
            f"(process_count={jax.process_count()}, "
            f"local_device_count={jax.local_device_count()}, "
            f"num_rdma_ranks={buf.num_rdma_ranks})",
            flush=True,
        )

    # Generate synthetic data on this process's only local GPU.
    key = jax.random.PRNGKey(jax.process_index())
    x = jax.random.normal(
        key, (args.num_tokens, args.hidden), dtype=jnp.bfloat16
    )
    scores = jnp.abs(
        jax.random.normal(
            key, (args.num_tokens, args.num_experts), dtype=jnp.float32
        )
    ) + 1.0
    _topk_vals, topk_idx = jax.lax.top_k(scores, args.num_topk)
    topk_idx = topk_idx.astype(jnp.int32)
    topk_weights = jnp.abs(
        jax.random.normal(
            key, (args.num_tokens, args.num_topk), dtype=jnp.float32
        )
    )

    # ---------------------------------------------------------------
    # 1) Forward: jit(dispatch -> expert placeholder -> combine).
    # ---------------------------------------------------------------
    @jax.jit
    def fwd(x, topk_idx, topk_weights):
        recv_x, _recv_ti, recv_tw, handle = ucx.moe_dispatch(
            x, topk_idx, topk_weights,
            num_experts=args.num_experts, num_ranks=world_size,
        )
        expert_out = recv_x  # placeholder for the real expert MLP
        return ucx.moe_combine(
            expert_out, handle, topk_weights=recv_tw,
            num_ranks=world_size,
        )

    combined = fwd(x, topk_idx, topk_weights)
    combined.block_until_ready()

    # ---------------------------------------------------------------
    # 2) Backward: jax.grad exercises both custom_vjp rules.
    # ---------------------------------------------------------------
    @jax.jit
    def loss(x, topk_idx, topk_weights):
        return fwd(x, topk_idx, topk_weights).sum().astype(jnp.float32)

    grad_x = jax.grad(loss, argnums=0)(x, topk_idx, topk_weights)
    grad_x.block_until_ready()

    if jax.process_index() == 0:
        print(
            f"Process 0 done: combined.shape={combined.shape}, "
            f"grad_x.shape={grad_x.shape}",
            flush=True,
        )

    ucx.shutdown(buf)


if __name__ == "__main__":
    main()
