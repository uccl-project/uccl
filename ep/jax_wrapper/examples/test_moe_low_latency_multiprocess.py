"""Multi-process ``moe_low_latency_dispatch/combine`` example.

Launch this script with ``torchrun`` / ``mpirun`` / ``jax.distributed``
bootstrap similar to the NVIDIA/TransformerEngine and ROCm/mori tests:

    # Example: 8 ranks on a single host, one process per GPU.
    jax_dist_init="jax.distributed.initialize()" \\
    torchrun --standalone --nproc_per_node=8 test_moe_low_latency_multiprocess.py \\
        --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=32
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp

import uccl_ep_jax as ucx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=32)
    args = parser.parse_args()

    # NB: The user is responsible for calling ``jax.distributed.initialize()``
    # in their launcher; we don't do it here to avoid double-init issues.
    if jax.process_count() <= 1:
        raise SystemExit(
            "This example requires jax.process_count() > 1. Initialize JAX "
            "distributed before running it."
        )

    assert ucx.detect_execution_mode() is ucx.JaxExecutionMode.MULTI_PROCESS

    world_size = jax.process_count() * jax.local_device_count()
    num_rdma_bytes = ucx.get_low_latency_rdma_size_hint(
        args.num_tokens, args.hidden, world_size, args.num_experts
    )
    buf = ucx.initialize(
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_experts=args.num_experts,
        num_qps_per_rank=max(args.num_experts // world_size, 1),
        explicitly_destroy=True,
    )

    key = jax.random.PRNGKey(jax.process_index())
    x = jax.random.normal(key, (args.num_tokens, args.hidden), dtype=jnp.bfloat16)
    scores = jnp.abs(jax.random.normal(key, (args.num_tokens, args.num_experts), dtype=jnp.float32)) + 1
    _, topk_idx = jax.lax.top_k(scores, args.num_topk)
    topk_idx = topk_idx.astype(jnp.int64)
    topk_weights = jnp.abs(
        jax.random.normal(key, (args.num_tokens, args.num_topk), dtype=jnp.float32)
    )

    # Primitive API: the XLA custom call is issued synchronously on
    # the stream XLA picked for this process's GPU; no ``event`` /
    # ``hook`` return values are needed.
    @jax.jit
    def dispatch_combine(x, topk_idx, topk_weights):
        recv_x, recv_count, handle = ucx.moe_low_latency_dispatch(
            x, topk_idx,
            num_max_dispatch_tokens_per_rank=args.num_tokens,
            num_experts=args.num_experts,
            num_ranks=world_size,
            use_fp8=False,
        )
        return ucx.moe_low_latency_combine(recv_x, topk_idx, topk_weights, handle)

    combined = dispatch_combine(x, topk_idx, topk_weights)
    combined.block_until_ready()

    if jax.process_index() == 0:
        print(
            f"Process 0 done: combined shape={combined.shape}, "
            f"world_size={world_size}"
        )

    ucx.shutdown(buf)


if __name__ == "__main__":
    main()
