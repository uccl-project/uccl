"""Single-process, multi-thread low-latency dispatch/combine example.

Run with JAX configured for a single host:

    python test_low_latency_multithread.py \
        --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=32

The script spawns one Python thread per local GPU, rendezvous happens
through the in-process KV store shipped with ``uccl_ep_jax``.
"""

from __future__ import annotations

import argparse
import threading
import traceback

import jax
import jax.numpy as jnp
import numpy as np

import uccl_ep_jax as ucx


def worker(
    local_rank: int,
    world_size: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    errors: list,
) -> None:
    try:
        assert (
            ucx.detect_execution_mode() is ucx.JaxExecutionMode.SINGLE_PROCESS
        ), "This example expects jax.process_count() == 1"

        num_rdma_bytes = ucx.get_low_latency_rdma_size_hint(
            num_tokens, hidden, world_size, num_experts
        )
        buf = ucx.initialize(
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_experts=num_experts,
            local_rank=local_rank,
            global_rank=local_rank,
            global_world_size=world_size,
            local_world_size=world_size,
            num_qps_per_rank=max(num_experts // world_size, 1),
            explicitly_destroy=True,
        )

        device = jax.local_devices()[local_rank]
        key = jax.random.PRNGKey(local_rank)
        x = jax.device_put(
            jax.random.normal(key, (num_tokens, hidden), dtype=jnp.bfloat16),
            device=device,
        )
        scores = jax.random.normal(key, (num_tokens, num_experts), dtype=jnp.float32)
        scores = jnp.abs(scores) + 1.0
        topk_vals, topk_idx = jax.lax.top_k(scores, num_topk)
        topk_idx = jax.device_put(topk_idx.astype(jnp.int64), device=device)
        topk_weights = jax.device_put(
            jnp.abs(jax.random.normal(key, (num_tokens, num_topk), dtype=jnp.float32)),
            device=device,
        )

        # Primitive API: the XLA custom call is issued synchronously
        # on the stream XLA picked for ``buf.cuda_device_index``; no
        # ``event`` / ``hook`` return values are needed.
        @jax.jit
        def dispatch_combine(x, topk_idx, topk_weights):
            recv_x, recv_count, handle = ucx.low_latency_dispatch(
                x, topk_idx,
                num_max_dispatch_tokens_per_rank=num_tokens,
                num_experts=num_experts,
                num_ranks=world_size,
                use_fp8=False,
            )
            return ucx.low_latency_combine(
                recv_x, topk_idx, topk_weights, handle,
            )

        combined = dispatch_combine(x, topk_idx, topk_weights)
        combined.block_until_ready()
        print(
            f"[rank {local_rank}] dispatch/combine OK, shape={combined.shape}",
            flush=True,
        )

        ucx.shutdown(buf)
    except Exception as exc:  # pragma: no cover - runtime path
        traceback.print_exc()
        errors.append((local_rank, exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=32)
    args = parser.parse_args()

    world_size = jax.local_device_count()
    print(
        f"Running with {world_size} local devices; "
        f"process_count={jax.process_count()}",
        flush=True,
    )

    errors: list = []
    threads = [
        threading.Thread(
            target=worker,
            args=(
                r,
                world_size,
                args.num_tokens,
                args.hidden,
                args.num_topk,
                args.num_experts,
                errors,
            ),
            name=f"ep-worker-{r}",
        )
        for r in range(world_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise SystemExit(f"Workers failed: {errors}")
    print("All ranks completed successfully.")


if __name__ == "__main__":
    main()
