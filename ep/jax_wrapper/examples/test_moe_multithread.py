"""Single-process, multi-thread ``moe_dispatch`` / ``moe_combine`` example.

Spawns one Python thread per local GPU. Each thread owns a dedicated
``uccl_ep_jax.Buffer`` via ``initialize(local_rank=i, ...)``, and
drives a jitted dispatch+combine round trip. The test also exercises
``jax.grad`` so the ``custom_vjp`` path (combine backward ->
cached-mode dispatch replay) is validated end-to-end.

Run with JAX configured for a single host (no ``jax.distributed`` /
``torchrun`` wrapping):

    python test_moe_multithread.py \\
        --num-tokens=4096 --hidden=7168 --num-topk=8 --num-experts=32
"""

from __future__ import annotations

import argparse
import threading
import traceback

import jax
import jax.numpy as jnp

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

        # High-throughput moe_dispatch / moe_combine drive kernels
        # that need a per-device NVLink scratch buffer; the low-
        # latency kernels additionally want an RDMA scratch buffer.
        hidden_bytes = hidden * 2  # bf16 input
        num_nvl_bytes = ucx.get_nvl_buffer_size_hint(hidden_bytes, world_size)
        num_rdma_bytes = ucx.get_low_latency_rdma_size_hint(
            num_tokens, hidden, world_size, num_experts
        )
        buf = ucx.initialize(
            num_rdma_bytes=num_rdma_bytes,
            num_nvl_bytes=num_nvl_bytes,
            low_latency_mode=False,
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
            jax.random.normal(
                key, (num_tokens, hidden), dtype=jnp.bfloat16
            ),
            device=device,
        )
        scores = jnp.abs(
            jax.random.normal(key, (num_tokens, num_experts), dtype=jnp.float32)
        ) + 1.0
        _topk_vals, topk_idx = jax.lax.top_k(scores, num_topk)
        topk_idx = jax.device_put(topk_idx.astype(jnp.int32), device=device)
        topk_weights = jax.device_put(
            jnp.abs(
                jax.random.normal(
                    key, (num_tokens, num_topk), dtype=jnp.float32
                )
            ),
            device=device,
        )

        # ------------------------------------------------------------
        # 1) Forward: jitted dispatch -> (fake expert work) -> combine
        # ------------------------------------------------------------
        @jax.jit
        def fwd(x, topk_idx, topk_weights):
            recv_x, _recv_ti, recv_tw, handle = ucx.moe_dispatch(
                x, topk_idx, topk_weights,
                num_experts=num_experts, num_ranks=world_size,
            )
            # Placeholder for the expert MLP: identity in bf16. Real
            # users would do a GroupedGEMM or similar here.
            expert_out = recv_x
            combined = ucx.moe_combine(
                expert_out, handle, topk_weights=recv_tw,
                num_ranks=world_size,
            )
            return combined

        combined = fwd(x, topk_idx, topk_weights)
        combined.block_until_ready()

        # ------------------------------------------------------------
        # 2) Backward: jax.grad exercises both custom_vjp rules:
        #    * moe_dispatch_bwd = moe_combine(grad)
        #    * moe_combine_bwd  = moe_cached_dispatch(grad)
        # ------------------------------------------------------------
        @jax.jit
        def loss(x, topk_idx, topk_weights):
            out = fwd(x, topk_idx, topk_weights)
            return out.sum().astype(jnp.float32)

        grad_x = jax.grad(loss, argnums=0)(x, topk_idx, topk_weights)
        grad_x.block_until_ready()

        print(
            f"[rank {local_rank}] moe_dispatch/combine OK; "
            f"combined.shape={combined.shape}, grad_x.shape={grad_x.shape}",
            flush=True,
        )

        ucx.shutdown(buf)
    except Exception as exc:  # pragma: no cover -- runtime path
        traceback.print_exc()
        errors.append((local_rank, exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=32)
    args = parser.parse_args()

    world_size = jax.local_device_count()
    if world_size < 2:
        raise SystemExit(
            "moe_dispatch requires num_ranks >= 2; this example needs at "
            "least 2 local GPUs. Use examples/test_low_latency_* for the "
            "single-GPU path."
        )
    assert (
        args.num_experts % world_size == 0
    ), f"num_experts ({args.num_experts}) must be divisible by world_size ({world_size})"

    print(
        f"Running moe_dispatch/combine with {world_size} local devices; "
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
            name=f"ep-moe-worker-{r}",
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
