"""Multi-process ``moe_dispatch`` / ``moe_combine`` example.

This script is self-launching: run it with plain ``python`` and it will
spawn one child process per local GPU, initialize ``jax.distributed`` in
each child, and then run the real workload. No ``torchrun`` / ``mpirun``
needed.

Single host, 8 GPUs::

    python examples/test_moe_multiprocess.py \\
        --nproc-per-node=8 \\
        --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 32

Two hosts, 8 GPUs each (run on BOTH hosts)::

    # on host 0
    python examples/test_moe_multiprocess.py \\
        --nnodes=2 --node-rank=0 \\
        --master-addr=<host0_ip> --master-port=12355 \\
        --nproc-per-node=8 \\
        --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 64

    # on host 1
    python examples/test_moe_multiprocess.py \\
        --nnodes=2 --node-rank=1 \\
        --master-addr=<host0_ip> --master-port=12355 \\
        --nproc-per-node=8 \\
        --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 64

All launcher flags start with ``--nproc-per-node`` / ``--nnodes`` /
``--node-rank`` / ``--master-addr`` / ``--master-port``; everything else is
forwarded to the per-rank workload.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import traceback


# ---------------------------------------------------------------------------
# Launcher: spawn one child per local GPU, set up jax.distributed, run main.
# ---------------------------------------------------------------------------


def _parse_launcher_args(argv):
    """Split argv into launcher flags and workload flags.

    Returns (launcher_ns, workload_argv). The launcher flags are stripped
    from argv before the workload parser ever sees them.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nproc-per-node", type=int, default=8,
                        help="Processes (GPUs) to spawn on this node. "
                             "Defaults to the number of visible CUDA devices.")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=12355)
    parser.add_argument("--_worker", action="store_true",
                        help=argparse.SUPPRESS)
    ns, rest = parser.parse_known_args(argv)
    return ns, rest


def _detect_local_gpu_count() -> int:
    """Best-effort local GPU count without importing jax/cuda runtime."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([s for s in cvd.split(",") if s.strip() != ""])
    hvd = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
    if hvd:
        return len([s for s in hvd.split(",") if s.strip() != ""])
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        n = len([ln for ln in out.decode().splitlines() if ln.strip()])
        if n > 0:
            return n
    except Exception:
        pass
    return 1


def _worker_entry(local_rank: int, launcher_ns, workload_argv):
    """Child-process entry: init jax.distributed, then run main()."""
    nproc_per_node = launcher_ns.nproc_per_node
    nnodes = launcher_ns.nnodes
    node_rank = launcher_ns.node_rank
    world_size = nnodes * nproc_per_node
    global_rank = node_rank * nproc_per_node + local_rank

    # NOTE: intentionally do NOT set CUDA_VISIBLE_DEVICES=local_rank here.
    # uccl.ep registers proxies under the CUDA ordinal that
    # ``cudaGetDevice()`` returns inside the FFI handler, and uccl_ep_jax's
    # ``_get_device_index(local_rank)`` uses ``local_hardware_id`` -- both
    # must agree. If we pinched the visible set to a single GPU, every
    # child would see that one GPU as ordinal 0 and the registry lookup
    # in ``ep.Buffer(...)`` would fail for local_rank > 0.
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    # Make the true "per-node GPU count" visible to the workload.
    # ``jax.local_device_count()`` is NOT a reliable source here, because
    # in multi-process mode each process calls
    # ``jax.distributed.initialize(local_device_ids=[local_rank])`` and
    # therefore sees exactly one local device. uccl_ep's
    # ``num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS`` derivation needs
    # to know how many GPUs are actually NVLink-connected on this host,
    # which is exactly ``nproc_per_node`` for single-node jobs.
    os.environ["LOCAL_WORLD_SIZE"] = str(nproc_per_node)
    os.environ["NPROC_PER_NODE"] = str(nproc_per_node)
    os.environ["NNODES"] = str(nnodes)

    try:
        import jax

        coordinator = f"{launcher_ns.master_addr}:{launcher_ns.master_port}"
        # Only rank 0 binds the coordinator's listen socket; other ranks
        # are clients and should not try to bind the same port.
        if global_rank == 0:
            bind_addr = os.environ.get(
                "JAX_COORDINATOR_BIND_ADDRESS",
                f"0.0.0.0:{launcher_ns.master_port}",
            )
            init_kwargs = dict(coordinator_bind_address=bind_addr)
        else:
            init_kwargs = {}
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=world_size,
            process_id=global_rank,
            local_device_ids=[local_rank],
            **init_kwargs,
        )

        if global_rank == 0:
            print(
                f"[launcher] jax.distributed ready: world_size={world_size} "
                f"nnodes={nnodes} nproc_per_node={nproc_per_node} "
                f"coordinator={coordinator}",
                flush=True,
            )

        sys.argv = [sys.argv[0], *workload_argv]
        _run_workload()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)


def _spawn_and_wait(launcher_ns, workload_argv) -> int:
    """Spawn ``nproc_per_node`` children and wait for them."""
    ctx = mp.get_context("spawn")
    procs = []
    for local_rank in range(launcher_ns.nproc_per_node):
        p = ctx.Process(
            target=_worker_entry,
            args=(local_rank, launcher_ns, workload_argv),
            daemon=False,
        )
        p.start()
        procs.append(p)

    exit_code = 0
    try:
        for p in procs:
            p.join()
            if p.exitcode != 0 and exit_code == 0:
                exit_code = p.exitcode or 1
    except KeyboardInterrupt:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
        exit_code = 130
    return exit_code


# ---------------------------------------------------------------------------
# Workload: the actual moe_dispatch / moe_combine test.
# ---------------------------------------------------------------------------


def _parse_workload_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=32)
    return parser.parse_args()


def _run_workload():
    """Real moe_dispatch / moe_combine workload.

    Runs inside a child process that has already called
    ``jax.distributed.initialize``.
    """
    import jax
    import jax.numpy as jnp

    import uccl_ep_jax as ucx

    args = _parse_workload_args()

    if jax.process_count() <= 1:
        raise SystemExit(
            "This example requires jax.process_count() > 1. Run it via "
            "the built-in launcher, e.g. "
            "`python test_moe_multiprocess.py --nproc-per-node=8 ...`."
        )
    assert ucx.detect_execution_mode() is ucx.JaxExecutionMode.MULTI_PROCESS

    # -----------------------------------------------------------------
    # Topology: derive (world_size, nproc_per_node, nnodes) from the
    # launcher-injected env vars, then cross-check against what JAX
    # believes.  We rely on the launcher as the ground truth because
    # JAX has no way to know the physical NVLink domain size in
    # process-per-GPU mode (every process sees exactly one device).
    # -----------------------------------------------------------------
    nproc_per_node = int(os.environ.get("LOCAL_WORLD_SIZE",
                                        os.environ.get("NPROC_PER_NODE", "0")))
    nnodes = int(os.environ.get("NNODES", "0"))
    if nproc_per_node <= 0 or nnodes <= 0:
        raise SystemExit(
            "LOCAL_WORLD_SIZE/NPROC_PER_NODE and NNODES env vars not set. "
            "Launch via the built-in launcher: "
            "`python test_moe_multiprocess.py --nproc-per-node=N [--nnodes=M ...]`."
        )
    world_size = nnodes * nproc_per_node
    # ``jax.process_count()`` is the number of Python processes ACROSS
    # ALL hosts (the ``num_processes`` passed to
    # ``jax.distributed.initialize``). It is NOT a per-node quantity.
    # With one process per GPU, it must equal world_size; if it doesn't
    # the launcher and jax.distributed disagree on the topology and
    # nothing below will be correct.
    if jax.process_count() != world_size:
        raise SystemExit(
            f"Topology mismatch: launcher says world_size="
            f"{world_size} (nnodes={nnodes} x nproc_per_node={nproc_per_node}), "
            f"but jax.process_count()={jax.process_count()}."
        )
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
        num_rdma_bytes=0,
        num_nvl_bytes=num_nvl_bytes,
        low_latency_mode=False,
        num_experts=args.num_experts,
        num_qps_per_rank=max(args.num_experts // world_size, 1),
        explicitly_destroy=True,
        # CRITICAL: tell uccl_ep_jax the real per-node GPU count; the
        # default (``jax.local_device_count()``) would be 1 here and
        # the C++ side would compute num_rdma_ranks = world_size / 1,
        # treating a single-node run as if every GPU were on its own node.
        local_world_size=nproc_per_node,
    )

    if jax.process_index() == 0:
        print(
            f"Running moe_dispatch/combine with world_size={world_size} "
            f"(nnodes={nnodes}, nproc_per_node={nproc_per_node}, "
            f"jax.process_count={jax.process_count()}, "
            f"jax.local_device_count={jax.local_device_count()}, "
            f"num_rdma_ranks={buf.num_rdma_ranks})",
            flush=True,
        )

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


# ---------------------------------------------------------------------------
# Entry point: acts as launcher by default, as workload under --_worker.
# ---------------------------------------------------------------------------


def main():
    launcher_ns, workload_argv = _parse_launcher_args(sys.argv[1:])

    if launcher_ns.nproc_per_node is None:
        launcher_ns.nproc_per_node = _detect_local_gpu_count()

    if launcher_ns.nproc_per_node <= 0:
        raise SystemExit("--nproc-per-node must be >= 1")
    if launcher_ns.nnodes <= 0:
        raise SystemExit("--nnodes must be >= 1")
    if not (0 <= launcher_ns.node_rank < launcher_ns.nnodes):
        raise SystemExit(
            f"--node-rank ({launcher_ns.node_rank}) must be in "
            f"[0, --nnodes={launcher_ns.nnodes})"
        )

    print(
        f"[launcher] spawning {launcher_ns.nproc_per_node} process(es) "
        f"on node_rank={launcher_ns.node_rank}/{launcher_ns.nnodes}, "
        f"coordinator={launcher_ns.master_addr}:{launcher_ns.master_port}",
        flush=True,
    )
    sys.exit(_spawn_and_wait(launcher_ns, workload_argv))


if __name__ == "__main__":
    main()
