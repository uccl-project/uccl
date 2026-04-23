"""Torchrun-compatible launcher that calls ``jax.distributed.initialize``.

This wraps ``test_moe_multiprocess.py`` / ``test_moe_low_latency_multiprocess.py``
(which the author explicitly did not self-init) so they can be launched with a
plain ``torchrun --standalone --nproc_per_node=N`` invocation.

Usage (from inside ``ep/jax_wrapper``)::

    torchrun --standalone --nproc_per_node=8 \
        examples/_launch_with_jax_dist.py \
        examples/test_moe_multiprocess.py -- \
        --num-tokens 128 --hidden 1024 --num-topk 4 --num-experts 32

Everything after ``--`` is forwarded as ``sys.argv`` to the target script.
"""

from __future__ import annotations

import os
import runpy
import sys


def _read_torchrun_env():
    """Extract the distributed bootstrap info torchrun publishes.

    Torchrun already binds ``MASTER_PORT`` as its own TCPStore, so we
    cannot reuse it for JAX's gRPC coordinator. Instead we derive a
    dedicated, deterministic coordinator port from ``MASTER_PORT``
    (``MASTER_PORT + 1``) so every rank agrees on where the
    coordinator lives.
    """
    required = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing = [k for k in required if k not in os.environ]
    if missing:
        raise SystemExit(
            "This launcher must be invoked under torchrun; missing env vars: "
            + ", ".join(missing)
        )
    master_port = int(os.environ["MASTER_PORT"])
    jax_port = int(os.environ.get("JAX_COORDINATOR_PORT", master_port + 1))
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDR",
                                      os.environ["MASTER_ADDR"])
    return {
        "rank": int(os.environ["RANK"]),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "coordinator": f"{coordinator_addr}:{jax_port}",
        "port": jax_port,
    }


def main():
    # Split sys.argv around "--": argv[1:sep] is the target script, then
    # everything after is forwarded to it.
    try:
        sep = sys.argv.index("--")
    except ValueError:
        # No explicit separator: treat the first positional as the script
        # and pass the rest through.
        sep = 2 if len(sys.argv) >= 2 else 1

    if sep < 2:
        raise SystemExit(
            "Usage: _launch_with_jax_dist.py <script> [-- <script args...>]"
        )
    target = sys.argv[1]
    target_args = sys.argv[sep + 1 :] if sep + 1 <= len(sys.argv) else []

    env = _read_torchrun_env()

    import jax

    # Some JAX builds default ``coordinator_bind_address`` to ``[::]``
    # which fails on hosts without IPv6. Bind to the coordinator
    # address explicitly so the gRPC service listens on a routable v4
    # interface on every launcher.
    bind_addr = os.environ.get(
        "JAX_COORDINATOR_BIND_ADDRESS", f"0.0.0.0:{env['port']}"
    )
    jax.distributed.initialize(
        coordinator_address=env["coordinator"],
        num_processes=env["world_size"],
        process_id=env["rank"],
        local_device_ids=[env["local_rank"]],
        coordinator_bind_address=bind_addr,
    )

    if env["rank"] == 0:
        print(
            f"[launcher] jax.distributed.initialize ok: "
            f"world_size={env['world_size']} local_rank={env['local_rank']} "
            f"coordinator={env['coordinator']}",
            flush=True,
        )

    # Rewrite sys.argv so the target script sees a normal
    # ``script --arg value ...`` command line.
    sys.argv = [target, *target_args]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
