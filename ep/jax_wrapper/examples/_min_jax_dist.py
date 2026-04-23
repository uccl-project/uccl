"""Minimal jax.distributed.initialize smoke-test for torchrun."""
from __future__ import annotations

import os
import sys


def main():
    master_port = int(os.environ["MASTER_PORT"])
    jax_port = master_port + 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    addr = os.environ["MASTER_ADDR"]
    print(
        f"[rank={rank}] master_addr={addr} master_port={master_port} "
        f"jax_port={jax_port} world={world} local_rank={local_rank}",
        flush=True,
    )

    import jax
    try:
        jax.distributed.initialize(
            coordinator_address=f"{addr}:{jax_port}",
            num_processes=world,
            process_id=rank,
            local_device_ids=[local_rank],
            coordinator_bind_address=f"0.0.0.0:{jax_port}",
        )
    except Exception as e:
        print(f"[rank={rank}] initialize FAILED: {type(e).__name__}: {e}", flush=True)
        sys.exit(2)

    print(
        f"[rank={rank}] ok, process_count={jax.process_count()} "
        f"local_devs={jax.local_device_count()} devs={jax.devices()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
