from __future__ import annotations

import os

from . import Context, mpi_finalize, mpi_rank, mpi_world_size


def main() -> int:
    if (
        Context is None
        or mpi_finalize is None
        or mpi_rank is None
        or mpi_world_size is None
    ):
        raise RuntimeError("uccl_gin._uccl_gin extension is not built")

    rank = int(mpi_rank())
    world = int(mpi_world_size())
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    max_message_bytes = int(os.environ.get("UCCL_GIN_CONTEXT_BYTES", str(1 << 20)))
    ifname = os.environ.get("NCCL_SOCKET_IFNAME", "enp71s0")

    ctx = Context(
        max_message_bytes=max_message_bytes,
        local_world_size=local_world,
        ifname=ifname,
    )
    resources = ctx.resources()
    assert ctx.max_message_bytes == max_message_bytes
    assert ctx.window_bytes == 2 * max_message_bytes
    assert ctx.num_queues > 0
    assert resources["num_queues"] == ctx.num_queues
    assert resources["num_lanes"] > 0
    assert resources["num_scaleup_ranks"] == local_world
    assert resources["num_scaleout_ranks"] == max(1, world // local_world)
    assert resources["window_base"] != 0
    assert resources["atomic_tail_base"] != 0
    ctx.close()

    if rank == 0:
        print(
            "uccl_gin Context smoke PASS "
            f"world={world} local_world={local_world} "
            f"max_message_bytes={max_message_bytes}"
        )
    mpi_finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
