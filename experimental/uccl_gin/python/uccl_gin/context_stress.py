from __future__ import annotations

import os

from . import Context, mpi_finalize, mpi_rank


def main() -> int:
    if Context is None or mpi_finalize is None or mpi_rank is None:
        raise RuntimeError("uccl_gin._uccl_gin extension is not built")

    rank = int(mpi_rank())
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    iterations = int(os.environ.get("UCCL_GIN_CONTEXT_STRESS_ITERS", "100"))
    max_message_bytes = int(os.environ.get("UCCL_GIN_CONTEXT_BYTES", str(1 << 20)))
    ifname = os.environ.get("NCCL_SOCKET_IFNAME", "enp71s0")

    for _ in range(iterations):
        ctx = Context(
            max_message_bytes=max_message_bytes,
            local_world_size=local_world,
            ifname=ifname,
        )
        assert ctx.num_queues > 0
        ctx.close()

    if rank == 0:
        print(f"uccl_gin Context stress PASS iterations={iterations}")
    mpi_finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
