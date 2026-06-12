"""put + quiet correctness driver (AMD/RoCE and NVIDIA/EFA).

Paired-remote: rank r talks to peer (r + local_world) % world, i.e. the same
local rank on a different node. Each rank put()s a rank-tagged pattern to its
peer, quiet()s, then verifies its own recv window holds the peer's pattern.
Only put + quiet are exercised (no atomics), so this is the validation vehicle
on non-EFA RC transports where the ordered-atomic path is unavailable.

Launch (2 nodes, 1 rank/node is the simplest rail-valid shape):

    LOCAL_WORLD_SIZE=1 NCCL_SOCKET_IFNAME=<iface> \
    mpirun --host <node0>:1,<node1>:1 -np 2 -npernode 1 \
      -x LD_LIBRARY_PATH -x PYTHONPATH -x LOCAL_WORLD_SIZE -x NCCL_SOCKET_IFNAME \
      python -m uccl_gin.put_quiet_smoke
"""

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
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    max_message_bytes = int(os.environ.get("UCCL_GIN_CONTEXT_BYTES", str(1 << 20)))
    ifname = os.environ.get("NCCL_SOCKET_IFNAME", "enp49s0f1np1")
    sizes = [
        int(s)
        for s in os.environ.get("UCCL_GIN_SMOKE_SIZES", "1024,65536,1048576").split(",")
        if s
    ]

    if world % local_world != 0 or world // local_world < 2:
        raise RuntimeError(
            f"put/quiet smoke needs >=2 nodes (world={world}, local_world={local_world})"
        )

    peer = (rank + local_world) % world

    ctx = Context(
        max_message_bytes=max_message_bytes,
        local_world_size=local_world,
        ifname=ifname,
    )

    failures = []
    for nbytes in sizes:
        if nbytes > max_message_bytes:
            continue
        ok = ctx.put_quiet_smoke(peer, nbytes)
        if not ok:
            failures.append(nbytes)
        if rank == 0:
            print(f"  put/quiet {nbytes:>9} B: {'PASS' if ok else 'FAIL'}", flush=True)

    # Optional bandwidth sweep (UCCL_GIN_BENCH=1). per-rank GB/s plus a crude
    # aggregate (per-rank * world) printed by rank 0; all ranks run in lockstep.
    if os.environ.get("UCCL_GIN_BENCH") == "1" and not failures:
        iters = int(os.environ.get("UCCL_GIN_BENCH_ITERS", "50"))
        warmup = int(os.environ.get("UCCL_GIN_BENCH_WARMUP", "10"))
        if rank == 0:
            print(f"  -- bandwidth (iters={iters}, warmup={warmup}) --", flush=True)
        for nbytes in sizes:
            if nbytes > max_message_bytes:
                continue
            gbps = ctx.put_bench(peer, nbytes, iters, warmup)
            if rank == 0:
                agg = gbps * world if gbps > 0 else 0.0
                print(
                    f"  bench {nbytes:>9} B: per-rank {gbps:8.2f} GB/s  "
                    f"aggregate {agg:9.2f} GB/s",
                    flush=True,
                )

    ctx.close()

    # Reduce pass/fail across ranks via a tiny allreduce-by-min over a sentinel:
    # simplest is to let each rank print and rely on a nonzero exit on any FAIL.
    status = 0 if not failures else 2
    if rank == 0:
        print(
            f"uccl_gin put/quiet smoke {'PASS' if status == 0 else 'FAIL'} "
            f"world={world} local_world={local_world} peer_of_0={(0 + local_world) % world}"
            + (f" failures={failures}" if failures else "")
        )
    mpi_finalize()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
