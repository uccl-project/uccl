#!/usr/bin/env python3
"""Alternate the direct and host-multicast UCCL AllGather paths on two nodes."""

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    hosts = os.environ["HOSTS"]
    groups = ",".join(part.rsplit(":", 1)[0] + ":2"
                      for part in hosts.split(","))
    socket = os.environ["SOCKET_IFNAME"]
    mpirun = os.environ.get("MPIRUN", "mpirun")
    begin = os.environ.get("MIN_BYTES", "1024")
    end = os.environ.get("MAX_BYTES", "2097152")
    output = Path(os.environ.get("OUTPUT_DIR", ROOT / "build/host-multicast-paired"))
    output.mkdir(parents=True, exist_ok=True)
    binary = os.environ.get("HOST_MULTICAST_BIN", str(ROOT / "build/uccl_lite_multinode_host_allgather"))
    direct = ["bash", str(ROOT / "scripts/run_multinode.sh"),
              "multinode_allreduce", "--op", "allgather", "--ag-auto", "1",
              "-b", begin, "-e", end, "-w", "20", "-n", "10", "-c", "1",
              "--check-each", "1", "--gpu-timing", "0"]
    mpi = [mpirun, "--host", groups, "-np", "4", "--map-by", "ppr:2:node",
           "--bind-to", "none", "--mca", "pml", "ucx", "--mca", "coll",
           "^hcoll", "-x", f"NCCL_SOCKET_IFNAME={socket}",
           "-x", f"NCCL_IB_HCA={os.environ.get('HCA_DEVICES', 'mlx5_0,mlx5_1')}"]
    host = [binary, "-b", begin, "-e", end, "-k", "1048576",
            "-w", "20", "-n", "10", "-r", "1", "--nccl", "0",
            "--download", "sm"]
    commands = {
        "direct": direct,
        "host_auto": mpi + host,
        "host_put": mpi + ["-x", "UCX_RNDV_THRESH=0", "-x",
                             "UCX_RNDV_SCHEME=put_zcopy"] + host,
    }
    order = ("direct", "host_auto", "host_put")
    with (output / "commands.txt").open("w") as record:
        for repeat in range(3):
            for offset in range(3):
                backend = order[(repeat + offset) % 3]
                path = output / f"{backend}_run{repeat + 1}.log"
                if path.exists():
                    raise FileExistsError(path)
                cmd = commands[backend]
                record.write(f"{path.name}: {' '.join(cmd)}\n")
                record.flush()
                result = subprocess.run(cmd, env=os.environ.copy(),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        timeout=180)
                path.write_text(result.stdout)
                print(path.name, result.returncode, flush=True)
                if result.returncode:
                    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
