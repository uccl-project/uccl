#!/usr/bin/env python3
"""Select the measured host multicast AllGather path through 256 KiB."""

import csv
import math
from pathlib import Path

from figure_style import ROOT


TUNED = ROOT / "benchmarks/data/nccl_tuned_2x4l4_20261111/tuned_summary.csv"
HOST = ROOT / "benchmarks/data/ucx_eager_nccl_2x4l4_20261111/summary.csv"
OUTPUT = (ROOT / "benchmarks/data/host_multicast_selected_2x4l4_20261111"
          / "selected_summary.csv")
HOST_LIMIT = 256 << 10


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    original = {int(row["bytes"]): row for row in read_csv(TUNED)
                if row["collective"] == "allgather"}
    if sorted(original) != [1 << power for power in range(10, 31)]:
        raise ValueError("incomplete original AllGather sweep")
    host = {}
    for row in read_csv(HOST):
        if not row["case"].startswith("auto_"):
            continue
        size = int(row["bytes"])
        host.setdefault(size, {})[row["backend"]] = row
    if sorted(host) != [1 << power for power in range(10, 19)]:
        raise ValueError("incomplete UCX-auto sweep")
    selected = []
    for size, source in sorted(original.items()):
        row = dict(source)
        if size <= HOST_LIMIT:
            paired = host[size]
            if set(paired) != {"host-rdma-multicast", "nccl"}:
                raise ValueError(f"missing paired host/NCCL result at {size}")
            for name, backend in (("uccl", "host-rdma-multicast"),
                                  ("nccl", "nccl")):
                chosen = paired[backend]
                for stat in ("min", "median", "max"):
                    key = f"{stat}_run_mean_us"
                    value = float(chosen[key])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"bad {name} timing at {size}")
                    row[f"{name}_{stat}_us"] = value
            row["uccl_path"] = "host-multicast-ucx-auto"
            row["nccl_config"] = ("RING+SIMPLE" if size >= 128 << 10
                                  else "default")
            row["source"] = ("../ucx_eager_nccl_2x4l4_20261111/"
                             + ("auto_128k_256k.log" if size >= 128 << 10
                                else "auto_1k_64k.log"))
            row["mpi_processes"] = 4
        else:
            row["mpi_processes"] = 8
            row["source"] = "../nccl_tuned_2x4l4_20261111/" + row["source"]
        row["speedup"] = (float(row["nccl_median_us"])
                          / float(row["uccl_median_us"]))
        selected.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    print(f"wrote {len(selected)} AllGather sizes to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
