"""Reproduce the paired SHM sweep summary from archived benchmark stdout."""
import csv
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def main():
    result = []
    for operation, collective in (("allreduce", "all_reduce"),
                                  ("allgather", "all_gather"),
                                  ("reducescatter", "reduce_scatter")):
        runs = []
        for suffix in ("a", "b"):
            with (ROOT / "raw" / f"shm_{operation}_{suffix}.csv").open() as stream:
                rows = list(csv.DictReader(line for line in stream
                                           if not line.startswith("#")))
            assert len(rows) == 26
            runs.append({(int(row["bytes"]), row["backend"] == "fallback"): row
                         for row in rows})
        for size in (1 << power for power in range(10, 23)):
            uccl = [float(run[size, False]["median_us"]) for run in runs]
            nccl = [float(run[size, True]["median_us"]) for run in runs]
            result.append(dict(collective=collective, bytes=size,
                               placement="in_place" if operation == "allreduce" else "out_of_place",
                               uccl_run1_us=uccl[0], uccl_run2_us=uccl[1],
                               uccl_median_us=median(uccl),
                               nccl_run1_us=nccl[0], nccl_run2_us=nccl[1],
                               nccl_median_us=median(nccl),
                               speedup=median(nccl) / median(uccl)))
    with (ROOT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
