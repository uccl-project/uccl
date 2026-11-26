"""Require complete checked full-range AllReduce comparisons."""

import csv
import math
from pathlib import Path
from statistics import mean, median

DATA = Path(__file__).resolve().parent
SIZES = {1 << power for power in range(10, 31)}


def main():
    rows = []
    for mode in ("hybrid", "nccl"):
        rounds = {}
        for repeat in range(1, 4 if mode == "hybrid" else 3):
            path = DATA / "raw" / f"{mode}_{repeat}.log"
            samples = {}
            for line in path.read_text().splitlines():
                fields = line.split(",")
                if len(fields) != 6:
                    continue
                if mode == "nccl" and fields[:2] == ["nccl", "warmup"]:
                    assert int(fields[5]) == 0, path
                if mode == "nccl" and fields[:2] == ["nccl", "timed"]:
                    size, value = int(fields[3]), float(fields[4])
                    assert int(fields[5]) == 0, path
                elif mode == "hybrid" and fields[:2] == ["# sample", "allreduce"]:
                    size, value = int(fields[2]), float(fields[4])
                    assert float(fields[5]) <= 1e-5, path
                else:
                    continue
                assert math.isfinite(value) and value > 0, (path, size)
                samples.setdefault(size, []).append(value)
            assert set(samples) == SIZES, path
            for size, values in sorted(samples.items()):
                assert len(values) == 10, (path, size)
                rounds.setdefault(size, []).append(mean(values))
        for size, values in rounds.items():
            latency = median(values)
            rows.append((mode, size, latency, min(values), max(values),
                         1.75 * size / latency / 1000))
    with (DATA / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("mode", "input_bytes_per_gpu", "median_us", "min_us",
                         "max_us", "bus_GBps"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
