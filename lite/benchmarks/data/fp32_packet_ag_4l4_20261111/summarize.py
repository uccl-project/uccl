#!/usr/bin/env python3
"""Validate the FP32 four-L4 AllGather paired sweep and selected path."""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
SIZES = tuple(1 << power for power in range(10, 31))
SAMPLE = re.compile(
    r"^# sample,all_gather,(\d+),([a-z-]+),(\d+),(\d+),([\d.]+)$", re.M)
RESULT = re.compile(
    r"^all_gather,(\d+),([a-z-]+),([\d.]+),([\d.]+),([\d.]+)$", re.M)
PROFILE = re.compile(
    r"^profile GPU(\d) consumer ready0 ([\d.]+) copy0 ([\d.]+) "
    r"total ([\d.]+) us$", re.M)


def write_rows(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def expected_backends(size):
    if size <= 32 << 10:
        return {"peer", "pull", "push-serial", "nccl"}
    if size <= 2 << 20:
        return {"peer", "push-serial", "nccl"}
    return {"peer", "packet-router", "nccl"}


def packet_geometry(size):
    # The existing packetCapacity helper counts 16-bit storage units.
    elements = size // 8  # One GPU contributes size/4 bytes.
    rounded = lambda value: (value + 7) // 8 * 8
    capacity = min(rounded(elements),
                   min((4 << 20) // (2 * 4),
                       max((128 << 10) // (2 * 4),
                           rounded((elements + 1) // 2))))
    return capacity * 2, (elements + capacity - 1) // capacity


def main():
    content = (ROOT / "full.log").read_text()
    if ("NCCL version=22807; warmups=20 iterations=10 repeats=3" not in content
            or "# packet-router AG: FP32 bitwise copy" not in content
            or "peer AG/RS:" in content):
        raise ValueError("unexpected benchmark setup or correctness failure")
    grouped = defaultdict(dict)
    samples = []
    for size, backend, repeat, call, latency in SAMPLE.findall(content):
        size, repeat, call = int(size), int(repeat), int(call)
        latency = float(latency)
        key = (size, backend, repeat)
        if (size not in SIZES or backend not in expected_backends(size) or
                not 1 <= repeat <= 3 or not 1 <= call <= 10 or
                not math.isfinite(latency) or latency <= 0 or
                call in grouped[key]):
            raise ValueError(f"invalid timed call {key}, {call}")
        grouped[key][call] = latency
        samples.append(dict(operation="allgather", bytes=size,
                            backend=backend, repeat=repeat, call=call,
                            cuda_event_us=latency))
    if len(samples) != 2070:
        raise ValueError(f"wrong timed call count {len(samples)}")
    reported = {}
    for size, backend, middle, low, high in RESULT.findall(content):
        key = (int(size), backend)
        if key in reported:
            raise ValueError(f"duplicate result {key}")
        reported[key] = tuple(map(float, (middle, low, high)))
    candidates, selected = [], []
    for size in SIZES:
        stats = {}
        for backend in sorted(expected_backends(size)):
            run_means = []
            for repeat in range(1, 4):
                calls = grouped[size, backend, repeat]
                if sorted(calls) != list(range(1, 11)):
                    raise ValueError(f"missing calls {size}, {backend}, {repeat}")
                run_means.append(mean(calls.values()))
            actual = (median(run_means), min(run_means), max(run_means))
            if any(abs(a - b) > 0.002 for a, b in zip(actual, reported[size, backend])):
                raise ValueError(f"reported summary mismatch {size}, {backend}")
            stats[backend] = actual
            candidates.append(dict(operation="allgather", bytes=size,
                                   backend=backend, median_us=actual[0],
                                   min_us=actual[1], max_us=actual[2]))
        chosen = ("push-serial" if size <= 2 << 20 else
                  "peer" if size < 32 << 20 else "packet-router")
        if chosen not in stats or any(stats[chosen][0] > stats[other][0] + 0.002
                                     for other in stats if other != "nccl"):
            raise ValueError(f"selected path is not fastest tested UCCL at {size}")
        own, nccl = stats[chosen], stats["nccl"]
        selected.append(dict(operation="allgather", bytes=size,
                             dtype="fp32", gpus=4, placement="out-of-place",
                             uccl_path=chosen,
                             uccl_median_us=own[0], uccl_min_us=own[1],
                             uccl_max_us=own[2], nccl_median_us=nccl[0],
                             nccl_min_us=nccl[1], nccl_max_us=nccl[2],
                             nccl_version="2.28.7", nccl_config="default",
                             timing="slowest-gpu-cuda-event", source="full.log"))
    if len(grouped) != sum(len(expected_backends(size)) for size in SIZES) * 3:
        raise ValueError("extra sample groups")
    for name in ("pilot_1g.log", "tail_1008.log", "tail_large.log",
                 "default_peer_smoke.log"):
        text = (ROOT / name).read_text()
        if "peer AG/RS:" in text or "all_gather," not in text:
            raise ValueError(f"failed correctness smoke {name}")
    for name in ("tail_1008.log", "tail_large.log"):
        if ",packet-router," not in (ROOT / name).read_text():
            raise ValueError(f"packet router was not exercised in {name}")
    write_rows("samples.csv", samples)
    write_rows("candidates.csv", candidates)
    write_rows("selected.csv", selected)
    diagnostic = []
    lookup = {(int(row["bytes"]), row["backend"]): row for row in candidates}
    for mib in (4, 8, 16, 32):
        size = mib << 20
        content = (ROOT / f"profile_{mib}m.log").read_text()
        if "peer AG/RS:" in content:
            raise ValueError(f"failed profile at {mib} MiB")
        profiles = {int(gpu): tuple(map(float, (ready, copy, total)))
                    for gpu, ready, copy, total in PROFILE.findall(content)}
        if sorted(profiles) != list(range(4)):
            raise ValueError(f"missing GPU stage profile at {mib} MiB")
        packet_bytes, tiles = packet_geometry(size)
        diagnostic.append(dict(bytes=size, source_bytes=size // 4,
                               packet_bytes_per_source=packet_bytes,
                               tiles=tiles, gpu0_ready0_us=profiles[0][0],
                               gpu0_copy0_us=profiles[0][1],
                               gpu0_total_us=profiles[0][2],
                               slowest_profiled_gpu_total_us=max(p[2] for p in profiles.values()),
                               paired_router_us=lookup[size, "packet-router"]["median_us"],
                               paired_peer_us=lookup[size, "peer"]["median_us"],
                               source=f"profile_{mib}m.log"))
    write_rows("pipeline_diagnostic.csv", diagnostic)
    print(f"validated {len(samples)} timed calls and {len(selected)} selected sizes")


if __name__ == "__main__":
    main()
