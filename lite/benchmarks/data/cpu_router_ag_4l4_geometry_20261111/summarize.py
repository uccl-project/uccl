#!/usr/bin/env python3
"""Validate paired AllGather geometry sweeps and the final policy replays."""

import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
COMMANDS = ("commands.json", "followup_commands.json", "block_commands.json",
            "packet_gpu_commands.json", "manual_commands.json",
            "final_full_commands.json", "final_auto_commands.json")


def write_rows(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def argument(cmd, flag, default=None):
    return cmd[cmd.index(flag) + 1] if flag in cmd else default


def sizes_for(cmd):
    begin, end = int(argument(cmd, "--begin")), int(argument(cmd, "--end"))
    sizes = [1 << power for power in range(begin.bit_length() - 1,
                                           end.bit_length())]
    if sizes[0] != begin or sizes[-1] != end:
        raise ValueError("unexpected size range")
    return sizes


def sample_rows(name):
    with (ROOT / f"{name}_samples.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired(case):
    name, cmd = case["case"], case["cmd"]
    if case.get("exit_code", 0):
        raise ValueError(f"{name}: failed command")
    if "# correctness: PASS (every invocation, all ranks)" not in (ROOT / f"{name}.log").read_text():
        raise ValueError(f"{name}: output check failed")
    sizes = sizes_for(cmd)
    rows = sample_rows(name)
    if len(rows) != len(sizes) * 2 * 3 * 10:
        raise ValueError(f"{name}: wrong sample count {len(rows)}")
    if int(argument(cmd, "--warmup")) < 5 or int(argument(cmd, "--iterations")) != 10:
        raise ValueError(f"{name}: wrong timing protocol")
    grouped = defaultdict(dict)
    placement = int("--in-place" in cmd)
    for row in rows:
        size, backend = int(row["bytes"]), row["backend"]
        repeat, call = int(row["repeat"]), int(row["iteration"])
        latency = float(row["slowest_rank_us"])
        key = (size, backend, repeat)
        if (row["collective"] != "all_gather" or size not in sizes or
                int(row["in_place"]) != placement or
                not 0 <= repeat < 3 or not 0 <= call < 10 or
                not math.isfinite(latency) or latency <= 0 or
                call in grouped[key]):
            raise ValueError(f"{name}: invalid timed call {key}, {call}")
        grouped[key][call] = latency
    result = []
    for size in sizes:
        backends = {backend for b, backend, _ in grouped if b == size}
        if len(backends) != 2 or "nccl" not in backends:
            raise ValueError(f"{name}: missing paired control {size}")
        router = next(backend for backend in backends if backend != "nccl")
        stats = {}
        for backend in ("nccl", router):
            values = []
            for repeat in range(3):
                calls = grouped[size, backend, repeat]
                if sorted(calls) != list(range(10)):
                    raise ValueError(f"{name}: missing timed calls {size}, {backend}")
                values.append(median(calls.values()))
            stats[backend] = (min(values), median(values), max(values))
        control, candidate = stats["nccl"], stats[router]
        stable = control[1] / candidate[1] >= 1.03 and candidate[2] < control[0]
        for backend in ("nccl", router):
            low, middle, high = stats[backend]
            result.append(dict(case=name, bytes=size, in_place=placement,
                               backend=backend, packet_cap_bytes=int(argument(cmd, "--large-ag-packet-bytes")),
                               transport_blocks=int(argument(cmd, "--transport-blocks", "64")),
                               gpu_progress=int("--gpu-progress" in cmd),
                               workers_per_node=int(argument(cmd, "--workers-per-node")),
                               min_us=low, median_us=middle, max_us=high,
                               nccl_over_backend=control[1] / middle,
                               stable_router_gain=int(stable) if backend == router else 0,
                               source=f"{name}_samples.csv"))
    if len(grouped) != len(sizes) * 2 * 3:
        raise ValueError(f"{name}: extra sample groups")
    return result, len(rows)


def replay(case, expected):
    name, cmd = case["case"], case["cmd"]
    if case.get("exit_code", 0):
        raise ValueError(f"{name}: failed replay")
    if "# correctness: PASS (every invocation, all ranks)" not in (ROOT / f"{name}.log").read_text():
        raise ValueError(f"{name}: replay output check failed")
    rows = sample_rows(name)
    if len(rows) != 21 * 3 * 3:
        raise ValueError(f"{name}: wrong replay sample count")
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["collective"], int(row["bytes"]))
        repeat = int(row["repeat"])
        if (key[0] not in ("all_reduce", "all_gather", "reduce_scatter") or
                key[1] not in sizes_for(cmd) or not 0 <= repeat < 3 or
                repeat in grouped[key] or
                not math.isfinite(float(row["slowest_rank_us"])) or
                float(row["slowest_rank_us"]) <= 0):
            raise ValueError(f"{name}: invalid replay call")
        grouped[key][repeat] = row["backend"]
    result = []
    for (op, size), backends in sorted(grouped.items()):
        if sorted(backends) != [0, 1, 2] or len(set(backends.values())) != 1:
            raise ValueError(f"{name}: inconsistent replay {op}, {size}")
        backend = backends[0]
        placement = int("--in-place" in cmd)
        selected = int(backend != "nccl")
        if selected != expected.get((placement, op, size), 0):
            raise ValueError(f"{name}: selection disagrees with paired calibration")
        result.append(dict(case=name, collective=op, bytes=size,
                           in_place=placement, backend=backend,
                           cpu_router_selected=selected))
    if len(result) != 63:
        raise ValueError(f"{name}: incomplete replay")
    return result


def profile_rows():
    result = []
    for name, kernel in (("nsys_router_final_1g", "dispatchTransport"),
                         ("nsys_nccl_1g", "ncclDevKernel_AllGather_RING_LL")):
        path = ROOT / f"{name}.sqlite"
        with sqlite3.connect(path) as db:
            launches = list(db.execute(
                "SELECT k.start,k.end,k.gridX,k.blockX FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON s.id=k.shortName WHERE s.value=?", (kernel,)))
        if len(launches) != 16 or len({(x[2], x[3]) for x in launches}) != 1:
            raise ValueError(f"{name}: expected four calls on four GPUs")
        values = [end - start for start, end, _, _ in launches]
        result.append(dict(report=f"{name}.nsys-rep", kernel=kernel,
                           launches=len(values),
                           actual_grid_blocks=launches[0][2],
                           threads_per_block=launches[0][3],
                           mean_kernel_us=sum(values) / len(values) / 1000))
    return result


def main():
    commands = [case for file in COMMANDS for case in json.loads((ROOT / file).read_text())]
    if len(commands) != 27 or len({c["case"] for c in commands}) != len(commands):
        raise ValueError("unexpected case count")
    summary, timed = [], 0
    for case in commands:
        if "--profile" in case["cmd"]:
            continue
        rows, count = paired(case)
        summary.extend(rows)
        timed += count
    expected = {(r["in_place"], "all_gather", r["bytes"]): r["stable_router_gain"]
                for r in summary if r["case"].startswith("ag_full_cpu_blocks248_packet4m_")
                and r["backend"] != "nccl"}
    replay_rows = [r for case in commands if "--profile" in case["cmd"]
                   for r in replay(case, expected)]
    write_rows("summary.csv", summary)
    write_rows("auto_selection.csv", replay_rows)
    write_rows("nsys_kernel_summary.csv", profile_rows())
    print(f"validated {timed} paired calls, {len(summary)} summary rows, "
          f"{len(replay_rows)} automatic decisions")


if __name__ == "__main__":
    main()
