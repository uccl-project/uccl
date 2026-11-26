#!/usr/bin/env python3
"""Audit and summarize the last 20 AllReduce calls in an Nsight SQLite trace."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import mean


def union_ns(intervals):
    total, previous = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous))
        previous = max(previous, end)
    return total


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(path, kernel, size, kernels_per_call=1, aggregate_window=False):
    c = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    kernels = c.execute(
        "SELECT k.* FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
        "ON s.id=k.shortName WHERE s.value LIKE ? ORDER BY start",
        (kernel,),
    ).fetchall()
    copies = c.execute(
        "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE srcDeviceId BETWEEN 0 AND 7 "
        "AND dstDeviceId BETWEEN 0 AND 7 AND srcDeviceId!=dstDeviceId AND bytes>8 "
        "ORDER BY start"
    ).fetchall()
    result = dict(
        sqlite_sha256=file_sha(path),
        kernel=kernel,
        size=size,
        kernels_per_call=kernels_per_call,
        aggregate_window=aggregate_window,
        timed_calls=20,
        span_kind=(
            "global timed-batch span / 20"
            if aggregate_window
            else (
                "first payload copy through last phase kernel"
                if kernels_per_call > 1
                else "single collective kernel"
            )
        ),
        ranks=[],
        diagnostics=[
            dict(r)
            for r in c.execute(
                "SELECT DISTINCT severity,text FROM DIAGNOSTIC_EVENT WHERE severity>=2"
            )
        ],
    )
    windows = {}
    for rank in range(8):
        all_calls = [k for k in kernels if k["deviceId"] == rank]
        if len(all_calls) != 39 * kernels_per_call:
            raise ValueError(
                f"Expected {39 * kernels_per_call} collective kernels on GPU {rank}, got {len(all_calls)}"
            )
        rank_copies = [x for x in copies if x["deviceId"] == rank]
        if kernels_per_call > 1 and len(rank_copies) != 39 * kernels_per_call:
            raise ValueError(
                "Host-queued ring requires exactly one payload copy per phase kernel"
            )
        windows[rank] = []
        for call in range(19, 39):
            group = all_calls[call * kernels_per_call : (call + 1) * kernels_per_call]
            start = (
                rank_copies[call * kernels_per_call]["start"]
                if kernels_per_call > 1
                else group[0]["start"]
            )
            windows[rank].append(dict(start=start, end=max(k["end"] for k in group)))
    if aggregate_window:
        # NCCL's staging-copy lifetime can extend beyond one local kernel.
        # The global completed batch encloses every rank's final receive.
        lo = min(ks[0]["start"] for ks in windows.values())
        hi = max(ks[-1]["end"] for ks in windows.values())
        windows = {rank: [dict(start=lo, end=hi)] for rank in range(8)}
    divisor = 20 if aggregate_window else 1
    for rank in range(8):
        calls = []
        edges = collections.defaultdict(list)
        rank_copies = [x for x in copies if x["deviceId"] == rank]
        for k in windows[rank]:
            cc = [x for x in rank_copies if k["start"] <= x["start"] < k["end"]]
            payload = sum(x["bytes"] for x in cc)
            if payload != divisor * size * 7 // 4:
                raise ValueError(
                    f"Incomplete outgoing payload on GPU {rank}: {payload}"
                )
            if any(x["end"] > k["end"] for x in cc):
                raise ValueError(
                    "Outgoing copy extends beyond completed collective kernel"
                )
            calls.append(
                {
                    key: value / divisor
                    for key, value in dict(
                        kernel_ms=(k["end"] - k["start"]) / 1e6,
                        copy_busy_ms=union_ns((x["start"], x["end"]) for x in cc) / 1e6,
                        copy_sum_ms=sum(x["end"] - x["start"] for x in cc) / 1e6,
                        copies=len(cc),
                        payload=payload,
                    ).items()
                }
            )
            for x in cc:
                edges[x["srcDeviceId"], x["dstDeviceId"]].append(x)
        result["ranks"].append(
            dict(
                rank=rank,
                calls=calls,
                means={key: mean(row[key] for row in calls) for key in calls[0]},
                edges=[
                    dict(
                        src=src,
                        dst=dst,
                        copies=len(xs),
                        bytes_per_call=sum(x["bytes"] for x in xs) / 20,
                        ms_per_call=sum(x["end"] - x["start"] for x in xs) / 20 / 1e6,
                        GBps=sum(x["bytes"] for x in xs)
                        / sum(x["end"] - x["start"] for x in xs),
                    )
                    for (src, dst), xs in sorted(edges.items())
                ],
            )
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--kernel", required=True, help="Nsight short kernel name (SQL LIKE)"
    )
    p.add_argument("--size", type=int, default=1 << 30)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kernels-per-call", type=int, default=1)
    p.add_argument("--aggregate-window", action="store_true")
    a = p.parse_args()
    result = analyze(a.input, a.kernel, a.size, a.kernels_per_call, a.aggregate_window)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "rank0": result["ranks"][0]["means"],
                "diagnostics": result["diagnostics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
