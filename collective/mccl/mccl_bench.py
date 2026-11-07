"""Strict parsing and offline analysis of nccl-tests-style MCCL tables."""

import math
import json
import re
import statistics
from pathlib import Path


def parse_results(output):
    rows = []
    seen = set()
    for line_number, line in enumerate(output.splitlines(), 1):
        fields = line.split()
        if not fields or not fields[0].lstrip("-").isdigit():
            continue
        if len(fields) != 13:
            raise ValueError(f"unsupported result layout at line {line_number}")
        size, count = int(fields[0]), int(fields[1])
        if size < 0 or count < 0 or (size and not count):
            raise ValueError(f"negative size/count at line {line_number}")
        for mode, offset in (("out_of_place", 5), ("in_place", 9)):
            time, algbw, busbw = map(float, fields[offset : offset + 3])
            if not all(math.isfinite(x) and x >= 0 for x in (time, algbw, busbw)):
                raise ValueError(f"invalid measurement at line {line_number}")
            if size and time == 0:
                raise ValueError(f"zero duration at line {line_number}")
            wrong = fields[offset + 3]
            if wrong != "N/A" and (not wrong.isdigit() or int(wrong) != 0):
                raise ValueError(f"correctness error at line {line_number}: {wrong}")
            key = size, mode
            if size and key in seen:
                raise ValueError(f"duplicate result at line {line_number}")
            seen.add(key)
            rows.append(
                {
                    "bytes": size,
                    "count": count,
                    "dtype": fields[2],
                    "redop": fields[3],
                    "mode": mode,
                    "time_us": time,
                    "algbw_GBs": algbw,
                    "busbw_GBs": busbw,
                    "validation": "unsupported" if wrong == "N/A" else "passed",
                    "raw_line": line_number,
                }
            )
    if not any(row["bytes"] and row["validation"] == "passed" for row in rows):
        raise ValueError("no nonempty, numerically validated result")
    return rows


def size_sweep(begin, end, factor):
    def size(text):
        match = re.fullmatch(r"([1-9][0-9]*)([KMG]?)", str(text).upper())
        if not match:
            raise ValueError("size must be positive integer bytes or K/M/G")
        return int(match[1]) * (1024 ** ("", "K", "M", "G").index(match[2]))

    first, last = size(begin), size(end)
    if factor < 2 or first > last:
        raise ValueError("factor must be >=2 and min size <= max size")
    result = []
    while first <= last:
        if len(result) >= 256:
            raise ValueError("size sweep is too large")
        result.append(first)
        first *= factor
    return result


def expected_sizes(args, collective):
    sweep = size_sweep(args.min_bytes, args.max_bytes, args.factor)
    if args.coverage_file is None:
        return sweep
    with open(args.coverage_file) as stream:
        sizes = json.load(stream)[collective]
    if (
        not isinstance(sizes, list)
        or not sizes
        or any(type(x) is not int or x <= 0 for x in sizes)
        or len(set(sizes)) != len(sizes)
    ):
        raise ValueError("coverage file requires unique positive actual byte sizes")
    return sizes


def validate_coverage(rows, expected):
    actual = {row["bytes"] for row in rows if row["bytes"]}
    if actual != set(expected):
        raise ValueError(
            f"incomplete/changed size coverage: expected {expected}, got {sorted(actual)}"
        )
    for size in actual:
        if not any(
            row["bytes"] == size and row["validation"] == "passed" for row in rows
        ):
            raise ValueError(f"no numerically checked mode for size {size}")


def read_counters(root, device_port):
    device, port = device_port.split(":")
    if not device or Path(device).name != device or not port.isdigit() or int(port) < 1:
        raise ValueError("counter must be a device:port, e.g. mlx5_0:1")
    directory = Path(root) / device / "ports" / port / "counters"
    values = {
        "tx": int((directory / "port_xmit_data").read_text()),
        "rx": int((directory / "port_rcv_data").read_text()),
    }
    if min(values.values()) < 0:
        raise ValueError("negative counter")
    return {"source": str(directory), "unit": "4octets", "values": values}


def counter_delta(before, after, unit):
    multiplier = {"4octets": 4, "bytes": 1}.get(unit)
    if multiplier is None:
        raise ValueError("unknown counter unit")
    result = {}
    for direction in ("tx", "rx"):
        delta = after[direction] - before[direction]
        if delta < 0:
            raise ValueError("counter reset/wrap invalidates the sample")
        result[direction + "_bytes"] = delta * multiplier
    return result


def summarize(records):
    configurations = {
        json.dumps(
            [record.get("fixed_environment", {}), record.get("case_configuration", {})],
            sort_keys=True,
        )
        for record in records
    }
    if len(configurations) != 1:
        raise ValueError("cannot pair different experiment configurations")
    groups = {}
    for record in records:
        if record["backend"] not in ("native", "uccl"):
            raise ValueError("unknown backend")
        for row in record["rows"]:
            if row["validation"] != "passed" or not row["bytes"]:
                continue
            key = (
                record["collective"],
                row["bytes"],
                row["count"],
                row["dtype"],
                row["redop"],
                row["mode"],
            )
            samples = groups.setdefault(key, {"native": {}, "uccl": {}})
            runs = samples[record["backend"]]
            if record["run"] in runs:
                raise ValueError("duplicate run")
            duration = row["time_us"]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("invalid duration")
            runs[record["run"]] = duration
    if not groups:
        raise ValueError("no comparable results")
    result = []
    for key, samples in sorted(groups.items()):
        native, uccl = samples["native"], samples["uccl"]
        if not native or native.keys() != uccl.keys():
            raise ValueError(f"incomplete paired samples for {key}")
        change = [100 * (uccl[i] - native[i]) / native[i] for i in native]
        result.append(
            {
                "collective": key[0],
                "bytes": key[1],
                "count": key[2],
                "dtype": key[3],
                "redop": key[4],
                "mode": key[5],
                "pairs": len(native),
                "native_median_us": statistics.median(native.values()),
                "uccl_median_us": statistics.median(uccl.values()),
                "paired_latency_change_pct": statistics.median(change),
                "paired_change_range_pct": [min(change), max(change)],
                "independent_runs": len(native) >= 5,
            }
        )
    return result
