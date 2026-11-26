#!/usr/bin/env python3
"""Validate complete hardware-probe runs and summarize their bandwidth rows."""

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import median


def read_run(path: Path):
    text = path.read_text()
    topology = []
    settings = None
    rows = {}
    for line in text.splitlines():
        if line.startswith("topology gpu="):
            match = re.search(
                r'gpu=(\d+) name="([^"]+)" pci=([^ ]+) numa=([^ ]+).*peers=([01]+)$',
                line,
            )
            if not match:
                raise ValueError(f"invalid GPU topology line: {path}")
            topology.append(match.groups())
        elif line.startswith("# repeats="):
            fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
            settings = (int(fields["repeats"]), int(fields["warmups"]),
                        int(fields["iterations"]))
        elif line.startswith("bandwidth "):
            fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
            name = fields["metric"]
            if name in rows:
                raise ValueError(f"duplicate metric {name}: {path}")
            values = tuple(float(fields[key]) for key in
                           ("min_gbs", "median_gbs", "max_gbs"))
            if not all(math.isfinite(value) for value in values) or not (
                0 < values[0] <= values[1] <= values[2]
            ):
                raise ValueError(f"invalid metric {name}: {path}")
            rows[name] = (
                int(fields["bytes_per_participant"]),
                int(fields["participants"]),
                int(fields["iterations"]),
                *values,
            )
    if not topology or not settings or not rows:
        raise ValueError(f"incomplete hardware profile: {path}")
    return tuple(topology), settings, rows


def summarize(paths: list[Path], output: Path) -> None:
    runs = [read_run(path) for path in paths]
    topology, settings, first = runs[0]
    for current_topology, current_settings, rows in runs[1:]:
        if (current_topology != topology or current_settings != settings or
                set(rows) != set(first)):
            raise ValueError("hardware profile topology, settings, or metrics differ")
        for name, result in rows.items():
            if result[:3] != first[name][:3]:
                raise ValueError(f"metric settings differ: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("metric", "bytes_per_participant", "participants",
                         "iterations", "runs", "repeats_per_run", "min_gbs",
                         "median_gbs", "max_gbs"))
        for name in sorted(first):
            values = [rows[name] for _, _, rows in runs]
            writer.writerow((name, *first[name][:3], len(runs), settings[0],
                             min(value[3] for value in values),
                             median(value[4] for value in values),
                             max(value[5] for value in values)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    summarize(args.runs, args.output)
